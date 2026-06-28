# Implementation Guide

## 5.1 Code Structure

```
kinecmamba/
├── train.py                              # Training loop (wandb-only, layer-wise hooks)
├── configs/
│   └── pose3d/
│       └── PoseMamba_train_h36m_S.yaml   # Config (batch=4, accum=8, cosine LR)
├── lib/
│   ├── model/
│   │   ├── PoseMamba.py                  # Model definition (patch for fusion/adj)
│   │   ├── mambablocks.py                # BiSTSSMBlock (patch for forward_core)
│   │   ├── csms6s.py                     # CrossScan/CrossMerge variants
│   │   ├── loss.py                       # Loss functions
│   │   ├── DSTformer.py                  # Unchanged
│   │   ├── csm_triton.py                 # Unchanged
│   │   └── model_*.py                    # Unchanged
│   ├── data/
│   │   ├── dataset_motion_3d.py          # Data loader
│   │   └── datareader_h36m.py            # Data reader
│   └── utils/
│       ├── learning.py                   # load_backbone (register new models)
│       └── tools.py                      # Config parsing
└── kernels/
    └── selective_scan/                   # CUDA kernels (unchanged)
```

## 5.2 Changes per Experiment

### Exp 00: Baseline + Wandb + Layer-Wise Diagnostics

**Files to modify:**
- `kinecmamba/train.py` — Remove tensorboard, add layer-wise block hooks

**New code (~30 LOC):**

```python
# Layer-wise hooks during evaluate()
layer_outputs = {}
def make_layer_hook(name):
    def hook(module, input, output):
        layer_outputs[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return hook

# Register hooks on all blocks
hooks = []
for i, blk in enumerate(model.STEblocks):
    hooks.append(blk.register_forward_hook(make_layer_hook(f'ste_{i}')))
for i, blk in enumerate(model.TTEblocks):
    hooks.append(blk.register_forward_hook(make_layer_hook(f'tte_{i}')))

# After forward pass, compute per-layer MPJPE
for name, feat in layer_outputs.items():
    pred_3d = model.head(feat)
    mpjpe = torch.mean(torch.norm(pred_3d - target_3d, dim=-1)).item()
    wandb.log({f'layer_mpjpe/{name}': mpjpe})

# Remove hooks after eval
for h in hooks:
    h.remove()
```

**Wandb config logging:**
```python
wandb.config.update({
    'model': 'PoseMamba-S',
    'batch_size': cfg.BATCH_SIZE,
    'accum_steps': cfg.ACCUM_STEPS,
    'lr_scheduler': cfg.LR_SCHEDULER,
    'warmup_epochs': cfg.WARMUP_EPOCHS,
    'grad_clip_norm': cfg.GRAD_CLIP_NORM,
    'params': sum(p.numel() for p in model.parameters()),
    'effective_batch': cfg.BATCH_SIZE * cfg.ACCUM_STEPS,
})
```

---

### Exp 01: Learnable Skeleton Adjacency (H12)

**Files to modify:**
- `kinecmamba/lib/model/csms6s.py` — Add `CrossScan_learnable` and `CrossMerge_learnable`
- `kinecmamba/lib/model/mambablocks.py` — Register new forward_type `v2_learnable_adj`, add `learnable_adj` parameter

**New code (~40 LOC):**

```python
class CrossScan_learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        B, C, H, W_dim = x.shape  # H=T=243, W_dim=J=17
        # Path 0: skeleton-aware with learnable adjacency
        mixing = torch.einsum('bchw,ij->bcih', x, W)  # (B, C, H, J) @ (J, J) → (B, C, H, J)
        xs0 = (x + mixing).reshape(B, C, H * W_dim)
        # Path 1: column-major (standard)
        x_t = x.transpose(2, 3).contiguous()
        xs1 = x_t.reshape(B, C, H * W_dim)
        # Paths 2-3: flipped versions
        xs2 = torch.flip(xs0, dims=[-1])
        xs3 = torch.flip(xs1, dims=[-1])
        ctx.save_for_backward(x, W)
        return torch.stack([xs0, xs1, xs2, xs3], dim=1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, W = ctx.saved_tensors
        # grad from path 0: propagate through mixing (autograd handles W grad)
        gy = grad_outputs[0][:, 0].reshape_as(x)
        gy += grad_outputs[0][:, 2].flip(dims=[-1]).reshape_as(x)
        gW = torch.einsum('bchw,bcih->ij', x, gy)  # gradient for W
        return gy, gW

class CrossMerge_learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys):
        # Same as CrossMerge — sum all 4 paths
        B, K, C, L = ys.shape
        ys_merged = ys[:, 0] + ys[:, 1] + ys[:, 2].flip(dims=[-1]) + ys[:, 3].flip(dims=[-1])
        return ys_merged

    @staticmethod
    def backward(ctx, *grad_outputs):
        gy = grad_outputs[0]
        return gy[:, None].expand(-1, 4, -1, -1).contiguous()
```

**In BiSTSSM init:**
```python
self.learnable_adj = nn.Parameter(torch.zeros(17, 17))  # symmetric init
# Enforce symmetry: W = (W + W^T) / 2
```

---

### Exp 02: GCN-Mamba Dual-Stream (H1)

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add `GCNModule` class, modify forward pass
- `kinecmamba/lib/utils/learning.py` — Register new model variant

**New code (~80 LOC):**

```python
class GCNModule(nn.Module):
    """Lightweight spatio-temporal GCN for skeletal topology."""
    def __init__(self, in_dim, hidden_dim, num_joints=17):
        super().__init__()
        self.adj = self._build_skeleton_adjacency(num_joints)
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, T, J, C = x.shape
        x = self.gcn1(x, self.adj)   # (B, T, J, C) → (B, T, J, hidden)
        x = self.relu(x)
        x = self.gcn2(x, self.adj)   # (B, T, J, hidden) → (B, T, J, hidden)
        x = x.permute(0, 1, 3, 2)   # (B, T, hidden, J) for temporal conv
        x = self.temporal(x)         # temporal conv over joint dim
        x = x.permute(0, 1, 3, 2)   # (B, T, J, hidden)
        return x

    def _build_skeleton_adjacency(self, num_joints):
        adj = torch.zeros(num_joints, num_joints)
        edges = [(0,1),(0,4),(0,6),(1,2),(2,3),(3,5),(4,12),(12,13),
                 (13,14),(14,15),(6,8),(8,9),(9,10),(10,11)]
        for i, j in edges:
            adj[i, j] = adj[j, i] = 1  # undirected
        return adj
```

**Fusion in PoseMamba.forward:**
```python
gcn_out = self.gcn_module(x)        # parallel GCN stream
x = self.STE_forward(x)             # Mamba STE blocks
x = self.TTE_foward(x)              # Mamba TTE blocks
x = self.ST_foward(x)               # Mamba body blocks
x = self.alpha * x + (1 - self.alpha) * gcn_out  # adaptive fusion
x = self.head(x)
```

---

### Exp 03: SAMA-Style State-Level Fusion (H11)

**Files to modify:**
- `kinecmamba/lib/model/mambablocks.py` — Modify `forward_corev2` for SSI

**Modification (~50 LOC added to forward_corev2):**

```python
# In forward_corev2, after SelectiveScan but before CrossMerge:
# Apply Structure-aware State Integrator:
# Aggregate neighboring states through learnable topology weights
# This modifies the SSM state before the output is computed

# Key modification to the scan loop:
def state_level_fusion(states, topology_weights):
    # states: (B, K, D', L) — SSM hidden states for all 4 scan paths
    # topology_weights: (J, J) — learnable skeleton adjacency
    # For each time step t in the 1D scan:
    #   s_t = A * s_{t-1} + B * u_t                          (original)
    #   s_t = A * s_{t-1} + B * (u_t + W * s_{t-1}_spatial)  (SSI)
    # where s_{t-1}_spatial aggregates states of neighboring joints
    ...
```

**Full implementation:**
- Override the selective scan to add topology-weighted state aggregation
- Use the same learnable adjacency W from Exp 01 (shared parameter)
- Add motion-adaptive timescale modulation (simplified: learned per-joint dt scaling)

---

## 5.3 Experiment Branch Naming

| Branch | Hypothesis | Base |
|--------|-----------|------|
| `main` | Baseline reproduction + wandb + layer-wise | — |
| `exp01_learnable_adj` | H12 (learnable skeleton adjacency) | `main` |
| `exp02_gcn_mamba` | H1 (GCN-Mamba dual-stream) | `main` |
| `exp03_sama_fusion` | H11 (SAMA state-level fusion) | `exp02_gcn_mamba` |
| `exp04_compound` | H12+H1+H11 | `exp03_sama_fusion` |

## 5.4 Experiment Journal Template

Every experiment gets a journal at `experiments/exp<ID>/README.md`:

```markdown
## Exp <ID>: <Name>

**Status:** ⏳ Planned / ❌ Failed S3 / ✅ Merged / ❌ Reverted S4
**Branch:** exp<ID>
**Baseline:** X.X mm

### Hypothesis
<one line>

### Change
<what changed, where, LOC>

### Results
| Metric | Baseline | Ours | Δ | p |
|--------|----------|------|---|--|
| MPJPE  |          |      |   |   |

### Layer-Wise Analysis
| Block | Baseline MPJPE | Ours MPJPE | Δ |
|-------|---------------|------------|---|
| STE 0 |               |            |   |
| ...   |               |            |   |

### Analysis
<why it worked or didn't>

### Lessons
<what to repeat or avoid>
```

## 5.5 How to Run

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# Baseline with wandb (Exp 00)
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true

# Background with setsid (always use full conda python)
mkdir -p experiments/exp00_baseline && \
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/posemamba/bin/python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true \
  > experiments/exp00_baseline/train.log 2>&1 &

# Watch log
tail -f experiments/exp00_baseline/train.log
```

## 5.6 LOC Budget

| Component | LOC | File |
|-----------|-----|------|
| Layer-wise hooks | 30 | `train.py` |
| Learnable adjacency (CrossScan_learnable) | 40 | `csms6s.py` |
| Learnable adjacency (register forward_type) | 10 | `mambablocks.py` |
| GCNModule class | 50 | `PoseMamba.py` |
| GCN-Mamba fusion in forward | 20 | `PoseMamba.py` |
| SAMA state-level fusion (SSI) | 150 | `mambablocks.py` |
| W parameter + symmetry constraint | 5 | `mambablocks.py` |
| **Total** | **305** | |
