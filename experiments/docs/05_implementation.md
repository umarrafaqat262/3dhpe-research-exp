# Implementation Guide
*Updated: 2026-06-28 — Refined LOC estimates, corrected A2 (1 LOC not 5), added Phase 0 pre-check scripts, updated branch strategy*

## Changelog
- **2026-06-28:** A2 LOC revised from 5→1 (confidence already in pipeline). 
  A5 LOC revised from 60→15 (reuses `get_limb_lens` from loss.py). 
  Layer-wise hooks section removed (already implemented in train.py). 
  Added Phase 0 pre-check scripts. Added branch naming convention with experiment IDs.

---

## 5.1 Code Structure

```
kinecmamba/
├── train.py                              # Training loop (wandb-only, layer-wise hooks ALREADY IMPLEMENTED)
├── configs/
│   └── pose3d/
│       ├── PoseMamba_train_h36m_S.yaml   # S config (cosine, accum=8, warmup=5) — REFERENCE ✅
│       ├── PoseMamba_train_h36m_B.yaml   # B config — NEEDS UPGRADE (cosine, accum, warmup)
│       └── PoseMamba_train_h36m_L.yaml   # L config — NEEDS UPGRADE (cosine, accum, warmup)
├── lib/
│   ├── model/
│   │   ├── PoseMamba.py                  # Model definition (patch for in_chans, fusion, learnable adj)
│   │   ├── mambablocks.py                # BiSTSSMBlock (patch for forward_type, learnable adj param)
│   │   ├── csms6s.py                     # CrossScan/CrossMerge variants (+ learnable variants)
│   │   ├── loss.py                       # Loss functions (has get_limb_lens, get_angles — USE for A5)
│   │   ├── DSTformer.py                  # Unchanged
│   │   └── csm_triton.py                 # Unchanged
│   ├── data/
│   │   ├── dataset_motion_3d.py          # Data loader (confidence already in motion_2d tensor)
│   │   └── datareader_h36m.py            # Data reader (confidence already read at lines 45-57)
│   └── utils/
│       ├── learning.py                   # load_backbone (register new model variants)
│       └── tools.py                      # Config parsing
└── kernels/
    └── selective_scan/                   # CUDA kernels (unchanged)
```

---

## 5.2 Changes per Experiment (Updated with Corrected LOC)

### EXP A1: Baseline + Config Harmonization (0 LOC, Config Only)

**Files to modify:**
- `kinecmamba/configs/pose3d/PoseMamba_train_h36m_L.yaml` — Add missing training params

**What to add to L and B configs:**
```yaml
accum_steps: 8
lr_scheduler: cosine  # changed from exponential
warmup_epochs: 5
grad_clip_norm: 1.0
```

**Existing code available:**
- Layer-wise hooks: `train.py:72-85` (registration), `train.py:192-211` (computation) — **already implemented**
- Wandb logging: `train.py:312-325` (init), `train.py:496-513` (per-epoch log)

**No new code needed for baseline.** Only config fixes.

---

### EXP A2: Confidence Score as 3rd Input Channel (1 LOC, 🔵 100%)

**What to change:**

1. **`PoseMamba.py` line 38:** `in_chans=2` → `in_chans=3`

2. **Config:** Set `no_conf: False` (line 37 of config)

**Why this works:** The data pipeline already provides confidence:
- `DataReaderH36M.__init__` has `read_confidence=True` (default)
- `DataReaderH36M.read_2d()` lines 45-57: reads `confidence` from dataset, normalizes, concatenates as 3rd channel → `[N, 17, 3]`
- `MotionDataset3D.__getitem__` returns `motion_file["data_input"]` → `[T, J, 3]`
- `train.py:224`: `if args.no_conf: batch_input = batch_input[:, :, :, :2]` — this is what strips it

**LOC breakdown:**
- `PoseMamba.py`: change `in_chans=2→3` — 1 character changed
- Config: `no_conf: True → False` — 1 config flag
- **Total: literally 1 LOC**

---

### EXP A3: Learnable Skeleton Adjacency (40 LOC, 🟢 95%)

**Files to modify:**
- `kinecmamba/lib/model/csms6s.py` — Add `CrossScan_learnable` and `CrossMerge_learnable`
- `kinecmamba/lib/model/mambablocks.py` — Register new `forward_type='v2_learnable_adj'`, add `learnable_adj` parameter

**New CrossScan_learnable:**
```python
class CrossScan_learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        B, C, H, W_dim = x.shape
        # Path 0: skeleton-aware with learnable adjacency (proper autograd)
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
        gy = grad_outputs[0][:, 0].reshape_as(x)
        gy += grad_outputs[0][:, 2].flip(dims=[-1]).reshape_as(x)
        gW = torch.einsum('bchw,bcih->ij', x, gy)
        return gy, gW
```

**In BiSTSSM init:**
```python
self.learnable_adj = nn.Parameter(torch.zeros(17, 17))
# Enforce symmetry: W = (W + W^T) / 2
```

**Forward_type registration in FORWARD_TYPES dict (mambablocks.py line 312):**
```python
v2_learnable_adj=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
    CrossScan=CrossScan_learnable, SelectiveScan=SelectiveScanCore,
    CrossMerge=CrossMerge_learnable),
```

**LOC breakdown:**
- `CrossScan_learnable` class: ~25 LOC
- `CrossMerge_learnable` class: ~10 LOC
- Registration + parameter: ~5 LOC
- **Total: ~40 LOC**

---

### EXP A4: Per-Joint Timescale Modulator (25 LOC, 🟡 60%)

**Files to modify:**
- `kinecmamba/lib/model/mambablocks.py` — Modify `forward_corev2` delta computation

**Key challenge:** The SSM operates on 2D (T×J) flattened to 1D. Delta is computed per-channel in `dts = F.conv1d(...)`. Making it per-joint requires:
1. Keep joint identity through the flatten step
2. Compute motion magnitude per-joint: `motion_j = ||x_t - x_{t-1}||` over temporal dimension
3. Modulate delta: `delta_j = delta_global * softplus(conv1d(motion_j))`

**Simplified approach (if full MSM is too complex):**
```python
# Learnable per-joint scalar multiplier on global delta
per_joint_dt = nn.Parameter(torch.ones(17))  # 17 learnable scalars
# Applied as: delta_j = delta_global * per_joint_dt[j]
```

This simpler version is much easier to integrate (just multiply delta by a per-joint scalar) but captures the same hypothesis. If it works, the full MSM with motion dependence can be Phase 4.

**LOC breakdown:**
- Full MSM with motion conv: ~25 LOC
- Simplified per-joint scalar: ~5 LOC (recommended for initial test)

---

### EXP A5: Bone Vector Auxiliary Input (15 LOC, 🔵 95%)

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add bone computation in `forward()`, change `in_chans`

**Source code to reuse from `loss.py`:**
```python
# loss.py:100-114 — already computes bone lengths
def get_limb_lens(x):
    limbs_id = [[0,1], [1,2], [2,3], [0,4], [4,5], [5,6],
                [0,7], [7,8], [8,9], [9,10], [8,11], [11,12], [12,13],
                [8,14], [14,15], [15,16]]
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:] - limbs[:,:,:,1,:]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens  # [B, T, 16]
```

**Implementation:**
```python
# In PoseMamba.forward():
bones = x[:, :, CHILD_JOINTS, :2] - x[:, :, PARENT_JOINTS, :2]  # [B, T, 16, 2]
bone_len = torch.norm(bones, dim=-1, keepdim=True)                # [B, T, 16, 1]
bone_dir = bones / (bone_len + 1e-8)                              # [B, T, 16, 2]
# Broadcast to 17 joints (repeat at parent joint):
bone_len_j = bone_len.gather(2, parent_idx)                       # [B, T, 17, 1]
bone_dir_j = bone_dir.gather(2, parent_idx)                       # [B, T, 17, 2]
x = torch.cat([x, bone_len_j, bone_dir_j], dim=-1)               # [B, T, 17, 5]
```

**Parent joint mapping:** `PARENT_JOINTS = [0,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]`

**LOC breakdown:**
- Bone computation in forward: ~10 LOC
- `in_chans=2→5`: 1 character change
- Parent joint mapping constant: ~4 LOC
- **Total: ~15 LOC**

---

### EXP B2: Head-Aware Auxiliary Branch (25 LOC, 🟢 90%)

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add `HeadAwareBranch` class, modify forward and loss

```python
class HeadAwareBranch(nn.Module):
    def __init__(self, dm, heads=4):
        super().__init__()
        self.gat1 = GATConv(dm, dm // heads, heads=heads)
        self.gat2 = GATConv(dm, dm, heads=1)
        self.head_joints = [0, 7, 8, 9, 10]

    def forward(self, Z):  # Z: [T x J x dm]
        Z_head = Z[:, self.head_joints, :]
        Z_head = self.gat1(Z_head, head_edge_index)
        Z_head = self.gat2(Z_head, head_edge_index)
        return Z_head
```

**No GAT dependency in current codebase.** Need to add `torch_geometric` or implement a simple GAT manually (2 layers with self-attention over 5 nodes — trivial).

---

### EXP B1: HGMamba Dual-Stream (80 LOC, 🟡 60%)

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add `GCNModule`, modify forward

**LOC breakdown:**
- `GCNModule` class: ~50 LOC
- Fusion logic in forward: ~20 LOC
- Model registration: ~10 LOC
- **Total: ~80 LOC**

---

### EXP B3: Scan Order Grid Search (0 LOC, Config Only)

**No code changes.** 8 config files with different `forward_type` settings:
- O1: `v2_plus_poselimbs` (original)
- O2: `v2` (standard — no skeleton prior)
- O3-O8: various `v2_fs_ft`, `v2_fs_bt`, `v2_bs_ft`, `v2_bs_bt`, `Ab_1direction`, `Ab_2direction`

All forward types are already defined in `mambablocks.py:325-339`.

---

## 5.3 Experiment Branch Naming

| Branch | Experiment | Hypothesis | Base | Est. LOC |
|--------|-----------|-----------|------|----------|
| `main` | — | Baseline + config fix | — | 0 |
| `exp/A1-baseline-repro` | A1 | Baseline reproduction | main | 0 |
| `exp/A2-confidence-input` | A2 | Confidence score | main | **1** |
| `exp/A5-bone-vectors` | A5 | Bone vectors | main | **15** |
| `exp/A3-SSI-learnable-adjacency` | A3 | Learnable skeleton adjacency | main | **40** |
| `exp/B2-head-aware-GAT` | B2 | Head GAT branch | main | **25** |
| `exp/A4-MSM-timescale` | A4 | Per-joint delta | main | **25** |
| `exp/C1-SSI-MSM-combined` | C1 | A3 + A4 | A3+A4 | **55** |
| `exp/B1-hypergcn-dual-stream` | B1 | HyperGCN + Mamba | A3 | **80** |
| `exp/B3-scan-order-search` | B3 | Scan order grid search | main | **0** |
| `exp/A6-auglift-uadd` | A6 | AugLift depth | A2 | **10+pre** |
| `exp/C2-full-combined` | C2 | All P0 + C1 | A2+C1 | **60** |

**Merge strategy:** Each confirmed experiment merges into `main`. Branches are deleted after merge.

---

## 5.4 Experiment Journal Template

Every experiment creates `experiments/<exp_id>/journal.md`:

```markdown
## Exp <ID>: <Name>

**Status:** ✅ Merged / ❌ Reverted / ❌ Falsified at Phase 0
**Branch:** exp/<ID>
**Baseline:** X.X mm

### Hypothesis
<one line>

### Phase 0 Pre-Check Results
| Check | Result | Verdict |
|-------|--------|---------|
| Forward pass | PASS/FAIL | — |
| Init loss | XX mm | OK/FAIL |
| Gradient flow | PASS/FAIL | — |
| Kill signal | <value> | PASS/KILL |

### Change
<what changed, where, LOC>

### Results
| Metric | Baseline | Ours | Δ | p |
|--------|----------|------|---|--|
| MPJPE  |          |      |   |   |

### Analysis
<why it worked or didn't>

### Lessons
<what to repeat or avoid>
```

---

## 5.5 How to Run (Updated with Phase 0)

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# == Phase 0: Pre-check (every experiment) ==
# 1. Single-batch overfit test
python -c "
import torch
from lib.model.PoseMamba import PoseMamba
model = PoseMamba(num_frame=243, embed_dim_ratio=64, depth=10)
x = torch.randn(1, 243, 17, 3)  # with confidence
out = model(x)
loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
loss.backward()
# Check gradients
for name, p in model.named_parameters():
    if p.grad is None or p.grad.abs().sum() == 0:
        print(f'DEAD PARAM: {name}')
print(f'Init loss: {loss.item():.2f} (expected: 50-100mm equivalent)')
print('Gradient check: PASS' if all(p.grad is not None for p in model.parameters()) else 'FAIL')
"

# == A3-specific: Check adjacency matrix ==
python -c "
# After 1 training epoch:
W = model.module.learnable_adj.softmax(dim=-1)  # or model.learnable_adj
print(f'Mean adj weight: {W.mean().item():.4f} (expected: ~0.06 = 1/17)')
print(f'Max adj weight: {W.max().item():.4f}')
if abs(W.mean().item() - 1/17) < 0.01:
    print('KILL: adjacency collapsed to uniform')
"

# == A4-specific: Check delta diversity ==
python -c "
# After 1 training epoch:
delta = model.module.ste_blocks[0].op.dt_projs_bias
print(f'Delta std: {delta.std().item():.4f}')
if delta.std().item() < 0.01:
    print('KILL: no per-joint delta specialization')
"

# == Background run (setsid) ==
mkdir -p experiments/exp_A2_confidence && \
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/posemamba/bin/python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp_A2_confidence/PoseMamba_S \
  --seed 42 --wandb true \
  > experiments/exp_A2_confidence/train.log 2>&1 &

# Watch log
tail -f experiments/exp_A2_confidence/train.log
```

---

## 5.6 LOC Budget (Corrected)

| Experiment | Previous Est. | Actual | Reason for Difference |
|-----------|--------------|--------|----------------------|
| A1 (baseline) | 30 (hooks) | **0** | Hooks already implemented |
| A2 (confidence) | 5 | **1** | Already in data pipeline |
| A3 (learnable adj) | 40 | **40** | Same as estimated |
| A4 (MSM delta) | 25 | **25** (or 5 for simplified) | Same |
| A5 (bone vectors) | 60 | **15** | Reuses get_limb_lens |
| B2 (head GAT) | 25 | **25** | Same |
| B1 (GCN-Mamba) | 80 | **80** | Same |
| B3 (scan order) | — | **0** | Config only |
| A6 (AugLift) | 10 | **10** | Same |
| C1 (A3+A4) | 55 | **55** | Merge only |
| C2 (full) | 60 | **60** | Merge only |
| **Total** | **~390** | **~316** | **19% reduction** |
