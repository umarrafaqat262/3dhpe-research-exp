# Implementation Guide

## 5.1 Code Structure

```
kinecmamba/
├── train.py                              # Training loop (grad acc, cosine LR, wandb)
├── configs/
│   └── pose3d/
│       └── PoseMamba_train_h36m_S.yaml   # Config (batch=4, accum=8, lr_scheduler=cosine)
├── lib/
│   ├── model/
│   │   ├── PoseMamba.py                  # Model definition (patch for GCN fusion)
│   │   ├── mambablocks.py                # BiSTSSMBlock (patch for stride scan)
│   │   ├── csms6s.py                     # CrossScan/CrossMerge variants
│   │   ├── loss.py                       # Loss functions (add BLCE)
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

## 5.2 Changes per Hypothesis

### H1: GCN-Mamba Dual-Stream

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add `GCNModule` class, modify forward pass
- `kinecmamba/lib/utils/learning.py` — Register new model variant

**New code (~80 LOC):**

```python
class GCNModule(nn.Module):
    """Lightweight spatio-temporal GCN for skeletal topology."""
    def __init__(self, in_dim, hidden_dim, num_joints=17):
        super().__init__()
        # Spatial adjacency matrix (17×17)
        self.adj = self._build_skeleton_adjacency(num_joints)
        # GCN layers
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # Temporal conv
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Adaptive fusion weight (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, T, J, C)
        B, T, J, C = x.shape
        x = self.gcn1(x, self.adj)  # spatial GCN
        x = self.relu(x)
        x = self.gcn2(x, self.adj)  # second spatial GCN
        x = x.permute(0, 1, 3, 2)  # (B, T, C, J) for temporal conv
        x = self.temporal(x)        # temporal conv
        x = x.permute(0, 1, 3, 2)  # (B, T, J, C)
        return x

    def _build_skeleton_adjacency(self, num_joints):
        """H36M 17-joint skeleton adjacency matrix."""
        adj = torch.zeros(num_joints, num_joints)
        edges = [
            (0, 1), (0, 4), (0, 6),   # Hip → Spine, L_Hip, R_Hip
            (1, 2), (2, 3),            # Spine → Thorax → Neck
            (3, 5),                     # Neck → Head
            (4, 12), (12, 13), (13, 14), (14, 15),  # L_arm chain
            (6, 8), (8, 9), (9, 10), (10, 11),      # R_arm chain
            (0, 16), (16, 17), (17, 18),            # L_leg chain
            (4, 19), (19, 20), (20, 21),            # R_leg chain
        ]
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1  # undirected
        return adj
```

**Fusion in PoseMamba.forward:**
```python
gcn_out = self.gcn_module(x)
x = self.STE_forward(x)
x = self.TTE_foward(x)
x = self.ST_foward(x)
x = self.alpha * x + (1 - self.alpha) * gcn_out  # adaptive fusion
x = self.head(x)
```

### H2: Structure-Aware Stride Scan

**Files to modify:**
- `kinecmamba/lib/model/csms6s.py` — Add `CrossScan_stride` and `CrossMerge_stride`
- `kinecmamba/lib/model/mambablocks.py` — Register new forward_type

**New code (~100 LOC):**

```python
# Skeleton partitions for stride scan
SKELETON_PARTITIONS = [
    [12, 13, 14, 15],     # Head group
    [0, 1, 6, 7],          # Torso group
    [2, 3, 4, 5],          # Left arm
    [8, 9, 10, 11],        # Right arm
    [16, 17, 18],          # Left leg
    [19, 20, 21],          # Right leg
]

class CrossScan_stride:
    """Stride-based scan over skeleton partitions."""
    @staticmethod
    def apply(x):
        # x: (B, C, J, T)
        B, C, J, T = x.shape
        # Reorder joints by partition, then scan within each partition
        # Returns (B, K*dirs, C, L) where K=num_partitions
        ...
```

### H3: Cosine Annealing + Warmup

**Files modified:** `kinecmamba/train.py` (already done), `configs/pose3d/PoseMamba_train_h36m_S.yaml` (already done)

**Config change:**
```yaml
lr_scheduler: cosine  # 'exponential' or 'cosine'
warmup_epochs: 5
```

### H4: Gradient Accumulation + Clipping

**Files modified:** `kinecmamba/train.py` (already done), config (already done)

**Config change:**
```yaml
accum_steps: 8
grad_clip_norm: 1.0
```

### H6: Bone-Aware Module

**Files to modify:**
- `kinecmamba/lib/model/PoseMamba.py` — Add bone embedding in input processing

**New code (~60 LOC):**

```python
class BoneEmbedding(nn.Module):
    """Compute and embed bone vectors as topological prior."""
    def __init__(self, in_dim, bone_dim=16):
        super().__init__()
        self.bone_proj = nn.Linear(3, bone_dim)  # (dx, dy, dz) per bone
        self.fusion = nn.Linear(in_dim + bone_dim, in_dim)

    def forward(self, joints):
        # joints: (B, T, J, 3)
        bones = self._compute_bones(joints)  # (B, T, num_bones, 3)
        bone_feat = self.bone_proj(bones)    # (B, T, num_bones, bone_dim)
        # Aggregate bones to joints (each joint has incident bones)
        joint_bone_feat = self._aggregate(bone_feat, joints)
        return self.fusion(torch.cat([joints, joint_bone_feat], dim=-1))
```

## 5.3 Experiment Branch Naming

| Branch | Hypothesis | Base |
|--------|-----------|------|
| `exp00_baseline` | Baseline reproduction | `main` |
| `exp01_train_opt` | H3 + H4 (cosine + grad acc) | `main` |
| `exp02_gcn_mamba` | H1 (GCN-Mamba) | `main` |
| `exp03_stride_scan` | H2 (stride scan) | `main` |
| `exp04_bone_module` | H6 (bone-aware) | `main` |
| `exp05_decoupled_st` | H5 (decoupled S-T) | `main` |
| `exp06_compound` | H1+H2+H6 | `main` |

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

### Analysis
<why it worked or didn't>

### Lessons
<what to repeat or avoid>
```

## 5.5 How to Run

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# Baseline reproduction (Exp 00)
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true

# Background with nohup (always use full conda python)
mkdir -p experiments/exp00_baseline && \
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/posemamba/bin/python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true \
  > experiments/exp00_baseline/train.log 2>&1 &

# Watch log
tail -f experiments/exp00_baseline/train.log

# Overfit test (Stage 2): modify config for 16 samples, 300 epochs
# Short validation (Stage 3): train for 30 epochs
# Full validation (Stage 4): train 3 seeds × 120 epochs
```
