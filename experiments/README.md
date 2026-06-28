# KinecMamba — Experiment Plan

**Goal:** Beat PoseMamba (38.1mm P1 on H36M) with literature-backed architectural changes, targeting CVPR-level publication.

**Methodology:** One hypothesis per experiment, staged validation (correctness → overfit → short val → full val), full hypothesis bank in `RESEARCH_PLAN.md`.

**Train config:** batch_size=4, accum_steps=8 → effective batch=32 (matches paper), gradient clipping max_norm=1.0.

---

## Exp 00: Baseline Reproduction

| Field | Value |
|-------|-------|
| Status | ⏳ Planned |
| Model | PoseMamba-S (1.47M params, paper claims 0.9M) |
| Config | `configs/pose3d/PoseMamba_train_h36m_S.yaml` |
| Batch Size | 4 (accum 8 → effective 32) |
| LR Scheduler | exponential (γ=0.99, per paper) |
| Epochs | 120 |
| Target | 41.8mm P1 MPJPE |
| Branch | `main` |
| Command | See below |

**Expected time:** ~17 hrs on NVIDIA L4 (23GB) — 1759 batches/epoch, ~3.4 it/s

---

## Exp 01: Training Optimization (P3+P4)

| Field | Value |
|-------|-------|
| Hypothesis | Cosine annealing + warmup + gradient clipping improves over exp decay |
| Changes | `lr_scheduler: cosine`, `warmup_epochs: 5`, `grad_clip_norm: 1.0`, `accum_steps: 8` |
| Est. Δ | -0.4mm P1 MPJPE |
| Priority | **P1** (free lunch) |

---

## Exp 02: GCN-Mamba Dual-Stream (P0)

**Hypothesis:** PoseMamba's accuracy is bottlenecked by Mamba's inability to model local joint graph topology. Adding a parallel GCN stream with adaptive fusion (PoseMagic-style, AAAI 2025) will improve MPJPE by ≥0.9mm.

**Intervention:** Add a lightweight spatio-temporal GCN branch that:
- Spatial GCN: models skeletal adjacency (17 joints, pre-defined skeleton graph)
- Temporal GCN: 1D conv across time
- Adaptive fusion: learnable α to blend Mamba + GCN outputs

**Expected Δ:** -0.9mm P1 MPJPE

**Validation plan:**
| Stage | What | Time | Pass/Fail |
|-------|------|------|-----------|
| 1 | Correctness (CPU, 5min) | 5 min | Loss decreases, no NaN |
| 2 | Overfit (16 samples, 300 epochs) | 10 min | Near-zero loss |
| 3 | Short val (25% epochs) | ~2.5 hrs | Δ < -0.2mm → Stage 4; Δ > +0.1mm → reject |
| 4 | Full val (120 epochs × 3 seeds) | ~36 hrs | p < 0.05 AND Δ < -0.1mm → merge |

---

## Exp 03: Structure-Aware Stride Scan (P1)

**Hypothesis:** Flat 4-direction scan destroys joint adjacency. Stride-based scanning over skeleton topology (SasMamba-style, WACV 2026) preserves local structure.

**Intervention:** Replace `CrossScan` with grouped stride scan over skeleton partitions (head, torso, arms, legs).

**Expected Δ:** -0.5mm P1 MPJPE

---

## Exp 04: Bone-Aware Module (P2)

**Hypothesis:** Bone vectors (direction + length) provide stronger spatial inductive bias for Mamba (Mamba-Driven Topology Fusion-style).

**Intervention:** Compute bone vectors from joints, fuse via Bone-Joint Fusion Embedding before SSM.

**Expected Δ:** -0.4mm P1 MPJPE

---

## Exp 05: Compound (P0+P1+P2)

**Hypothesis:** Combining all architectural interventions yields additive improvement.

**Est. cumulative Δ:** -1.5mm P1 MPJPE (41.8 → 40.3mm)

---

## Exp 06: Hybrid SSM-Attention (P7)

**Hypothesis:** Adding 2 sparse attention layers to SSM pipeline improves long-range modeling.

**Est. Δ:** -0.3mm P1 MPJPE

---

## Leaderboard

| Exp | Model | P1 MPJPE ↓ | P-MPJPE ↓ | Params | Δ | Status |
|-----|-------|-----------|-----------|--------|---|--------|
| — | PoseMamba-S (paper) | 41.8 | — | 0.9M | — | Published |
| — | PoseMamba-B (paper) | 40.8 | — | 3.4M | — | Published |
| — | PoseMamba-L (paper) | 38.1 | — | 6.7M | — | Published |
| — | PoseMagic (Mamba+GCN) | 40.9 | — | ~1.2M | -0.9 | AAAI 2025 |
| — | DBMambaPose | 40.5 | — | ~2.0M | -1.3 | arXiv 2025 |
| — | SasMamba | 40.2 | — | ~1.5M | -1.6 | WACV 2026 |
| 00 | PoseMamba-S (ours) | — | — | — | — | ⏳ Planned |
| 01 | + Training Opt | — | — | — | — | ⏳ Planned |
| 02 | + GCN-Mamba | — | — | — | — | ⏳ Planned |
| 03 | + Stride Scan | — | — | — | — | ⏳ Planned |
| 04 | + Bone Module | — | — | — | — | ⏳ Planned |
| 05 | + Compound | — | — | — | — | ⏳ Planned |
| 06 | + Hybrid Attention | — | — | — | — | ⏳ Planned |

---

## How to Run

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# EXP 00: Baseline (exponential LR, batch=4, accum=8, effective batch=32)
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0

# With wandb
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true

# EXP 01: Training Optimization (cosine LR + warmup + grad clip)
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp01_train_opt/PoseMamba_S \
  --seed 0 --wandb true

# Background (nohup) — always use full conda python path
mkdir -p experiments/exp00_baseline && \
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/posemamba/bin/python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true \
  > experiments/exp00_baseline/train.log 2>&1 &
```
