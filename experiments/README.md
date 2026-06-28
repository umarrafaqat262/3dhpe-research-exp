# KinecMamba — Experiment Plan

**Goal:** Beat PoseMamba (38.1mm P1 on H36M) with literature-backed architectural changes + novel compound architecture, targeting CVPR-level publication.

**Methodology:** One hypothesis per experiment, staged validation (S1 correctness → S2 overfit → S3 short val → S4 full ×3 seeds). Layer-wise diagnostics guide engineering decisions.

**Train config:** batch_size=4, accum_steps=8 → effective batch=32, cosine annealing (warmup=5), gradient clipping max_norm=1.0. **Wandb-only** logging.

**Comprehensive docs:**
- [`docs/01_weakness_analysis.md`](docs/01_weakness_analysis.md) — 13 architectural weaknesses with code-level analysis
- [`docs/02_literature_review.md`](docs/02_literature_review.md) — Hybrid Mamba taxonomy (3 fusion types)
- [`docs/03_hypotheses.md`](docs/03_hypotheses.md) — 12 hypotheses with falsification, novelty assessment
- [`docs/04_experimental_design.md`](docs/04_experimental_design.md) — Wandb-only, layer-wise diagnostics, power analysis
- [`docs/05_implementation.md`](docs/05_implementation.md) — Code structure, LOC budget per experiment
- [`docs/06_references.md`](docs/06_references.md) — Full bibliography (38 references)

**Key architectural discoveries:**
1. STE/TTE blocks are **identical** — the "spatial/temporal" distinction is cosmetic
2. `plus_poselimbs` has **no gradient flow** — straight-through estimator bypasses skeleton prior
3. The `indices` array is **asymmetric** — left joints blended heavily, right joints mostly untouched

---

## Phase 1: Foundation + Novel Fix

### Exp 00: Baseline + Wandb + Layer-Wise Diagnostics

| Field | Value |
|-------|-------|
| Status | ⏳ Planned |
| Model | PoseMamba-S (1.47M params) |
| Config | `configs/pose3d/PoseMamba_train_h36m_S.yaml` |
| Batch Size | 4 (accum 8 → effective 32) |
| LR Scheduler | cosine (warmup=5) |
| Epochs | 120 |
| Target | 41.8mm P1 MPJPE |
| Branch | `main` |
| Changes | Remove tensorboard, add layer-wise hooks (30 LOC) |

**Expected time:** ~17 hrs on NVIDIA L4 (23GB) — 1759 batches/epoch, ~3.4 it/s

### Exp 01: Learnable Skeleton Adjacency (H12, P0, NOVEL)

| Field | Value |
|-------|-------|
| Hypothesis | Hardcoded `plus_poselimbs` indices are asymmetric + have no gradient flow. Learnable `W ∈ ℝ^(17×17)` with proper backward is a strict improvement |
| Change | New `CrossScan_learnable`/`CrossMerge_learnable` classes (~40 LOC) |
| Est. Δ | **-0.4mm** P1 MPJPE |
| Novelty | **YES** — No Mamba-based HPE paper uses learnable skeleton adjacency within the SSM scan |
| Branch | `exp01_learnable_adj` |

### Exp 02: GCN-Mamba Dual-Stream (H1, P0)

| Field | Value |
|-------|-------|
| Hypothesis | Pure SSM ignores skeletal topology. Parallel GCN stream with adaptive fusion addresses this |
| Change | GCNModule class + fusion logic in PoseMamba.py (~80 LOC) |
| Est. Δ | **-0.9mm** P1 MPJPE |
| Evidence | **Level 5** — PoseMagic (AAAI 2025) |
| Branch | `exp02_gcn_mamba` |

### Exp 03: SAMA-Style State Fusion within Dual-Stream (H11, P1, NOVEL)

| Field | Value |
|-------|-------|
| Hypothesis | State-level fusion (modifying SSM state transition with topology) adds value beyond feature-level fusion (GCN branch). **No paper combines both** |
| Change | Structure-aware State Integrator in `forward_corev2` (~150 LOC) |
| Est. Δ | **-0.6mm** additive beyond H1 |
| Novelty | **HIGH** — First combination of state-level + feature-level fusion for Mamba HPE |
| Branch | `exp03_sama_fusion` (stacks on `exp02_gcn_mamba`) |

---

## Phase 2: Compound

### Exp 04: Full Compound (H12+H1+H11)

| Field | Value |
|-------|-------|
| Changes | No new code — merge all branches |
| Target Δ | **-1.9mm** (41.8 → **39.9mm** P1) |
| Parameters | ~1.9M (vs 6.7M for PoseMamba-L) |
| Target | Beat PoseMamba-L (38.1mm) with 3.5× fewer params |

---

## Phase 3: Incremental (Deferred)

| Exp | Hypothesis | Est. Δ | Priority |
|-----|-----------|--------|----------|
| — | Structure-aware stride scan (H2) | -0.5mm | P2 |
| — | Bone-aware module (H6) | -0.4mm | P2 |
| — | Sparse hybrid attention (H8) | -0.3mm | P2 |
| — | Decoupled S-T scans (H5) | -0.5mm | P3 |
| — | Mamba-2 SSD N=64 (H9) | -0.2mm | P3 |

---

## Leaderboard

| Exp | Model | P1 MPJPE ↓ | P-MPJPE ↓ | Params | Δ | Status |
|-----|-------|-----------|-----------|--------|---|--------|
| — | PoseMamba-S (paper) | 41.8 | — | 0.9M* | — | Published |
| — | PoseMamba-B (paper) | 40.8 | — | 3.4M | — | Published |
| — | PoseMamba-L (paper) | 38.1 | — | 6.7M | — | Published |
| — | PoseMagic (GCN+Mamba) | 40.9 | — | ~1.2M | -0.9 | AAAI 2025 |
| — | DBMambaPose | 40.5 | — | ~2.0M | -1.3 | arXiv 2025 |
| — | SasMamba | 40.2 | — | ~1.5M | -1.6 | WACV 2026 |
| — | HGMamba (HyperGCN+Mamba) | 38.65 | — | — | — | arXiv 2025 |
| — | VIMCAN (Mamba+CrossAttn+IMU) | 45.3† | — | — | — | CVPR 2026 |
| — | AGMamba (Attn-GCN+Mamba) | **SOTA** | — | — | — | SIVP 2026 |
| 00 | PoseMamba-S (ours, baseline) | — | — | 1.47M | — | ⏳ Planned |
| 01 | + Learnable Adj (H12) | — | — | 1.47M | **-0.4** | ⏳ Planned |
| 02 | + GCN-Mamba (H1) | — | — | ~1.6M | **-0.9** | ⏳ Planned |
| 03 | + SAMA Fusion (H11) | — | — | ~1.9M | **-0.6** | ⏳ Planned |
| 04 | + Compound (H12+H1+H11) | **39.9** | — | **~1.9M** | **-1.9** | **⏳ Planned** |

*PoseMamba paper claims 0.9M but actual count is 1.47M for S variant
†VIMCAN uses IMU data — not directly comparable

---

## How to Run

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# EXP 00: Baseline (cosine LR, accum=8, grad clip, wandb-only)
python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true

# Background (setsid) — always use full conda python path
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
