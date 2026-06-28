# KinecMamba — Experiment Plan

**Goal:** Beat PoseMamba (38.1mm P1 on H36M) with literature-backed architectural changes, targeting CVPR-level publication.

**Methodology:** One hypothesis per experiment, staged validation (S1 correctness → S2 overfit → S3 short val → S4 full ×3 seeds), full hypothesis bank in `docs/03_hypotheses.md`.

**Train config:** batch_size=4, accum_steps=8 → effective batch=32, cosine annealing (warmup=5), gradient clipping max_norm=1.0.

**Comprehensive docs:**
- [`docs/01_weakness_analysis.md`](docs/01_weakness_analysis.md) — Weakness-to-fix mapping
- [`docs/02_literature_review.md`](docs/02_literature_review.md) — Full literature sweep with hybrid taxonomy
- [`docs/03_hypotheses.md`](docs/03_hypotheses.md) — 11 hypotheses (H1–H11) with falsification
- [`docs/04_experimental_design.md`](docs/04_experimental_design.md) — Protocol, power analysis, statistics
- [`docs/05_implementation.md`](docs/05_implementation.md) — Code structure map, LOC estimates
- [`docs/06_references.md`](docs/06_references.md) — Full bibliography (38 references)

---

## Exp 00: Baseline Reproduction

> **NOTE:** The `nohup` session issue caused 3 silent background failures — foreground training verified OK.
> Use `setsid` for background runs (see below).

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

## Exp 06: Hybrid SSM-Attention (H8)

**Hypothesis:** Adding 2 sparse attention layers to SSM pipeline improves long-range modeling (inspired by VIMCAN CVPR 2026, Jamba-1.5, Nemotron-H).

**Est. Δ:** -0.3mm P1 MPJPE

---

## Exp 07: SAMA-Style State-Level Fusion (H11)

**Hypothesis:** Feature-level dual-stream fusion (H1) is post-hoc. State-level fusion — modifying the SSM state transition to incorporate joint topology — yields additional improvement beyond feature-level fusion alone.

**Mechanism:** SAMA-style Structure-aware State Integrator: before state update, aggregate neighboring joint states via learned adjacency weights.

**Est. Δ:** -0.6mm P1 MPJPE (combined with H1: up to -1.5mm)

**Novelty:** No paper combines state-level (SAMA) + feature-level (PoseMagic) fusion. Novel architecture if both H1 and H11 confirm.

---

## Exp 08: RoPE + QKNorm (H10)

**Hypothesis:** SSM lacks explicit position encoding. RoPE + QKNorm (Mamba-3 style) improves order awareness.

**Est. Δ:** -0.2mm P1 MPJPE

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
| — | HGMamba (HyperGCN+Mamba) | 38.65 | — | — | — | arXiv 2025 |
| — | VIMCAN (Mamba+CrossAttn+IMU) | 45.3* | — | — | — | CVPR 2026 |
| — | PoseMamba-L (paper) | 38.1 | — | 6.7M | — | Published |
| — | AGMamba (Attention-GCN+Mamba) | **SOTA** | — | — | — | SIVP 2026 |
| 00 | PoseMamba-S (ours) | — | — | 1.47M | — | ⏳ Training |
| 01 | + Training Opt | — | — | — | — | ⏳ Planned |
| 02 | + GCN-Mamba (H1) | — | — | — | — | ⏳ Planned |
| 03 | + Stride Scan (H2) | — | — | — | — | ⏳ Planned |
| 04 | + Bone Module (H6) | — | — | — | — | ⏳ Planned |
| 05 | + Compound (H1+H2+H6) | — | — | — | — | ⏳ Planned |
| 06 | + Hybrid Attention (H8) | — | — | — | — | ⏳ Planned |
| 07 | + SAMA Fusion (H11) | — | — | — | — | ⏳ Planned |
| 08 | + RoPE+QKNorm (H10) | — | — | — | — | ⏳ Planned |

*VIMCAN uses IMU data — not directly comparable

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

# Background (setsid) — always use full conda python path
# NOTE: nohup doesn't fully detach from session — use setsid instead
mkdir -p experiments/exp00_baseline && \
setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/ubuntu/miniforge3/envs/posemamba/bin/python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp00_baseline/PoseMamba_S \
  --seed 0 --wandb true \
  > experiments/exp00_baseline/train.log 2>&1 &
```
