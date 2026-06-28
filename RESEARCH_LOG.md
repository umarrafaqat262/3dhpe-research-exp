# Research Log — PoseMamba Improvement

## Experiments

| # | Code | Experiment | Status | S MPJPE | L MPJPE | Branch |
|---|------|-----------|--------|---------|---------|--------|
| 1 | A1 | Baseline Reproduce | planned | - | - | `exp/A1-baseline-repro` |
| 2 | A2 | Confidence Input | planned | - | - | `exp/A2-confidence` |
| 3 | A5 | Bone Vectors | planned | - | - | `exp/A5-bone-vectors` |
| 4 | A3 | SSI Learnable Adj | planned | - | - | `exp/A3-ssi-learnable-adj` |
| 5 | A4 | MSM Per-joint Δ | planned | - | - | `exp/A4-msm-delta` |
| 6 | C1 | SSI+MSM Fusion | planned | - | - | `exp/C1-ssi-msm` |
| 7 | B2 | Head GAT | planned | - | - | `exp/B2-head-gat` |
| 8 | B1 | HyperGCN | planned | - | - | `exp/B1-hypergcn` |
| 9 | B3 | Scan Order | planned | - | - | `exp/B3-scan-order` |
| 10 | C2 | Full System | planned | - | - | `exp/C2-full-system` |

## Progress

### 2026-06-28 — Phase 0 Infrastructure Setup
- Fixed `learning.py` to pass `in_chans` from config
- Fixed L config with accum/warmup/cosine parameters
- Created `validate.py` (Phase 1 sanity checks)
- Created experiment scripts (sanity_check, stage3, stage4)
- Created A1 baseline config
- Pushed to `main`
