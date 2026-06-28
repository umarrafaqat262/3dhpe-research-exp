# RESEARCH_PLAN.md — PoseMamba Improvement Loop
*Baseline: 41.8 mm MPJPE (PoseMamba-S, SH-detected 2D, Human3.6M P1)*

## Code Audit Findings
1. `plus_poselimbs.backward()` is straight-through estimator — **skeleton prior has zero gradient**
2. Confidence is already in data pipeline (`no_conf: True` strips it) — **A2 is 1 LOC**
3. Bone computation (`get_limb_lens`) exists in `loss.py` — **A5 reuses it, 15 LOC**
4. Layer-wise hooks already in `train.py` — **no re-implementation needed**

## Experiment Order (S Variant)

| Phase | Exp | Change | LOC | Est Δ | Feasibility |
|-------|-----|--------|-----|-------|-------------|
| 1 | A1 | Baseline repro | 0 | 41.8mm | 🔵 100% |
| 1 | A2 | Enable confidence | 1 | -0.5mm | 🔵 100% |
| 1 | A5 | Bone vectors | 15 | -0.3mm | 🔵 95% |
| 2 | A3 | Learnable adjacency (fix gradient bug) | 40 | -0.4mm | 🟢 95% |
| 3 | B2 | Head-aware GAT | 25 | -1.5mm head | 🟢 90% |
| 3 | A4 | Per-joint delta MSM | 25 | -0.3mm | 🟡 60% |
| 4 | C1 | A3+A4 combined | 55 | -0.8mm | 🟡 70% |
| 4 | B1 | HyperGCN dual-stream | 80 | -0.4mm | 🟡 60% |

**Phase 1** = Zero risk, trivial code. **Phase 2** = Bug fix. **Phase 3-4** = Conditional (pre-check required).

## Branch Strategy
```
main → exp/A1-baseline-repro → exp/A2-confidence → exp/A5-bone-vectors → exp/A3-learnable-adj ...
```
Each branch merges to main when confirmed. Delete after merge.

## Pre-Checks (before any full training)
1. Single-batch overfit: 2 clips × 100 epochs → MPJPE < 5mm
2. Gradient flow: all params have non-zero gradient after 1 backward
3. Init loss: 50-100mm (dataset variance)
4. Experiment-specific kill signals (e.g., W_adj uniform → reject A3)

## How to Run
```bash
cd /home/ubuntu/experiments
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

python -u kinecmamba/train.py \
  --config kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml \
  --checkpoint experiments/exp/A1_baseline/PoseMamba_S \
  --seed 42 --wandb true
```
