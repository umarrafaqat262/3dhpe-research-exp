# KinecMamba — Experiment Plan

**Goal:** Beat PoseMamba (38.1mm P1 on H36M) with literature-backed architectural changes + novel compound architecture, targeting CVPR-level publication.

**Methodology:** One hypothesis per experiment, staged validation (Phase 0 pre-check → S1 correctness → S2 overfit → S3 short val → S4 full ×3 seeds). Fast-kill signals prevent wasted GPU on doomed experiments.

**Critical code audit findings (2026-06-28):**
1. `CrossScan_plus_poselimbs.backward()` is broken — skeleton prior has zero gradient. **Bug fix first.**
2. Confidence is already in data pipeline — `no_conf: True` strips it. **A2 is 1 LOC.**
3. L config missing accum/warmup/cosine — must fix before baseline.
4. Layer-wise hooks already implemented in `train.py`.
5. Bone computation (`get_limb_lens`) exists in `loss.py` — A5 reuses it.

**Feasibility:** 🔵 100% = zero risk | 🟢 90-95% = trivial | 🟡 50-70% = needs pre-check | 🔴 <50% = speculative

---

## Phase 1: Foundation + Zero-Risk Experiments

### EXP A1: Baseline Reproduction + Config Harmonization

| Field | Value |
|-------|-------|
| Status | ⏳ Planned |
| Model | PoseMamba-S (1.47M) + PoseMamba-L (6.7M) |
| Config | `PoseMamba_train_h36m_S.yaml` + upgraded `L.yaml` |
| **Fix** | L config needs `accum_steps: 8, warmup: 5, cosine LR, grad_clip: 1.0` |
| Target | S: 41.8mm, L: 38.1mm ±0.5mm |
| Branch | `exp/A1-baseline-repro` |
| Feasibility | 🔵 100% |

### EXP A2: Confidence Score as 3rd Input Channel (1 LOC)

| Field | Value |
|-------|-------|
| Hypothesis | Confidence score `c ∈ [0,1]` is already in data. Setting `no_conf: False` + `in_chans=3` reduces MPJPE ≥ 0.3mm |
| Change | 1 LOC: `in_chans=2→3` in `PoseMamba.py` + config `no_conf: False` |
| Est. Δ | **-0.5mm** P1 MPJPE |
| Feasibility | 🔵 **100%** — confidence already flowing through pipeline, being actively stripped |
| Branch | `exp/A2-confidence-input` |

### EXP A5: Bone Vector Auxiliary Input (15 LOC)

| Field | Value |
|-------|-------|
| Hypothesis | Bone direction/length extends input representation. `get_limb_lens` already exists in `loss.py` |
| Change | Compute 16 bone vectors → broadcast to 17 joints → concat with (x,y) → `in_chans=2→5` |
| Est. Δ | **-0.3 to -0.5mm** P1 MPJPE |
| Feasibility | 🔵 **95%** — reuses existing limb computation code |
| Branch | `exp/A5-bone-vectors` |

---

## Phase 2: Bug Fix + Proven Architecture Changes

### EXP A3: Learnable Skeleton Adjacency (40 LOC)

| Field | Value |
|-------|-------|
| Hypothesis | Replace broken `plus_poselimbs` (no gradient flow, asymmetric indices) with learnable `W ∈ ℝ^(17×17)` |
| Change | New `CrossScan_learnable`/`CrossMerge_learnable` classes, proper `einsum` backward |
| Est. Δ | **-0.4 to -0.9mm** P1 MPJPE |
| Feasibility | 🟢 **95%** — bug fix with proven benefit |
| Kill signal | `W.softmax(dim=-1).mean() ≈ 1/17` → uniform, no learning |
| Branch | `exp/A3-SSI-learnable-adjacency` |

---

## Phase 3: Conditional Experiments

### EXP B2: Head-Aware GAT (25 LOC)

| Field | Value |
|-------|-------|
| Hypothesis | GAT on {0,7,8,9,10} subgraph with weighted loss reduces head/neck MPJPE ≥ 1.5mm |
| Change | `HeadAwareBranch` class, aux loss |
| Est. Δ | **-1.5mm** head/neck |
| Feasibility | 🟢 **90%** — standalone, independent of SSM |
| Kill signal | `λ_head * head_loss / total_loss < 0.01` |
| Branch | `exp/B2-head-aware-GAT` |

### EXP A4: Per-Joint Timescale Modulator (MSM, 25 LOC)

| Field | Value |
|-------|-------|
| Hypothesis | Per-joint Δ from local motion magnitude reduces MPJPE ≥ 0.3mm |
| Change | Modify SSM delta computation to be joint-aware in `mambablocks.py` |
| Est. Δ | **-0.3 to -0.5mm** |
| Feasibility | 🟡 **60%** — SSM operates on 2D flattened scan; disentangling joint dim is non-trivial |
| Kill signal | `delta_j.std() ≈ 0` after 1 epoch → no specialization |
| Branch | `exp/A4-MSM-timescale` |

### EXP C1: SSI + MSM Combined (55 LOC)

| Field | Value |
|-------|-------|
| Prerequisite | A3 + A4 both Confirmed |
| Change | Merge A3+A4 branches |
| Est. Δ | **-0.8 to -1.2mm** |
| Feasibility | 🟡 70% |
| Branch | `exp/C1-SSI-MSM-combined` |

---

## Phase 4: Speculative

| Exp | Hypothesis | Est. Δ | Feasibility | Kill Signal | Branch |
|-----|-----------|--------|-------------|-------------|--------|
| B1 | HyperGCN dual-stream (80 LOC) | -0.4mm | 🟡 60% | α → 0 | `exp/B1-hypergcn-dual-stream` |
| B3 | Scan order grid search (config) | -0.3mm | 🔵 100% | Bottom 6 eliminated at Phase 2 | `exp/B3-scan-order-search` |
| A6 | AugLift depth UADD | -0.3mm ID | 🟡 50% | depth-GT Z corr > 0.95 | `exp/A6-auglift-uadd` |
| C2 | A2 + C1 combined | -1.0mm | 🟢 85% | Depends on A2+C1 | `exp/C2-full-combined` |

---

## Leaderboard

| Exp | Model | P1 MPJPE ↓ | Δ | Feasibility | Status |
|-----|-------|-----------|---|-------------|--------|
| — | PoseMamba-S (paper) | 41.8 | — | — | Published |
| — | PoseMamba-L (paper) | 38.1 | — | — | Published |
| — | PoseMagic (GCN+Mamba) | 40.9 | -0.9 | — | AAAI 2025 |
| — | SasMamba | 40.2 | -1.6 | — | WACV 2026 |
| — | HGMamba | 38.65 | — | — | arXiv 2025 |
| **A1** | **Baseline (ours)** | **TBD** | **0.0** | 🔵 100% | ⏳ Planned |
| **A2** | **+ Confidence (1 LOC)** | **TBD** | **-0.5** | 🔵 100% | ⏳ Planned |
| **A5** | **+ Bone vectors** | **TBD** | **-0.3** | 🔵 95% | ⏳ Planned |
| **A3** | **+ Learnable adj** | **TBD** | **-0.4** | 🟢 95% | ⏳ Planned |
| **B2** | **+ Head GAT** | **TBD** | **-1.5mm head** | 🟢 90% | ⏳ Planned |
| **A4** | **+ MSM delta** | **TBD** | **-0.3** | 🟡 60% | ⏳ Planned |
| **C1** | **+ SSI+MSM** | **TBD** | **-0.8** | 🟡 70% | ⏳ Planned |
| **Target** | **Full compound** | **≤ 37.0** | **-2.0+** | — | — |

---

## How to Run

```bash
# Activate environment
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh && conda activate posemamba

# Phase 0: Pre-check (runs before every experiment)
python -c "python experiments/scripts/precheck.py --config config.yaml"

# EXP A2: Confidence input (1 LOC)
vim kinecmamba/lib/model/PoseMamba.py  # change in_chans=2→3
vim kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml  # no_conf: False

# Background (setsid) — always use full conda python path
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
