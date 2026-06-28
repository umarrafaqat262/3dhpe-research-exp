# RESEARCH_PLAN.md — PoseMamba Improvement Loop
*Follows the Autonomous Research Scientist Loop Engineering Guide*
*Last updated: 2026-06-28 | Baseline: 38.1 mm MPJPE (PoseMamba-L, SH-detected 2D, Human3.6M P1)*

---

## 0. Project Summary

PoseMamba (AAAI 2025) is a purely SSM-based 2D-to-3D lifting network. Its core innovation is a bidirectional global-local spatio-temporal Mamba block with a skeleton-geometry-aware local scan reordering, achieving 38.1 mm MPJPE on Human3.6M at 36% of MotionAGFormer's compute. The literature has since produced several papers that directly attack PoseMamba's known weaknesses: static scan topology (SAMA, ICCV 2025), lack of GCN-style local structure (HGMamba, 2025), poor OOD robustness due to ignored 2D detector confidence (AugLift, 2025), and uniform treatment of joint motion dynamics (SAMA's MSM). 

**Goal:** Port the best ideas from this literature *into* the PoseMamba codebase — each as a clean, independently testable experiment — and beat SAMA-L (current SOTA: ~37.5 mm) while preserving PoseMamba's efficiency advantage.

**Critical code audit findings (June 2026):**
1. `CrossScan_plus_poselimbs.backward()` breaks gradient flow — skeleton prior is a dead weight
2. Confidence score is already in the data pipeline (`no_conf: True` strips it) — A2 is 1 LOC
3. L config uses inferior training recipe (exponential LR, no accum/warmup) — must harmonize
4. Layer-wise diagnostics already implemented in `train.py` — no re-implementation needed
5. Bone computation (`get_limb_lens`) already exists in `loss.py` — A5 can reuse

---

## 1. Baseline State

| Model | P1 (mm) | P2 (mm) | MACs/frame | Params |
|---|---|---|---|---|
| PoseMamba-L (paper) | 38.1 | 32.5 | 127M | 6.7M |
| PoseMamba-X (paper) | 37.7 | — | ~200M | ~12M |
| SAMA-L (ICCV 2025) | ~37.5 | — | < PoseMamba-X | < PoseMamba-X |
| HGMamba-B (2025) | 38.65 | — | — | — |
| Pose3DM-L (2025) | 37.9 | 32.1 | 127M | — |
| **Target** | **≤ 37.0** | **≤ 31.5** | **< 150M** | **< 10M** |

Reproduction requirement: our PoseMamba-L run must land within ±0.5 mm of 38.1 mm before any experiment proceeds. **L config must be upgraded** to S config's recipe (cosine LR, accum_steps=8, warmup=5, grad_clip=1.0) for fair comparison.

---

## 2. Literature → Limitations Table

| Limitation | Source Papers | Effect Size | Proven Fix | Fix Cost | Feasibility |
|---|---|---|---|---|---|
| plus_poselimbs backward is broken (gradient = 0 for skeleton prior) | Code audit | Unknown (lost learning signal) | Learnable adjacency matrix (or fix backward) | ~30 LOC | 🔵 95% — bug fix |
| Static scan order ignores joint topology dynamics | SAMA (ICCV 2025) | −0.6→−1.5 mm | Learnable adjacency matrix fused into SSM state (SSI) | ~30 LOC | 🟢 90% |
| L config training recipe outdated | Config audit | Unfair comparison | Add accum_steps=8, warmup=5, cosine LR, grad_clip=1.0 | Config | 🔵 100% |
| Uniform Δ timescale across joints ignores motion heterogeneity | SAMA (ICCV 2025) | −0.4→−0.9 mm | Motion-Adaptive State Modulator (MSM) — per-joint Δ from 1D conv | ~25 LOC | 🟡 60% — SSM dimension entanglement |
| 2D input is (x, y) only — detector confidence discarded | AugLift (arXiv 2508.07112) | 4–10% MPJPE OOD, 4% ID | Set `no_conf: False`, change `in_chans=2→3` | ~1 LOC | 🔵 100% — already in pipeline |
| Local scan uses fixed geometric reordering; does not adapt to action | SAMA; HGMamba | ~0.3–0.6 mm | Learned or searched topology ordering vs. hard-coded reordering | Config-level test | 🔵 100% |
| No hypergraph local modeling — global SSM underfits local limb structure | HGMamba (2504.06638) | up to 0.5 mm on GT 2D | Dual-stream: Hyper-GCN + Mamba with softmax-weighted adaptive fusion | ~80 LOC | 🟡 60% — complex, α may collapse |
| Bone vectors not used as auxiliary input | Mamba Topology Fusion (2505.20611) | ~0.3 mm | Add bone direction + length as extra input channels (reuse `get_limb_lens`) | ~15 LOC | 🔵 95% — code exists |
| Head/neck joints underperform | PoseMamba (paper) | ~2–3 mm per-joint | Local auxiliary branch on {hip_center, spine, thorax, neck, head} subgraph | ~25 LOC | 🟢 90% |
| No OOD evaluation | AugLift | 40–60mm degradation expected | AugLift UADD + scale normalisation enables OOD generalization | ~10 LOC + depth precompute | 🟡 50% — disk + leakage risk |
| Layer-wise diagnostics already present | Code audit | N/A — no re-implementation needed | Just use existing hooks in `train.py` | 0 LOC | 🔵 100% |
| Single training run reported, no variance | Statistical | Unclear significance of 0.3 mm margins | 3–5 seeds with bootstrap CI | Config | 🔵 100% |
| Old environment (PyTorch 1.13, CUDA 11.7) | GitHub repo | Reproducibility barrier | Port to PyTorch 2.x | ~1 day | ✅ Done |

---

## 3. Hypothesis Bank

Statuses: `Ready` → `In Progress` → `Confirmed` | `Falsified`

**Feasibility scores:** 🔵 100% (zero risk) → 🟢 90-95% (trivial) → 🟡 50-70% (needs pre-check) → 🔴 <50% (speculative)

| ID | Hypothesis | Ev | Est Δ | Cost | Priority | Feasibility | Dep |
|----|---|---|---|---|---|---|---|
| **A1** | Reproducing PoseMamba on PyTorch 2.x within ±0.5 mm confirms the baseline is stable | 5 | 0 mm (must match) | Config fix | **P0** | 🔵 100% | — |
| **A2** | Setting `no_conf: False` + `in_chans=3` (confidence is already in data) reduces MPJPE ≥ 0.3 mm | 3 | −0.5 mm | ~1 LOC | **P0** | 🔵 100% | A1 |
| **A5** | Adding bone direction vectors (spherical coords) as auxiliary input reduces MPJPE ≥ 0.3 mm | 3 | −0.3 to −0.5 mm | ~15 LOC (reuse get_limb_lens) | **P0** | 🔵 95% | A1 |
| **A3** | Fixing plus_poselimbs broken backward + learnable adjacency reduces MPJPE ≥ 0.3 mm | 5 | −0.4 to −0.9 mm | ~40 LOC | **P0** | 🟢 95% | A1 |
| **B2** | Head-aware GAT auxiliary branch reduces head/neck MPJPE ≥ 1.5 mm without degrading overall | 2 | −1.5 mm (head/neck) | ~25 LOC | **P1** | 🟢 90% | A1 |
| **B3** | Scan order grid search (8 orderings) finds one ≥ 0.3 mm better than original reordering | 2 | −0.3 mm | Config | **P2** | 🔵 100% | A3 |
| **A4** | Per-joint Δ (MSM) reduces MPJPE ≥ 0.3 mm | 5 | −0.3 to −0.5 mm | ~25 LOC | **P1** | 🟡 60% — SSM entanglement | A1 |
| **B1** | Hyper-GCN dual-stream reduces MPJPE ≥ 0.4 mm | 3 | −0.4 to −0.8 mm | ~80 LOC | **P1** | 🟡 60% — α may collapse | A3 |
| **A6** | AugLift UADD depth improves ID ≥ 0.3 mm + OOD ≥ 5% | 3 | −0.3 mm ID | ~10 LOC + precompute | **P2** | 🟡 50% — leakage + disk | A2 |
| **C1** | A3 (SSI) + A4 (MSM) combined achieves ≥ 0.8 mm improvement | 5 | −0.8 to −1.2 mm | ~55 LOC | **P1** | 🟡 70% | A3+A4 |
| **C2** | A2 + C1 achieves ≥ 1.0 mm improvement | 3 | −1.0 mm | ~60 LOC | **P2** | 🟢 85% | A2+C1 |

---

## 4. Revised Run Order (with Feasibility Gate)

```
Phase 0: Fix Baseline Config
  [Harmonize L config: accum_steps=8, warmup=5, cosine LR, grad_clip=1.0]

Phase 1: Zero-Risk Experiments (100% success rate — run NOW)
  A1 → A2 (1 LOC) → A5 (reuses existing code)

Phase 2: Bug Fix + Proven Architecture Changes (95% success)
  A3 — Fix plus_poselimbs broken gradient + learnable adjacency

Phase 3: Conditional Experiments (60-90%, with pre-checks)
  B2 (90%) → A4 (60%, kill if delta uniform) → C1 (only if A3+A4 pass)

Phase 4: Speculative (50-60%)
  B1 (60%, kill if α→0) → B3 (config) → A6 (50%, leakage risk)
```

---

## 5. Fast-Kill Strategy (Pre-Training Feasibility Checks)

Every experiment gets a **feasibility verdict before full training**:

### Phase 0 Pre-Checks (all experiments, <15 min each)

| Pre-Check | Measure | Fail Threshold | Time |
|-----------|---------|----------------|------|
| Gradient flow | All params have non-zero gradient | Any zero-gradient param | 1 min |
| Init loss | Starting MPJPE should be ~dataset variance | <10mm or >200mm | 1 min |
| Single-batch overfit | Model can memorize 2 clips → MPJPE < 5mm | >5mm at epoch 100 | 5 min |

### Experiment-Specific Kill Signals

```python
# A4 (MSM): Check per-joint delta diversity
# If std(delta_j) ≈ 0 across joints → no specialization → hypothesis FALSIFIED
delta_std = model.ste_blocks[0].op.dt_projs_bias.view(-1).std().item()
if delta_std < 0.01: REJECT

# A3 (learnable adj): Check adjacency matrix doesn't collapse
# If W_adj.mean() ≈ 1/J after softmax → uniform → hypothesis FALSIFIED
W_mean = model.learnable_adj.softmax(dim=-1).mean().item()
if abs(W_mean - 1/17) < 0.01: REJECT

# B1 (GCN-Mamba): Check fusion weight α doesn't → 0
if alpha.mean().item() < 0.1 after 5 epochs: REJECT

# B2 (Head GAT): Check λ_head × loss_head doesn't → 0
if lambda_head * head_loss / total_loss < 0.01: REJECT

# A6 (AugLift): Check depth doesn't leak GT 3D
# If depth channels correlate > 0.95 with GT Z → ID gains are inflated
```

---

## 6. Experiment Specifications (Ranked Order)

---

### EXP A1 — Baseline Reproduction + Config Harmonization

**Branch:** `exp/A1-baseline-repro`
**Hypothesis:** PoseMamba-L can be reproduced within ±0.5 mm on PyTorch 2.x with upgraded config.

**Critical fix:** L config missing `accum_steps: 8`, `warmup_epochs: 5`, `lr_scheduler: cosine`, `grad_clip_norm: 1.0`. Must harmonize with S config before running.

**Checklist before running:**
- [ ] L config updated with S config's training recipe
- [ ] H3.6M data in MotionBERT format, SH detections downloaded
- [ ] Slice clips with `tools/convert_h36m.py`
- [ ] Single-batch overfit test passes (MPJPE → < 5 mm in 100 steps on 2 clips)
- [ ] Init loss ≈ 50–100 mm
- [ ] All parameter gradients non-zero after 1 backward
- [ ] `no_conf: True` (baseline strips confidence — A2 changes this)

**Expected:** 38.1 ± 0.5 mm (S: 41.8 mm)
**Action if pass:** Tag as `baseline-A1`, all subsequent experiments branch from this commit.

---

### EXP A2 — Confidence Score as 3rd Input Channel

**Branch:** `exp/A2-confidence-input`
**Hypothesis:** Setting `no_conf: False` + `in_chans: 2→3` reduces MPJPE ≥ 0.3 mm.

**What changes (1 LOC):**
1. `PoseMamba.py` line 38: `in_chans=2` → `in_chans=3`
2. Config: `no_conf: True` → `no_conf: False`

**That's it.** The data pipeline (`DataReaderH36M.read_2d`, `MotionDataset3D`) already reads confidence and returns `[N, 17, 3]`. No data loading changes needed.

**Feasibility:** 🔵 100% — confidence is already in the data, being actively stripped
**Risk:** None. Zero risk of regression.
**Stage 3 decision:** Δ > +0.1 mm → REJECT (unlikely — this setting was standard in the original MotionBERT pipeline)

---

### EXP A3 — Learnable Adjacency Matrix (Fix plus_poselimbs bug + SSI from SAMA)

**Branch:** `exp/A3-SSI-learnable-adjacency`
**Hypothesis:** Replacing the broken `plus_poselimbs` (which has zero gradient flow for its skeleton prior) with a learnable adjacency matrix with proper backward reduces MPJPE ≥ 0.4 mm.

**Bug details:** `CrossScan_plus_poselimbs.backward()` (csms6s.py:163) is identical to standard `CrossScan.backward()`. The `indices = [0,0,1,2,3,0,4,5,6,8,11,12,13,8,14,15,16]` forward blending has NO gradient path. Additionally, the indices are asymmetric — right-side joints (14-16) map to themselves (no topology mixing), while left-side joints are heavily blended.

**What changes:**
1. Create `CrossScan_learnable`/`CrossMerge_learnable` with `torch.einsum('bchw,ij->bcih', x, W)` for proper gradient flow
2. Add `self.learnable_adj = nn.Parameter(torch.zeros(17, 17))` in BiSTSSM
3. Register new `forward_type='v2_learnable_adj'`
4. Apply symmetry constraint: `W = (W + W.T) / 2`

**Kill signal:** If `W.softmax(dim=-1).mean() ≈ 1/17` (uniform), no learning happening — REJECT
**Feasibility:** 🟢 95% — any fix is better than broken gradient

---

### EXP A4 — Per-Joint Timescale Modulator (MSM from SAMA)

**Branch:** `exp/A4-MSM-timescale`
**Hypothesis:** Per-joint Δ (discretisation step) reduces MPJPE ≥ 0.3 mm because different joints have different motion dynamics.

**Complexity:** HIGH. The SSM operates on 2D (T×J) flattened to 1D. Delta is per-channel, not per-joint. Need to reshape delta computation to be joint-aware.

**Kill signal after 1 epoch:** `delta_j.view(17, -1).std(dim=0).mean() < 0.01` → joint delta is uniform → FALSIFIED
**Feasibility:** 🟡 60% — literature proves it works, but integration into PoseMamba's 2D flattened scan is non-trivial

---

### EXP A5 — Bone Vector Auxiliary Input

**Branch:** `exp/A5-bone-vectors`
**Hypothesis:** Adding bone direction (spherical coordinates: length l, elevation θ, azimuth φ per bone) as extra input channels reduces MPJPE ≥ 0.3 mm.

**What changes:**
1. Reuse `get_limb_lens` from `loss.py` to compute bone vectors
2. Map 16 bone features → 17 joint features (broadcast from parent joint)
3. Change `in_chans=2` → `in_chans=5` (x, y, l, θ, φ) or `in_chans=3` → `in_chans=6` if combined with A2
4. Concatenate bone features to joint features in `PoseMamba.forward`

**Feasibility:** 🔵 95% — `get_limb_lens` and `get_angles` already exist in `loss.py:100-114` and `loss.py:150-184`

---

### EXP B2 — Head-Aware Auxiliary Branch (GAT on head subgraph)

**Branch:** `exp/B2-head-aware-GAT`
**Hypothesis:** A 2-layer GAT on {0,7,8,9,10} (hip_center, spine, thorax, neck, head) with weighted loss reduces head/neck MPJPE ≥ 1.5 mm.

**Kill signal:** If `λ_head * head_loss / total_loss < 0.01`, auxiliary branch contributes nothing → REJECT
**Feasibility:** 🟢 90% — standalone module, no interaction with SSM internals

---

### EXP B1 — HGMamba Dual-Stream (Hyper-GCN + Mamba)

**Branch:** `exp/B1-hypergcn-dual-stream`
**Prerequisite:** A3 Confirmed.
**Hypothesis:** Parallel Hyper-GCN stream with adaptive fusion reduces MPJPE ≥ 0.4 mm.

**Kill signal after 5 epochs:** If fusion weight α → 0, GCN stream contributes nothing → REJECT
**Feasibility:** 🟡 60% — 80 LOC, complex interaction, high risk of α collapse

---

### EXP A6 — AugLift UADD (Full Depth + Confidence)

**Branch:** `exp/A6-auglift-uadd`
**Prerequisite:** A2 Confirmed.
**Hypothesis:** Expanding input to 6D (x, y, c, d, d_min, d_max) using Depth Anything v2 reduces ID ≥ 0.3 mm + OOD ≥ 5%.

**Risks:**
1. Disk space: Depth precompute requires ~50GB on a 98%-full disk
2. Leakage: On H3.6M (indoor lab), depth may correlate with GT Z → inflated ID gains
3. Mitigation: Check correlation depth vs GT Z before training. If > 0.95, signal is not genuine.

**Feasibility:** 🟡 50%

---

## 7. Leaderboard

| Rank | Exp ID | MPJPE (mm) | Δ vs A1 | Feasibility | Status |
|---|---|---|---|---|---|
| — | A1 (baseline) | TBD | 0.0 | 🔵 100% | Pending |
| — | A2 | TBD | TBD | 🔵 100% | Pending |
| — | A5 | TBD | TBD | 🔵 95% | Pending |
| 1 | A3 | TBD | TBD | 🟢 95% | Pending |
| 2 | B2 | TBD | TBD | 🟢 90% | Pending |
| 3 | A4 | TBD | TBD | 🟡 60% | Pending |
| 4 | C1 | TBD | TBD | 🟡 70% | Pending |
| 5 | B1 | TBD | TBD | 🟡 60% | Pending |
| 6 | B3 | TBD | TBD | 🔵 100% | Pending |
| 7 | A6 | TBD | TBD | 🟡 50% | Pending |
| 8 | C2 | TBD | TBD | 🟢 85% | Pending |

*Update after each Stage 4 completes.*

---

## 8. Lessons Learned

### Code Audit Findings (2026-06-28)
| Finding | Impact | Action |
|---------|--------|--------|
| `plus_poselimbs` backward is straight-through estimator | Critical bug — skeleton prior has zero gradient | A3: learnable adjacency with proper backward |
| Confidence is already in data pipeline | A2 is 1 LOC, not 5 | Move A2 to immediate Phase 1 |
| L config has no accum_steps/warmup/cosine | Baseline unfair to smaller configs | Fix L config before A1 |
| Layer-wise hooks already in train.py | No re-implementation needed | Remove from docs, reference existing code |
| `get_limb_lens` and `get_angles` in loss.py | A5 can reuse | Reduce A5 estimate to 15 LOC |

---

## 9. Stopping Condition

Stop when **at least 3 of 4** are met:

| # | Condition | Current State |
|---|---|---|
| 1 | Last 3 experiments all falsified (Stage 3 rejected or Stage 4 reverted) | 0/3 |
| 2 | No remaining hypothesis with Priority ≤ P1 | 5 × P0/P1 remain |
| 3 | No new hypotheses generated in last 2 cycles | N/A |
| 4 | Total improvement < 0.2 mm over last 5 experiments | N/A |

Current status: **Far from stopping.** 5 experiments at P0/P1, all with feasibility ≥ 90% in Phase 1-2.
