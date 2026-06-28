# Hypotheses Bank
*Updated: 2026-06-28 — Added feasibility assessments, kill signals, pre-check protocols, and revised priorities*

## Changelog
- **2026-06-28:** Added `Feasibility` and `Kill Signal` columns to all hypotheses.
  A2 moved to P0 (1 LOC — confidence already in data). A5 moved to P0 (reuses existing code).
  Added pre-check protocol for each experiment. New W16 finding deepens confidence in A3.

---

## Methodology

All hypotheses follow the loop engineering framework:
- **Falsifiable:** State exactly what experimental result would disprove the hypothesis
- **One intervention per experiment:** A+B combined cannot be interpreted
- **Minimum viable change:** Simplest code change that tests the hypothesis
- **Feasibility pre-check:** Before full training, run fast diagnostic. If kill signal triggers, REJECT.

## Priority Definitions

| Priority | Criteria | Action |
|----------|----------|--------|
| **P0** | Evidence level ≥ 4, Δ > 0.3mm, **feasibility ≥ 95%** | Do immediately |
| **P1** | Evidence ≥ 3, Δ > 0.2mm, feasibility ≥ 60% | Do after P0 |
| **P2** | Evidence ≥ 1, Δ > 0.1mm, any feasibility | Do if P0/P1 exhausted |
| **P3** | Speculative or high cost | Defer indefinitely |

## Feasibility Scores
- 🔵 **100%** — zero risk, guaranteed to run. Config change or 1 LOC.
- 🟢 **90-95%** — trivial implementation, reuses existing code, low risk.
- 🟡 **50-70%** — non-trivial integration into SSM internals. Must run pre-check before full training.
- 🔴 **<50%** — high implementation risk or data dependency. Only if high-ROI phases exhausted.

---

## H12/A3: Learnable Skeleton Adjacency (P0, HIGHEST ROI)

| Field | Value |
|-------|-------|
| **Hypothesis** | The hardcoded `plus_poselimbs` indices create an asymmetric, non-learnable skeleton prior with **zero gradient flow** (confirmed bug — backward is straight-through estimator). Replacing with a learnable adjacency matrix with proper backward propagation improves topology modeling and per-joint accuracy |
| **Mechanism** | Replace `CrossScan_plus_poselimbs` (broken backward) with `CrossScan_learnable` using `torch.einsum('bchw,ij->bcih', x, W)` for proper autograd. W ∈ ℝ^(17×17) with symmetry constraint. |
| **Evidence** | **Level 4** — Direct code analysis proves gradient gap exists. W16 confirms backward is straight-through. |
| **Est. Δ** | **-0.4mm** P1 MPJPE (from fixing gradient gap + data-driven topology) |
| **Cost** | ~40 LOC |
| **Feasibility** | 🟢 **95%** |
| **Kill Signal** | After 1 epoch: `W.softmax(dim=-1).mean() ≈ 1/17` (uniform weights) → no topology learning → REJECT |
| **Pre-check** | `python -c "print(model.learnable_adj.softmax(dim=-1).mean().item())"` — should NOT be 1/17 |

**Falsification:** If Δ > -0.1mm, then the specific `plus_poselimbs` indices are near-optimal and gradient flow is unnecessary.

---

## H1/B1: GCN-Mamba Dual-Stream (P1, FEASIBILITY GATED)

| Field | Value |
|-------|-------|
| **Hypothesis** | PoseMamba's accuracy is bottlenecked by Mamba's inability to model local joint graph topology. Adding a parallel GCN stream with learnable adaptive fusion will improve MPJPE by ≥0.9mm |
| **Mechanism** | Spatial GCN (17-node skeleton adjacency, 2 layers) + temporal conv + learnable α fusion. If α → 0, GCN contributes nothing. |
| **Evidence** | **Level 5** — PoseMagic (AAAI 2025) proves this exact intervention works on the same H36M benchmark, confirmed by AGMamba, HGMamba |
| **Est. Δ** | **-0.9mm** P1 MPJPE |
| **Cost** | ~80 LOC (highest of all P0/P1 experiments) |
| **Feasibility** | 🟡 **60%** |
| **Kill Signal** | After 5 epochs: `α.mean() < 0.1` → GCN branch ignored → REJECT |
| **Pre-check** | Monitor `α` during training. Log to wandb every epoch. |

**Falsification:** If α → 0 or Δ < 0.2mm, GCN-Mamba dual-stream does not transfer to PoseMamba's SSM variant.

---

## H11/C1: SAMA-Style State-Level Fusion (P1)

| Field | Value |
|-------|-------|
| **Hypothesis** | State-level fusion (modifying SSM state transition with topology) adds value beyond feature-level fusion. |
| **Mechanism** | Structure-aware State Integrator: modify `forward_corev2` to incorporate topology-weighted state aggregation before selective scan output. |
| **Evidence** | **Level 3** — SAMA (ICCV 2025), MambaTopFusion GEM (CVIU 2026) |
| **Est. Δ** | **-0.6mm** additive beyond H1 |
| **Cost** | ~50 LOC added to `forward_corev2` |
| **Feasibility** | 🟡 **70%** |
| **Kill Signal** | If combined with A3 and the adjacency matrix already captures topology, SSI may be redundant. Check: does SSI change the output at all in first forward pass? If `||SSI_output - original_output|| < 1e-6`, no effect → REJECT |

---

## A2: Confidence Score as 3rd Input Channel (P0, 1 LOC)

| Field | Value |
|-------|-------|
| **Hypothesis** | Confidence score `c ∈ [0,1]` is already in the data pipeline (`DataReaderH36M` reads and concatenates it). Setting `no_conf: False` + `in_chans=2→3` adds this signal to the model, allowing it to discount low-confidence joints. |
| **Mechanism** | 1 LOC change: switch `in_chans=2→3` in `PoseMamba.py` line 38, set `no_conf: False` in config. Confidence is already normalized and available as 3rd channel of motion_2d tensors. |
| **Evidence** | **Level 4** — Code audit proves confidence is in the data. AugLift (2025) proves the value of confidence for MPJPE. |
| **Est. Δ** | **-0.5mm** P1 MPJPE |
| **Cost** | **1 LOC** (not 5 as previously estimated) + 1 config flag |
| **Feasibility** | 🔵 **100%** |
| **Kill Signal** | N/A — zero risk of regression. If Δ > +0.1mm (impossible without bug), it would mean the model cannot utilize confidence — confidence signal is too noisy. |

**Why this was underestimated:** All previous estimates said "add confidence column from SH detection output." The code already does this — `DataReaderH36M.read_2d()` has `if self.read_confidence` block at lines 45-57. The 3rd channel is present in every batch; `no_conf: True` strips it at `train.py:224`.

---

## A5: Bone Vector Auxiliary Input (P0, 15 LOC)

| Field | Value |
|-------|-------|
| **Hypothesis** | Bone direction (θ, φ) + length (l) per bone provides explicit topology structure. |
| **Mechanism** | Compute 16 bone vectors using existing `get_limb_lens` from `loss.py`, broadcast to 17 joints, concatenate with (x, y) → `in_chans=2→5`. |
| **Evidence** | **Level 3** — MambaTopFusion (CVIU 2026) |
| **Est. Δ** | **-0.3 to -0.5mm** |
| **Cost** | ~15 LOC |
| **Feasibility** | 🔵 **95%** |
| **Kill Signal** | Δ > +0.1mm at Stage 3. If bone vectors are collinear with joint coordinates (they shouldn't be), no gain. |

**Key efficiency:** `loss.py:100-114` already has `get_limb_lens()`. `loss.py:150-184` has `get_angles()`. Both compute bone-specific features. A5 can import these directly rather than re-implementing skeleton math.

---

## B2: Head-Aware Auxiliary Branch (P1, 25 LOC)

| Field | Value |
|-------|-------|
| **Hypothesis** | GAT on {0,7,8,9,10} subgraph with weighted loss reduces head/neck MPJPE ≥ 1.5mm. |
| **Mechanism** | 2-layer GAT on head subgraph, auxiliary loss `λ_head * MPJPE_head` added to main loss. |
| **Evidence** | **Level 2** — PoseMamba supplementary acknowledges head/neck underperformance. |
| **Est. Δ** | **-1.5mm** head/neck |
| **Cost** | ~25 LOC |
| **Feasibility** | 🟢 **90%** |
| **Kill Signal** | `λ_head * head_loss / total_loss < 0.01` after 5 epochs → branch contributes nothing → REJECT |

---

## A4: Per-Joint Timescale Modulator / MSM (P1, 25 LOC, HIGH RISK)

| Field | Value |
|-------|-------|
| **Hypothesis** | Per-joint Δ (discretisation step) from local motion magnitude reduces MPJPE ≥ 0.3mm. |
| **Mechanism** | Compute motion = x_t - x_{t-1}, pass through 1D conv to get per-joint Δ, use as SSM time step. |
| **Evidence** | **Level 5** — SAMA (ICCV 2025) proves this. |
| **Est. Δ** | **-0.3 to -0.5mm** |
| **Cost** | ~25 LOC |
| **Feasibility** | 🟡 **60%** |
| **Kill Signal** | After 1 epoch: `delta_j.view(17, -1).std(dim=0).mean() < 0.01` → joints have uniform Δ → no per-joint specialization → REJECT |

**Risk detail:** The SSM in PoseMamba operates on a 2D (T×J) tensor flattened to 1D for the selective scan. Per-joint delta requires reshaping the delta computation to maintain joint identity. The `forward_corev2` function computes delta via `dts = F.conv1d(...)` on the flattened 1D representation, where all joints are interleaved. Disentangling this is the core challenge.

---

## B3: Scan Order Grid Search (P2, Config Only)

| Field | Value |
|-------|-------|
| **Hypothesis** | At least one of 8 alternative scan orderings is ≥ 0.3mm better than original. |
| **Mechanism** | 8 configs with different scan orders (BFS, DFS, bilateral, variance-sorted, etc.). Run Phase 2 overfit ranker first, then top-2 survivors get Stage 3. |
| **Evidence** | **Level 2** — SAMA mentions scan order sensitivity. |
| **Est. Δ** | **-0.3mm** |
| **Cost** | Config only |
| **Feasibility** | 🔵 **100%** |
| **Kill Signal** | Phase 2 overfit rank: bottom 6 eliminated. Top-2 must show Δ < -0.1mm vs baseline at Phase 2. |

---

## A6: AugLift UADD Depth (P2, 10 LOC + Precompute)

| Field | Value |
|-------|-------|
| **Hypothesis** | 6D input (x,y,c,d,d_min,d_max) improves ID ≥ 0.3mm + OOD ≥ 5%. |
| **Mechanism** | Precompute depth with Depth Anything v2 (requires ~50GB disk + ~3hrs GPU). Concatenate depth stats as extra channels. |
| **Evidence** | **Level 3** — AugLift (2025). |
| **Est. Δ** | **-0.3mm ID, -5% OOD** |
| **Cost** | ~10 LOC + precompute pipeline |
| **Feasibility** | 🟡 **50%** |
| **Kill Signal** | Check correlation between depth channels and GT Z on H3.6M. If > 0.95, depth leaks GT → ID gains are inflated → REJECT (or only evaluate on OOD). |

---

## Hypothesis Bank Summary (Updated)

| ID | Hypothesis | Ev | Est Δ | LOC | Priority | Feasibility | Kill Signal | Status |
|----|-----------|----|-------|-----|----------|-------------|-------------|--------|
| **A2** | **Confidence input** | **4** | **-0.5mm** | **1** | **P0** | 🔵 **100%** | None — zero risk | **Ready** |
| **A5** | **Bone vectors** | **3** | **-0.3mm** | **15** | **P0** | 🔵 **95%** | Δ > +0.1mm | **Ready** |
| **A3** | **Learnable adj (bug fix)** | **4** | **-0.4mm** | **40** | **P0** | 🟢 **95%** | W_adj uniform | **Ready** |
| B2 | Head-aware GAT | 2 | -1.5mm (head) | 25 | P1 | 🟢 90% | λ→0 | Ready |
| A4 | MSM per-joint Δ | 5 | -0.3mm | 25 | P1 | 🟡 60% | δ_std ≈ 0 | Ready |
| C1 | A3+A4 combined | 5 | -0.8mm | 55 | P1 | 🟡 70% | Depends on A3+A4 | Pending |
| B1 | HyperGCN dual-stream | 5 | -0.9mm | 80 | P1 | 🟡 60% | α→0 | Ready |
| B3 | Scan order search | 2 | -0.3mm | 0 | P2 | 🔵 100% | Bottom 6 cut | Ready |
| A6 | AugLift depth | 3 | -0.3mm ID | 10+pre | P2 | 🟡 50% | depth-GT corr | Ready |
| C2 | A2+C1 combined | 3 | -1.0mm | 60 | P2 | 🟢 85% | Depends | Deferred |
| H2 | Stride scan | 3 | -0.5mm | 100 | P2 | 🟡 50% | — | Deferred |
| H5 | Decoupled S-T | 3 | -0.5mm | 100 | P3 | 🟡 40% | — | Deferred |
| H9 | Mamba-2 SSD | 4 | -0.2mm | 200 | P3 | 🔴 30% | — | Deferred |
| H10 | RoPE + QKNorm | 3 | -0.2mm | 40 | P3 | 🟡 50% | — | Deferred |

---

## Execution Order (Final, June 2026)

```
Phase 1 — Zero Risk (DO NOW, feasibility ≥ 95%)
  Step 1: A1 — Baseline repro (fix L config first!)
  Step 2: A2 — Confidence input (1 LOC)     ← can batch with A1?
  Step 3: A5 — Bone vectors (15 LOC)
  
Phase 2 — Bug Fix (proven improvement)
  Step 4: A3 — Learnable adjacency (fixes plus_poselimbs gradient bug)
  
Phase 3 — Conditional (feasibility 60-90%, with pre-checks)
  Step 5: B2 — Head-aware GAT (90%, independent)
  Step 6: A4 — MSM per-joint delta (60%, kill if δ_j uniform)
  Step 7: C1 — A3+A4 combined (70%, only if both pass)
  
Phase 4 — Speculative (feasibility 50-60%)
  Step 8: B1 — HyperGCN (60%, kill if α→0)
  Step 9: B3 — Scan order (100% easy, low Δ expected)
  Step 10: A6 — AugLift depth (50%, high disk + leakage risk)
```

**Decision rule:** After Phase 2, if cumulative Δ ≥ -1.2mm (A2+A5+A3), skip Phase 3-4 and go directly to publication sprint (ablation study, 3DPW eval, paper writing).
