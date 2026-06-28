# Hypotheses Bank

## Methodology

All hypotheses follow the loop engineering framework:
- **Falsifiable:** State exactly what experimental result would disprove the hypothesis
- **One intervention per experiment:** A+B combined cannot be interpreted
- **Minimum viable change:** Simplest code change that tests the hypothesis

## Priority Definitions

| Priority | Criteria | Action |
|----------|----------|--------|
| **P0** | Evidence level ≥ 4, Δ > 0.3mm, ≤ 80 LOC | Do immediately |
| **P1** | Evidence ≥ 3, Δ > 0.2mm, novelty or compound value | Do after P0 |
| **P2** | Evidence ≥ 1, Δ > 0.1mm | Do if P0/P1 exhausted |
| **P3** | Speculative or high cost | Defer indefinitely |

**Evidence scores:** 5 = proven on same benchmark, 3 = multiple papers same domain,
1 = single paper or related domain, 0 = speculative.

---

## H12: Learnable Skeleton Adjacency (P0, NEW)

**Code:** `exp01_learnable_adj`

| Field | Value |
|-------|-------|
| **Hypothesis** | The hardcoded `plus_poselimbs` indices create an asymmetric, non-learnable skeleton prior with no gradient flow. Replacing them with a learnable adjacency matrix with proper backward gradient propagation improves topology modeling and per-joint accuracy |
| **Mechanism** | Replace `indices = [0,0,1,2,3,0,4,5,6,8,11,12,13,8,14,15,16]` with a learnable `W ∈ ℝ^(17×17)` parameter. Path 0 becomes `x + einsum('bchw,ij->bcih', x, W)` instead of `x + x[..., indices]`. Fix `CrossScan_learnable.backward()` to propagate gradients through W via `scatter_add_` or `einsum` backward (autograd handles this naturally if implemented with `torch.einsum`). Symmetry constraint: W = (W + W^T) / 2 to enforce undirected skeleton graph |
| **Evidence** | **Level 4** — Direct code analysis proves gradient gap exists. Literature: every GCN paper shows learnable adjacency improves over fixed. This is a strict superset of the existing hardcoded approach |
| **Est. Δ** | **-0.4mm** P1 MPJPE (from fixing gradient gap + data-driven topology) |
| **Cost** | ~40 LOC (new `CrossScan_learnable`/`CrossMerge_learnable` classes, `W` parameter in BiSTSSM, backward fix) |
| **Dependency** | None — replaces plus_poselimbs at leaf level |
| **Novelty** | **YES** — No Mamba-based HPE paper uses learnable skeleton adjacency within the SSM scan. All existing work uses hardcoded indices (PoseMamba), GCN feature fusion (PoseMagic), or stride partitions (SasMamba) |

**Falsification:** If Δ > -0.1mm, then the specific `plus_poselimbs` indices are near-optimal and gradient flow is unnecessary.

**Implementation sketch:**
```python
class CrossScan_learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        # Path 0: x + x @ W^T  (learnable adjacency mixing)
        mixing = torch.einsum('bchw,ij->bcih', x, W)
        xs0 = (x + mixing).reshape(B, C, -1)
        # Paths 1-3: standard transpose + flips
        ...
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        # Autograd handles W gradient through einsum backward
        # Standard backward for paths 1-3
        ...
```

---

## H1: GCN-Mamba Dual-Stream (P0)

**Code:** `exp02_gcn_mamba`

| Field | Value |
|-------|-------|
| **Hypothesis** | PoseMamba's accuracy is bottlenecked by Mamba's inability to model local joint graph topology. Adding a parallel GCN stream with learnable adaptive fusion will improve MPJPE by ≥0.9mm |
| **Mechanism** | Spatial GCN (17-node skeleton adjacency, 2 layers) + temporal conv (1D over T, kernel=3) + learnable α fusion weight |
| **Evidence** | **Level 5** — PoseMagic (AAAI 2025) proves this exact intervention works on the same H36M benchmark, confirmed by AGMamba (SIVP 2026), HGMamba (arXiv 2025) |
| **Est. Δ** | **-0.9mm** P1 MPJPE |
| **Cost** | ~80 LOC (new GCNModule class, fusion logic in PoseMamba.py) |
| **Dependency** | None — runs in parallel with existing Mamba stream |

**Falsification:** If Δ < 0.2mm, then GCN-Mamba dual-stream does not transfer to PoseMamba's SSM variant (PoseMagic uses a different Mamba implementation).

**Simplest test:**
```python
self.gcn = GCN(in_dim=64, hidden_dim=64, num_joints=17)
self.temporal_conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
self.alpha = nn.Parameter(torch.tensor(0.5))
# Fusion during forward
gcn_out = self.temporal_conv(self.gcn(x).permute(0,2,1)).permute(0,2,1)
out = self.alpha * mamba_out + (1 - self.alpha) * gcn_out
```

---

## H11: SAMA-Style State-Level Fusion (P1)

**Code:** `exp03_sama_fusion`

| Field | Value |
|-------|-------|
| **Hypothesis** | Feature-level dual-stream fusion (H1) is post-hoc. State-level fusion — modifying the SSM state transition to incorporate joint topology — yields additional improvement beyond feature-level fusion alone. Combining both (H1+H11) is novel and additive |
| **Mechanism** | Structure-aware State Integrator (adapted from SAMA, ICCV 2025): before SSM state update, aggregate neighboring joint states via learned adjacency weights. State transition becomes `sₜ = A·sₜ₋₁ + B·(xₜ + Σⱼ wᵢⱼ · sⱼ₋₁)` where `wᵢⱼ` are topology weights |
| **Evidence** | **Level 3** — SAMA (ICCV 2025), MambaTopFusion GEM (CVIU 2026). The combination with H1 is **unexplored** in literature |
| **Est. Δ** | **-0.6mm** additive beyond H1 (cumulative with H12+H1: up to -1.9mm) |
| **Cost** | ~150 LOC (modify `forward_corev2` in mambablocks.py) |
| **Dependency** | Test after H1 proven. If H1 confirmed, test H11 on top of H1 baseline |
| **Novelty** | **YES** — No paper combines state-level fusion (SAMA) + feature-level fusion (PoseMagic). This is a novel compound architecture |

**Falsification:** If H11 combined with H1 yields Δ < 0.2mm beyond H1 alone, state-level and feature-level fusion address the same bottleneck and are redundant.

---

## H2: Structure-Aware Stride Scan (P2)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | Flat 4-direction scan destroys joint adjacency. Stride-based scanning over skeleton preserves local structure |
| **Mechanism** | Replace CrossScan with grouped stride scan over 4 body-part partitions |
| **Evidence** | **Level 3** — SasMamba (WACV 2026) |
| **Est. Δ** | -0.5mm P1 MPJPE |
| **Cost** | ~100 LOC |
| **Dependency** | Deferred — test after compound baseline |

---

## H6: Bone-Aware Module (P2)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | Bone direction/length vectors provide spatial inductive bias beyond joint coordinates |
| **Mechanism** | Compute 16 bone vectors, embed via Linear, fuse with joint features pre-SSM |
| **Evidence** | **Level 3** — MambaTopFusion (CVIU 2026) |
| **Est. Δ** | -0.4mm |
| **Cost** | ~60 LOC |
| **Dependency** | Deferred — test after compound baseline |

---

## H8: Sparse Hybrid Attention (P2)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | 2 sparse attention layers improve long-range temporal modeling over pure SSM |
| **Mechanism** | Replace 2 of 10 BiSTSSMBlocks with multi-head self-attention |
| **Evidence** | **Level 3** — VIMCAN (CVPR 2026), Jamba-1.5 (2025) |
| **Est. Δ** | -0.3mm |
| **Cost** | ~60 LOC |
| **Dependency** | Deferred |

---

## H9: Mamba-2 SSD Kernel (P3)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | Increasing state dimension N=16→64 with Mamba-2 SSD improves capacity |
| **Mechanism** | Replace selective_scan kernel with Mamba-2 SSD |
| **Evidence** | **Level 4** — Mamba-2 (ICML 2024) |
| **Est. Δ** | -0.2mm |
| **Cost** | ~200 LOC |
| **Dependency** | Deferred — significant engineering effort |

---

## H5: Decoupled S-T Scans (P3)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | Separate spatial and temporal SSM passes improve dimension separation over entangled 2D scan |
| **Mechanism** | Two forward_corev2 passes: one spatial, one temporal |
| **Est. Δ** | -0.5mm |
| **Cost** | ~100 LOC |
| **Dependency** | Deferred |

---

## H7: Head-Aware Branch (P3)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | GCN-Mamba already subsumes head/neck improvements (H7) |
| **Mechanism** | N/A — skip if H1 confirmed |
| **Est. Δ** | Subsumed by H1 |
| **Dependency** | Cancelled if H1 confirms |

---

## H10: RoPE + QKNorm (P3)

**Code:** Deferred

| Field | Value |
|-------|-------|
| **Hypothesis** | SSM lacks explicit position encoding. RoPE + QKNorm improves order awareness |
| **Est. Δ** | -0.2mm |
| **Cost** | ~40 LOC |
| **Dependency** | Deferred |

---

## Hypothesis Bank Summary

| ID | Hypothesis | Ev | Est Δ | LOC | Priority | Status |
|----|-----------|----|-------|-----|----------|--------|
| **H12** | **Learnable skeleton adjacency (NEW)** | **4** | **-0.4mm** | **40** | **P0** | Ready |
| H1 | GCN-Mamba dual-stream | 5 | -0.9mm | 80 | **P0** | Ready |
| H11 | SAMA-style state-level fusion | 3 | -0.6mm* | 150 | P1 | Ready |
| H12+H1+H11 | **Compound (novel)** | **4** | **-1.9mm** | **270** | **P0** | **Design** |
| H2 | Structure-aware stride scan | 3 | -0.5mm | 100 | P2 | Deferred |
| H6 | Bone-aware module | 3 | -0.4mm | 60 | P2 | Deferred |
| H8 | Sparse hybrid attention | 3 | -0.3mm | 60 | P2 | Deferred |
| H5 | Decoupled S-T scans | 3 | -0.5mm | 100 | P3 | Deferred |
| H9 | Mamba-2 SSD (N=64) | 4 | -0.2mm | 200 | P3 | Deferred |
| H7 | Head-aware branch | 1 | subsumed | — | P3 | Cancelled |
| H10 | RoPE + QKNorm | 3 | -0.2mm | 40 | P3 | Deferred |

*H11 Δ is additive beyond H1; H12 is independent and stacks with H1+H11.

---

## Execution Order

```
Phase 1 (Immediate — proven + novel, highest ROI)
  Step 1: Baseline (Exp 00) with wandb-only + layer-wise diagnostics
  Step 2: H12 → Learnable skeleton adjacency (Exp 01)   [40 LOC, Δ=-0.4mm]
  Step 3: H1  → GCN-Mamba dual-stream (Exp 02)          [80 LOC, Δ=-0.9mm]
  Step 4: H12+H1 compound (no extra code)                [cumulative Δ=-1.3mm]

Phase 2 (Novel compound)
  Step 5: H11 → SAMA-style state fusion (Exp 03)        [150 LOC, Δ=-0.6mm*]
  Step 6: H12+H1+H11 full compound                       [cumulative Δ=-1.9mm target]

Phase 3 (If time permits — incremental)
  Step 7: H2  → Stride scan                             [Δ=-0.5mm]
  Step 8: H6  → Bone module                              [Δ=-0.4mm]
  Step 9: H8  → Hybrid attention                         [Δ=-0.3mm]
```

Key insight: H12 and H1 address different weaknesses (gradient flow vs missing topology).
They are additive. H11 addresses a third dimension (state-level vs feature-level).
The full compound is novel — no paper combines all three.
