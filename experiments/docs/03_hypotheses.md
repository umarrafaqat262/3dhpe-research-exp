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
| **P1** | Evidence ≥ 3, Δ > 0.2mm, ≤ 2 days | Do after P0 |
| **P2** | Evidence ≥ 1, Δ > 0.1mm | Do if P0/P1 exhausted |
| **P3** | Speculative or high cost | Defer indefinitely |

**Evidence scores:** 5 = proven on same benchmark, 3 = multiple papers same domain,
1 = single paper or related domain, 0 = speculative.

---

## H1: GCN-Mamba Dual-Stream (P0)

**Code:** `exp02_gcn_mamba`

| Field | Value |
|-------|-------|
| **Hypothesis** | PoseMamba's accuracy is bottlenecked by Mamba's inability to model local joint graph topology. Adding a parallel GCN stream with learnable adaptive fusion will improve MPJPE by ≥0.9mm |
| **Mechanism** | Spatial GCN (17-node skeleton adjacency) + temporal conv (1D over T) + learnable α fusion |
| **Evidence** | **Level 5** — PoseMagic (AAAI 2025) proves this exact intervention works on the same benchmark |
| **Est. Δ** | **-0.9mm** P1 MPJPE |
| **Cost** | ~80 LOC (new GCNModule class, fusion logic in PoseMamba.py) |
| **Dependency** | None — runs in parallel with existing Mamba stream |

**Falsification:** If Δ < 0.2mm improvement over baseline at Stage 4, then the hypothesis
that Mamba's local modeling is a bottleneck is false. (PoseMagic might not transfer.)

**Conflict check:** GCN runs in parallel — no interaction with existing BiSTSSMBlock internals.

**Simplest test:** Add GCNModule with:
```python
self.spatial_gcn = GCN(in_dim, hidden_dim, num_joints=17)  # graph conv
self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable fusion weight
```

---

## H2: Structure-Aware Stride Scan (P1)

**Code:** `exp03_stride_scan`

| Field | Value |
|-------|-------|
| **Hypothesis** | Flat 4-direction scan destroys joint adjacency. Stride-based scanning over skeleton topology preserves local structure and improves MPJPE |
| **Mechanism** | Replace CrossScan with grouped stride scan over 6 skeleton partitions |
| **Evidence** | **Level 3** — SasMamba (WACV 2026) |
| **Est. Δ** | **-0.5mm** P1 MPJPE |
| **Cost** | ~100 LOC (new CrossScan_stride variants in csms6s.py) |
| **Dependency** | None (scan is at leaf level, no interaction with other changes) |

**Falsification:** If Δ > -0.1mm after Stage 4, hypothesis is false.

**Skeleton partitions for stride scan:**
- Partition 1: Head (joints 12-15: Neck, Head, nose?)
- Partition 2: Torso (joints 0, 1, 6, 7: Hip, Spine, Thorax, Neck)
- Partition 3: Left arm (joints 2, 3, 4, 5: L_Shoulder, L_Elbow, L_Wrist, L_Hand)
- Partition 4: Right arm (joints 8, 9, 10, 11: R_Shoulder, R_Elbow, R_Wrist, R_Hand)
- Partition 5: Left leg (joints 13, 14, 15: L_Hip, L_Knee, L_Ankle, L_Foot)
- Partition 6: Right leg (joints 17, 18, 19, 20: R_Hip, R_Knee, R_Ankle, R_Foot)

---

## H3: Cosine Annealing + Warmup (P3)

**Code:** Config change only

| Field | Value |
|-------|-------|
| **Hypothesis** | Cosine annealing with linear warmup improves convergence over exponential decay (γ=0.99) |
| **Mechanism** | Linear warmup from 0→lr over 5 epochs, then cosine decay to 0 over 115 epochs |
| **Evidence** | **Level 3** — HPE literature (Sapiens, MotionBERT all use cosine) |
| **Est. Δ** | **-0.3mm** P1 MPJPE |
| **Cost** | Config change + 15 LOC in train.py (already implemented) |
| **Dependency** | Independent |

**Falsification:** If Δ > -0.1mm, paper's exponential decay is already near-optimal.

---

## H4: Gradient Accumulation + Clipping (P4)

**Code:** Config change only

| Field | Value |
|-------|-------|
| **Hypothesis** | Gradient accumulation (8 steps) with max_norm=1.0 clipping improves stability and enables effective batch=32 |
| **Mechanism** | Accumulate over 8 micro-batches of 4, then step optimizer; clip gradients |
| **Evidence** | **Level 5** — Standard practice |
| **Est. Δ** | **-0.1mm** P1 MPJPE |
| **Cost** | Config change + 20 LOC (already implemented) |
| **Dependency** | Independent |

---

## H5: Decoupled Spatial-Temporal Scans (P2 → P5)

**Code:** `exp05_decoupled_st`

| Field | Value |
|-------|-------|
| **Hypothesis** | Flattening J×T → H×W entangles space and time. Separate spatial and temporal SSM passes improve dimension separation |
| **Mechanism** | Run two forward_corev2 passes: one with CrossScan over spatial dim, one over temporal dim, with separate SSM params |
| **Evidence** | **Level 3** — DBMambaPose (arXiv 2025) |
| **Est. Δ** | **-0.5mm** P1 MPJPE |
| **Cost** | ~100 LOC |
| **Dependency** | BiSTSSMBlock forward_type refactoring |

**Falsification:** If Δ > -0.1mm, entangled scan is not a bottleneck.

---

## H6: Bone-Aware Module (P2)

**Code:** `exp04_bone_module`

| Field | Value |
|-------|-------|
| **Hypothesis** | Bone direction/length vectors provide stronger spatial inductive bias than joint coordinates alone |
| **Mechanism** | Compute 16 bone vectors from 17 joints, embed via linear projection, fuse with joint features before SSM |
| **Evidence** | **Level 3** — MambaTopFusion (arXiv 2026) |
| **Est. Δ** | **-0.4mm** P1 MPJPE |
| **Cost** | ~60 LOC |
| **Dependency** | None (pre-SSM input transformation) |

---

## H7: Head-Aware Branch (P6)

**Code:** `exp07_head_aware`

| Field | Value |
|-------|-------|
| **Hypothesis** | PoseMamba underperforms on head/neck because SSM state decays over joint index distance. A local GAT on the 5-joint head subgraph fixes this |
| **Mechanism** | 2-layer GAT over {Hip_center, Spine, Thorax, Neck, Head}; output summed with main branch |
| **Evidence** | **Level 1** — Inference from architectural analysis |
| **Est. Δ** | -1.5mm on head/neck, -0.2mm overall |
| **Cost** | ~50 LOC |
| **Dependency** | Test after GCN-Mamba (may overlap in benefit) |

---

## H8: Sparse Hybrid Attention (P7)

**Code:** `exp08_hybrid_attention`

| Field | Value |
|-------|-------|
| **Hypothesis** | Adding 2 sparse attention layers improves long-range temporal modeling over pure SSM |
| **Mechanism** | Replace 2 of 10 BiSTSSMBlocks with standard multi-head self-attention |
| **Evidence** | **Level 3** — Jamba, TransMamba, Nemotron |
| **Est. Δ** | -0.3mm P1 MPJPE |
| **Cost** | ~60 LOC |
| **Dependency** | Compound experiment |

---

## H9: Mamba-2 SSD Kernel (P8)

**Code:** `exp09_mamba2_ssd`

| Field | Value |
|-------|-------|
| **Hypothesis** | Increasing state dimension (N=16→64) with Mamba-2's SSD layer improves model capacity |
| **Mechanism** | Replace selective_scan kernel with Mamba-2 SSD (scalar A, chunked matmul) |
| **Evidence** | **Level 4** — Mamba-2 paper (ICML 2024) |
| **Est. Δ** | -0.2mm P1 MPJPE |
| **Cost** | ~200 LOC (kernel replacement) |
| **Dependency** | Low priority — significant engineering effort |

---

## H10: QKNorm + RoPE (P9)

**Code:** `exp10_qkrope`

| Field | Value |
|-------|-------|
| **Hypothesis** | SSM lacks explicit position encoding. RoPE + QKNorm (Mamba-3 style) improves order awareness |
| **Mechanism** | Apply RoPE to SSM input projections; add LayerNorm to B/C projections |
| **Evidence** | **Level 3** — Mamba-3 (2026) |
| **Est. Δ** | -0.2mm P1 MPJPE |
| **Cost** | ~40 LOC |
| **Dependency** | Low priority |

## H11: SAMA-Style State-Level Fusion (New)

**Code:** `exp11_sama_fusion`

| Field | Value |
|-------|-------|
| **Hypothesis** | Feature-level dual-stream fusion (PoseMagic-style) is post-hoc. State-level fusion (SAMA-style) — modifying the SSM state transition to incorporate joint topology — yields additional improvement beyond feature-level fusion alone |
| **Mechanism** | Implement SAMA's Structure-aware State Integrator (SSI): before state update, aggregate neighboring joint states via learned adjacency weights. The state transition becomes sₜ = A·sₜ₋₁ + B·(xₜ + Σⱼ wᵢⱼ · sⱼ₋₁) where wᵢⱼ are learned topology weights |
| **Evidence** | **Level 3** — SAMA (ICCV 2025), MambaTopFusion GEM (CVIU 2026) |
| **Est. Δ** | -0.6mm P1 MPJPE (beyond GCN-Mamba alone) |
| **Cost** | ~150 LOC (modify forward_corev2 in mambablocks.py) |
| **Dependency** | Test after H1 (GCN-Mamba) — if both confirmed, combine for compound gain |

**Falsification:** If H11 combined with H1 yields Δ < 0.2mm beyond H1 alone, then state-level fusion does not add value beyond feature-level fusion.

**Novelty:** No paper combines state-level fusion (SAMA) + feature-level fusion (PoseMagic).
If both H1 and H11 confirm, the compound model represents a novel architecture.

---

## Hypothesis Bank Summary

| ID | Hypothesis | Ev | Est Δ | LOC | Priority | Status |
|----|-----------|----|-------|-----|----------|--------|
| H1 | GCN-Mamba dual-stream | 5 | -0.9mm | 80 | **P0** | Ready |
| H2 | Structure-aware stride scan | 3 | -0.5mm | 100 | P1 | Ready |
| H3 | Cosine annealing + warmup | 3 | -0.3mm | 10 | P1 | Implemented |
| H4 | Gradient accumulation + clip | 5 | -0.1mm | 20 | P1 | Implemented |
| H5 | Decoupled S-T scans | 3 | -0.5mm | 100 | P2 | Ready |
| H6 | Bone-aware module | 3 | -0.4mm | 60 | P1 | Ready |
| H7 | Head-aware branch | 1 | -1.5mm head | 50 | P2 | Ready |
| H8 | Sparse hybrid attention | 3 | -0.3mm | 60 | P2 | Ready |
| H9 | Mamba-2 SSD (N=64) | 4 | -0.2mm | 200 | P2 | Ready |
| H10 | RoPE + QKNorm | 3 | -0.2mm | 40 | P3 | Ready |
| **H11** | **SAMA-style state-level fusion** | **3** | **-0.6mm** | **150** | **P2** | Ready |

## Execution Order

```
Phase 1 (Training fixes, independent)
  H3 → cosine annealing + warmup        [config only]
  H4 → gradient accumulation + clip      [config only]

Phase 2 (Core architecture, prioritized)
  H1 → GCN-Mamba dual-stream             [P0: highest Δ, proven by PoseMagic]
  H2 → structure-aware stride scan       [P1: topological scan fix]

Phase 3 (Topological priors)
  H6 → bone-aware module                 [P1]
  H11 → SAMA-style state fusion          [P2: novel, after H1]
  H5 → decoupled S-T scans               [P2]

Phase 4 (Compound + advanced)
  H1 + H2 + H6 → compound model
  H7 → head-aware branch                 [P2]
  H8 → sparse hybrid attention           [P2]
  H1 + H11 → novel state+feature compound

Phase 5 (Deferred)
  H9 → Mamba-2 SSD kernel               [P2]
  H10 → RoPE + QKNorm                    [P3]
```
