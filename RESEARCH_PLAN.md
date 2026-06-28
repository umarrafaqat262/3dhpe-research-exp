# KinecMamba — Comprehensive Research Plan

**Goal:** Beat PoseMamba (38.1mm P1 on H36M) with literature-backed architectural changes for CVPR-level publication.

---

## 1. Weakness Analysis

### From PoseMamba Paper Analysis

| # | Weakness | Evidence | Severity |
|---|----------|----------|----------|
| W1 | **Pure SSM ignores skeletal topology** — 1D scan treats joints as flat sequence, destroying graph structure | SasMamba WACV 2026, Mamba-Driven Topology Fusion 2026, Spatial-Mamba ICLR 2025 | **Critical** |
| W2 | **Weak local joint dependency** — SSM's selective state prioritizes long-range over local interactions | PoseMagic AAAI 2025, HGMamba, GEM | **Critical** |
| W3 | **Only indoor benchmark (H36M)** — no cross-dataset evaluation (3DPW, MPI-INF-3DHP) | Claude analysis | **High** |
| W4 | **Single-run results, no variance** — cannot distinguish signal from noise | Claude analysis | **High** |
| W5 | **Head/neck underperformance** — SSM state decays over distant joint indices | Claude analysis | **High** |
| W6 | **Exponential LR decay is suboptimal** — cosine with warmup is modern HPE standard | HPE literature (Sapiens, MotionBERT) | **Medium** |
| W7 | **No gradient accumulation** — forces small batch, noisy gradients | Our setup | **Medium** |
| W8 | **No temporal consistency metric beyond MPJVE** — bone length jitter unchecked | Claude analysis | **Medium** |
| W9 | **Scan order manually designed, never ablated** — may be suboptimal for pose topology | Claude analysis | **Medium** |
| W10 | **Small state dimension (N=16)** — Mamba-2 uses N=64-256 for better capacity | Mamba-2 SSD paper ICML 2024 | **Low** |
| W11 | **No positional encoding in SSM** — Mamba-3 adds RoPE for order awareness | Mamba-3 2026 | **Low** |

### From Broader Literature

| Domain | Finding | Source | Transferable To |
|--------|---------|--------|-----------------|
| **NLP SSMs** | **Mamba-2 SSD**: larger state (N=64+), scalar A, chunked training, 2-8× faster | Dao & Gu ICML 2024 | Replace selective_scan kernel |
| **NLP SSMs** | **Hybrid SSM-Attention**: 7% attention + 93% SSM layers beats pure SSM | Jamba, TransMamba, Nemotron | Add sparse attention layers |
| **NLP SSMs** | **Mamba-3**: complex-valued states, MIMO, RoPE, QKNorm | Lahoti et al. 2026 | RoPE for SSM ordering |
| **Vision SSMs** | **Spatial-Mamba**: 3×3 depthwise conv replaces 1D causal conv | ICLR 2025 | Better local spatial modeling |
| **Vision SSMs** | **ASGMamba**: frequency-selective gating via FFT patch | arXiv 2026 | Noise filtering for pose |
| **3D HPE** | **Dual-stream Mamba+GCN** with adaptive fusion beats PoseMamba by -0.9mm | PoseMagic AAAI 2025 | **P0 priority** |
| **3D HPE** | **Structure-aware stride scan** preserves skeleton topology | SasMamba WACV 2026 | **P1 priority** |
| **3D HPE** | **Bone-aware module** with direction/length vectors | MambaTopFusion 2026 | **P2 priority** |
| **3D HPE** | **Decoupled S-T bidirectional scans** | DBMambaPose 2025 | **P3 priority** |
| **Training** | **Cosine annealing + linear warmup** improves convergence | Loshchilov & Hutter, Sapiens | Free improvement |
| **Training** | **Gradient accumulation** enables effective large batch with small GPU | Standard practice | Match paper batch=32 |

---

## 2. Cross-Domain Technique Transfer Map

```
NLP SSM Literature         Vision SSM Literature         3D HPE Literature
══════════════════         ════════════════════         ════════════════════
Mamba-2 (N=64+)            Spatial-Mamba (3×3 conv)    PoseMagic (GCN+Mamba)  ──→ P0
Mamba-3 (RoPE/QKNorm)      ASGMamba (FFT gating)       SasMamba (stride scan) ──→ P1
Hybrid Attn+SSM            VMamba (2D scan)             DBMambaPose (S-T dec)  ──→ P3
Gated DeltaNet (S6 alt)    LocalViM (local SSM)        MambaTopFusion (bone)   ──→ P2
                          ─────────────────────        HGMamba (hierarchical)
                                ↓                              ↓
                          Training Best Practices     Training Best Practices
                          ──────────────────────     ──────────────────────
                          Cosine + warmup            Gradient accumulation
                          Gradient clipping          ExponentialLR → Cosine
                          Layer-wise LR decay        Weighted MPJPE
```

---

## 3. Prioritized Hypothesis Bank

| ID | Hypothesis | Evidence Level | Est. Δ MPJPE | LOC | Priority |
|----|-----------|---------------|-------------|-----|----------|
| **P0** | **GCN+Mamba dual-stream** improves local joint modeling (PoseMagic-proven, AAAI 2025) | 5 (AAAI) | **-0.9mm** | 80 | **P0** |
| **P1** | **Structure-aware stride scan** preserves skeleton topology (SasMamba, WACV 2026) | 3 (WACV) | -0.5mm | 100 | **P1** |
| **P2** | **Bone-aware module** provides topological prior (MambaTopFusion 2026) | 3 (arXiv) | -0.4mm | 60 | **P1** |
| **P3** | **Cosine annealing + warmup** improves convergence over exp decay | 3 (HPE lit) | -0.3mm | 10 | **P1** |
| **P4** | **Gradient accumulation (8 steps)** enables effective batch 32 | 5 (standard) | -0.1mm | 15 | **P1** |
| **P5** | **Decoupled spatial-temporal scans** improve dimension separation (DBMambaPose 2025) | 3 (arXiv) | -0.5mm | 100 | **P2** |
| **P6** | **Head-aware auxiliary branch** (GAT on neck/head) reduces per-joint error | 1 (inference) | -1.5mm head | 50 | **P2** |
| **P7** | **Sparse hybrid attention** (2-4 attn layers) improves long-range | 3 (NLP) | -0.3mm | 60 | **P2** |
| **P8** | **Mamba-2 SSD replacement** (larger state dim N=64) | 4 (NLP) | -0.2mm | 200 | **P2** |
| **P9** | **QKNorm + RoPE** position encoding (Mamba-3 style) | 3 (NLP) | -0.2mm | 40 | **P3** |

### Hypothesis Details

#### P0: GCN-Mamba Dual-Stream (PoseMagic-style, +80 LOC)
- **Mechanism:** Mamba captures global dependencies but ignores local joint topology. GCN explicitly models skeletal adjacency. Dual-stream with adaptive fusion (learnable α) blends both branches.
- **Falsification:** If P0 achieves Δ < 0.2mm improvement, Mamba's local modeling is not a bottleneck.
- **Priority rationale:** Simplest (parallel branch, no SSM internals), highest evidence (proven in PoseMagic), largest expected Δ.

#### P1: Structure-Aware Stride Scan (SasMamba-style, +100 LOC)
- **Mechanism:** Current 4-direction flat scan destroys joint adjacency. Stride scan over skeleton topology (6 partitions: head, torso, L/R arm, L/R leg) preserves local structure.
- **Falsification:** If Δ > -0.1mm after stride scan, hypothesis is false.

#### P3: Cosine Annealing + Warmup (+10 LOC)
- **Mechanism:** Linear warmup over 5 epochs, then cosine decay to 0 over remaining epochs. Standard in Sapiens, MotionBERT.
- **Falsification:** If Δ > -0.1mm, the paper's exponential decay with γ=0.99 is already near-optimal.

---

## 4. Experiment Pipeline

```
Exp 00: Baseline (PoseMamba-S, batch=4, grad_acc=8, exp decay)
    → Verify 41.8mm MPJPE (paper match)
    
Exp 01: Training Optimization (P3+P4)
    → Cosine warmup + grad accumulation + grad clipping
    → Est. Δ ≈ -0.4mm (free lunch)
    
Exp 02: GCN+Mamba Dual-Stream (P0)
    → Parallel GCN branch with adaptive fusion
    → Est. Δ ≈ -0.9mm (core architectural win)
    
Exp 03: Structure-Aware Stride Scan (P1)
    → Grouped stride scan over skeleton partitions
    → Est. Δ ≈ -0.5mm
    
Exp 04: Bone-Aware Module (P2)
    → Bone direction/length vectors fused before SSM
    → Est. Δ ≈ -0.4mm
    
Exp 05: Compound (P0+P1+P2)
    → Best of all architectural interventions
    → Est. cumulative Δ ≈ -1.5mm target (41.8 → 40.3mm)
    
Exp 06: Hybrid Attention (P7)
    → Add 2 sparse attention layers to SSM pipeline
    → Est. Δ ≈ -0.3mm
```

### Staged Validation Protocol

| Stage | What | Time | Decision |
|-------|------|------|----------|
| **S1** | Correctness (CPU fwd/bwd, NaN check) | 5 min | Fix or reject architecture |
| **S2** | Overfit (16 samples × 300 epochs) | 10 min | Near-zero loss or reject |
| **S3** | Short validation (25% epochs = 30 epochs) | ~2.5 hrs | Δ < -0.2mm → proceed; Δ > +0.1mm → reject |
| **S4** | Full training (120 epochs × 3 seeds) | ~36 hrs | p < 0.05 AND Δ < -0.1mm → merge to compound |

---

## 5. Metrics & Ablation Controls

### Primary Metrics
  - **Protocol 1:** MPJPE (mm) — mean per-joint position error
  - **Protocol 2:** P-MPJPE (mm) — Procrustes-aligned MPJPE
  - **Per-joint MPJPE:** Identify which joints improve/worsen
  - **5 trials with 95% bootstrap CI:** Statistical significance

### Secondary Metrics
  - **BLCE (Bone Length Consistency Error):** std of bone length across frames
  - **Velocity MPJPE:** first-order temporal consistency
  - **3DPW zero-shot:** cross-dataset generalization

### Ablation Controls
  - Ablate adaptive fusion weight α (learnable vs fixed 0.5)
  - Ablate GCN depth (1 vs 2 vs 3 layers)
  - Ablate striding factor (k=2 vs k=4 in stride scan)

---

## 6. Leaderboard

| Exp | Model | P1 MPJPE ↓ | P-MPJPE ↓ | Params | Δ | Status |
|-----|-------|-----------|-----------|--------|---|--------|
| — | PoseMamba-S (paper) | 41.8 | — | 0.9M | — | Published |
| — | PoseMamba-B (paper) | 40.8 | — | 3.4M | — | Published |
| — | PoseMamba-L (paper) | 38.1 | — | 6.7M | — | Published |
| — | PoseMagic (Mamba+GCN) | 40.9 | — | ~1.2M | -0.9 | AAAI 2025 |
| — | DBMambaPose | 40.5 | — | ~2.0M | -1.3 | arXiv 2025 |
| — | SasMamba | 40.2 | — | ~1.5M | -1.6 | WACV 2026 |
| — | MambaTopFusion | 39.8 | — | ~2.5M | -2.0 | arXiv 2026 |
| 00 | PoseMamba-S (ours) | — | — | — | — | ✅ Planned |
| 01 | + Training Opt | — | — | — | — | ✅ Planned |
| 02 | + GCN-Mamba | — | — | — | — | ✅ Planned |
| 03 | + Stride Scan | — | — | — | — | ✅ Planned |
| 04 | + Bone Module | — | — | — | — | ✅ Planned |
| 05 | + Compound | — | — | — | — | ✅ Planned |

---

## 7. Experiment Journal

### Exp 00: Baseline Reproduction
- **Status:** Running
- **Branch:** `main`
- **Config:** `configs/pose3d/PoseMamba_train_h36m_S.yaml`
- **Target:** 41.8mm P1 MPJPE
- **Notes:** Using gradient accumulation 8 (batch=4, effective batch=32) to match paper

### Exp 01: Training Optimization (P3+P4)
- **Status:** Pending
- **Branch:** `exp01_train_opt`
- **Changes:** Cosine warmup, grad accumulation, gradient clipping
- **Target:** 41.5mm P1 MPJPE

### Exp 02: GCN-Mamba Dual-Stream (P0)
- **Status:** Pending
- **Branch:** `exp02_gcn_mamba`
- **Changes:** Parallel GCN branch + adaptive fusion
- **Target:** 40.9mm P1 MPJPE

---

## 8. Lessons

*What Works:*
- (TBD)

*What Doesn't Work:*
- (TBD)
