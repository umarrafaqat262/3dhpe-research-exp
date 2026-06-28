# Literature Review: Beyond PoseMamba

## 2.1 SSM Evolution (NLP Origins)

### Mamba-1 (Gu & Dao, 2023)
- **Key innovation:** Selective state space models — input-dependent parameterization of SSM dynamics (B, C matrices)
- **Limitation for HPE:** 1D causal scanning destroys spatial topology; small state dimension (N=16)
- **Relevance:** PoseMamba is built on Mamba-1; inherits all its limitations

### Mamba-2 (Dao & Gu, ICML 2024)
- **Key innovations:**
  - **Structured State-Space Duality (SSD):** Unified view of SSMs and attention; enables 2-8x faster training
  - **Scalar A parameter:** Replaces diagonal A with scalar times identity; simpler, more stable
  - **Head dimension:** Uses larger state dim (N=64-256 vs N=16 in Mamba-1)
  - **Chunked training:** Processes sequences in chunks for memory efficiency
- **Not yet in HPE:** No published adaptation to 3D pose estimation
- **Relevance:** State dimension increase (N=16→64) may improve capacity for complex pose dynamics
- **Reference:** Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024

### Mamba-3 (Lahoti et al., 2026)
- **Key innovations:**
  - **Complex-valued state representation:** A, B, C matrices in complex domain for richer dynamics
  - **MIMO SSM:** Multiple-input multiple-output processing instead of MISO
  - **RoPE integration:** Rotary Position Embedding for SSM (previously SSMs had no explicit position encoding)
  - **QKNorm:** Query-key normalization for training stability (borrowed from attention)
  - **Inference-first design:** Architecture optimized for autoregressive inference
- **Relevance:** RoPE and QKNorm can be directly applied to PoseMamba's spatial/temporal blocks
- **Reference:** Lahoti et al., "Mamba-3: Overcoming the Mamba-Kernel Paradox", arXiv 2026

### Hybrid SSM-Attention Architectures

| Model | SSM% | Attn% | Key Idea | Reference |
|-------|------|-------|----------|-----------|
| **Jamba** | 93% | 7% | Interleave Mamba layers with sparse attention every 8th layer | AI21 Labs, 2024 |
| **TransMamba** | 67% | 33% | Dual-path: one SSM, one attention, fuse outputs | arXiv 2025 |
| **Nemotron** | 95% | 5% | 1 attention layer for every 19 Mamba layers; production-grade | NVIDIA, 2025 |
| **Samba** | 50% | 50% | Alternating Mamba and attention layers | arXiv 2024 |

**Key finding:** Even 5-7% attention layers significantly improve long-range dependency modeling
over pure SSM. For HPE with T=243 frames, adding 2 sparse attention layers (≈10% of depth)
may improve temporal coherence without quadratic cost.

## 2.2 Vision SSMs

### VMamba (Liu et al., 2024)
- **Innovation:** 2D cross-scan for image processing — scans rows and columns separately
- **Relevance:** PoseMamba's scan strategy (flatten J×T to H×W) is inspired by VMamba

### Spatial-Mamba (ICLR 2025)
- **Innovation:** Replaces 1D causal conv in SSM with 3×3 depthwise conv for spatial locality
- **Key result:** Significantly better local feature extraction in vision tasks
- **For HPE:** Joint-level spatial conv (17×17 depthwise) could model local adjacency better than 1D causal conv

### ASGMamba (arXiv 2026)
- **Innovation:** Adaptive spectral gating via patch-level FFT; selects frequency bands adaptively
- **For HPE:** Motion dynamics are frequency-selective (slow torso, fast limbs) — FFT gating could filter noise while preserving motion

### LocalViM (arXiv 2025)
- **Innovation:** Windowed local scanning for vision; processes patches in windows before global scan
- **For HPE:** Windowed scanning over joint groups (arm, leg, torso windows) before full-body scan

## 2.3 3D HPE with SSMs (Direct Competitors)

### PoseMamba (Huang et al., AAAI 2025)
- **Our baseline:** 38.1mm P1 MPJPE on H36M (L variant)
- **Architecture:** Bidirectional global-local spatio-temporal SSM
- **Weaknesses:** See `01_weakness_analysis.md`

### PoseMagic (AAAI 2025)
- **Key idea:** Dual-stream GCN + Mamba with adaptive fusion
- **Δ over PoseMamba:** -0.9mm P1 MPJPE on H36M
- **Why it works:** GCN branch explicitly models skeletal adjacency that Mamba ignores
- **Implementation:** Parallel spatial GCN (17-node skeleton graph) + temporal conv, learnable α to fuse
- **Reference:** "PoseMagic: Dual-stream GCN-Mamba for 3D Human Pose Estimation", AAAI 2025

### SasMamba (WACV 2026)
- **Key idea:** Structure-aware stride scan over skeleton topology (6 partitions)
- **Δ over PoseMamba:** -1.6mm P1 MPJPE (estimated)
- **Why it works:** Stride scan preserves local joint adjacency by scanning within skeleton groups
- **Reference:** "SasMamba: Structure-Aware Stride State Space Model for 3D Human Pose Estimation", WACV 2026

### DBMambaPose (arXiv 2025)
- **Key idea:** Decoupled spatial and temporal bidirectional Mamba scans with separate parameters
- **Δ over PoseMamba:** -1.3mm P1 MPJPE (estimated)
- **Why it works:** Flattening J×T → H×W (as PoseMamba does) entangles dimensions; separating them lets each SSM specialize
- **Reference:** "DBMambaPose: Decoupled Bidirectional Mamba for 3D Human Pose Estimation", arXiv 2025

### Mamba-Driven Topology Fusion (arXiv 2026)
- **Key idea:** Bone-aware module with direction/length vectors fused with joints before SSM
- **Δ over PoseMamba:** -2.0mm P1 MPJPE (estimated)
- **Why it works:** Bone vectors provide structural prior that pure joint coordinates lack
- **Reference:** "Mamba-Driven Topology Fusion for 3D Human Pose Estimation", arXiv 2026

### HGMamba (arXiv 2025)
- **Key idea:** Hierarchical GCN + Mamba with multi-scale fusion
- **Δ over PoseMamba:** -0.7mm P1 MPJPE (estimated)

## 2.4 Cross-Domain Transfer Map

```
NLP SSM Literature         Vision SSM Literature         3D HPE Literature
══════════════════         ════════════════════         ════════════════════
Mamba-2: SSD, N=64+        Spatial-Mamba: 3×3 conv     PoseMagic: GCN+Mamba  ──→ P0
Mamba-3: RoPE, QKNorm      ASGMamba: FFT gating        SasMamba: stride scan ──→ P1
Hybrid Attn+SSM            VMamba: 2D cross-scan        DBMambaPose: S-T dec  ──→ P3
Gated DeltaNet             LocalViM: windowed scan      MambaTopFusion: bone  ──→ P2
                                                       HGMamba: hierarchical ──→ compound
                               ↓                              ↓
                         Training Best Practices     Training Best Practices
                         ──────────────────────     ──────────────────────
                         Cosine + warmup            Gradient accumulation
                         Gradient clipping          ExponentialLR → Cosine
                         Layer-wise LR decay        Effective batch size
```

## 2.5 Hybrid Mamba Architectures: Taxonomy and Insights

A systematic analysis of the Mamba-based 3D HPE literature (2025-2026) reveals
**three distinct hybrid integration strategies**, each with different mechanisms,
strengths, and unexplored combinations.

### Type 1: Feature-Level Fusion (Dual-Stream)

**How it works:** Two parallel streams (Mamba + GCN/Attention) process the input
independently, then fuse at the feature level via adaptive weighting.

| Model | Stream 1 | Stream 2 | Fusion | Δ vs PoseMamba | Venue |
|-------|----------|----------|--------|----------------|-------|
| **PoseMagic** | Mamba (global) | GCN (local) | Adaptive α | -0.9mm | AAAI 2025 |
| **HGMamba** | Mamba (global) | HyperGCN (local) | Adaptive α | -0.4mm | arXiv 2025 |
| **AGMamba** | Mamba (global+frequency) | Attention-GCN | Adaptive fusion | SOTA | SIVP 2026 |
| **PoseRWGCN** | RWKV (global) | GCN (local) | Learned α | 42.2mm P1 | Complex & Intel. 2026 |

**Strengths:** Modular, easy to implement, proven across 4+ papers, Δ is additive.
**Weakness:** Fusion is post-hoc — streams do not interact during computation.

### Type 2: State-Level Fusion (Embedded)

**How it works:** GCN or topology-aware processing is embedded *inside* the SSM state
update, modifying how the state integrates information at each step.

| Model | Mechanism | Δ | Venue |
|-------|-----------|---|-------|
| **SAMA** | Structure-aware State Integrator (SSI) fuses joint topology into state space; Motion-adaptive State Modulator (MSM) adjusts timescale per joint | SOTA | **ICCV 2025** |
| **MambaTopFusion GEM** | Bidirectional GCN embedded before SSM's causal conv + SSM, providing local features directly to state update | -2.0mm | CVIU 2026 |

**Strengths:** More fundamental integration — topology affects every state transition.
**Weakness:** Higher implementation complexity, harder to ablate.

**Key insight:** SAMA's SSI operates at the *state level* (modifying how SSM state evolves
based on joint topology), while PoseMagic's GCN operates at the *feature level* (modifying
representations before/after SSM). **These are complementary — no paper combines both.**
This represents a novel research gap.

### Type 3: Mamba-Attention Hybrid

**How it works:** Interleaves Mamba layers with sparse attention layers, or uses Mamba
for temporal processing and Attention for spatial processing.

| Model | Design | Mamba:Attention | Domain | Venue |
|-------|--------|----------------|--------|-------|
| **VIMCAN** | Mamba temporal + Cross-Attention spatial fusion | ~1:1 per block | Visual-Inertial HPE | **CVPR 2026** |
| **Jamba-1.5** | Interleaved Mamba + Attention + MoE layers | 7:1 | NLP | AI21, 2025 |
| **TransMamba** | Shared QKV/CBx params, dynamic switching | Variable per token | NLP | **AAAI 2026** |
| **Nemotron-H** | 92% Mamba2 + 8% attention layers | 12:1 | NLP | NVIDIA, 2025 |
| **Bamba-9B** | Mamba2 + Transformer, 2× throughput | ~4:1 | NLP | IBM, 2025 |

**Key finding from Jamba-1.5:** In hybrid architectures, **Mamba-1 + Attention outperforms
Mamba-2 + Attention**. The advantages of Mamba-2 (larger state) are less significant when
attention layers are present, since attention can pool information from the entire context.

**For 3D HPE:** VIMCAN (CVPR 2026) proves the Mamba-Attention hybrid works for HPE,
but it uses IMU data — the approach is unproven for pure 2D→3D lifting.

## 2.6 Key Insight: Consensus Across Literature

Every paper surveyed (PoseMagic, SasMamba, DBMambaPose, MambaTopFusion, HGMamba,
Spatial-Mamba, SAMA, AGMamba) independently identifies the **same fundamental
limitation**:

> **Mamba's 1D sequential scanning destroys the spatial topology of the skeleton.**
> Pure SSM-based pose estimation underperforms on local joint relationships, particularly
> at limbs and extremities.

The proposed fixes fall into the three categories above:

| Fix Type | Papers | Example |
|----------|--------|---------|
| Feature-level dual-stream | PoseMagic, HGMamba, AGMamba | Mamba + GCN in parallel |
| State-level embedding | SAMA, MambaTopFusion GEM | Topology-aware SSM state update |
| Scan-level modification | SasMamba, DBMambaPose | Stride scan, decoupled S-T |

**Unexplored combinations:**
1. **State-level + Feature-level fusion** (SAMA × PoseMagic) — novel, could yield -1.5mm+
2. **Mamba-Attention hybrid for 2D→3D lifting** (VIMCAN style without IMU)
3. **Mamba-2 SSD + GCN dual-stream** (no paper has tried Mamba-2 for HPE)

This taxonomy strengthens the case for prioritizing architectural interventions (H1, H2, H6)
and suggests H11 (SAMA-style state fusion) as a promising follow-up.
