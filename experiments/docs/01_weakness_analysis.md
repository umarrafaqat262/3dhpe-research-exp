# Weakness Analysis: PoseMamba and Broader Literature

## 1.1 From PoseMamba Paper Analysis

The following weaknesses were identified by systematically analyzing the PoseMamba paper
(Huang et al., AAAI 2025), its supplementary material, codebase, and comparison with
subsequent SSM literature.

| # | Weakness | Type | Severity | Evidence | Impact on Claims |
|---|----------|------|----------|----------|------------------|
| W1 | **Pure SSM ignores skeletal topology** — 1D scan treats joints as flat sequence, destroying graph structure | Architectural | **Critical** | SasMamba (WACV 2026), MambaTopFusion (2026), Spatial-Mamba (ICLR 2025) all identify this as the fundamental SSM limitation for pose | SSM's linear scan cannot model joint adjacency; limbs predicted independently |
| W2 | **Weak local joint dependency** — SSM's selective state prioritizes long-range over local interactions | Architectural | **Critical** | PoseMagic (AAAI 2025), HGMamba, GEM — all propose GCN/Mamba hybrid to fix | Local limb geometry is poorly captured, causing higher per-joint error on extremities |
| W3 | **Only indoor benchmark (H36M)** — no wild video evaluation | Scope | **High** | Claude analysis; no 3DPW, EMDB, or Fit3D anywhere | Generalization to real-world deployment untested |
| W4 | **No cross-dataset evaluation** — train on H36M, test only on H36M | Scope | **High** | Claude analysis; no train-on-A, test-on-B protocol | Overfitting to dataset-specific biases undiscovered |
| W5 | **Single-run results, no variance** — tables report single MPJPE numbers | Statistical | **High** | Claude analysis; no mean ± std, no confidence intervals | SOTA margins often sub-1 mm — significance unconfirmable |
| W6 | **No statistical significance testing** — improvements asserted without tests | Statistical | **High** | Claude analysis; no t-tests, bootstrap, or Wilcoxon | Claims of "state-of-the-art" rest on single-point estimates |
| W7 | **Known per-joint failure: head/neck** — underperforms on head and neck | Design | **High** | Supplementary material, Claude analysis | Practical applications (face/gaze tracking) unaddressed |
| W8 | **2D detector not ablated** — only Stacked Hourglass used | Fairness | **Moderate** | Claude analysis; HRNet used by some baselines (asterisked) | Inconsistent comparison conditions |
| W9 | **No ablation of local scan order** — geometric reordering not varied | Design | **Moderate** | Claude analysis; only "with vs. without" tested, not which ordering | Optimal scan topology undetermined |
| W10 | **No bone-length constraint or temporal smoothness analysis** — MPJVE only | Scope | **Moderate** | Claude analysis; BLCE not measured | Temporal coherence beyond aggregate error unchecked |
| W11 | **Small state dimension (N=16)** — Mamba-2 uses N=64-256 | Capacity | **Low** | Mamba-2 SSD paper (Dao & Gu, ICML 2024) | SSM state may be bottleneck for complex pose dynamics |

## 1.2 From Broader Literature

| Domain | Finding | Source | Transferable To | Evidence Level |
|--------|---------|--------|-----------------|----------------|
| **NLP SSMs** | **Mamba-2 SSD**: larger state (N=64+), scalar A param, chunked training 2-8x faster | Dao & Gu, ICML 2024 | Replace selective_scan kernel; increase state dim | 4 (ICML) |
| **NLP SSMs** | **Hybrid SSM-Attention**: 7% attention + 93% SSM beats pure SSM | Jamba (AI21, 2024), TransMamba, Nemotron | Add sparse attention layers to PoseMamba | 4 (AI21 production) |
| **NLP SSMs** | **Mamba-3**: complex-valued states, MIMO SSM, RoPE, QKNorm | Lahoti et al., 2026 | RoPE for SSM ordering; QKNorm for stability | 3 (arXiv 2026) |
| **NLP SSMs** | **Gated DeltaNet**: replacement for selective scan with gating | Yang et al., 2024 | Alternative to Mamba block; simpler training | 2 (ICLR 2025 under review) |
| **Vision SSMs** | **Spatial-Mamba**: 3x3 depthwise conv replaces 1D causal conv for spatial locality | ICLR 2025 | Better local spatial modeling in SSM | 4 (ICLR) |
| **Vision SSMs** | **ASGMamba**: frequency-selective gating via patch-level FFT | arXiv 2026 | Noise filtering for pose dynamics | 2 (arXiv 2026) |
| **Vision SSMs** | **LocalViM**: local vision Mamba with windowed scanning | Huang et al., 2025 | Windowed scan for local joint groups | 3 (arXiv 2025) |
| **3D HPE** | **Dual-stream Mamba+GCN** with adaptive fusion beats PoseMamba by -0.9mm | PoseMagic (AAAI 2025) | **P0** — simplest highest-impact intervention | **5** (AAAI, same benchmark) |
| **3D HPE** | **Structure-aware stride scan** preserves skeleton topology | SasMamba (WACV 2026) | **P1** — topological fix for SSM scan | **3** (WACV) |
| **3D HPE** | **Bone-aware module** with direction/length vectors as topological prior | MambaTopFusion (arXiv 2026) | **P2** — strong inductive bias for skeleton | **3** (arXiv 2026) |
| **3D HPE** | **Decoupled S-T bidirectional scans** — separate SSM passes for space and time | DBMambaPose (arXiv 2025) | **P3** — cleaner dimension separation | **3** (arXiv 2025) |
| **3D HPE** | **Hierarchical GCN+Mamba** with multi-scale fusion | HGMamba (arXiv 2025) | Multi-scale approach for compound | **3** (arXiv 2025) |
| **Training** | **Cosine annealing + linear warmup** improves convergence over exponential decay | Loshchilov & Hutter, Sapiens (CVPR 2024) | Free lunch: config change only | **5** (standard practice) |
| **Training** | **Gradient accumulation** enables large effective batch on small GPU | Standard practice | Match paper batch=32 with L4 | **5** (engineering standard) |
| **Training** | **Gradient clipping (max_norm=1.0)** prevents gradient explosion | Sapiens, standard practice | Training stability | **4** (standard) |

## 1.3 Weakness-to-Fix Mapping

| Weakness | Fix | Experiment | Priority |
|----------|-----|------------|----------|
| W1 (topology ignored) | GCN-Mamba dual-stream (P0) | Exp 02 | P0 |
| W2 (local dependency) | Structure-aware stride scan (P1) | Exp 03 | P1 |
| W3 (only indoor) | Add 3DPW, EMDB zero-shot eval | Cross-cutting | High |
| W4 (no cross-dataset) | Train H36M → test MPI-INF-3DHP | Cross-cutting | High |
| W5 (single-run) | 5-seed protocol, report mean±std, 95% CI | Cross-cutting | High |
| W6 (no significance) | Wilcoxon signed-rank, Holm-Bonferroni | Cross-cutting | High |
| W7 (head/neck failure) | Head-Aware Branch (P6) + GCN-Mamba | Exp 02 + P6 | P2 |
| W8 (detector not ablated) | Add HRNet comparison | Cross-cutting | Medium |
| W9 (scan order) | Implicitly fixed by GCN-Mamba | Exp 02 | P0 |
| W10 (no BLCE) | Add BLCE metric | Cross-cutting | Medium |
| W11 (small state) | Increase d_state from 16 to 64 (P8) | Exp 08 | P2 |

## 1.4 Criticality Assessment

**Critical weaknesses** (must fix for publication):
- W1, W2: These are architectural — they limit the model's ceiling regardless of training improvements
- W3, W4: Without cross-dataset eval, reviewers will question generalization

**High-severity weaknesses** (strongly recommended):
- W5, W6: Without statistics, comparisons lack credibility
- W7: Head/neck errors are visually obvious in qualitative results

**Moderate weaknesses** (should address):
- W8, W9, W10: Important for thoroughness but not blocking

**Low-severity weaknesses** (nice-to-have):
- W11: Can be bundled with compound experiments
