# Weakness Analysis: PoseMamba
*Updated: 2026-06-28 — Added 5 new code audit findings (W12-W16)*

## Changelog
- **2026-06-28:** Added W12 (L config mismatch), W13 (confidence stripped), W14 (bone code exists),
  W15 (layer hooks already done), W16 (plus_poselimbs backward analysis deepened).
  Updated severity assessments with feasibility scores.

---

## 1.1 Architectural Weaknesses (Code-Level Analysis)

| # | Weakness | Location | Severity | Description | Feasibility to Fix |
|---|---|---|---|---|---|
| W1 | **No true spatio-temporal separation** — STE and TTE blocks use identical `BiSTSSMBlock` with same `forward_type='v2_plus_poselimbs'`. Both process `(B, T, J, D)` identically. The "spatial" vs "temporal" naming is cosmetic — only `blocks[0]` of each type receives positional encoding. | `PoseMamba.py` lines 95-131, `mambablocks.py` | **Critical** | After block 0, every block does the same 2D scan over the full T×J grid. No dedicated spatial-only or temporal-only pathway exists. This wastes parameters: 20 identical blocks instead of specialized spatio-temporal processing. | Deferred (P2) |
| W2 | **plus_poselimbs has NO gradient flow** — `CrossScan_plus_poselimbs.backward()` is IDENTICAL to standard `CrossScan.backward()`. The skeleton blending `x + x[..., indices]` on path 0 is injected forward but gradient completely bypasses it via straight-through estimator. The skeleton prior contributes NOTHING to learning. | `csms6s.py` lines 149-192 | **Critical** | **[DEEPENED ANALYSIS]** The forward pass does: `xs[:, 0] = (x + x[..., indices]).flatten(2,3)`. The backward (line 163-169): `ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])... y = ys[:, 0] + ys[:, 1]...`. This is identical to standard CrossScan backward. The `indices` operation is invisible to autograd — zero gradient for the skeleton prior. The entire topology computation at path 0 is dead computation. | **A3: Learnable adjacency — 🟢 95%** |
| W3 | **Asymmetric skeleton prior** — `indices = [0,0,1,2,3,0,4,5,6,8,11,12,13,8,14,15,16]` heavily blends left-side joints (1-6 → mapped to 0,2,3; 11-13 → mapped to 8) while right-side joints (14-16) map to themselves. Joint 8 (thorax) is referenced twice as a target. | `csms6s.py` line 156 | **High** | **[DEEPENED ANALYSIS]** Right arm/shoulder joints (14=RShoulder, 15=RElbow, 16=RWrist) map to themselves — no topology mixing at all. Left arm (11=LShoulder, 12=LElbow, 13=LWrist) maps to [8, 8, 8] (all → thorax). Root (0) used as source for 3 joints (1,4,6) but also maps to itself. No physically meaningful pattern. | **A3: Learnable replacement — 🟢 95%** |
| W4 | **Very shallow head** — Single `LayerNorm + Linear(D, 3)` maps backbone features to 3D coordinates. No refinement, no residual, no multi-scale aggregation. All heavy lifting is on the backbone. | `PoseMamba.py` lines 89-92 | **High** | A linear projection from 64-dim features to 3D coordinates is extremely limited. Minor feature noise directly corrupts output. | Deferred (P2) |
| W5 | **No position encoding in blocks after block 0** — Positional encoding (spatial and temporal) is added once before the first block of each stack. Blocks 1..N-1 receive no positional signal. | `PoseMamba.py` lines 95-131 | **Moderate** | Deeper blocks operate on position-agnostic representations. For T=243, positional information may be lost by block 5-6. | Deferred (P3) |
| W6 | **Small state dimension (N=16)** — Each SSM channel has only 16 hidden states. Mamba-2 uses N=64-256. With K=4 scan groups, total effective state = 128×16=2048 for S variant. | `mambablocks.py` line 247 | **Moderate** | Limited state may bottleneck long-range temporal modeling for T=243 sequences. | Deferred (P3) |
| W7 | **Memory blowup from 4× CrossScan** — Creates 4 copies of `(B, D', H, W)` tensor. For B variant with batch=4: ~64MB per SSM layer. 40 layers → ~2.5GB for scan buffers alone. | `csms6s.py` lines 4-75 | **Moderate** | GPU memory pressure limits batch size and model scale. | Deferred (P3) |

## 1.2 NEW — Code Audit Findings (2026-06-28)

| # | Finding | Location | Severity | Fix | Feasibility |
|---|---------|----------|----------|-----|-------------|
| **W12** | **L config uses inferior training recipe** — L config has no `accum_steps`, `warmup_epochs`, `lr_scheduler: cosine`, or `grad_clip_norm`. Uses `lr_decay: 0.99` (exponential) while S config uses cosine + warmup. Any comparison between S and L is invalid. | `configs/pose3d/PoseMamba_train_h36m_L.yaml` | **High** | Copy S config's training params into L config | 🔵 100% |
| **W13** | **Confidence score stripped from data** — `DataReaderH36M.read_2d()` reads confidence and concatenates as 3rd channel `[N, 17, 3]`. `no_conf: True` strips it at `train.py:224`. The model's `in_chans=2` doesn't expect the 3rd channel. | `PoseMamba.py:38`, `train.py:224` | **Low** (easy fix) | Set `no_conf: False`, change `in_chans=2→3` | 🔵 **100%** |
| **W14** | **Bone computation exists but unused** — `loss.py:100-114` has `get_limb_lens()`, lines 150-184 have `get_angles()`. These compute bone vectors and angles already. A5 can import and reuse them — no need to re-implement bone math. | `loss.py` lines 100-114, 150-184 | **Informational** | Import `get_limb_lens` for A5 bone vectors | 🔵 95% |
| **W15** | **Layer-wise hooks already implemented** — `train.py` lines 72-85 register forward hooks on STE/TTE blocks, lines 192-211 compute per-layer MPJPE during eval. Already logged to wandb. | `train.py` lines 72-85, 192-211 | **Informational** | No re-implementation needed. Docs should reference existing code. | 🔵 100% |
| **W16** | **W2 backward analysis deepened** — The `CrossScan_plus_poselimbs.backward` (line 163-169) has `ys[:, 0:2]` hard-coded, ignoring that forward path 0 contains `x + x[..., indices]` while standard path 0 contains just `x`. The backward computes the standard gradient but the expected gradient should be `grad[:,0] + scatter_add(grad[:,0], indices)`. This is a confirmed gradient bug. | `csms6s.py` lines 162-169 | **Critical** | Replace with learnable adjacency that uses `einsum` for proper autograd | 🟢 95% |

## 1.3 Literature-Backed Weaknesses

| # | Weakness | Evidence | Δ Reported | Source |
|---|----------|----------|-----------|--------|
| W8 | **Pure SSM ignores skeletal topology** — 1D/2D scan treats joints as flat sequence, destroying graph structure | PoseMagic: Δ=-0.9mm, SasMamba: Δ=-1.6mm, MambaTopFusion: Δ=-2.0mm | -0.9 to -2.0mm | PoseMagic (AAAI 2025), SasMamba (WACV 2026), MambaTopFusion (CVIU 2026) |
| W9 | **Weak local joint dependency** — SSM selective state prioritizes long-range over local interactions | PoseMagic, HGMamba, GEM all propose GCN/Mamba hybrid | -0.9mm | PoseMagic (AAAI 2025), HGMamba (arXiv 2025), GEM (2026) |
| W10 | **Scan order is not skeleton-aware** — Flat row-major + column-major is optimal for images, not skeletons | SasMamba shows stride scan over body parts improves results | -0.5mm | SasMamba (WACV 2026) |
| W11 | **No explicit bone/length modeling** — Joint coordinates alone lack bone direction and length information | MambaTopFusion bone module gives additional gain | -0.4mm | MambaTopFusion (CVIU 2026) |

## 1.4 Weakness-to-Fix Mapping (Updated)

| Weakness | Fix | Experiment | Est. Δ | Feasibility | Priority |
|----------|-----|------------|--------|-------------|----------|
| W12 (L config broken) | Copy S training recipe | A1 | 0mm (fairness) | 🔵 100% | P0 |
| W13 (confidence stripped) | `no_conf: False`, `in_chans=3` | A2 | -0.5mm | 🔵 100% | P0 |
| W14 (bone code unused) | Import `get_limb_lens`, concat input | A5 | -0.3mm | 🔵 95% | P0 |
| W2+W3+W16 (broken backward, asymmetry) | Learnable adjacency (einsum, proper grad) | A3 | -0.4mm | 🟢 95% | P0 |
| W8 (no topology) | GCN-Mamba dual-stream (but high risk of α→0) | B1 | -0.9mm | 🟡 60% | P1 |
| W1 (no S-T separation) | SAMA-style state-level fusion | A3 (partial) | -0.6mm additive | 🟢 90% | P1 |
| W4 (shallow head) | Deeper head with residual refinement | Compound | -0.1mm | 🟢 90% | P1 |
| W10 (scan order) | Structure-aware stride scan (H2) | B3 | -0.5mm | 🔵 100% | P2 |
| W11 (no bone modeling) | Bone-aware module (H6) | A5 | -0.4mm | 🔵 95% | P0 |
| W5 (position encoding decay) | Add positional encoding to deeper blocks | Deferred | -0.1mm | — | P3 |
| W6 (small state) | Mamba-2 SSD with N=64 (H9) | Deferred | -0.2mm | 🔴 40% | P3 |
| W7 (memory) | Fused kernel or cascade2d variant | Deferred | N/A | — | P3 |

## 1.5 Criticality Assessment (Updated)

**Critical weaknesses** (block publication without address):
- **W2+W3+W16**: plus_poselimbs has zero gradient flow + asymmetric → addressed by A3 (learnable adj)
- **W12**: L config training recipe broken → addressed by A1 (config fix) 
- **W8**: Pure SSM without topology modeling → addressed by A3 (SSI) + B1 (dual-stream)

**High-severity** (significant impact):
- **W13**: Confidence discarded → addressed by A2 (1 LOC)
- **W4**: Shallow head loses precision → deferred
- **W1**: Wasted parameters on identical blocks → partially addressed by A3

**Moderate** (should address in compound):
- W5, W6, W10, W11 (W11 addressed by A5)

**Informational** (resources already available, no fix needed):
- **W14** (bone code exists — use it for A5)
- **W15** (layer hooks already done — reference existing code)
