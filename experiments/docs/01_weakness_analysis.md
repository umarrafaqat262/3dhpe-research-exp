# Weakness Analysis: PoseMamba

## 1.1 Architectural Weaknesses (Code-Level Analysis)

| # | Weakness | Location | Severity | Description |
|---|----------|----------|----------|-------------|
| W1 | **No true spatio-temporal separation** — STE and TTE blocks use identical `BiSTSSMBlock` with same `forward_type='v2_plus_poselimbs'`. Both process `(B, T, J, D)` identically. The "spatial" vs "temporal" naming is cosmetic — only `blocks[0]` of each type receives positional encoding. | `PoseMamba.py` lines 95-131, `mambablocks.py` | **Critical** | After block 0, every block does the same 2D scan over the full T×J grid. No dedicated spatial-only or temporal-only pathway exists. This wastes parameters: 20 identical blocks instead of specialized spatio-temporal processing. |
| W2 | **plus_poselimbs has no gradient flow** — `CrossScan_plus_poselimbs.backward()` is identical to standard `CrossScan.backward()`. The skeleton blending `x + x[..., indices]` on path 0 is injected forward but gradient bypasses it via straight-through estimator. The skeleton prior is non-learnable. | `csms6s.py` lines 149-192 | **Critical** | The main spatial inductive bias (joint adjacency) cannot be refined through training. The backward treats path 0 as identity, missing the `scatter_add` gradient from the indices operation. |
| W3 | **Asymmetric skeleton prior** — The `indices` array heavily blends left-side joints (1-6, 11-13) while leaving right-side joints (14-16) mostly untouched. Some connections are bizarre (R Ankle→L Hip, L Wrist→R Shoulder). | `csms6s.py` line 156 | **High** | Creates unintended left-right processing asymmetry. Right arm/shoulder joints (14-16) map to themselves — no topology mixing at all. |
| W4 | **Very shallow head** — Single `LayerNorm + Linear(D, 3)` maps backbone features to 3D coordinates. No refinement, no residual, no multi-scale aggregation. All heavy lifting is on the backbone. | `PoseMamba.py` lines 89-92 | **High** | A linear projection from 64-dim features to 3D coordinates is extremely limited. Minor feature noise directly corrupts output. |
| W5 | **No position encoding in blocks after block 0** — Positional encoding (spatial and temporal) is added once before the first block of each stack. Blocks 1..N-1 receive no positional signal. | `PoseMamba.py` lines 95-131 | **Moderate** | Deeper blocks operate on position-agnostic representations. For T=243, positional information may be lost by block 5-6. |
| W6 | **Small state dimension (N=16)** — Each SSM channel has only 16 hidden states. Mamba-2 uses N=64-256. With K=4 scan groups, total effective state = 128×16=2048 for S variant. | `mambablocks.py` line 247 | **Moderate** | Limited state may bottleneck long-range temporal modeling for T=243 sequences. |
| W7 | **Memory blowup from 4× CrossScan** — Creates 4 copies of `(B, D', H, W)` tensor. For B variant with batch=4: ~64MB per SSM layer. 40 layers → ~2.5GB for scan buffers alone. | `csms6s.py` lines 4-75 | **Moderate** | GPU memory pressure limits batch size and model scale. |

## 1.2 Literature-Backed Weaknesses

| # | Weakness | Evidence | Δ Reported | Source |
|---|----------|----------|-----------|--------|
| W8 | **Pure SSM ignores skeletal topology** — 1D/2D scan treats joints as flat sequence, destroying graph structure | PoseMagic: Δ=-0.9mm, SasMamba: Δ=-1.6mm, MambaTopFusion: Δ=-2.0mm | -0.9 to -2.0mm | PoseMagic (AAAI 2025), SasMamba (WACV 2026), MambaTopFusion (CVIU 2026) |
| W9 | **Weak local joint dependency** — SSM selective state prioritizes long-range over local interactions | PoseMagic, HGMamba, GEM all propose GCN/Mamba hybrid | -0.9mm | PoseMagic (AAAI 2025), HGMamba (arXiv 2025), GEM (2026) |
| W10 | **Scan order is not skeleton-aware** — Flat row-major + column-major is optimal for images, not skeletons | SasMamba shows stride scan over body parts improves results | -0.5mm | SasMamba (WACV 2026) |
| W11 | **No explicit bone/length modeling** — Joint coordinates alone lack bone direction and length information | MambaTopFusion bone module gives additional gain | -0.4mm | MambaTopFusion (CVIU 2026) |

## 1.3 Weakness-to-Fix Mapping

| Weakness | Fix | Experiment | Est. Δ | Priority |
|----------|-----|------------|--------|----------|
| W2+W3 (no gradient flow, asymmetric prior) | Learnable skeleton adjacency (H12) | Exp 01 | -0.4mm | **P0** |
| W8+W9 (no spatial topology) | GCN-Mamba dual-stream (H1) | Exp 02 | -0.9mm | **P0** |
| W1 (no spatio-temporal separation) | SAMA-style state-level fusion inside dual-stream (H11) | Exp 03 | -0.6mm additive | P1 |
| W4 (shallow head) | Deeper head with residual refinement | Compound | -0.1mm | P1 |
| W10 (scan order) | Structure-aware stride scan (H2) | Deferred | -0.5mm | P2 |
| W11 (no bone modeling) | Bone-aware module (H6) | Deferred | -0.4mm | P2 |
| W5 (position encoding decay) | Add positional encoding to deeper blocks | Deferred | -0.1mm | P2 |
| W6 (small state) | Mamba-2 SSD with N=64 (H9) | Deferred | -0.2mm | P3 |
| W7 (memory) | Fused kernel or cascade2d variant | Deferred | N/A | P3 |

## 1.4 Criticality Assessment

**Critical weaknesses** (block publication without address):
- W2+W3: The main spatial inductive bias is broken (no gradient flow) and asymmetric
- W8: Pure SSM without topology modeling is the fundamental limitation proven by 3+ papers

**High-severity** (significant impact on results):
- W1: Wasted parameters on identical blocks
- W4: Shallow head loses precision

**Moderate** (should address in compound):
- W5, W6, W10, W11

**Deferred** (engineering optimizations):
- W7, W9 (resolved by H1)
