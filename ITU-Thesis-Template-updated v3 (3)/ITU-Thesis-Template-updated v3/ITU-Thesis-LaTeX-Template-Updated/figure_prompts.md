# KinecMamba thesis — image-generation prompts for all figures

One prompt per figure the thesis references. Generate each image, then place it under
`figures/` with the filename noted, and (for the two `\includegraphics` slots) uncomment
the `\includegraphics` line in the `.tex`.

Thesis figures:
| # | Label in .tex | File to save | Status in thesis |
|---|---------------|--------------|------------------|
| 1 | `fig:architecture-pipeline` (methodology.tex) | `figures/architecture.pdf` | image slot (uncomment `\includegraphics`) |
| 2 | `fig:kinematic-tree` (methodology.tex) | `figures/kinematic_tree.pdf` | currently TikZ; replace with image if desired |
| 3 | `fig:bistssm-layer` (methodology.tex) | `figures/bistssm_layer.pdf` | currently TikZ; replace with image if desired |
| 4 | `fig:scan-order-comparison` (methodology.tex) | `figures/scan_order.pdf` | currently TikZ; replace with image if desired |
| 5 | `fig:results-plot` (results.tex) | `figures/results_plot.pdf` | image slot; DATA-driven (see note) |

**Facts to keep consistent across every figure** (do not change):
- 17 joints (Human3.6M order): 0 Pelvis, 1 Right Hip, 2 Right Knee, 3 Right Ankle,
  4 Left Hip, 5 Left Knee, 6 Left Ankle, 7 Spine, 8 Thorax, 9 Neck, 10 Head,
  11 Left Shoulder, 12 Left Elbow, 13 Left Wrist, 14 Right Shoulder, 15 Right Elbow,
  16 Right Wrist.
- 16 bones (parent -> child): 0-1,1-2,2-3 (right leg); 0-4,4-5,5-6 (left leg);
  0-7,7-8,8-9,9-10 (spine/head); 8-11,11-12,12-13 (left arm); 8-14,14-15,15-16 (right arm).
- BFS kinematic order pi = 0 1 4 7 2 5 8 3 6 9 11 14 10 12 15 13 16.
- Token width C = 64; SSM hidden state N = 16; frames T = 243; joints J = 17; 20 blocks (10 x [spatial + temporal]).

**Shared visual style (prepend or assume for every prompt):** publication-quality figure
for a top computer-vision conference (CVPR/ICCV/ECCV/NeurIPS), looks hand-made in Adobe
Illustrator or Figma, not AI-generated, not a PowerPoint flowchart. Flat vector graphics,
white background, thin dark-gray outlines, rounded rectangles, soft blue for feature
tensors, soft gray for computation modules, muted amber only for the proposed
contribution, Helvetica/Arial typography, perfect alignment and even spacing, generous
whitespace, clean thin arrows (tensor-flow ribbons where natural). No gradients, no
shadows, no 3D rendering, no clip art, no decorative elements, no photorealism. Use
exactly the labels given; do not invent extra captions, equations, or legends.

────────────────────────────────────────────────────────────────────────
## FIGURE 1 — Architecture pipeline  (save as figures/architecture.pdf)
────────────────────────────────────────────────────────────────────────

Create a publication-quality architecture diagram for a top-tier computer vision conference paper (CVPR, ICCV, ECCV, NeurIPS). The figure must look manually designed in Adobe Illustrator or Figma by a researcher, not AI-generated, and not like a PowerPoint flowchart.

Use a clean landscape (16:9) layout with a modern horizontal network design. The main backbone runs left to right across the center. Place a compact module inset in the upper-right connected with a thin callout line, and place the proposed contribution panel across the bottom. Resemble architecture diagrams from recent Transformer, Mamba, MotionBERT, or Vision Transformer papers.

Style: flat vector graphics; white background; thin dark-gray outlines; rounded rectangles; soft blue for feature tensors; soft gray for computation modules; muted amber only for the proposed contribution panel; professional Helvetica/Arial typography; perfect alignment and equal spacing; large whitespace; tensor-flow ribbons or clean feature arrows; no gradients; no shadows; no 3D rendering; no clip art; no decorative elements; no photorealism.

MAIN NETWORK (center):
- Far left: a small 2D human pose skeleton (stick figure with joints) labeled **2D HPE**. Next to it: **Input: 2D keypoints (B, T, J, 2)**.
- Connect to **Spatial Token Embedding (STE)**, then **Temporal Token Embedding (TTE)**. Small annotation above the stream: **C = 64**.
- Backbone: one repeated stage enclosed by a long dashed bracket labeled **×10**. Inside one stage: **Spatial BiSTSSM Block** then (↓) **Temporal BiSTSSM Block**. Below the bracket: **10 repetitions = 20 blocks total**.
- Continue to **Output head: LayerNorm then Linear C to 3**.
- Far right: a small 3D human pose skeleton in perspective labeled **3D HPE**. Below it: **Output: 3D pose (B, T, J, 3)**.

MODULE INSET (upper right, thin callout line from the Spatial BiSTSSM Block), title **BiSTSSM layer**: Input Features (↓) **Four-directional Cross-Scan** (↓) a dashed container labeled **×4 (one per direction)** holding four small parallel blocks labeled **Selective SSM**; merge into **Cross-Merge** (↓) **Output Features**. Keep it conceptual and clean.

PROPOSED CONTRIBUTION (bottom, muted amber border), title **BFS kinematic-tree scan order**:
- Left: a clean rooted kinematic tree with labeled joints and thin edges: Pelvis → {Right Hip → Right Knee → Right Ankle}, {Left Hip → Left Knee → Left Ankle}, {Spine → Thorax}; Thorax → {Neck → Head}, {Left Shoulder → Left Elbow → Left Wrist}, {Right Shoulder → Right Elbow → Right Wrist}.
- Right: two horizontal strips. **Naive index order**: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16. Below (↓) **BFS kinematic order**: 0 1 4 7 2 5 8 3 6 9 11 14 10 12 15 13 16.
- Between the backbone and this panel: **permute joints before scan, inverse-permute after merge**.

Use exactly these labels. The final figure should be indistinguishable from a manually created architecture figure in a recent CVPR/ICCV paper.

────────────────────────────────────────────────────────────────────────
## FIGURE 2 — Kinematic tree  (save as figures/kinematic_tree.pdf)
────────────────────────────────────────────────────────────────────────

[Apply the shared visual style above.] A clean rooted-tree diagram titled nothing (no
title inside the image) showing the 17-joint Human3.6M skeleton as a kinematic tree.
Each joint is a rounded rectangle node labeled with its index and name; the pelvis node
(root, index 0) is shaded soft gray to mark the root, all others white. Thin dark-gray
edges connect parents to children (each edge is a bone). Lay the tree top-to-bottom, root
at top, in the exact structure:

- 0 Pelvis (root) has three children: 1 Right Hip, 4 Left Hip, 7 Spine.
- 1 Right Hip → 2 Right Knee → 3 Right Ankle.
- 4 Left Hip → 5 Left Knee → 6 Left Ankle.
- 7 Spine → 8 Thorax.
- 8 Thorax has three children: 9 Neck, 11 Left Shoulder, 14 Right Shoulder.
- 9 Neck → 10 Head.
- 11 Left Shoulder → 12 Left Elbow → 13 Left Wrist.
- 14 Right Shoulder → 15 Right Elbow → 16 Right Wrist.

Keep the left/right legs on the outer sides and the spine chain in the middle so the tree
is symmetric and uncluttered. Use exactly these 17 index+name labels; add no other text.

────────────────────────────────────────────────────────────────────────
## FIGURE 3 — BiSTSSM layer  (save as figures/bistssm_layer.pdf)
────────────────────────────────────────────────────────────────────────

[Apply the shared visual style above.] A clean vertical module-pipeline diagram (portrait)
of one BiSTSSM layer, top to bottom, using rounded rectangles and thin downward arrows:

1. **Input: Z (T × J × C), post-LayerNorm** (soft blue tensor block)
2. ↓ **Four-directional cross-scan** — unfold the T × J grid into 4 sequences (soft gray)
3. A dashed container labeled **×4 (one per direction)** enclosing this vertical sub-stack:
   - **Input projection: Linear C → 2C**
   - ↓ **Depthwise Conv1D (kernel 3)**
   - ↓ **Selective SSM: input-dependent Δ, B, C; diagonal A; state size N = 16**
4. ↓ **Cross-merge: recombine the 4 directional outputs** (soft gray)
5. ↓ **Output projection: Linear 2C → C**
6. ↓ **Output: O (T × J × C)** (soft blue tensor block)

Emphasize the concept, keep it compact. Use exactly these labels; no extra annotations.

────────────────────────────────────────────────────────────────────────
## FIGURE 4 — Scan-order comparison  (save as figures/scan_order.pdf)
────────────────────────────────────────────────────────────────────────

[Apply the shared visual style above.] A compact figure with two horizontal strips of 17
small rounded square cells each, left-aligned and vertically stacked with a thin downward
arrow between them. Each cell shows an abbreviated joint name with its index as a subscript.

Top strip, title **Naive index order** (white cells), left to right:
Pel(0) RHp(1) RKn(2) RAn(3) LHp(4) LKn(5) LAn(6) Spn(7) Thx(8) Nck(9) Hd(10) LSh(11) LEl(12) LWr(13) RSh(14) REl(15) RWr(16)

Bottom strip, title **BFS kinematic order** (cells lightly shaded soft blue), left to right:
Pel(0) RHp(1) LHp(4) Spn(7) RKn(2) LKn(5) Thx(8) RAn(3) LAn(6) Nck(9) LSh(11) RSh(14) Hd(10) LEl(12) REl(15) LWr(13) RWr(16)

The point the figure conveys: under the BFS order, joints that are close in the kinematic
tree (e.g. the two hips and the spine, all children of the pelvis) become adjacent, while
the naive order interleaves unrelated joints. Use exactly these labels; no extra text.

────────────────────────────────────────────────────────────────────────
## FIGURE 5 — Results plot  (save as figures/results_plot.pdf)  [DATA-DRIVEN]
────────────────────────────────────────────────────────────────────────

NOTE: this figure encodes real numbers, so it should be produced from data (e.g. a
matplotlib script) rather than an image model, once the final results are available. Do
NOT invent data points. A matching matplotlib spec:

- Scatter plot. X axis: number of parameters (millions, log scale). Y axis: MPJPE (mm),
  lower is better. Title: none. Grid: light. One marker per method, each annotated with
  the method name.
- Points (from the thesis Table): MixSTE (33.6, 40.9), STCFormer (4.7, 41.0),
  MotionBERT (42.3, 39.2), MotionAGFormer-L (19.0, 38.4), Pose Magic (14.4, 37.5),
  PoseMamba-S (0.9, 41.8), PoseMamba-L (6.7, 38.1), SasMamba (0.64, 41.48),
  SasMamba-large (4.1, 39.77).
- KinecMamba (ours): plot at (1.26, <your measured MPJPE>) as a filled amber star,
  clearly highlighted. Leave a `# TODO: MPJPE` placeholder until the number is final.
- Style: clean, minimal, publication-ready; soft-blue baseline markers, amber highlight
  for ours; no chartjunk.

If you must generate it as an image instead, describe the same axes/points/highlight and
mark the ours-point as pending — but a plotted matplotlib figure is strongly preferred for
correctness.

────────────────────────────────────────────────────────────────────────
## OPTIONAL — building blocks and qualitative figure
────────────────────────────────────────────────────────────────────────

- **2D and 3D pose skeleton icons** (only if you want them standalone, not just inside
  Figure 1): a flat 2D stick-figure skeleton (frontal, 17 joints as dots, bones as thin
  lines) labeled **2D pose**; and a 3D skeleton drawn in light perspective labeled **3D
  pose**. Same flat vector style, no photorealism.
- **Qualitative results** (future work, `figures/qualitative.pdf`): a row of 3–4 sample
  frames, each showing the input 2D pose, the predicted 3D skeleton, and the ground-truth
  3D skeleton side by side, clean vector skeletons on white. Use only your own model
  outputs; do not fabricate poses.
