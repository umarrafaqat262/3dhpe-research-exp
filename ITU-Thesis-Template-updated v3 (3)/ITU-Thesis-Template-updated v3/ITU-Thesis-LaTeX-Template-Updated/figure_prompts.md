# KinecMamba thesis — optional architecture-figure sketch aid

NOTE: the thesis figures are drawn as editable vector graphics directly in LaTeX
(TikZ for the architecture, kinematic tree, BiSTSSM layer, and scan-order figures;
pgfplots for the results plot), which is the standard academic approach and requires
no external images. This file is OPTIONAL: use the prompt below only if you want to
sketch or redraw the architecture figure in an image tool (e.g. draw.io / Illustrator)
and export a vector PDF to `figures/architecture.pdf`. If you do, replace the TikZ in
methodology.tex (`fig:architecture-pipeline`) with `\includegraphics`. Otherwise the
TikZ figure already in the thesis is sufficient.

**Facts to keep consistent with the thesis text** (do not change):
- 17 joints (Human3.6M order): 0 Pelvis, 1 Right Hip, 2 Right Knee, 3 Right Ankle,
  4 Left Hip, 5 Left Knee, 6 Left Ankle, 7 Spine, 8 Thorax, 9 Neck, 10 Head,
  11 Left Shoulder, 12 Left Elbow, 13 Left Wrist, 14 Right Shoulder, 15 Right Elbow,
  16 Right Wrist.
- BFS kinematic order pi = 0 1 4 7 2 5 8 3 6 9 11 14 10 12 15 13 16.
- Token width C = 64; SSM hidden state N = 16; frames T = 243; joints J = 17;
  20 blocks (10 x [Spatial BiSTSSM Block + Temporal BiSTSSM Block]).

────────────────────────────────────────────────────────────────────────
## PROMPT  (save the result as figures/architecture.pdf)
────────────────────────────────────────────────────────────────────────

Create a publication-quality architecture diagram for a top-tier computer vision conference paper (CVPR, ICCV, ECCV, NeurIPS). The figure must look manually designed in Adobe Illustrator or Figma by a researcher, not AI-generated, and not like a PowerPoint flowchart.

Use a clean landscape (16:9) layout with a modern horizontal network design. The main backbone runs left to right across the center. Place a compact module inset in the upper-right connected with a thin callout line, and place the proposed-contribution panel across the bottom. Resemble architecture diagrams from recent Transformer, Mamba, MotionBERT, or Vision Transformer papers.

Style: flat vector graphics; white background; thin dark-gray outlines; rounded rectangles; soft blue for feature tensors; soft gray for computation modules; muted amber only for the proposed-contribution panel; professional Helvetica/Arial typography; perfect alignment and equal spacing; large whitespace; tensor-flow ribbons or clean feature arrows; no gradients; no shadows; no 3D rendering; no clip art; no decorative elements; no photorealism.

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

Use exactly these labels where specified. Do not invent additional captions, equations, legends, or annotations. The final figure should be visually indistinguishable from a manually created architecture figure in a recent CVPR or ICCV paper, using clean vector graphics, compact repeated stages, tensor-flow visualization, and a modern scientific layout.
