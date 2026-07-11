# KinecMamba architecture-figure prompt (publication-quality)

Paste the prompt below into an image generator (e.g. nano-banana / Gemini image).
Export the result and place it at `figures/architecture.pdf` (or .png), then uncomment
the `\includegraphics` line in `methodology.tex` (Figure `fig:architecture-pipeline`).

---

Create a publication-quality architecture diagram for a top-tier computer vision conference paper (CVPR, ICCV, ECCV, NeurIPS). The figure must look manually designed in Adobe Illustrator or Figma by a researcher, not AI-generated, and not like a PowerPoint flowchart.

Use a clean **landscape (16:9)** layout with a modern horizontal network design. The main backbone should run from left to right across the center of the figure. Place a compact module inset in the upper-right connected with a thin callout line, and place the proposed contribution panel across the bottom. The overall figure should resemble architecture diagrams from recent Transformer, Mamba, MotionBERT, or Vision Transformer papers.

Style:
- Flat vector graphics
- White background
- Thin dark-gray outlines
- Rounded rectangles
- Soft blue for feature tensors
- Soft gray for computation modules
- Muted amber only for the proposed contribution panel
- Professional Helvetica/Arial typography
- Perfect alignment and equal spacing
- Large whitespace
- Tensor-flow ribbons or clean feature arrows instead of simple flowchart arrows
- No gradients
- No shadows
- No 3D rendering
- No clip art
- No decorative elements
- No photorealism

────────────────────────────────────────

MAIN NETWORK (Center)

On the far left, draw a small **2D human pose skeleton** (stick figure with joints) labeled **2D HPE**.

Next to it write:

**Input: 2D keypoints (B, T, J, 2)**

Connect it to:

**Spatial Token Embedding (STE)**

Then to:

**Temporal Token Embedding (TTE)**

Add a small annotation above the feature stream:

**C = 64**

The backbone occupies most of the figure width.

Represent it as one repeated stage enclosed by a long dashed bracket labeled:

**×10**

Inside one stage place:

**Spatial BiSTSSM Block**

↓

**Temporal BiSTSSM Block**

Use compact modern neural-network modules instead of ordinary rectangles.

Below the repetition bracket write:

**10 repetitions = 20 blocks total**

Continue to:

**Output head: LayerNorm then Linear C to 3**

Finally, on the far right, draw a small **3D human pose skeleton** in perspective labeled **3D HPE**.

Below it write:

**Output: 3D pose (B, T, J, 3)**

────────────────────────────────────────

MODULE INSET (Upper Right)

Connect this inset to the Spatial BiSTSSM Block using a thin callout line.

Title:

**BiSTSSM layer**

Draw a compact conceptual module rather than a detailed implementation.

Input Features

↓

Four-directional Cross-Scan

↓

A dashed container labeled:

**×4 (one per direction)**

Inside the dashed container, show four small parallel processing blocks labeled:

**Selective SSM**

Merge the four branches into:

**Cross-Merge**

↓

**Output Features**

Keep this inset clean and simple, emphasizing the concept rather than every internal operation.

────────────────────────────────────────

PROPOSED CONTRIBUTION (Bottom)

Highlight this panel using a muted amber border.

Title:

**BFS kinematic-tree scan order**

On the left, draw a clean rooted kinematic tree using labeled joints connected by thin edges:

Pelvis

├── Right Hip
│   └── Right Knee
│       └── Right Ankle

├── Left Hip
│   └── Left Knee
│       └── Left Ankle

└── Spine
    └── Thorax
        ├── Neck
        │   └── Head
        ├── Left Shoulder
        │   └── Left Elbow
        │       └── Left Wrist
        └── Right Shoulder
            └── Right Elbow
                └── Right Wrist

On the right, show two compact horizontal strips.

Title:

**Naive index order**

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16

↓

Title:

**BFS kinematic order**

0 1 4 7 2 5 8 3 6 9 11 14 10 12 15 13 16

Between the backbone and this panel place:

**permute joints before scan, inverse-permute after merge**

────────────────────────────────────────

Use exactly these labels where specified. Do not invent additional captions, equations, legends, or annotations.

The final figure should be visually indistinguishable from a manually created architecture figure in a recent CVPR or ICCV paper, using clean vector graphics, compact repeated stages, tensor-flow visualization, and modern scientific layout instead of a generic flowchart.
