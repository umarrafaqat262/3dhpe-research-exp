# Nano-banana prompt — KinecMamba architecture figure

Paste the block below into nano-banana (Gemini image). It describes the KinecMamba
pipeline exactly as written in `methodology.tex`. Generate at high resolution, portrait
or 4:5. If the labels come out garbled, regenerate and add "render all text crisply,
spelled exactly as given" — image models often mangle small text, so keep the label
list short and re-run rather than accept misspelled boxes.

---

## PROMPT (copy from here)

A clean, professional academic architecture diagram for a research paper, flat vector /
textbook illustration style. White background, thin dark-grey arrows, rounded-rectangle
boxes with a muted palette (soft blue, soft grey, one soft amber accent for the
contribution). Legible sans-serif labels. No photorealism, no 3D renders, no human
photos, no gradients, no drop shadows. Top-to-bottom data flow. Three grouped panels.

PANEL A — main pipeline (single vertical column, top to bottom):
1. Box "Input: 2D keypoints (B, T, J, 2)".
2. Arrow down to box "Spatial Token Embedding (STE): Linear 2 to C, + spatial position
   embedding".
3. Arrow down to box "Temporal Token Embedding (TTE): + temporal position embedding".
4. Arrow down into a large dashed container labelled "x 10" on its top-right corner.
   Inside the container, stacked vertically: box "Spatial BiSTSSM Block" then an arrow
   down to box "Temporal BiSTSSM Block". (Caption note under it: "10 repetitions = 20
   blocks total".)
5. Arrow down from the container to box "Output head: LayerNorm then Linear C to 3".
6. Arrow down to box "Output: 3D pose (B, T, J, 3)".
Set C = 64 as a small annotation near the token width.

PANEL B — inset, "BiSTSSM layer" (to the right of the block, connected by a thin
callout line from "Spatial BiSTSSM Block"), a vertical mini-pipeline:
- "Four-directional cross-scan (frame-major and joint-major, each forward and backward)"
- arrow to "Input projection: Linear C to 2C"
- arrow to "Depthwise Conv1D, kernel 3"
- arrow to "Selective SSM (per direction): input-dependent B, C, delta; diagonal A;
  state size N = 16"
- wrap the previous three boxes in a small dashed box labelled "x 4 (one per direction)"
- arrow to "Cross-merge (recombine 4 directions)"
- arrow to "Output projection: Linear 2C to C"

PANEL C — inset, "BFS kinematic-tree scan order" (bottom, the amber-accented
contribution panel):
- On the left, a small skeleton drawn as a rooted tree of labelled circles: Pelvis at
  top (root), branching down to Right Hip, Left Hip, Spine; then Right Knee, Left Knee,
  Thorax; then Right Ankle, Left Ankle, Neck, Left Shoulder, Right Shoulder; then Head,
  Left Elbow, Right Elbow; then Left Wrist, Right Wrist. Draw the bones as edges.
- On the right, show two horizontal strips of 17 small numbered cells:
  strip 1 titled "Naive index order": 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
  strip 2 titled "BFS kinematic order": 0,1,4,7,2,5,8,3,6,9,11,14,10,12,15,13,16
- Label the transformation between the layer and this panel: "permute joints before
  scan, inverse-permute after merge".

Use EXACTLY these text labels, spelled as written, and do not invent any other text,
numbers, equations, or captions. Keep math notation minimal (only the shapes and the
numbers listed). Overall look: a figure you would see in a top computer-vision
conference paper.

## END PROMPT

---

### Label cross-check (must match methodology.tex)
- Token width C = 64; SSM state N = 16; frames T = 243; joints J = 17.
- 10 macro-stages, each = Spatial + Temporal BiSTSSM Block => 20 blocks.
- BiSTSSM steps: cross-scan -> Linear C->2C -> DW Conv1D k=3 -> selective SSM -> merge
  -> Linear 2C->C.
- BFS order pi = [0,1,4,7,2,5,8,3,6,9,11,14,10,12,15,13,16] (matches code `BFS_ORDER`).
- Joint 8 = Thorax (shoulders branch here), 9 = Neck, 10 = Head.
