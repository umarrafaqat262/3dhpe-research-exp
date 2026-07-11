# KinecMamba thesis — GPT image-generation prompts

Prompts for generating the four diagram figures with GPT (GPT-4o / DALL·E image). Generate
each, save it at the path in the table, and the thesis will pick it up automatically — each
figure in the `.tex` uses `\IfFileExists{figures/<name>.pdf}{...}{placeholder}`, so once the
file is present it is included; until then a labelled placeholder box shows. PNG works too
(change the filename in the `.tex` if you export PNG).

| Figure | Save as | `.tex` label |
|--------|---------|--------------|
| Architecture pipeline | `figures/architecture.pdf` | `fig:architecture-pipeline` (methodology.tex) |
| Kinematic tree (full-body skeleton) | `figures/kinematic_tree.pdf` | `fig:kinematic-tree` (methodology.tex) |
| BiSTSSM layer | `figures/bistssm_layer.pdf` | `fig:bistssm-layer` (methodology.tex) |
| Naive vs BFS scan order | `figures/scan_order.pdf` | `fig:scan-order-comparison` (methodology.tex) |

The results figure (`fig:results-plot`, results.tex) is a **data plot** and stays as
pgfplots — do not generate it with GPT (an image model would fabricate the numbers).

**GPT image tips (important):** image models garble text and math. In every prompt: ask for
a flat vector, white-background, black-and-soft-color diagram; keep text minimal and require
the labels to be spelled EXACTLY as given and rendered legibly; no gradients, shadows, 3D, or
photorealism; 300+ DPI, transparent or white background. If labels come out wrong, regenerate
or add the text afterward in a vector editor.

**Facts to keep consistent (do not change):**
- 17 joints (order): 0 Pelvis, 1 Right Hip, 2 Right Knee, 3 Right Ankle, 4 Left Hip,
  5 Left Knee, 6 Left Ankle, 7 Spine, 8 Thorax, 9 Neck, 10 Head, 11 Left Shoulder,
  12 Left Elbow, 13 Left Wrist, 14 Right Shoulder, 15 Right Elbow, 16 Right Wrist.
- 16 bones (parent→child): 0-1,1-2,2-3; 0-4,4-5,5-6; 0-7,7-8,8-9,9-10; 8-11,11-12,12-13;
  8-14,14-15,15-16.
- BFS order: 0 1 4 7 2 5 8 3 6 9 11 14 10 12 15 13 16.
- C = 64, N = 16, T = 243, J = 17, 20 blocks (10 × [Spatial + Temporal BiSTSSM Block]).

════════════════════════════════════════════════════════════════════════
## PROMPT 1 — Architecture pipeline  →  figures/architecture.pdf
════════════════════════════════════════════════════════════════════════

A clean, publication-quality neural-network architecture diagram for a computer-vision
research paper, flat vector style, white background, thin dark-gray outlines, rounded
rectangles, soft blue for data tensors and soft gray for computation blocks, one muted amber
accent for the proposed part, Helvetica/Arial labels, evenly spaced, lots of whitespace, no
gradients, no shadows, no 3D, no photorealism. Landscape 16:9. Left-to-right flow.

Left: a small 2D stick-figure pose skeleton labeled "2D HPE", with the caption "Input: 2D
keypoints (B, T, J, 2)". An arrow to a box "Spatial Token Embedding (STE)", then an arrow to
"Temporal Token Embedding (TTE)"; a small note "C = 64" above the stream. Then a long dashed
rounded box labeled "×10" containing two stacked boxes: "Spatial BiSTSSM Block" above
"Temporal BiSTSSM Block", with a small caption "10 repetitions = 20 blocks total". An arrow
to "Output head: LayerNorm, Linear C→3". Right: a small 3D stick-figure pose skeleton in
light perspective labeled "3D HPE", with the caption "Output: 3D pose (B, T, J, 3)". Use
exactly these labels and nothing else. It must look hand-designed in Illustrator, not like a
flowchart.

════════════════════════════════════════════════════════════════════════
## PROMPT 2 — Kinematic tree (full-body skeleton)  →  figures/kinematic_tree.pdf
════════════════════════════════════════════════════════════════════════

A clean flat-vector diagram of a full-body human skeleton, front-facing, standing upright
with arms slightly out and legs apart, drawn ONLY as small filled circles (joints) connected
by thin straight lines (bones). White background, soft-blue joints, thin dark-gray bones, no
muscles or silhouette, no shading, no 3D, no photorealism. Mark the pelvis joint with a
slightly larger amber circle (the root). Label every joint with its index and name in a
small neat caption next to its circle, using exactly: 0 Pelvis, 1 Right Hip, 2 Right Knee,
3 Right Ankle, 4 Left Hip, 5 Left Knee, 6 Left Ankle, 7 Spine, 8 Thorax, 9 Neck, 10 Head,
11 Left Shoulder, 12 Left Elbow, 13 Left Wrist, 14 Right Shoulder, 15 Right Elbow, 16 Right
Wrist. Place: Head at top, then Neck, Thorax, Spine, Pelvis down the center; from the Thorax
the two shoulders out to the sides then elbows then wrists down each arm; from the Pelvis the
two hips then knees then ankles down each leg. Bones must be exactly these pairs: 0-1,1-2,2-3
(right leg), 0-4,4-5,5-6 (left leg), 0-7,7-8,8-9,9-10 (spine and head), 8-11,11-12,12-13
(left arm), 8-14,14-15,15-16 (right arm). The skeleton's bones should read as a tree rooted at
the pelvis. Labels spelled exactly; no extra text.

════════════════════════════════════════════════════════════════════════
## PROMPT 3 — BiSTSSM layer  →  figures/bistssm_layer.pdf
════════════════════════════════════════════════════════════════════════

A clean flat-vector block diagram of one neural-network layer, vertical top-to-bottom flow,
white background, rounded rectangles, thin dark-gray arrows, soft blue for tensors and soft
gray for operations, no gradients/shadows/3D/photorealism, portrait orientation. Boxes top to
bottom with downward arrows: "Input Z (T × J × C), post-LayerNorm"; "Four-directional
cross-scan"; then a dashed container labeled "×4 (one per direction)" holding a vertical
sub-stack of three boxes: "Input projection: Linear C→2C", "Depthwise Conv1D, kernel 3",
"Selective SSM (Δ, B, C input-dependent; diagonal A; N = 16)"; then "Cross-merge (recombine
4 directions)"; then "Output projection: Linear 2C→C"; then "Output O (T × J × C)". Use
exactly these labels; keep it conceptual and clean; no extra text.

════════════════════════════════════════════════════════════════════════
## PROMPT 4 — Naive vs BFS scan order  →  figures/scan_order.pdf
════════════════════════════════════════════════════════════════════════

A clean flat-vector figure comparing two orderings, white background, small rounded square
cells, thin dark-gray outlines, Helvetica/Arial, no gradients/shadows/3D. Two horizontal
rows of 17 cells each, one below the other, with a thin downward arrow between them. Each
cell holds a short joint abbreviation with its index as a subscript. Top row titled "Naive
index order", left to right: Pel(0) RHp(1) RKn(2) RAn(3) LHp(4) LKn(5) LAn(6) Spn(7) Thx(8)
Nck(9) Hd(10) LSh(11) LEl(12) LWr(13) RSh(14) REl(15) RWr(16). Bottom row titled "BFS
kinematic order", cells lightly shaded soft blue, left to right: Pel(0) RHp(1) LHp(4) Spn(7)
RKn(2) LKn(5) Thx(8) RAn(3) LAn(6) Nck(9) LSh(11) RSh(14) Hd(10) LEl(12) REl(15) LWr(13)
RWr(16). The figure shows that in the BFS row, joints close in the body (e.g. the two hips
and the spine) sit adjacent. Use exactly these labels; no extra text.
