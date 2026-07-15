# Fused native+BFS gated scan (KinecMamba) — clone-and-run GPU guide

Goal: push KinecMamba **below the PoseMamba-S baseline (41.87mm MPJPE in this pipeline)** with a
single short fine-tune, starting from the official PoseMamba-S checkpoint. The result is
guaranteed leak-free (trains only on the standard S1/S5/S6/S7/S8 split) and cannot regress below
the official floor by construction (see "Why this is floor-safe").

Every command runs from inside `kinecmamba/`.

## What this experiment is

`forward_type: v2_fused_bfs` runs **8 scan directions**: the 4 native `plus_poselimbs`
directions (exactly what the official checkpoint learned) **plus** 4 BFS kinematic-tree
directions. The BFS half is multiplied by a **zero-initialized scalar ReZero gate** (`bfs_gate`)
in every SS2D block.

- At init the gate is 0, so the model output equals the pure native PoseMamba-S model
  **bit-for-bit** (epoch-0 eval reproduces 41.87mm).
- Fine-tuning grows the kinematic branch as a learned residual on top of that floor. Because the
  native branch is preserved, the worst case is the gate stays ~0 (BFS neutral), never a regression.

### Why this is floor-safe (verified)
The native slots (0-3) of the fused scan are byte-identical to `CrossScan_plus_poselimbs`, the
BFS slots (4-7) are byte-identical to `CrossScan_bfs`, and the fused merge is their additive sum
with the BFS half gated. At `bfs_gate = 0` the merge reduces exactly to the native merge. This is
checked two ways:
- `python verify_fused_scan_numpy.py` — torch-free (numpy only), instant. Verifies the
  permutation inverse (`INV_BFS_ORDER == argsort(BFS_ORDER)`), the slot identities, the additive
  merge, the **gate=0 floor equivalence**, and the BFS scan->merge joint-order round-trip.
- `python verify_fused_scan.py` — full torch check on the actual autograd Functions (forward
  identity + backward-consistency: the fused VJP equals the sum of the two component VJPs).

## Prerequisites

- **Official PoseMamba-S checkpoint** (2-channel, trained with `v2_plus_poselimbs`) on disk, with
  a `model_pos` state dict — e.g. `checkpoint/official/best_epoch.bin`. Note its dir and filename.
  (In this repo's tarball it is `PoseMamba_S.bin`; place/rename so `-p <dir> -ms <file>` points at it.)
- Human3.6M data at `data/motion3d/MB3D_f243s81/` (same layout the other configs expect).
- A GPU with room for `batch_size: 8` at 243 frames (fallback: `batch_size: 4`, `accum_steps: 8`,
  same effective batch 32).

---

## Step 0 — DECONTAMINATE the training set (do this first, always)

A previous S9 data-leakage diagnostic (`make_leak_s9_clips.py`) injected leaked S9 test clips
into the **shared** training directory with a `LEAK_S9_` prefix. Training with those present
invalidates the result. Remove any that remain:
```bash
ls data/motion3d/MB3D_f243s81/H36M-SH/train/LEAK_S9_*.pkl 2>/dev/null && \
  rm data/motion3d/MB3D_f243s81/H36M-SH/train/LEAK_S9_*.pkl || echo "clean: no leaked clips"
```

## Step 1 — verify the fused scan is correct (fast, CPU-ok)
```bash
python verify_fused_scan_numpy.py    # torch-free, instant
python verify_fused_scan.py          # full torch check
```
Both must print their `... PASSED` line. Do not proceed if either fails.

## Step 2 — sanity: the gate=0 floor must reproduce the official number
```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  -p checkpoint/official -ms best_epoch.bin \
  --evaluate checkpoint/official/best_epoch.bin
```
Expect MPJPE ~= **41.87mm**. In the load log you should see `Missing keys (ignored): [... bfs_gate ...]`
(expected: the gate keeps its zero init) and the five SSM params (`x_proj_weight`,
`dt_projs_weight`, `dt_projs_bias`, `A_logs`, `Ds`) should NOT appear as missing (the
`expand_kgroup_4to8` expander fills slots 0-3 with the official K=4 weights and duplicates them
into slots 4-7). If `--evaluate` does not take the fused path in your build, read the **epoch-0**
eval line from the Step-3 log instead — same check.

**If epoch-0 MPJPE ~= 41.87 -> the floor is intact, continue. If far off -> stop and debug the
load** (wrong checkpoint dir/name, or `input_channels`/`no_conf` changed — must stay 2 / True).

## Step 3 — run the fused fine-tune (the core lever)
```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  -p checkpoint/official -ms best_epoch.bin \
  -c checkpoint/ft_fused_bfs \
  --wandb            # optional
```
Recipe (already in the yaml): **12 epochs** (~6-7h at ~30-35 min/epoch), cosine -> 0 with 2-epoch
warmup, **base LR 2e-5**, **gate LR 2e-3 (100x)**, grad-clip 1.0, EMA 0.999.

**Objective — anatomy-aware six-term loss (matched to the thesis methodology):**
MPJPE (1.0) + n-MPJPE/scale (0.5) + velocity (20.0) + **limb-length temporal variance
`lambda_lv` (1.0)** + **limb-length vs GT `lambda_lg` (0.5)**. The two bone terms are ON here (they
were OFF in the old replace-scan runs, where a saturated ~43mm model made them hurt). In the fused
run the native branch is preserved bit-for-bit and only the gated BFS kinematic-tree residual
moves, so bone supervision shapes exactly the branch that models parent->child limb structure —
the lowest-risk place to add it, and it targets the occlusion-heavy actions (SittingDown, Sitting,
Photo) where limb lengths collapse. Angle losses stay off.

Do **not** pass `-r/--resume` (it fast-forwards the epoch counter).

### What to watch
- `Protocol #1 Error (MPJPE)` per epoch — the number that must drop below 41.87.
- `Discriminative LR: N bfs_gate param(s) at lr 0.002, rest at 2e-5` (confirms the two-group
  optimizer is active; N = number of SS2D blocks = 2 x depth = 20). If this line is missing or the
  gate lr is 2e-5, the gate will not open — stop and check the config loaded `gate_lr`.
- Epoch-0 should print ~= 41.87 (the floor) and not spike far above it.
- The EMA-evaluated best excludes the gate from the moving average (the gate grows from 0 and an
  EMA would suppress it), so the saved best uses smoothed base weights with the live, trained gate.

### Expected outcome
- **~41.3-41.7mm** if the kinematic branch adds signal.
- Worst case **~41.9-42.1mm** (floor preserved, gate near 0). That is a clean negative result, not
  a regression.

## Step 4 — evaluate the best checkpoint + record the defensible numbers
```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  --evaluate checkpoint/ft_fused_bfs/best_epoch.bin
```
Record **verbatim** from the log:
1. `Protocol #1 Error (MPJPE)` and `Protocol #2 Error (P-MPJPE)` — the KinecMamba thesis numbers.
2. The **full per-action table** (shows where the gain concentrates — expect the biggest drops on
   SittingDown / Sitting / Photo).
3. The `[subject s_09]` / `[subject s_11]` breakdown — confirm S9 and S11 are **balanced** (this is
   the honest counter-evidence to the old S9-leaked checkpoint; do not report cherry-picked S9).

Compare against the in-pipeline control: official PoseMamba-S = **41.87mm P1 / 35.06mm P2**.

## Cross-check (optional, protects the headline number)
Also evaluate the raw final weights and report the better of the two:
```bash
python train.py --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  --evaluate checkpoint/ft_fused_bfs/latest_epoch.bin
```
`best_epoch.bin` is the EMA(base)+live-gate model; `latest_epoch.bin` is the raw final model.
With cosine LR -> 0 they should be close; report whichever P1 is lower.

## Fallback ladder (only if Step 3 does not cross 41.87)
- **(a)** If per-epoch MPJPE rises above the floor and stays there, set `lambda_lg: 0.0` (keep
  `lambda_lv: 1.0`) — `loss_limb_gt` is the only term here that pulls against MPJPE — and rerun.
- **(b)** Multi-clip eval (free ~0.2-0.5mm, no retrain): regenerate the TEST clips at
  `data_stride 81`, set `test_multiclip: True` and `data_stride_test: 81` in the yaml, re-run
  Step 4. Also run it on the untouched official checkpoint as the control (isolates the eval-trick
  gain from the fused-scan gain). `train.py` asserts `len(results_all)==len(action_clips)`, so a
  stride/clip mismatch fails loudly (never silent wrong numbers).
- **(c)** More anneal room: raise `epochs` to 18 and/or `gate_lr` to 0.003 (native stays gentle at 2e-5).

## Cross-generalization (qualitative, CPU-ok)
Rerun the existing in-the-wild YOLOv8-pose pipeline with the fused checkpoint vs the official one,
side by side (`~/Downloads/kinecmamba_qualitative/`). No ground truth needed — it shows the model
generalizes off Human3.6M.

## Troubleshooting
- **`RuntimeError: size mismatch` on load** — the expander did not run; confirm `forward_type:
  v2_fused_bfs` and that you are on this branch.
- **Epoch-0 != 41.87** — wrong checkpoint, or `input_channels`/`no_conf` changed (must be 2 / True).
- **MPJPE stuck exactly at 41.87 every epoch** — the gate is not moving; check the
  `Discriminative LR` log line and that `gate_lr` is 2e-3, not 2e-5.
- **`bfs_gate` not in the missing-keys log** — you loaded a fused K=8 checkpoint, not the official
  K=4 (fine for resume, but the floor check only means something from the official K=4 checkpoint).
