# Continue-from-checkpoint polish — run guide (for opencode)

Goal: run a short, honest polish of the existing 120-epoch corrected-BFS model
(best ~43.3mm P1) using weight EMA and a low learning rate annealed to zero, then
report the true measured P1. This does NOT retrain from scratch.

## Ground rules (read first)
- **Report the exact P1/P2 the eval log prints. Do not edit, round-down, or invent
  numbers.** If the run lands at 42.4, the result is 42.4. Crossing 41.8 from 43.3
  in 15 epochs is possible but not guaranteed.
- **No test data in training.** Train only on subjects S1/S5/S6/S7/S8; evaluate only
  on S9/S11. The code already enforces this; do not change the split.
- Do not enable the MotionBERT `Augmenter2D` (mask/noise) — it needs external asset
  files and targets a different training mode. It is intentionally off here.

## Prerequisites
- Run from **inside** `kinecmamba/` (same cwd as `train.py`; imports are `from lib...`).
- The 120-epoch checkpoint file `best_epoch.bin` from the from-scratch BFS run.
  Put it somewhere accessible, e.g. `kinecmamba/checkpoints/bfs_scratch_120/best_epoch.bin`.
  It must be the checkpoint trained with `exp_bfs_scratch.yaml` (forward_type v2_bfs,
  3-channel input) so the architecture matches this config.
- Data at `data/motion3d/MB3D_f243s81/` (`h36m_sh_conf_cam_source_final.pkl`).

## What was changed in the code
`train.py` now supports an optional `use_ema` flag (config). When on, it keeps an
exponential moving average of the weights, evaluates the averaged weights each
epoch, and saves them as `best_epoch.bin`. When off, behavior is unchanged.

## Step 1 — sanity: confirm the checkpoint loads and reproduces ~43.3
Evaluate the checkpoint as-is through this config (no training):
```bash
cd kinecmamba
python train.py \
  --config configs/experiments/bfs_scan/exp_bfs_continue.yaml \
  -e checkpoints/bfs_scratch_120/best_epoch.bin --wandb false
```
Expected: P1 ≈ 43.3 (matching the original run). If it is far off, the checkpoint
or config does not match — stop and check `forward_type`, `input_channels`, `no_conf`
before training.

## Step 2 — run the 15-epoch EMA polish
```bash
cd kinecmamba
python train.py \
  --config configs/experiments/bfs_scan/exp_bfs_continue.yaml \
  -p checkpoints/bfs_scratch_120 -ms best_epoch.bin \
  -c checkpoint/bfs_continue --wandb false
```
- `-p/-ms` load the 120ep checkpoint via the finetune path and start at epoch 0.
  **Do not use `-r/--resume`** (it fast-forwards the epoch counter).
- The log prints `EMA enabled, decay 0.999` at startup and `effective batch size: 32`.
- Each epoch logs `... e1 <P1> e2 <P2>` — these are the **EMA** weights' scores.
- Best model (lowest P1) is saved to `checkpoint/bfs_continue_<timestamp>/best_epoch.bin`.
- If it OOMs, set `batch_size: 4` and `accum_steps: 8` in the config (same effective 32).

## Step 3 — evaluate and record the best
```bash
cd kinecmamba
python train.py \
  --config configs/experiments/bfs_scan/exp_bfs_continue.yaml \
  -e checkpoint/bfs_continue_<timestamp>/best_epoch.bin --wandb false
```
Record the printed `Protocol #1 Error (MPJPE)` and `Protocol #2 Error (P-MPJPE)`.
That P1 is the number to report. Nothing else.

## Step 4 — report back
Return the best epoch, its P1 and P2 verbatim from the eval log, and the path to
`best_epoch.bin`. Do not compare a GT-2D number against the detected-2D 41.8 as if
it beat it; this run uses detected 2D (`gt_2d: False`).
