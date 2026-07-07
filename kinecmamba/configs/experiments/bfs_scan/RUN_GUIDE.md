# Corrected BFS Kinematic Scan — Run Guide

Fine-tune the official PoseMamba-S checkpoint with a **correctly-wired** BFS
kinematic selective-scan, aiming to beat the 41.8mm MPJPE (P1) SOTA on
Human3.6M. Zero new parameters — the only change is a correct joint reordering
of the Mamba scan plus a proper checkpoint load and fine-tune recipe.

## What was broken (and is now fixed)

Every prior BFS run plateaued near ~49mm because of three compounding bugs, all
fixed on this branch (`exp/bfs-corrected`):

1. **Forward joint scramble** — `CrossScan_bfs` permuted joints into BFS order
   but `CrossMerge_bfs` never un-permuted them, so each spatial block's output
   was misaligned with its residual add and `Spatial_pos_embed`.
   Fixed in `kinecmamba/lib/model/csms6s.py` (`INV_BFS_ORDER` un-permute in
   `CrossMerge_bfs.forward` + its adjoint in `.backward`).
2. **Gradient mis-routing** — the backward assumed `BFS_ORDER` is its own
   inverse; it fails at joints 10–15 (arms/head). Now uses
   `INV_BFS_ORDER = argsort(BFS_ORDER)` in `CrossScan_bfs.backward`.
3. **Checkpoint never loaded** — the old bfs configs had `finetune: False` and
   dead `freeze_backbone`/`checkpoint_freeze` keys, so they trained from scratch.
   `exp_ft_bfs_corrected.yaml` sets `finetune: True` and loads via CLI.

## Prerequisites

- A machine with a GPU and the project's PyTorch environment (this repo's Mamba
  deps). The verify step and training require `torch` with CUDA.
- **Official checkpoint**: place `PoseMamba_S.bin` at
  `kinecmamba/checkpoints/PoseMamba_S.bin`. Confirm its top-level key is
  `model_pos` (`python -c "import torch; print(torch.load('kinecmamba/checkpoints/PoseMamba_S.bin', map_location='cpu').keys())"`).
  It must be a **2-channel** model — do not switch `no_conf`/`input_channels`.
- **Data**: Human3.6M SH at `data/motion3d/MB3D_f243s81/`
  (`dt_file: h36m_sh_conf_cam_source_final.pkl`), the standard MotionBERT layout.

Run all commands **from the repo root** so relative `configs/`/`data/` paths resolve.

## Step 1 — Sanity gate (must pass before training)

Verifies the scan fix: inverse-permutation correctness, `merge(scan(x)) == 4*x`
round-trip, and `torch.autograd.gradcheck` on both hand-written autograd Functions.

```bash
python kinecmamba/verify_bfs_scan.py
# expect: ... ALL BFS SCAN CHECKS PASSED
```

## Step 2 — Zero-train eval (load check)

Evaluate the untouched official checkpoint through the corrected `v2_bfs` code
with no training. With the bugs fixed and a clean load it should already sit
**near 41.8mm**. A large gap here means a checkpoint-key/config mismatch, not a
modeling problem — fix it before training.

```bash
python kinecmamba/train.py \
  --config kinecmamba/configs/experiments/bfs_scan/exp_ft_bfs_corrected.yaml \
  -e kinecmamba/checkpoints/PoseMamba_S.bin --wandb false
```

## Step 3 — Fine-tune

```bash
python kinecmamba/train.py \
  --config kinecmamba/configs/experiments/bfs_scan/exp_ft_bfs_corrected.yaml \
  -p kinecmamba/checkpoints -ms PoseMamba_S.bin \
  -c checkpoint/bfs_ft_corrected --wandb false
```

- `-p/-ms` load the checkpoint via the finetune path (`train.py:388-395`, starts
  at epoch 0). **Do not** use `-r/--resume` — it fast-forwards the epoch counter
  and corrupts the schedule.
- Best model → `checkpoint/bfs_ft_corrected_<timestamp>/best_epoch.bin`.
- The log prints `effective batch size: 32` at startup (`train.py:430`) — confirm it.

## Step 4 — Evaluate the fine-tuned model

```bash
python kinecmamba/train.py \
  --config kinecmamba/configs/experiments/bfs_scan/exp_ft_bfs_corrected.yaml \
  -e checkpoint/bfs_ft_corrected_<timestamp>/best_epoch.bin --wandb false
```

**Target: MPJPE (P1) < 41.8mm.** Compare against the committed baselines
(B1 HyperGCN 46.4mm, C1 SSI+MSM 46.25mm) and the 41.8mm SOTA.

## Batch size / gradient accumulation

Effective batch = `batch_size * accum_steps`. The config uses **8 × 4 = 32** to
match the paper's batch while fitting a single GPU at 243 frames.

`train.py` accumulation is mathematically equivalent to a true large batch:
- `train.py:278` divides the loss by `accum_steps`,
- `train.py:279` backprops every micro-batch (grads accumulate),
- `train.py:281-285` clips + steps the optimizer + zeroes grads every
  `accum_steps` micro-batches,
- `train.py:483-488` flushes a trailing partial group at epoch end.

Tuning knobs (all give effective batch 32):

| VRAM situation            | batch_size | accum_steps |
|---------------------------|:----------:|:-----------:|
| Plenty (fastest)          | 32         | 1           |
| Single GPU (default here) | 8          | 4           |
| Tight / OOM at 8          | 4          | 8           |

**LR note:** `learning_rate` is linear-scaled to `2e-4` for effective batch 32
(2× the `1e-4` used at effective 16). If epoch-1 eval spikes far above 41.8mm
right after the scan swap, drop to `1e-4`. Warmup is intentionally `0` — it is
only honored by the cosine scheduler (`train.py:466-471`), not exponential.

## Fallbacks (if the swapped-scan fine-tune plateaus above 41.8)

1. Lower LR (`1e-4` / `5e-5`), 20–30 epochs.
2. Train corrected-BFS **from scratch** with the paper recipe (`finetune:False`,
   120 epochs to convergence, `lr:5e-4`, exponential-0.99, `grad_clip:0`,
   `lambda_diff:0`, and for from-scratch only `no_conf:False`/`input_channels:3`).
3. Control: same fine-tune with `forward_type: v2_plus_poselimbs` after fixing
   its indices to the true H36M parent tree
   `[0,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]` (`csms6s.py:158,181`) — tells you
   whether the win is BFS-specific or from any correct kinematic bias.
