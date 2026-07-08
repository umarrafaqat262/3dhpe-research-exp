# Corrected BFS Kinematic Scan — Run & Validation Guide

Goal: decide, as cheaply as possible, whether the corrected BFS kinematic
selective-scan can beat PoseMamba-S (41.8mm MPJPE P1 on Human3.6M) — and only
then spend the full training budget.

> **Run everything from INSIDE the `kinecmamba/` directory** (same cwd as
> `train.py`). Paths like `configs/...`, `data/...`, `checkpoint/...` are relative
> to `kinecmamba/`, and imports are `from lib...`.
> ```bash
> cd kinecmamba
> ```

## What was broken (now fixed on `exp/bfs-corrected`)

Every prior BFS run plateaued near ~49mm because of three bugs, all fixed:
1. **Forward joint scramble** — `CrossMerge_bfs` never un-permuted joints after
   the scan → misaligned with the residual add / `Spatial_pos_embed`.
2. **Gradient mis-routing** — backward assumed `BFS_ORDER` is its own inverse
   (false at joints 10–15). Now uses `INV_BFS_ORDER = argsort(BFS_ORDER)`.
3. **Checkpoint never loaded** — old configs had `finetune: False` / dead freeze
   keys, so they trained from scratch with the scrambling bugs.
(All in `lib/model/csms6s.py`.)

## Why not just fine-tune the official checkpoint?

Tried it (`exp_ft_bfs_corrected.yaml`): it plateaus at **42.53mm** and never
reaches the 41.8 baseline. The Mamba SSM is joint-sequence-order-specific — the
official checkpoint learned its recurrence in the NATIVE order, so re-ordering to
BFS at fine-tune time disrupts it (~44mm at epoch 1). **BFS can only pay off if
the SSM learns BFS order from scratch.** Hence the staged plan below.

## Prerequisites

- The project PyTorch/Mamba env (as in the training box).
- Data: Human3.6M SH at `data/motion3d/MB3D_f243s81/`
  (`dt_file: h36m_sh_conf_cam_source_final.pkl`).
- (Only for the fine-tune configs) official `checkpoints/PoseMamba_S.bin`.

---

## Stage 0 — Scan correctness (seconds)

```bash
python verify_bfs_scan.py     # expect: ALL BFS SCAN CHECKS PASSED
```
Confirms the inverse permutation, the `merge(scan(x)) == 4*x` round-trip, and
`gradcheck` on both autograd Functions. Do not train until this passes.

## Stage 1 — Small-sample overfit race (~5–10 min)

Trains **native vs BFS** on a small sample with a train/val split for a few
epochs; whichever drives the sample's train error down faster has the better-fit
inductive bias. Cheap go/no-go screen.

```bash
python overfit_probe.py \
  --config configs/experiments/bfs_scan/exp_bfs_scratch.yaml \
  --n_samples 128 --val_frac 0.25 --epochs 40 --seed 0
```
Reads a matched-epoch train-MPJPE table and a `VERDICT` line.
- **BFS fits faster (and val not far worse)** → go to Stage 2.
- **Native clearly faster** → BFS's bias isn't helping here; stop and rethink
  before spending GPU-days.

> Caveat: this measures memorization speed on a tiny sample — it *correlates*
> with, but doesn't *prove*, better generalization. It's a screen, not the verdict.

## Stage 2 — 30-epoch A/B pilot (~2 × 10h)

Train both scans from scratch, short, **same seed**, paper recipe (effective
batch 32 via `batch_size 8 × accum_steps 4`):

```bash
# BFS
python train.py --config configs/experiments/bfs_scan/exp_bfs_pilot.yaml \
  -sd 0 -c checkpoint/bfs_pilot --wandb false
# Native control
python train.py --config configs/experiments/bfs_scan/exp_native_pilot.yaml \
  -sd 0 -c checkpoint/native_pilot --wandb false
```
Compare epoch-by-epoch:
```bash
python compare_runs.py \
  checkpoint/bfs_pilot_<ts>/log.txt \
  checkpoint/native_pilot_<ts>/log.txt --labels bfs native
```
**Decision gate:** extend to Stage 3 only if BFS P1 is **at or below** native at
matched epochs. The native run doubles as your recipe-reproduction check (it
should be heading toward ~41.8).

> The claim "BFS beats PoseMamba" is measured against THIS native control — never
> against the paper's 41.8 or the frozen-backbone 41.7 (neither is a from-scratch
> control in your pipeline).

## Stage 3 — Full 120-epoch BFS run (~39h)

Only if Stage 2 passes. BFS only:

```bash
python train.py --config configs/experiments/bfs_scan/exp_bfs_scratch.yaml \
  -sd 0 -c checkpoint/bfs_scratch --wandb false
```
Best model → `checkpoint/bfs_scratch_<ts>/best_epoch.bin`. Evaluate:
```bash
python train.py --config configs/experiments/bfs_scan/exp_bfs_scratch.yaml \
  -e checkpoint/bfs_scratch_<ts>/best_epoch.bin --wandb false
```
**Target: P1 < 41.8mm** (and below your native-from-scratch baseline).

---

## Configs on this branch

| Config | Scan | finetune | epochs | in_ch | Purpose |
|---|---|:---:|:---:|:---:|---|
| `exp_bfs_scratch.yaml`    | v2_bfs            | no  | 120 | 3 | Stage 3 — full BFS run |
| `exp_native_scratch.yaml` | v2_plus_poselimbs | no  | 120 | 3 | full native baseline (optional) |
| `exp_bfs_pilot.yaml`      | v2_bfs            | no  | 30  | 3 | Stage 2 — BFS pilot |
| `exp_native_pilot.yaml`   | v2_plus_poselimbs | no  | 30  | 3 | Stage 2 — native control |
| `exp_ft_bfs_corrected.yaml` | v2_bfs          | yes | 40  | 2 | fine-tune (plateaus 42.5mm) |
| `exp_ft_bfs_lowlr.yaml`   | v2_bfs            | yes | 20  | 2 | cheap fine-tune LR check |

All from-scratch configs use effective batch 32 (8 × accum 4) and the paper recipe
(lr 5e-4, exponential-0.99, no grad clip, 3-channel confidence input,
`lambda_diff: 0`).

## Notes on batch / accumulation

`train.py:278-285` divides the loss by `accum_steps` and steps every `accum_steps`
micro-batches, so 8 × 4 is mathematically equivalent to true batch 32. If you have
VRAM, `batch_size:32 / accum_steps:1` is faster; if 8 OOMs, use `4 / 8`. Startup
logs `effective batch size: 32` — confirm it.

## Helper scripts (in `kinecmamba/`)

- `verify_bfs_scan.py` — Stage 0 scan-correctness check.
- `overfit_probe.py` — Stage 1 small-sample overfit race.
- `compare_runs.py` — Stage 2 matched-epoch P1/P2 comparison of two logs.
