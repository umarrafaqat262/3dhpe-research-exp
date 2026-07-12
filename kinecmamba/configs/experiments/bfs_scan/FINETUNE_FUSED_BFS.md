# Fine-tuning the FUSED native+BFS scan (KinecMamba) — GPU runbook

This is a step-by-step guide to run on the GPU machine. Goal: push KinecMamba **below the
PoseMamba-S baseline of 41.8mm MPJPE**. Every command runs from inside `kinecmamba/`.

## What this changes and why

Replacing the native raster scan with BFS never beat 41.8mm (best ~42.5–43.3mm) because the
Mamba SSM is **joint-order specific** — the official 41.8mm weights learned their recurrence in
native order, and reordering to BFS breaks them. Bone loss + EMA polish made it *worse*
(42.95 → 43.6mm).

New approach — **fused gated scan** (`forward_type: v2_fused_bfs`):
- Run **8** scan directions: 4 native (`plus_poselimbs`, exactly what the official checkpoint
  learned) **+** 4 BFS kinematic-tree directions.
- Multiply the BFS half by a **zero-initialized scalar gate** (ReZero) in every SS2D block.
- At init `gate = 0`, so the model output equals the pure native 41.8mm model **bit-for-bit**.
  Fine-tuning grows the kinematic branch as a learned residual on top of that floor.

Because the native branch is preserved, a short fine-tune only has to teach the new branch to
help — the worst case is the gate stays ~0 (BFS neutral), not a regression.

Code touched (branch `exp/fused-bfs-scan`):
- `lib/model/csms6s.py` — `CrossScan_fused_bfs`, `CrossMerge_fused_bfs`.
- `lib/model/mambablocks.py` — `v2_fused_bfs` type, `k_group=8`, `bfs_gate`, gated merge.
- `train.py` — `expand_kgroup_4to8` (loads a K=4 checkpoint into the K=8 model), two-group
  optimizer (gate at higher LR), per-group cosine LR, config-gated multi-clip eval.
- `configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml` — the fine-tune recipe.
- `verify_fused_scan.py` — correctness checks.

## Prerequisites

- The **official PoseMamba-S checkpoint** (2-channel, trained with `v2_plus_poselimbs`) on disk,
  e.g. `checkpoint/official/best_epoch.bin` with a `model_pos` state dict. Note its dir and filename.
- The Human3.6M data at `data/motion3d/MB3D_f243s81/` (as the other configs expect).
- A GPU with enough VRAM for `batch_size: 8` at 243 frames (fallback: `batch_size: 4`,
  `accum_steps: 8` — same effective batch 32).

---

## Step 0 — verify the fused scan is correct (fast, CPU-ok)

```bash
cd kinecmamba
python verify_fused_scan.py
```
Expect `ALL FUSED SCAN CHECKS PASSED`. This confirms slots 0–3 == native, slots 4–7 == BFS, and
that the fused backward equals the sum of the two component backwards. **Do not proceed if this fails.**

## Step 1 — sanity: the gate=0 floor must reproduce the official number

Run a 1-shot **evaluate** of the fused model loaded from the official checkpoint. With `gate=0`
the reported MPJPE must equal the official PoseMamba-S number (~41.8mm). This proves the
checkpoint expander loaded the native weights cleanly and the BFS branch is off at init.

```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  -p checkpoint/official -ms best_epoch.bin \
  --evaluate checkpoint/official/best_epoch.bin
```
> If `--evaluate` doesn't take the fused path in your build, instead run the fine-tune (Step 2)
> and read the **epoch-0** eval line from the log — same check. In the load log you should see
> `Missing keys (ignored): [... bfs_gate ...]` (expected — the gate keeps its zero init) and the
> five SSM params (`x_proj_weight`, `dt_projs_weight`, `dt_projs_bias`, `A_logs`, `Ds`) should
> NOT appear as missing (the expander fills them).

**If epoch-0 MPJPE ≈ 41.8mm → the floor is intact, continue. If it's far off → stop and debug
the load** (wrong checkpoint dir/name, or `input_channels`/`no_conf` mismatch — must stay 2ch).

## Step 2 — run the fused fine-tune

```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  -p checkpoint/official -ms best_epoch.bin \
  -c checkpoint/ft_fused_bfs \
  --wandb            # optional
```
Recipe (already in the yaml): 18 epochs, cosine→0 with 2-epoch warmup, **base LR 2e-5**,
**gate LR 2e-3 (100×)**, grad-clip 1.0, EMA 0.999 (EMA weights are evaluated and saved as
`best_epoch.bin`), loss = MPJPE + 0.5·scale + 20·velocity (bone loss OFF).

Do **not** pass `-r/--resume` (it fast-forwards the epoch counter). Do not change the loss.

### What to watch
- `Protocol #1 Error (MPJPE)` per epoch — this is the number that must drop below 41.8.
- The load log line `Discriminative LR: N bfs_gate param(s) at lr 0.002, rest at 2e-5`
  (confirms the two-group optimizer is active; N = number of SS2D blocks = 2·depth).
- Epoch-0 should print ≈41.8 (the floor). It should not spike far above it.

### Expected outcome
- Crosses to **~41.3–41.7mm** if the kinematic branch adds signal.
- Worst case **~41.9–42.1mm** (floor preserved, gate near 0 = BFS neutral). That is a clean
  negative result, not a regression.

## Step 3 — evaluate the best checkpoint

```bash
python train.py \
  --config configs/experiments/bfs_scan/exp_ft_fused_bfs.yaml \
  --evaluate checkpoint/ft_fused_bfs/best_epoch.bin
```
Record the `Protocol #1 Error (MPJPE)` — this is the KinecMamba number for the thesis table.

---

## Optional — free eval-time gain: multi-clip prediction averaging

Averages the predictions of every overlapping 243-window that covers a frame (a test-time
ensemble), stacking on the flip-averaging already on. Typical gain ~0.2–0.5mm, **no retrain**.
It is OFF by default and is a no-op unless the test clips actually overlap.

To enable it you must regenerate the **test** clips at the overlap stride so the loader and the
`datareader` agree (`train.py` asserts `len(results_all)==len(action_clips)` — a mismatch fails
loudly, it never produces silent wrong numbers):
1. Regenerate the test split at `data_stride 81` (or 27 for denser averaging).
2. In `exp_ft_fused_bfs.yaml` set `test_multiclip: True` and `data_stride_test: 81`.
3. Re-run Step 3. Eval time scales with the overlap factor (~3× at stride 81).

Apply this to whichever checkpoint you ship. As a control, also run it on the **untouched
official** checkpoint to see the eval-only floor (~41.4–41.6mm) — this isolates how much of any
gain is the eval trick vs. the fused scan.

---

## Experiment sequence (minimum runs to cross 41.8)

1. **Step 0 + Step 1** — verify + confirm the 41.8 floor (no training).
2. **Step 2** — the fused fine-tune (the core lever).
3. **Step 3** — report; optionally add multi-clip eval for the extra free margin.

### Fallback ladder (if Step 2 alone doesn't cross 41.8)
- **(a)** Turn on multi-clip eval (above) — usually carries a ~41.9–42.1 result under 41.8.
- **(b)** Rerun Step 2 at **25–30 epochs** (more cosine anneal room) and/or raise `gate_lr` to
  `0.003`. More training on the fresh branch only; native stays gentle at 2e-5.
- **(c)** EMA settle: take the Step-2 output and run a 15-epoch `lr 2e-5` cosine→0 pass with
  `use_ema: True` and `gate_lr` back at `2e-5` (pure settling, no branch changes).
- **(d)** Honest floor: report the multi-clip eval number on the official checkpoint.

## Troubleshooting

- **Load errors / `RuntimeError: size mismatch`**: the expander didn't run — confirm you're on
  branch `exp/fused-bfs-scan` and `forward_type: v2_fused_bfs`.
- **Epoch-0 ≠ 41.8**: wrong checkpoint, or `input_channels`/`no_conf` changed (must be 2 / True).
- **`bfs_gate` not in the log's missing keys**: you loaded a *fused* checkpoint (K=8) not the
  official (K=4) — that's fine for resume, but the floor check only means anything from the
  official K=4 checkpoint.
- **MPJPE stuck exactly at 41.8 every epoch**: the gate isn't moving — check the
  `Discriminative LR` log line appeared and `gate_lr` is 2e-3, not 2e-5.
