# S9 data-leakage experiment (diagnostic)

**Purpose:** deliberately put test subject **S9 into the training set** while still evaluating
on the standard test set (S9 + S11), using the **corrected-BFS** checkpoint (`v2_bfs`, *not* the
fused model). This is a controlled demonstration of how train/test overlap inflates the metric —
**the resulting number is NOT a valid result to report.** It only exists to observe the behaviour.

Standard H36M split: train S1/S5/S6/S7/S8, test **S9/S11**. We copy S9 (from the test data) into
the training clips, so S9 is seen during training and then scored at test time.

## Run it (GPU/data box, from `kinecmamba/`)

1. **Inject S9 into training** (adds prefixed clip files; base data untouched):
   ```bash
   python make_leak_s9_clips.py \
     --out_dir data/motion3d/MB3D_f243s81/H36M-SH/train \
     --subject s_09 --stride 81
   ```
   Use `--dry_run` first to just count how many S9 clips would be written.

2. **Fine-tune the corrected-BFS checkpoint with the leaked data:**
   ```bash
   python train.py \
     --config configs/experiments/bfs_scan/exp_bfs_leak_s9.yaml \
     -p <corrected-bfs-ckpt-dir> -ms best_epoch.bin \
     -c checkpoint/leak_s9
   ```
   - Point `-p/-ms` at your ~43mm corrected-BFS checkpoint.
   - **Channel match:** the config defaults to 3-channel (`input_channels: 3`, `no_conf: False`),
     matching the from-scratch corrected BFS. If your checkpoint is the 2-channel fine-tune, set
     `input_channels: 2` and `no_conf: True` in the config.

3. **Read the result.** Each eval prints, in addition to the overall `Protocol #1 Error (MPJPE)`:
   ```
   [subject s_09] frame-mean MPJPE: <low>mm   ...   (leaked — should drop a lot)
   [subject s_11] frame-mean MPJPE: <normal>mm ...  (clean — should stay ~unchanged)
   ```
   The gap between the S9 (leaked) and S11 (clean) lines is the leakage effect. The overall
   MPJPE will look artificially good because half the test set was memorized.

4. **Undo the leak** when done (removes only the injected files):
   ```bash
   rm data/motion3d/MB3D_f243s81/H36M-SH/train/LEAK_S9_*.pkl
   ```

## What to expect
- **S9 (leaked)** MPJPE drops well below the honest ~43mm — the model has seen those exact clips.
- **S11 (clean)** MPJPE stays roughly the same as before.
- The **overall** number drops (inflated), purely from the S9 memorization — the point of the demo.
- More epochs / lower stride (`--stride 27`) = more S9 clips = stronger memorization.

## Baseline comparison (optional)
Before injecting, run a plain `--evaluate` of the same corrected-BFS checkpoint to log the honest
per-subject S9 and S11 numbers. Compare against step 3 to quantify exactly how much leakage moved S9.

## Notes / caveats
- This reuses the standard eval, so `flip` TTA is on and clip order is preserved
  (`shuffle: False`), keeping `len(results_all) == len(action_clips)` valid.
- Generated S9 clips are normalized identically to the real train clips (same `DataReaderH36M`
  path), so they are indistinguishable in format from genuine training data.
