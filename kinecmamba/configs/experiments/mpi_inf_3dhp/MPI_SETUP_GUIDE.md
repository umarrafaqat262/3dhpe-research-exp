# MPI-INF-3DHP setup + run guide (for opencode, on the GPU box)

This branch (`exp/mpi-inf-3dhp`) adds MPI-INF-3DHP train+eval to KinecMamba, keeping the
BFS kinematic scan and the anatomy-aware bone losses. The **code is written and
syntax-checked, but has NOT been run against real data** (no GPU/data on the authoring
box). Your job: obtain + preprocess the dataset, verify the data-format assumptions
below, run training + eval, and report the measured MPJPE / PCK / AUC. Do not fabricate
numbers; the thesis MPI cells stay TBD until you produce them.

## What was added (all under `kinecmamba/`)
- `lib/data/dataset_3dhp.py` — `MPITrainDataset3D` (81-frame clips) and `MPITestDataset3D`
  (per-sequence, valid-frame masked). Handles 2D screen normalization, root-relative 3D
  (root = joint **14**), and left/right flip groups.
- `lib/eval/metrics_3dhp.py` — `mpjpe_mm`, `pck` (@150mm), `auc` (0–150mm, 5mm steps), and
  `export_inference_mat` (writes `inference_data.mat` for the official MATLAB scorer).
- `train_mpi.py` — train+eval entry (mirrors `train.py`: EMA, bone losses, `-e` eval-only,
  `-p/-ms` warm-start). Calls `csms6s.set_bfs_order('mpi')` when `forward_type: v2_bfs`.
- `lib/model/csms6s.py` — added `BFS_ORDER_MPI` + `set_bfs_order()`; H36M default unchanged.
- `lib/model/loss.py` — added `loss_limb_gt_mpi` / `loss_limb_var_mpi` (true MPI bone table).
- `configs/experiments/mpi_inf_3dhp/exp_mpi_bfs.yaml` — the run config (recipe below).

## 1. Get the data (this repo ships none)
1. Download the raw **MPI-INF-3DHP** train + test sets from the official site
   (`https://vcai.mpi-inf.mpg.de/3dhp-dataset/`) using their provided download scripts
   (train sequences S1–S8; the `mpi_inf_3dhp_test_set`).
2. Preprocess to the `.npz` format this loader expects using **P-STMO's** scripts
   (`https://github.com/paTRICK-swk/P-STMO`, `common/data_to_npz_3dhp.py` and
   `common/data_to_npz_3dhp_test.py`). They produce `data_train_3dhp.npz` and
   `data_test_3dhp.npz`.
3. Place both files at:  `kinecmamba/data/motion3d/mpi_inf_3dhp/`
   (matches `data_root` / `train_file` / `test_file` in the config).
4. For the paper-exact PCK/AUC, also fetch P-STMO's official evaluator dir `3dhp_test/`
   (matlab utils + `annot-test.h5`); you'll run it on the exported `.mat` (step 4 below).

## 2. VERIFY these assumptions before trusting a run (they were coded to spec, not tested)
- **npz internal structure**: train `raw[(subject,seq)][0][cam]['data_3d'|'data_2d']`;
  test `raw['TSx']['data_3d'|'data_2d'|'valid']`. If P-STMO's current scripts use
  different keys/nesting, adjust `lib/data/dataset_3dhp.py` accordingly.
- **Units**: metrics assume 3D is in **millimetres**. The loader has a `_infer_mm_scale`
  heuristic (×1000 if coords look like metres). Confirm against your npz; hard-set if needed.
- **Train 2D resolution**: assumed 2048×2048 for all train sequences; test TS5/TS6 use
  1920×1080. Confirm per-subject train resolutions and fix in the loader if any differ.
- **Joint order / BFS**: MPI 17-joint order assumed as `0 head_top,1 neck,2 R_sho,3 R_elb,
  4 R_wri,5 L_sho,6 L_elb,7 L_wri,8 R_hip,9 R_kne,10 R_ank,11 L_hip,12 L_kne,13 L_ank,
  14 pelvis,15 spine,16 head`. `BFS_ORDER_MPI` and the bone table depend on this — verify
  the table matches your npz before trusting BFS/bone-loss on MPI.
- **csms6s switch**: `CrossScan_bfs`/`CrossMerge_bfs` read the module globals at call time,
  so `set_bfs_order('mpi')` in `train_mpi.py` switches the scan order. Confirm nothing caches
  the list at import.

## 3. Recipe (in the config)
81-frame window, **GT 2D** input (`input_channels: 2`, `no_conf: True`), seq2seq, effective
batch 160 (`batch_size 20` × `accum_steps 8`), `lr 7e-4`, exponential `lr_decay 0.97`,
60 epochs, EMA on, `forward_type: v2_bfs`, bone losses `lambda_lg 0.5` / `lambda_lv 1.0`.

## 4. Run
```bash
cd kinecmamba
# Train (from scratch on 3DHP; add -p <dir> -ms best_epoch.bin to warm-start from an H36M ckpt)
python train_mpi.py --config configs/experiments/mpi_inf_3dhp/exp_mpi_bfs.yaml \
  -c checkpoint/mpi_bfs

# Eval only -> prints MPJPE/PCK/AUC (python) and writes checkpoint/mpi_bfs_eval/inference_data.mat
python train_mpi.py --config configs/experiments/mpi_inf_3dhp/exp_mpi_bfs.yaml \
  -e checkpoint/mpi_bfs/best_epoch.bin -c checkpoint/mpi_bfs_eval
```
Sanity at startup: dataset lengths > 0, and (eval) it prints per-sequence + overall
`MPJPE .. PCK@150 .. AUC ..` and writes `inference_data.mat`.

## 5. Paper-exact PCK/AUC (official MATLAB scorer)
The python PCK/AUC is for monitoring. For the numbers other papers report, run P-STMO's
`3dhp_test/test_util/mpii_test_predictions_py.m` (MATLAB/Octave) on the exported
`inference_data.mat` with the official `annot-test.h5`. Report those as the headline
MPI-INF-3DHP results.

## 6. Report back
Per-sequence and overall MPJPE (mm), PCK@150, AUC — both the python numbers and the MATLAB
numbers — plus which VERIFY items (section 2) you had to adjust. Sanity band (not a
target): PCK high-80s–low-90s, AUC ~50s, MPJPE ~80–100mm for a small model. Anything wildly
off usually means a data-format mismatch from section 2.
