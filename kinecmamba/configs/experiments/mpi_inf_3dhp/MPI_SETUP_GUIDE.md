# MPI-INF-3DHP setup + run guide (for opencode, on the GPU box)

This branch (`exp/mpi-inf-3dhp`) adds MPI-INF-3DHP train+eval to KinecMamba, keeping the
BFS kinematic scan and the anatomy-aware bone losses. The **code is written and
syntax-checked, but has NOT been run against real data** (no GPU/data on the authoring
box). Your job: obtain + preprocess the dataset, verify the data-format assumptions
below, run training + eval, and report the measured MPJPE / PCK / AUC. Do not fabricate
numbers; the thesis MPI cells stay TBD until you produce them.

## Protocol (PoseMamba-S, following MotionBERT)
- **27 frames** per clip (PoseMamba-S MPI setting; Base uses 81).
- **GT 2D** input (`input_channels: 2`, `no_conf: True`).
- **MPJPE** is the primary reported metric (PCK@150 / AUC kept as secondary prints).
- **H36M-order 17-joint skeleton, root = joint 0** — MotionBERT's unified skeleton. This is
  the SAME order the H36M pipeline uses, so MPI reuses the existing H36M BFS scan
  (`csms6s.BFS_ORDER`) and H36M bone table (`loss.get_limb_lens`) with no MPI-specific code.

## What was added (all under `kinecmamba/`)
- `lib/data/dataset_3dhp.py` — `MPITrainDataset3D` (27-frame clips) and `MPITestDataset3D`
  (per-sequence, valid-frame masked). Handles 2D screen normalization, root-relative 3D
  (root = joint **0**, H36M order), and left/right flip groups (`joints_left=[4,5,6,11,12,13]`,
  `joints_right=[1,2,3,14,15,16]`, matching `lib/utils/utils_data.flip_data`).
- `lib/eval/metrics_3dhp.py` — `mpjpe_mm`, `pck` (@150mm), `auc` (0–150mm, 5mm steps), and
  `export_inference_mat` (writes `inference_data.mat` for the official MATLAB scorer).
- `train_mpi.py` — train+eval entry (mirrors `train.py`: EMA, bone losses, `-e` eval-only,
  `-p/-ms` warm-start). Uses the default H36M BFS scan order and H36M bone losses.
- `configs/experiments/mpi_inf_3dhp/exp_mpi_bfs.yaml` — the run config (recipe below).

`lib/model/csms6s.py` and `lib/model/loss.py` are UNCHANGED from the H36M pipeline (no
MPI-specific BFS order or bone table — MPI shares the H36M order).

## 1. Get the data (this repo ships none)
1. Download the raw **MPI-INF-3DHP** train + test sets from the official site
   (`https://vcai.mpi-inf.mpg.de/3dhp-dataset/`) using their provided download scripts
   (train sequences S1–S8; the `mpi_inf_3dhp_test_set`).
2. Preprocess to the `.npz` format this loader expects. **NOTE:** PoseMamba and MotionBERT
   never released their MPI-INF-3DHP data-prep, so there is no canonical script to copy.
   You must produce `data_train_3dhp.npz` / `data_test_3dhp.npz` with the joints **remapped
   to MotionBERT's 17-joint Human3.6M order (root = joint 0)** — the same order the H36M
   pipeline uses. This is a hard requirement: the BFS scan, bone table, and left/right flip
   groups all assume that order. (A convenient starting point is P-STMO's
   `common/data_to_npz_3dhp*.py`, but its native joint order is root=14 and MUST be remapped
   to the H36M order before use.)
3. Place both files at:  `kinecmamba/data/motion3d/mpi_inf_3dhp/`
   (matches `data_root` / `train_file` / `test_file` in the config).
4. For the paper-exact PCK/AUC, also fetch the official MPI evaluator dir `3dhp_test/`
   (matlab utils + `annot-test.h5`); you'll run it on the exported `.mat` (step 5 below).

## 2. VERIFY these assumptions before trusting a run (they were coded to spec, not tested)
- **npz internal structure**: train `raw[(subject,seq)][0][cam]['data_3d'|'data_2d']`;
  test `raw['TSx']['data_3d'|'data_2d'|'valid']`. If P-STMO's current scripts use
  different keys/nesting, adjust `lib/data/dataset_3dhp.py` accordingly.
- **Units**: metrics assume 3D is in **millimetres**. The loader has a `_infer_mm_scale`
  heuristic (×1000 if coords look like metres). Confirm against your npz; hard-set if needed.
- **Train 2D resolution**: assumed 2048×2048 for all train sequences; test TS5/TS6 use
  1920×1080. Confirm per-subject train resolutions and fix in the loader if any differ.
- **Joint order (root = joint 0, H36M / MotionBERT)**: the npz MUST be remapped to
  MotionBERT's unified 17-joint Human3.6M order — `0 pelvis(root),1 R_hip,2 R_knee,3 R_ankle,
  4 L_hip,5 L_knee,6 L_ankle,7 spine,8 thorax,9 neck/nose,10 head,11 L_shoulder,12 L_elbow,
  13 L_wrist,14 R_shoulder,15 R_elbow,16 R_wrist`. The BFS scan (`csms6s.BFS_ORDER`), the bone
  table (`loss.get_limb_lens`), and the flip groups all assume this exact order. Verify your
  preprocessing emits it; a wrong order silently corrupts BFS, bone loss, and flip-TTA.

## 3. Recipe (in the config)
**27-frame** window (PoseMamba-S), **GT 2D** input (`input_channels: 2`, `no_conf: True`),
seq2seq, MPJPE primary metric, effective batch 160 (`batch_size 20` × `accum_steps 8`),
`lr 7e-4`, exponential `lr_decay 0.97`, 60 epochs, EMA on, `forward_type: v2_bfs` (default
H36M BFS order), bone losses `lambda_lg 0.5` / `lambda_lv 1.0`.

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
