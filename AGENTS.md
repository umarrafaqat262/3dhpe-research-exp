# KinecMamba — Agent Instructions

## Environment
- Conda env: `posemamba` (path: `/home/ubuntu/miniforge3/envs/posemamba/bin/python`)
- GPU: NVIDIA L4 (23GB, sm_89)
- CUDA kernels: `selective_scan_cuda_oflex` and `selective_scan_cuda_core` compiled for sm_89
- Project root: `/home/ubuntu/experiments`

## Critical: Disk Space
- Root: 48GB total, ~47GB used (98%).
- Never run `git fetch` or `git pull` without checking disk first.
- Clean wandb/ and old checkpoints regularly.
- On disk warning (98%), use `du -h -d2 / 2>/dev/null | sort -rh | head -20` to find space hogs.

## Critical: Git
- **Do NOT commit** data files, checkpoints, wandb/, build artifacts, or experiment output directories.
- Orphan branch created (fb93b1e0, root commit 29cf99a3). Only ~50 tracked files.
- `.gitignore` covers: data symlinks, wandb/, build artifacts, `__pycache__`, `*.so`, experiment directories.

## Training
- Config: `kinecmamba/configs/pose3d/PoseMamba_train_h36m_S.yaml`
- Pipeline has gradient accumulation (batch=4, accum 8 → effective batch=32), cosine annealing (5-epoch warmup), gradient clipping (max_norm=1.0).
- **Foreground training works.** Silent crashes in background were nohup/session issues.
- Use `setsid` (not nohup) for background runs.
- Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for background runs.
- Always use full conda python path for background runs.

## Experiments
- All experiments under `experiments/exp{NN}_{name}/`
- Staged protocol: S1 (correctness) → S2 (overfit) → S3 (short val) → S4 (full ×3 seeds)
- Full documentation in `experiments/docs/`
- Hypothesis bank: `experiments/docs/03_hypotheses.md`

## Docs Updated
- `experiments/docs/02_literature_review.md` — Updated with hybrid Mamba taxonomy (3 types: feature-level, state-level, Mamba-Attention) and novel research gap (state-level + feature-level fusion unexplored)
- `experiments/docs/03_hypotheses.md` — Added H11 (SAMA-style state-level fusion, Δ=-0.6mm, P2 priority)
- `experiments/docs/06_references.md` — Added 12 new references (VIMCAN, SAMA, AGMamba, HGMamba, Jamba, TransMamba, Nemotron-H, Bamba-9B)
