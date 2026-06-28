#!/bin/bash
# sanity_check.sh — Phase 1: Validate model, config, and data pipeline
# Usage: bash experiments/scripts/sanity_check.sh <exp_name> [config_path]

set -e

EXP_NAME=${1:-A1_baseline}
CONFIG=${2:-configs/experiments/exp_${EXP_NAME}.yaml}

PYTHON=/home/ubuntu/miniforge3/envs/posemamba/bin/python
BASE_DIR=$(dirname "$0")/../..

echo "============================================"
echo " Sanity Check: $EXP_NAME"
echo " Config:      $CONFIG"
echo "============================================"

# Step 1: Model validation
echo ""
echo "[Step 1/3] Model validation (forward/backward)"
$PYTHON -m kinecmamba.validate --config "$CONFIG"

# Step 2: Data pipeline check (1 batch)
echo ""
echo "[Step 2/3] Data pipeline check"
$PYTHON -c "
import sys, os
sys.path.insert(0, 'kinecmamba')
from lib.utils.tools import get_config
from lib.data.dataset_motion_3d import MotionDataset3D
from torch.utils.data import DataLoader

cfg = get_config('$CONFIG')
ds = MotionDataset3D(cfg, cfg.subset_list, 'train')
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
batch_input, batch_gt = next(iter(dl))
print(f'  Batch input shape: {batch_input.shape}')
print(f'  Batch GT shape:    {batch_gt.shape}')
assert batch_input.shape[0] == cfg.batch_size, f'Expected batch size {cfg.batch_size}'
assert batch_input.shape[3] == 2 or batch_input.shape[3] == 3 or batch_input.shape[3] == 5
print('  ✓ Data pipeline OK')
" || { echo '  ✗ Data pipeline FAILED'; exit 1; }

# Step 3: Full train loop (1 epoch, test mode)
echo ""
echo "[Step 3/3] Training loop (1 epoch, no save)"
$PYTHON -c "
import sys, os
sys.path.insert(0, 'kinecmamba')
sys.path.insert(0, 'kinecmamba/lib')
from lib.utils.tools import get_config
cfg = get_config('$CONFIG')
cfg.epochs = 1          # 1 epoch only
cfg.no_eval = True      # skip expensive eval
cfg.checkpoint_frequency = 1000

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.utils.learning import load_backbone
from lib.model.loss import loss_mpjpe, n_mpjpe, loss_velocity

ds = MotionDataset3D(cfg, cfg.subset_list, 'train')
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
model = load_backbone(cfg)
model = torch.nn.DataParallel(model)
model = model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

for i, (batch_input, batch_gt) in enumerate(dl):
    batch_input = batch_input.cuda()
    batch_gt = batch_gt.cuda()
    if cfg.no_conf:
        batch_input = batch_input[:, :, :, :(cfg.get('input_channels', 2))]
    pred = model(batch_input)
    loss = loss_mpjpe(pred, batch_gt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i == 0:
        print(f'  First batch loss: {loss.item():.4f}')
    if i >= 5:
        break
print('  ✓ Training loop OK (5 batches completed)')
" || { echo '  ✗ Training loop FAILED'; exit 1; }

echo ""
echo "============================================"
echo " All sanity checks PASSED for $EXP_NAME"
echo "============================================"
