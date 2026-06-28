#!/bin/bash
# sweep_scan_orders.sh — Phase B3: sweep over SSM scan order variants
# Usage: bash experiments/scripts/sweep_scan_orders.sh
# Output: experiments/exp/B3-sweep/{variant}/results.json

set -e

PYTHON=/home/ubuntu/miniforge3/envs/posemamba/bin/python
BASE_DIR=$(dirname "$0")/../..
OUTPUT_DIR=$BASE_DIR/experiments/exp/B3-sweep

mkdir -p "$OUTPUT_DIR"

VARIANTS=(
    v2
    v2_fs_ft
    v2_fs_bt
    v2_bs_ft
    v2_bs_bt
    v31d
    v32d
)

echo "============================================"
echo " Sweep: SSM Scan Order Variants"
echo " Variants: ${VARIANTS[*]}"
echo "============================================"

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "--- Running: $variant ---"
    CONFIG="configs/experiments/exp_B3_${variant}.yaml"
    VARIANT_DIR="$OUTPUT_DIR/$variant"
    mkdir -p "$VARIANT_DIR"

    $PYTHON -c "
import sys, os
sys.path.insert(0, '$BASE_DIR/kinecmamba')
sys.path.insert(0, '$BASE_DIR/kinecmamba/lib')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.data.dataset_motion_3d import MotionDataset3D
from lib.model.loss import loss_mpjpe

cfg = get_config('$BASE_DIR/kinecmamba/$CONFIG')
cfg.epochs = 1
cfg.no_eval = True
cfg.checkpoint_frequency = 1000

ds = MotionDataset3D(cfg, cfg.subset_list, 'train')
dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
model = load_backbone(cfg)
model = torch.nn.DataParallel(model)
model = model.cuda()
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)

total_loss = 0.0
num_batches = 0
for i, (batch_input, batch_gt) in enumerate(dl):
    batch_input = batch_input.cuda()
    batch_gt = batch_gt.cuda()
    if cfg.no_conf:
        batch_input = batch_input[:, :, :, :cfg.get('input_channels', 2)]
    pred = model(batch_input)
    loss = loss_mpjpe(pred, batch_gt)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += loss.item()
    num_batches += 1
    if i >= 5:
        break

avg_loss = total_loss / num_batches
print(f'{variant} avg_loss={avg_loss:.4f} num_batches={num_batches}')

with open('$VARIANT_DIR/results.txt', 'w') as f:
    f.write(f'avg_loss={avg_loss:.4f}\n')
    f.write(f'num_batches={num_batches}\n')
" 2>&1 | tee "$VARIANT_DIR/log.txt"

    echo "--- Completed: $variant ---"
done

echo ""
echo "============================================"
echo " Sweep complete. Results in $OUTPUT_DIR"
echo "============================================"
