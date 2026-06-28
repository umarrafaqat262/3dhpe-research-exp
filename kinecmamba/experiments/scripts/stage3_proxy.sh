#!/bin/bash
# stage3_proxy.sh — Stage 3: Short validation run (24 epochs on S)
# Usage: bash experiments/scripts/stage3_proxy.sh <exp_name> [extra_args...]

set -e

EXP_NAME=$1
shift || true

PYTHON=/home/ubuntu/miniforge3/envs/posemamba/bin/python
BASE_DIR=$(dirname "$0")/../..
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$BASE_DIR/experiments/exp/${EXP_NAME}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo " Stage 3 — Proxy Run: $EXP_NAME"
echo " Output:   $OUTPUT_DIR"
echo " Config:   configs/experiments/exp_${EXP_NAME}.yaml"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON "$BASE_DIR/kinecmamba/train.py" \
    --config "$BASE_DIR/kinecmamba/configs/experiments/exp_${EXP_NAME}.yaml" \
    -c "$OUTPUT_DIR" \
    "$@"

echo ""
echo "============================================"
echo " Stage 3 complete for $EXP_NAME"
echo " Check $OUTPUT_DIR for results"
echo "============================================"
