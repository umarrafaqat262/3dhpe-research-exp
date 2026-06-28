#!/bin/bash
# stage4_full.sh — Stage 4: Full 120-epoch run with 3 seeds
# Usage: bash experiments/scripts/stage4_full.sh <exp_name> [seed1 seed2 seed3]

set -e

EXP_NAME=$1
SEED1=${2:-0}
SEED2=${3:-1}
SEED3=${4:-2}

PYTHON=/home/ubuntu/miniforge3/envs/posemamba/bin/python
BASE_DIR=$(dirname "$0")/../..

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SEED in $SEED1 $SEED2 $SEED3; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$BASE_DIR/experiments/exp/${EXP_NAME}_s${SEED}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"

    echo "============================================"
    echo " Stage 4 — Full Run: $EXP_NAME (seed=$SEED)"
    echo " Output: $OUTPUT_DIR"
    echo "============================================"

    setsid $PYTHON "$BASE_DIR/kinecmamba/train.py" \
        --config "$BASE_DIR/kinecmamba/configs/experiments/exp_${EXP_NAME}.yaml" \
        -c "$OUTPUT_DIR" \
        -sd "$SEED" \
        > "$OUTPUT_DIR/stdout.log" 2>&1 &

    echo "  PID: $!"
    echo "  Log: tail -f $OUTPUT_DIR/stdout.log"
done

echo ""
echo "Started 3 seeds for $EXP_NAME in background"
echo "Monitor with: watch -n 30 'tail -5 */stdout.log'"
