#!/bin/bash
# Train on Part 2 of 5 (51,419 samples each)
# REQUIRES: NVIDIA GPU with CUDA support

set -e # Exit on error

# Change to project root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

echo "============================================================"
echo "Training on Part 2/5 (51,419 samples)"
echo "GPU-OPTIMIZED with fp16 Mixed Precision"
echo "============================================================"

# Check GPU availability
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available! GPU is required for training."
    echo "Please check your CUDA installation and GPU drivers."
    exit 1
}

echo "GPU check passed. Starting training..."
echo ""

SPLITS_DIR="data/processed/splits_5_parts"
VAL_FILE="data/processed/combined_raw_with_mpc_val.json"
TRAIN_FILE="$SPLITS_DIR/train_part_2.json"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

echo "Training on: $TRAIN_FILE"
echo "Validation: $VAL_FILE"
echo ""

python src/training/trainer.py "$TRAIN_FILE" "$VAL_FILE"

if [ $? -ne 0 ]; then
    echo "Error in Part 2 training!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Part 2/5 training completed!"
echo "============================================================"


