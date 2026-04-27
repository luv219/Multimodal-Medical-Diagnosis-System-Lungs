#!/bin/bash
# run.sh — Full pipeline: install → train → evaluate → explain
#
# Usage:
#   ./run.sh                       # DenseNet-121 with AUCM loss (defaults)
#   ./run.sh swin aucm             # Swin Transformer with AUCM loss
#   ./run.sh hybrid focal          # Hybrid with focal loss
#   NIH_DATA_ROOT=/data ./run.sh   # Override data root inline

set -e  # exit immediately on any error

# =============================================================================
# Configuration
# =============================================================================
MODEL=${1:-densenet}        # first arg: model name  (densenet | swin | hybrid)
LOSS=${2:-aucm}             # second arg: loss name  (aucm | focal)
DATA_ROOT=${NIH_DATA_ROOT:-data}

echo "========================================"
echo " NIH ChestX-ray14 — Training Pipeline"
echo "========================================"
echo "  Model     : $MODEL"
echo "  Loss      : $LOSS"
echo "  Data root : $DATA_ROOT"
echo "========================================"

# =============================================================================
# Validate environment
# =============================================================================
if [ -z "$NIH_DATA_ROOT" ]; then
    echo ""
    echo "WARNING: NIH_DATA_ROOT is not set."
    echo "  Defaulting to ./data/ — this will fail unless the dataset is there."
    echo "  Set the variable before running:"
    echo "    export NIH_DATA_ROOT=/path/to/your/nih_data"
    echo "  Then re-run: ./run.sh $MODEL $LOSS"
    echo ""
fi

# =============================================================================
# Install dependencies
# =============================================================================
echo ""
echo "[1/5] Installing dependencies..."
pip install -r requirements.txt

# =============================================================================
# Verify dataset paths
# =============================================================================
echo ""
echo "[2/5] Verifying dataset setup..."
python config.py

# =============================================================================
# Train
# =============================================================================
echo ""
echo "[3/5] Training $MODEL with $LOSS loss..."
python train.py --model "$MODEL" --loss "$LOSS"

# =============================================================================
# Evaluate on test split and save JSON report
# =============================================================================
echo ""
echo "[4/5] Evaluating on test split..."
python evaluate.py \
    --model "$MODEL" \
    --checkpoint "checkpoints/best.pth" \
    --split test \
    --save-report \
    --output-dir results/

# =============================================================================
# Explain — GradCAM++ on the first available test image
# =============================================================================
echo ""
echo "[5/5] Generating GradCAM++ saliency maps..."

SAMPLE_IMAGE=$(python -c "
import pandas as pd
from pathlib import Path
from config import PATHS
try:
    df = pd.read_csv(PATHS['labels_csv'], usecols=['Image Index'])
    img_path = Path(PATHS['images_dir']) / df['Image Index'].iloc[0]
    print(img_path)
except Exception:
    pass
" 2>/dev/null || echo "")

if [ -n "$SAMPLE_IMAGE" ] && [ -f "$SAMPLE_IMAGE" ]; then
    python explain.py \
        --model "$MODEL" \
        --checkpoint "checkpoints/best.pth" \
        --image "$SAMPLE_IMAGE" \
        --top-k 3 \
        --save-dir "results/"
else
    echo "  No sample image found — skipping saliency map generation."
    echo "  Run manually: python explain.py --model $MODEL --checkpoint checkpoints/best.pth --image <path>"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo " Done.  Artifacts:"
echo "========================================"
echo "  Checkpoint : checkpoints/best.pth"
echo "  Report     : results/${MODEL}_test_*.json"
echo "  Saliency   : results/*.png"
echo ""
echo "Next steps:"
echo "  Repeat for swin and hybrid, then generate the comparison dashboard:"
echo "    ./run.sh swin $LOSS"
echo "    ./run.sh hybrid $LOSS"
echo "    python dashboard.py --results-dir results/ --output results/dashboard.png"
echo "========================================"
