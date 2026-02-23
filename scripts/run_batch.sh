#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch adversarial image generation via CLIP optimization
# Run on a GPU machine with the model checkpoint downloaded
# ============================================================

# --- Configuration (edit these) ---
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-models/clip-vit-l-14-336.safetensors}"
CLIP_MODEL="${CLIP_MODEL:-ViT-L-14-336}"
EPOCHS="${EPOCHS:-5000}"
LR="${LR:-0.1}"
DEVICE="${DEVICE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/batch}"
# ----------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  VLM Safety - Batch Adversarial Generation"
echo "============================================"
echo "  CLIP model:      $CLIP_MODEL"
echo "  Checkpoint:       $CLIP_CHECKPOINT"
echo "  Epochs:           $EPOCHS"
echo "  LR:               $LR"
echo "  Device:           $DEVICE"
echo "  Output:           $OUTPUT_DIR"
echo "============================================"

# Check checkpoint exists
if [ "$CLIP_CHECKPOINT" != "" ] && [ ! -f "$CLIP_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CLIP_CHECKPOINT"
    echo "Download it first:"
    echo "  wget https://huggingface.co/timm/vit_large_patch14_clip_336.openai/resolve/main/open_clip_model.safetensors -O $CLIP_CHECKPOINT"
    exit 1
fi

# Install deps
echo ""
echo ">>> Installing dependencies..."
uv sync --native-tls 2>&1 | tail -1

# Run batch
echo ""
echo ">>> Starting batch generation..."
uv run python scripts/run_batch.py \
    --clip-model "$CLIP_MODEL" \
    --clip-checkpoint "$CLIP_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo ">>> Results in: $OUTPUT_DIR"
echo ">>> Summary:    $OUTPUT_DIR/summary.json"
