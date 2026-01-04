#!/bin/bash
# Activation script for ML training environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate uv virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv directory not found. Run setup_env.sh first."
    exit 1
fi

# Set environment variables
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export MODELS_DIR="$PROJECT_ROOT/models"
export DATASETS_DIR="$PROJECT_ROOT/datasets"
export RUNS_DIR="$PROJECT_ROOT/runs"
export EVAL_DIR="$PROJECT_ROOT/eval"

echo "Environment activated!"
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "MODELS_DIR: $MODELS_DIR"
