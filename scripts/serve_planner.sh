#!/bin/bash
# vLLM serving script - OpenAI-compatible endpoint

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate environment
if [ -f "$PROJECT_ROOT/scripts/activate_env.sh" ]; then
    source "$PROJECT_ROOT/scripts/activate_env.sh"
else
    echo "Error: activate_env.sh not found. Run setup first."
    exit 1
fi

# Default configuration
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-localtoken}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"

echo "Starting vLLM OpenAI-compatible server..."
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "API Key: $API_KEY"
echo ""

# Check if vllm is installed
if ! command -v vllm &> /dev/null; then
    echo "Error: vllm not found. Please run 'make setup' first to install vLLM."
    exit 1
fi

# Verify PyTorch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
PYTORCH_BASE=$(echo "$PYTORCH_VERSION" | cut -d+ -f1)
echo "PyTorch version: $PYTORCH_VERSION"

# Check for torchao compatibility issue and fix if needed
# This handles the case where vLLM was installed but torchao is outdated
if python -c "import torchao.quantization" 2>/dev/null; then
    if ! python -c "from torchao.quantization import Float8WeightOnlyConfig" 2>/dev/null; then
        echo "Fixing torchao compatibility issue..."
        uv pip install --upgrade "torchao>=0.6.0" || true
        echo "Please restart the server if the issue persists."
    fi
fi

# Verify flash-attn is working with current PyTorch
# Test import with proper error handling (suppress stderr to avoid noise)
if python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "✓ flash-attn is working correctly with PyTorch $PYTORCH_VERSION"
    echo "vLLM will use flash-attn for optimized attention."
elif python -c "import flash_attn" 2>/dev/null; then
    echo "Note: flash-attn package installed but CUDA extension import check failed"
    echo "vLLM will attempt to use flash-attn, falling back to xformers/eager if needed."
else
    echo "Note: flash-attn is not available"
    echo "vLLM will automatically use fallback attention (xformers or eager)."
fi

# Memory management settings
# Reduce GPU memory utilization to avoid OOM during warmup
# Lower max_num_seqs to reduce memory pressure
# Use PYTORCH_ALLOC_CONF instead of deprecated PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Memory settings:"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max concurrent sequences: $MAX_NUM_SEQS"
echo ""

# Start vLLM server with memory-optimized settings
exec vllm serve "$MODEL" \
    --dtype "$DTYPE" \
    --api-key "$API_KEY" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS"
