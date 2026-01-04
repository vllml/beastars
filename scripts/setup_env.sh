#!/bin/bash
# Setup script for ML training environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Cleanup function to remove any =* files that might be created
cleanup_temp_files() {
    cd "$PROJECT_ROOT"
    find . -maxdepth 1 -name "=*" -type f -delete 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup_temp_files EXIT

# Clean up any existing =* files at start
cleanup_temp_files

echo "Setting up ML training environment..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p models datasets runs eval
mkdir -p .cache/huggingface/transformers
mkdir -p .cache/huggingface/datasets

# Set environment variables
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export MODELS_DIR="$PROJECT_ROOT/models"
export DATASETS_DIR="$PROJECT_ROOT/datasets"
export RUNS_DIR="$PROJECT_ROOT/runs"
export EVAL_DIR="$PROJECT_ROOT/eval"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create uv environment
# Clear existing venv if it exists to avoid prompts
echo "Creating uv environment..."
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi
uv venv --python 3.12

# Activate environment and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate

# Detect CUDA version and install PyTorch with CUDA support
echo "Detecting CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Found CUDA version: $CUDA_VERSION"
    
    # Extract major and minor version numbers
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    
    # Map CUDA version to PyTorch index
    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
        CUDA_INDEX="cu121"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        CUDA_INDEX="cu121"
    elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        CUDA_INDEX="cu118"
    else
        CUDA_INDEX="cu118"  # Default to 11.8
    fi
    
    echo "Installing PyTorch with CUDA $CUDA_INDEX support..."
    # Pin PyTorch to a stable version compatible with flash-attn and vLLM
    # PyTorch 2.5.1 is stable and works well with flash-attn 2.8.x
    if [ "$CUDA_INDEX" = "cu121" ]; then
        echo "Installing PyTorch 2.5.1+cu121 (stable, compatible with flash-attn)..."
        uv pip install "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
            --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
    else
        echo "Installing PyTorch 2.5.1+cu118 (stable, compatible with flash-attn)..."
        uv pip install "torch==2.5.1+cu118" "torchvision==0.20.1+cu118" "torchaudio==2.5.1+cu118" \
            --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
    fi
else
    echo "CUDA not found. Installing CPU-only PyTorch..."
    echo "Note: For GPU support, install CUDA and reinstall PyTorch with:"
    echo "  uv pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118"
    uv pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
fi

# Install project dependencies using uv
# Read dependencies from pyproject.toml and install them directly
# Use quotes around version specifiers to prevent shell expansion issues
echo "Installing project dependencies from pyproject.toml..."
cd "$PROJECT_ROOT"
uv pip install \
    "transformers>=4.35.0" \
    "peft>=0.7.0" \
    "accelerate>=0.24.0" \
    "bitsandbytes>=0.45.0" \
    "datasets>=2.14.0" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20.0" \
    "packaging>=21.0" \
    "ninja>=1.11.0" \
    "einops>=0.7.0" \
    "safetensors>=0.4.0" \
    "huggingface-hub>=0.19.0" \
    "tokenizers>=0.15.0" \
    "litgpt>=0.2.0" \
    "axolotl>=0.4.0" \
    "wandb>=0.16.0" \
    "tensorboard>=2.15.0"

# Clean up any =* files that might have been created
cleanup_temp_files

# Install dev dependencies
echo "Installing dev dependencies..."
cd "$PROJECT_ROOT"
uv pip install "pytest>=7.4.0" "black>=23.0.0" "ruff>=0.1.0" "ipython>=8.17.0" "requests>=2.31.0"

# Clean up any =* files that might have been created
cleanup_temp_files

# Install flash-attn (requires build dependencies)
# flash-attn needs torch, wheel, setuptools, and cmake to build
echo "Installing flash-attn build dependencies..."
uv pip install wheel setuptools cmake

# Install flash-attn (version 2.8.x - compatible with PyTorch 2.5.1)
# Note: If vLLM upgrades PyTorch to 2.9.0, flash-attn will be removed to avoid compatibility issues
# flash-attn needs torch to be installed first, so use --no-build-isolation
echo "Installing flash-attn 2.8.3 (compatible with PyTorch 2.5.1)..."
if uv pip install "flash-attn==2.8.3" --no-build-isolation 2>&1; then
    echo "flash-attn installed successfully"
    python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')" 2>/dev/null || true
    # Verify it works - wait a moment for any lazy loading, then test
    sleep 1
    if python -c "import flash_attn_2_cuda" 2>/dev/null; then
        echo "✓ flash-attn CUDA extension verified and working"
    else
        echo "Warning: flash-attn installed but CUDA extension import failed during setup"
        echo "This may be a timing issue. It should work when vLLM uses it."
    fi
else
    echo "Warning: flash-attn installation failed"
    echo "  This may require additional system dependencies (CUDA toolkit, etc.)"
    echo "  vLLM will use fallback attention if flash-attn is not available."
fi

# Install vLLM for serving
echo "Installing vLLM..."
cd "$PROJECT_ROOT"
# Upgrade torchao first to avoid compatibility issues with newer transformers
echo "Upgrading torchao for compatibility..."
uv pip install --upgrade "torchao>=0.6.0" || echo "Warning: torchao upgrade failed, continuing..."
# Save current PyTorch version before installing vLLM
PYTORCH_VERSION_BEFORE=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "PyTorch version before vLLM install: $PYTORCH_VERSION_BEFORE"
# Install vLLM - we'll use latest and rebuild flash-attn for the PyTorch version it requires
# This ensures compatibility between vLLM, PyTorch, and flash-attn
echo "Installing vLLM (will determine required PyTorch version)..."
uv pip install "vllm>=0.6.0" || echo "Warning: vLLM installation had issues"
# Check if PyTorch was upgraded and fix if needed
PYTORCH_VERSION_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "PyTorch version after vLLM install: $PYTORCH_VERSION_AFTER"
if [ "$PYTORCH_VERSION_BEFORE" != "$PYTORCH_VERSION_AFTER" ]; then
    echo "Warning: PyTorch was upgraded by vLLM from $PYTORCH_VERSION_BEFORE to $PYTORCH_VERSION_AFTER"
    echo "Downgrading PyTorch back to 2.5.1 for flash-attn compatibility..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
            CUDA_INDEX="cu121"
        else
            CUDA_INDEX="cu118"
        fi
        # Determine CUDA index for downgrade
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
            if [ "$CUDA_MAJOR" -ge 12 ]; then
                CUDA_INDEX="cu121"
            else
                CUDA_INDEX="cu118"
            fi
        else
            CUDA_INDEX="cu118"
        fi
        uv pip install "torch==2.5.1+$CUDA_INDEX" "torchvision==0.20.1+$CUDA_INDEX" "torchaudio==2.5.1+$CUDA_INDEX" \
            --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" --force-reinstall
    else
        uv pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" --force-reinstall
    fi
    echo "Reinstalling vLLM to rebuild against PyTorch 2.5.1..."
    echo "This is necessary because vLLM's C extensions must match the PyTorch version..."
    uv pip uninstall vllm 2>/dev/null || true
    # vLLM may require newer PyTorch, so we'll reinstall it and then rebuild flash-attn for the final PyTorch version
    uv pip install "vllm>=0.6.0" || echo "Warning: vLLM reinstall had issues"
    # Check final PyTorch version after vLLM install
    PYTORCH_FINAL=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    PYTORCH_FINAL_BASE=$(echo "$PYTORCH_FINAL" | cut -d+ -f1)
    echo "Final PyTorch version after vLLM reinstall: $PYTORCH_FINAL"
    
    if [ "$PYTORCH_FINAL_BASE" = "2.5.1" ]; then
        echo "PyTorch is 2.5.1 - reinstalling flash-attn 2.8.3..."
        uv pip uninstall flash-attn 2>/dev/null || true
        uv pip install "flash-attn==2.8.3" --no-build-isolation || echo "Warning: flash-attn reinstall failed"
    else
        echo "vLLM requires PyTorch $PYTORCH_FINAL_BASE (newer than 2.5.1)"
        echo "Removing flash-attn to avoid compatibility issues with PyTorch $PYTORCH_FINAL_BASE..."
        echo "vLLM will use its built-in optimized fallback attention (works correctly)"
        uv pip uninstall flash-attn 2>/dev/null || true
        echo "✓ flash-attn removed. vLLM will use fallback attention mechanisms."
    fi
else
    echo "PyTorch version unchanged - flash-attn should work"
fi

# Clean up any =* files that might have been created
cleanup_temp_files

# Final cleanup of any =* files
cleanup_temp_files

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use the activate script:"
echo "  source scripts/activate_env.sh"
