# beastars
colonelGPT training course

## ML Training Environment Setup

A reproducible local ML environment suitable for isolated operation with support for:
- PyTorch + CUDA
- Axolotl for config-based finetuning
- Transformers + PEFT for code-based finetuning
- LitGPT for pretrain-finetune-inference lifecycle

## Quick Start

### 1. Initial Setup

Run the setup script to create the environment and install dependencies:

```bash
make setup
```

Or manually:

```bash
./scripts/setup_env.sh
```

### 2. Activate Environment

Activate the environment and set cache directories:

```bash
source scripts/activate_env.sh
```

Or use make:

```bash
make activate
```

Or manually:

```bash
source .venv/bin/activate
export HF_HOME=$(pwd)/.cache/huggingface
export TRANSFORMERS_CACHE=$(pwd)/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$(pwd)/.cache/huggingface/datasets
export MODELS_DIR=$(pwd)/models
export DATASETS_DIR=$(pwd)/datasets
export RUNS_DIR=$(pwd)/runs
export EVAL_DIR=$(pwd)/eval
```

### 3. Run Inference Sanity Check

Test GPU utilization and mixed precision with a 3B model:

```bash
make test
```

Or manually:

```bash
source scripts/activate_env.sh
python tests/test_inference.py
```

### 4. vLLM Serving Test

**Workflow:** The vLLM serving test requires a two-step process:
1. **First:** Start the server with `make serve` (must be running)
2. **Then:** Run the test with `make serve-test` (in another terminal)

#### Step 1: Start the Server

**IMPORTANT:** You must start the server first before running tests!

In terminal 1, start the vLLM OpenAI-compatible server:

```bash
make serve
```

Or manually:

```bash
./scripts/serve_planner.sh
```

The server will start on `http://localhost:8000` with API key `localtoken`.

**Configuration via environment variables:**
- `VLLM_MODEL` - Model to serve (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `VLLM_API_KEY` - API key (default: `localtoken`)
- `VLLM_HOST` - Host address (default: `0.0.0.0`)
- `VLLM_PORT` - Port number (default: `8000`)
- `VLLM_DTYPE` - Data type (default: `auto`)

**Example with custom settings:**
```bash
VLLM_MODEL="microsoft/Phi-3-mini-4k-instruct" VLLM_PORT=8001 make serve
```

#### Step 2: Run the Baseline Test

**IMPORTANT:** The server must be running from Step 1!

In terminal 2 (while server is running in terminal 1), run the test:

```bash
make serve-test
```

Or manually:

```bash
source scripts/activate_env.sh
python tests/test_vllm_baseline.py
```

**What the test does:**
1. **Checks prerequisites:**
   - Verifies vllm is installed
   - Verifies required libraries are available
   - Checks if server is running on the configured port
2. **Waits for server to be ready** (if needed)
3. **Runs golden prompts** against the server (chat and completion endpoints)
4. **Captures baseline outputs** to `tests/baseline_outputs.jsonl`

**If prerequisites fail**, the test will provide detailed instructions on what to do.

#### Step 3: Run Performance Metrics Test (Optional)

Benchmark vLLM performance and compare hardware compatibility:

```bash
make serve-metrics
```

Or manually:

```bash
source scripts/activate_env.sh
python tests/test_vllm_metrics.py
```

**What the metrics test does:**
1. **Collects system information:**
   - Python version, CUDA version, PyTorch version
   - GPU details (name, memory, compute capability)
2. **Tests single request latency:**
   - Measures latency for different prompt lengths
   - Calculates tokens per second
   - Records time to first token (if available)
3. **Tests batch processing:**
   - Concurrent request handling (1, 2, 4 requests)
   - Throughput measurement
   - Batch efficiency metrics
4. **Monitors GPU memory usage:**
   - Memory allocation and utilization
   - Per-GPU statistics
5. **Saves results** to `tests/vllm_metrics_results.json` for comparison

**Customization:**
```bash
# Custom API endpoint
VLLM_API_BASE=http://localhost:8001 make serve-metrics

# More requests per test
python tests/test_vllm_metrics.py --num-requests 10

# Longer generations
python tests/test_vllm_metrics.py --max-tokens 200
```

#### Step 4: Stop the Server (when done)

When you're finished testing, stop the server:

```bash
make serve-stop
```

This will:
- Find and kill all vLLM server processes
- Clean up processes using the configured port
- Gracefully terminate the server

**Example serve command (direct vllm usage):**
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype auto --api-key localtoken
```

**Hardware Compatibility Comparison:**

The `make serve-metrics` test generates a JSON report (`tests/vllm_metrics_results.json`) that includes:
- Token generation speed (tokens/second)
- Latency metrics (average, per-request)
- Batch processing throughput
- GPU memory utilization
- System specifications

You can compare these metrics across different hardware setups to evaluate performance characteristics.

#### Troubleshooting

**If you see `AttributeError: module 'torchao.quantization' has no attribute 'Float8WeightOnlyConfig'`:**

This is a compatibility issue between torchao and transformers. The serve script automatically fixes this, but if you encounter it:

```bash
source scripts/activate_env.sh
uv pip install --upgrade "torchao>=0.6.0"
```

Then restart the server with `make serve`.

**Version Compatibility:**

The setup script installs compatible versions:
- **PyTorch**: 2.9.0 (required by vLLM 0.13.0)
- **vLLM**: 0.13.0 (latest stable version)
- **flash-attn**: Installed but may use fallback attention

**Current Status:**
- vLLM works correctly with PyTorch 2.9.0
- Server starts successfully
- vLLM automatically uses fallback attention (xformers or eager) if flash-attn isn't fully compatible
- This provides full functionality, though may be slightly slower than with optimized flash-attn

**Flash-attn Compatibility:**
- vLLM 0.13.0 requires PyTorch 2.9.0
- flash-attn 2.8.3 was built for PyTorch 2.5.1
- The setup script automatically removes flash-attn when PyTorch is upgraded to 2.9.0
- vLLM uses its built-in optimized fallback attention (xformers/eager) which works correctly

**To enable flash-attn (optional, advanced):**
If you want flash-attn optimized attention, you need to rebuild it from source for PyTorch 2.9.0. This requires CUDA toolkit and compilation tools:

```bash
source scripts/activate_env.sh
# Ensure build dependencies
uv pip install packaging wheel setuptools ninja cmake
# Rebuild from source (takes several minutes, requires CUDA toolkit)
pip install flash-attn --no-build-isolation --no-binary flash-attn
```

**Note:** vLLM works perfectly fine without flash-attn. The fallback attention is well-optimized and provides good performance.

**Memory Management:**
The serve script includes memory-optimized settings for GPUs with limited memory:
- `--gpu-memory-utilization 0.85`: Uses 85% of GPU memory (default is 90%)
- `--max-num-seqs 64`: Limits concurrent sequences to 64 (default is 256)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Reduces memory fragmentation

If you encounter CUDA out of memory errors, you can further reduce these values by setting environment variables:
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=32
make serve
```

**If you see `ImportError: undefined symbol` with flash-attn:**

This should not happen with the pinned versions, but if it does:

```bash
source scripts/activate_env.sh
# Verify PyTorch version
python -c "import torch; print(torch.__version__)"  # Should be 2.5.1
# Reinstall flash-attn
uv pip uninstall flash-attn
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Directory Structure

```
beastars/
├── scripts/        # Setup and activation scripts
├── requirements/   # Requirements files
├── tests/          # Test scripts
├── models/         # Local model checkpoints
├── datasets/       # Local datasets
├── runs/           # Training runs and logs
├── eval/           # Evaluation results
└── .cache/         # Hugging Face cache
    └── huggingface/
        ├── transformers/
        └── datasets/
```

## Environment Variables

The environment uses the following cache and directory variables:

- `HF_HOME`: Base Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers model cache
- `HF_DATASETS_CACHE`: Datasets cache
- `MODELS_DIR`: Local model storage
- `DATASETS_DIR`: Local dataset storage
- `RUNS_DIR`: Training run outputs
- `EVAL_DIR`: Evaluation outputs

## Tools and Frameworks

### PyTorch + CUDA
- Full PyTorch support with CUDA acceleration
- Mixed precision training (FP16/BF16)
- GPU memory management

### Axolotl
Config-based finetuning framework. See [docs.axolotl.ai](https://docs.axolotl.ai)

### Transformers + PEFT
Code-based finetuning with Parameter-Efficient Fine-Tuning. See [Hugging Face PEFT docs](https://huggingface.co/docs/transformers/en/peft)

### LitGPT
Pretrain-finetune-inference lifecycle management. See [LitGPT tutorial](https://raw.githubusercontent.com/Lightning-AI/litgpt/main/tutorials/0_to_litgpt.md)

### vLLM
High-performance LLM serving with OpenAI-compatible API. See [vLLM OpenAI-Compatible Server docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) and [LoRA Adapters docs](https://docs.vllm.ai/en/latest/serving/lora.html)

## Python Version

This project uses **Python 3.12** for compatibility with PyTorch and ML libraries.

## Dependencies

All dependencies are managed via `uv` and defined in `pyproject.toml`. Key packages include:

- `torch>=2.0.0` - PyTorch
- `transformers>=4.35.0` - Hugging Face Transformers
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning
- `litgpt>=0.2.0` - LitGPT framework
- `axolotl>=0.4.0` - Axolotl finetuning
- `accelerate>=0.24.0` - Hugging Face Accelerate
- `bitsandbytes>=0.41.0` - Quantization support
- `flash-attn>=2.3.0` - Flash Attention v2 (installed with build dependencies)
- `vllm>=0.6.0` - vLLM for high-performance LLM serving

## Development

Install development dependencies:

```bash
uv sync --group dev
```

Or using uv directly:

```bash
uv pip install -e ".[dev]"
```

## Maintenance

### Cleanup Temporary Files

If temporary `=*` files appear in the project root (they shouldn't with the updated setup script), you can clean them up:

```bash
make clean-temp
```

Or manually:

```bash
./scripts/cleanup_temp_files.sh
```

## Notes

- The environment is designed for isolated, offline-capable operation
- All models and datasets are cached locally
- GPU utilization is automatically detected and configured
- Mixed precision is enabled when supported by hardware
- Temporary files are automatically cleaned up during setup