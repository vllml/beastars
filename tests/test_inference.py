#!/usr/bin/env python3
"""Inference sanity check for 3B model with GPU utilization and mixed precision."""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def check_gpu_availability():
    """Check GPU availability and CUDA version."""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        return False
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    return True


def test_inference_3b_model(model_name="microsoft/Phi-3-mini-4k-instruct", use_quantization=True):
    """Test inference with a 3B parameter model."""
    print("\n" + "=" * 60)
    print("Inference Test: 3B Model")
    print("=" * 60)
    
    gpu_available = torch.cuda.is_available()
    
    # Configure quantization for memory efficiency
    # Check if bitsandbytes is available and working
    bnb_available = False
    if use_quantization and gpu_available:
        try:
            import bitsandbytes as bnb
            # Try to import the validation function to check if bnb is fully functional
            from transformers.integrations import validate_bnb_backend_availability
            validate_bnb_backend_availability()
            bnb_available = True
        except (ImportError, ModuleNotFoundError, Exception) as e:
            print(f"\nBitsAndBytes not available or incompatible: {e}")
            print("Falling back to full precision...")
            bnb_available = False
    
    if bnb_available:
        print("\nUsing 4-bit quantization (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
    else:
        print("\nUsing full precision (FP16/BF16)...")
        if gpu_available:
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "device_map": "auto",
            }
        else:
            model_kwargs = {"torch_dtype": torch.float32}
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    print(f"Loading model: {model_name}")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If quantization failed, try without quantization
        if bnb_available and "quantization" in str(e).lower() or "bitsandbytes" in str(e).lower():
            print("Quantization failed, retrying without quantization...")
            if gpu_available:
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    "device_map": "auto",
                }
            else:
                model_kwargs = {"torch_dtype": torch.float32}
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **model_kwargs
            )
        else:
            print("Trying alternative model: microsoft/phi-2")
            model_name = "microsoft/phi-2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                **model_kwargs
            )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    # Check memory usage
    if gpu_available:
        print("\nGPU Memory After Loading:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Prepare input
    test_prompt = "Write a short Python function to calculate fibonacci numbers:"
    print(f"\nTest Prompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if gpu_available:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    print("\nGenerating response...")
    print("-" * 60)
    
    # Use torch.amp.autocast instead of deprecated torch.cuda.amp.autocast
    autocast_dtype = torch.bfloat16 if (gpu_available and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.amp.autocast(device_type="cuda" if gpu_available else "cpu", enabled=gpu_available, dtype=autocast_dtype):
        start_time = time.time()
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except AttributeError as e:
                if "seen_tokens" in str(e) or "DynamicCache" in str(e):
                    # Compatibility issue with some models - try without past_key_values
                    print(f"Warning: Generation compatibility issue ({e}), retrying with adjusted settings...")
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable KV cache to avoid compatibility issues
                    )
                else:
                    raise
        generation_time = time.time() - start_time
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    print("-" * 60)
    
    # Performance metrics
    num_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Generation time: {generation_time:.2f} seconds")
    print(f"  Tokens generated: {num_tokens}")
    print(f"  Tokens per second: {tokens_per_second:.2f}")
    
    if gpu_available:
        print(f"\nGPU Memory After Generation:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Verify mixed precision
    print(f"\nMixed Precision Check:")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  CUDA BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"  Autocast enabled: {gpu_available}")
    
    return model, tokenizer


def main():
    """Main function."""
    print("=" * 60)
    print("ML Training Environment - Inference Sanity Check")
    print("=" * 60)
    
    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    print(f"  MODELS_DIR: {os.environ.get('MODELS_DIR', 'Not set')}")
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Test inference
    try:
        model, tokenizer = test_inference_3b_model(use_quantization=gpu_available)
        print("\n" + "=" * 60)
        print("SUCCESS: Inference test completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
