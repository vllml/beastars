#!/usr/bin/env python3
"""vLLM performance metrics and hardware compatibility benchmarking."""

import os
import sys
import time
import json
import requests
from typing import Dict, List, Optional
import torch


def get_system_info() -> Dict:
    """Collect system and hardware information."""
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
    }
    
    # PyTorch and CUDA info
    if torch.cuda.is_available():
        info["cuda"] = {
            "available": True,
            "version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
        }
        
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        info["cuda"]["devices"] = devices
    else:
        info["cuda"] = {"available": False}
    
    return info


def test_vllm_metrics(
    api_base: str = "http://localhost:8000",
    api_key: str = "localtoken",
    model: Optional[str] = None,
    num_requests: int = 5,
    max_tokens: int = 100,
) -> Dict:
    """Test vLLM performance metrics."""
    print("=" * 80)
    print("vLLM Performance Metrics Test")
    print("=" * 80)
    
    # Get system info
    system_info = get_system_info()
    print("\nSystem Information:")
    print(f"  Python: {system_info['python_version']}")
    if system_info["cuda"]["available"]:
        print(f"  CUDA: {system_info['cuda']['version']}")
        print(f"  PyTorch: {system_info['cuda']['pytorch_version']}")
        for device in system_info["cuda"]["devices"]:
            print(f"  GPU {device['index']}: {device['name']} ({device['total_memory_gb']} GB)")
    
    # Test prompts of varying lengths
    test_prompts = [
        "Hello, how are you?",
        "Write a Python function to calculate the factorial of a number.",
        "Explain the concept of machine learning in simple terms. Include examples of supervised and unsupervised learning.",
    ]
    
    results = {
        "system_info": system_info,
        "api_base": api_base,
        "model": model,
        "tests": [],
        "summary": {},
    }
    
    # Check if server is available
    try:
        response = requests.get(
            f"{api_base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        response.raise_for_status()
        models_data = response.json()
        if not model:
            model = models_data["data"][0]["id"] if models_data.get("data") else "unknown"
        results["model"] = model
        print(f"\nUsing model: {model}")
    except Exception as e:
        print(f"\nERROR: Cannot connect to vLLM server at {api_base}")
        print(f"  Error: {e}")
        print(f"  Make sure the server is running: make serve")
        sys.exit(1)
    
    # Test 1: Single request latency
    print("\n" + "=" * 80)
    print("Test 1: Single Request Latency")
    print("=" * 80)
    
    single_request_metrics = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i} ({len(prompt.split())} words):")
        print(f"  '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")
        
        for attempt in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                end_time = time.time()
                
                data = response.json()
                completion = data["choices"][0]["message"]["content"]
                tokens_generated = data["usage"]["completion_tokens"]
                total_tokens = data["usage"]["total_tokens"]
                
                latency = end_time - start_time
                tokens_per_second = tokens_generated / latency if latency > 0 else 0
                
                # Check for time_to_first_token if available
                if "time_to_first_token" in data.get("usage", {}):
                    ttft = data["usage"]["time_to_first_token"]
                else:
                    ttft = None
                
                single_request_metrics.append({
                    "prompt_index": i,
                    "prompt_length": len(prompt),
                    "attempt": attempt + 1,
                    "latency_seconds": round(latency, 3),
                    "tokens_generated": tokens_generated,
                    "total_tokens": total_tokens,
                    "tokens_per_second": round(tokens_per_second, 2),
                    "time_to_first_token": ttft,
                })
                
                if attempt == 0:  # Only print first attempt
                    print(f"  Latency: {latency:.3f}s")
                    print(f"  Tokens: {tokens_generated} (total: {total_tokens})")
                    print(f"  Speed: {tokens_per_second:.2f} tokens/s")
                    if ttft:
                        print(f"  Time to first token: {ttft:.3f}s")
            except Exception as e:
                print(f"  ERROR on attempt {attempt + 1}: {e}")
    
    # Calculate averages
    if single_request_metrics:
        avg_latency = sum(m["latency_seconds"] for m in single_request_metrics) / len(single_request_metrics)
        avg_tokens_per_second = sum(m["tokens_per_second"] for m in single_request_metrics) / len(single_request_metrics)
        avg_tokens_generated = sum(m["tokens_generated"] for m in single_request_metrics) / len(single_request_metrics)
        
        results["tests"].append({
            "name": "single_request_latency",
            "metrics": single_request_metrics,
            "summary": {
                "avg_latency_seconds": round(avg_latency, 3),
                "avg_tokens_per_second": round(avg_tokens_per_second, 2),
                "avg_tokens_generated": round(avg_tokens_generated, 1),
                "num_requests": len(single_request_metrics),
            },
        })
        
        print(f"\n  Average Latency: {avg_latency:.3f}s")
        print(f"  Average Speed: {avg_tokens_per_second:.2f} tokens/s")
        print(f"  Average Tokens Generated: {avg_tokens_generated:.1f}")
    
    # Test 2: Batch processing (concurrent requests)
    print("\n" + "=" * 80)
    print("Test 2: Batch Processing (Concurrent Requests)")
    print("=" * 80)
    
    import concurrent.futures
    
    batch_sizes = [1, 2, 4]
    batch_metrics = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size} concurrent requests")
        
        def make_request(request_id):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": test_prompts[0]}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                end_time = time.time()
                
                data = response.json()
                tokens_generated = data["usage"]["completion_tokens"]
                latency = end_time - start_time
                tokens_per_second = tokens_generated / latency if latency > 0 else 0
                
                return {
                    "request_id": request_id,
                    "latency_seconds": round(latency, 3),
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": round(tokens_per_second, 2),
                    "success": True,
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                }
        
        batch_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(make_request, i) for i in range(batch_size)]
            batch_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        batch_end = time.time()
        
        total_batch_time = batch_end - batch_start
        successful = [r for r in batch_results if r.get("success")]
        
        if successful:
            avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
            avg_tokens_per_second = sum(r["tokens_per_second"] for r in successful) / len(successful)
            total_tokens = sum(r["tokens_generated"] for r in successful)
            throughput = total_tokens / total_batch_time if total_batch_time > 0 else 0
            
            batch_metrics.append({
                "batch_size": batch_size,
                "total_batch_time_seconds": round(total_batch_time, 3),
                "successful_requests": len(successful),
                "failed_requests": len(batch_results) - len(successful),
                "avg_latency_seconds": round(avg_latency, 3),
                "avg_tokens_per_second": round(avg_tokens_per_second, 2),
                "total_tokens_generated": total_tokens,
                "throughput_tokens_per_second": round(throughput, 2),
                "requests": batch_results,
            })
            
            print(f"  Total batch time: {total_batch_time:.3f}s")
            print(f"  Successful: {len(successful)}/{batch_size}")
            print(f"  Average latency: {avg_latency:.3f}s")
            print(f"  Throughput: {throughput:.2f} tokens/s")
        else:
            print(f"  ERROR: All requests failed")
    
    if batch_metrics:
        results["tests"].append({
            "name": "batch_processing",
            "metrics": batch_metrics,
        })
    
    # Test 3: GPU Memory Usage (if available)
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("Test 3: GPU Memory Usage")
        print("=" * 80)
        
        memory_info = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1e9
            
            memory_info.append({
                "device_index": i,
                "device_name": props.name,
                "total_memory_gb": round(total, 2),
                "allocated_memory_gb": round(allocated, 2),
                "reserved_memory_gb": round(reserved, 2),
                "utilization_percent": round((allocated / total) * 100, 1) if total > 0 else 0,
            })
            
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total: {total:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB ({memory_info[-1]['utilization_percent']}%)")
            print(f"  Reserved: {reserved:.2f} GB")
        
        results["tests"].append({
            "name": "gpu_memory",
            "metrics": memory_info,
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if results["tests"]:
        for test in results["tests"]:
            if test["name"] == "single_request_latency" and "summary" in test:
                summary = test["summary"]
                results["summary"]["single_request"] = {
                    "avg_latency_seconds": summary["avg_latency_seconds"],
                    "avg_tokens_per_second": summary["avg_tokens_per_second"],
                }
                print(f"\nSingle Request Performance:")
                print(f"  Average Latency: {summary['avg_latency_seconds']}s")
                print(f"  Average Speed: {summary['avg_tokens_per_second']} tokens/s")
            
            elif test["name"] == "batch_processing" and test["metrics"]:
                best_batch = max(test["metrics"], key=lambda x: x.get("throughput_tokens_per_second", 0))
                results["summary"]["batch_processing"] = {
                    "best_batch_size": best_batch["batch_size"],
                    "max_throughput_tokens_per_second": best_batch["throughput_tokens_per_second"],
                }
                print(f"\nBatch Processing:")
                print(f"  Best batch size: {best_batch['batch_size']}")
                print(f"  Max throughput: {best_batch['throughput_tokens_per_second']} tokens/s")
    
    # Save results to JSON file
    output_file = "tests/vllm_metrics_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)
    
    return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Performance Metrics Test")
    parser.add_argument(
        "--api-base",
        default=os.environ.get("VLLM_API_BASE", "http://localhost:8000"),
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "localtoken"),
        help="API key for authentication",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of requests per test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per request",
    )
    
    args = parser.parse_args()
    
    try:
        results = test_vllm_metrics(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS: vLLM metrics test completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
