#!/usr/bin/env python3
"""vLLM baseline output capture test.

This script tests the vLLM OpenAI-compatible server and captures baseline outputs.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "localtoken")
BASELINE_OUTPUTS_FILE = PROJECT_ROOT / "tests" / "baseline_outputs.jsonl"
MAX_RETRIES = 30
RETRY_DELAY = 2


def wait_for_server(api_base, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Wait for vLLM server to be ready."""
    print(f"Waiting for server at {api_base}...")
    print(f"  (Start server with: make serve or ./scripts/serve_planner.sh)")
    for i in range(max_retries):
        try:
            response = requests.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5
            )
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            # Server not running yet
            pass
        except requests.exceptions.RequestException as e:
            # Other errors - might be server starting up
            pass
        
        if i < max_retries - 1:
            if (i + 1) % 5 == 0:  # Print every 5 retries
                print(f"  Retry {i+1}/{max_retries}... (still waiting)")
            time.sleep(retry_delay)
    
    print(f"\nError: Server not ready after {max_retries} retries ({max_retries * retry_delay} seconds)")
    print(f"\nTo start the server, run in another terminal:")
    print(f"  make serve")
    print(f"  # or")
    print(f"  ./scripts/serve_planner.sh")
    return False


def test_chat_completion(api_base, api_key, messages, model=None):
    """Test chat completion endpoint."""
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model or "default",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def test_completion(api_base, api_key, prompt, model=None):
    """Test completion endpoint."""
    url = f"{api_base}/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model or "default",
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def check_server_running(port=8000):
    """Check if a process is listening on the given port."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except Exception:
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")
    issues = []
    
    # Check if vllm is installed
    try:
        import subprocess
        result = subprocess.run(
            ["vllm", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            issues.append("vllm command not found or not working")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        issues.append("vllm is not installed. Run: make setup")
    
    # Check if requests is available
    try:
        import requests
    except ImportError:
        issues.append("requests library not installed. Run: make setup")
    
    # Check if server is running
    port = int(os.environ.get("VLLM_PORT", "8000"))
    if not check_server_running(port):
        issues.append(f"No server detected on port {port}")
    
    if issues:
        print("\n" + "=" * 60)
        print("PREREQUISITE CHECK FAILED")
        print("=" * 60)
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\n" + "=" * 60)
        print("INSTRUCTIONS:")
        print("=" * 60)
        print("\n1. Ensure environment is set up:")
        print("   make setup")
        print("\n2. Start the vLLM server in another terminal:")
        print("   make serve")
        print("   # or")
        print("   ./scripts/serve_planner.sh")
        print("\n3. Wait for server to be ready, then run this test again:")
        print("   make serve-test")
        print("\n" + "=" * 60)
        return False
    
    print("✓ All prerequisites met")
    return True


def capture_baseline_outputs():
    """Capture baseline outputs from golden prompts."""
    print("=" * 60)
    print("vLLM Baseline Output Capture Test")
    print("=" * 60)
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    print()
    
    # Wait for server to be ready
    port = int(os.environ.get("VLLM_PORT", "8000"))
    api_base = os.environ.get("VLLM_API_BASE", f"http://localhost:{port}/v1")
    
    if not wait_for_server(api_base):
        print("\n" + "=" * 60)
        print("ERROR: Server is not responding")
        print("=" * 60)
        print("\nThe server may still be starting up, or there may be an issue.")
        print("\nTroubleshooting:")
        print("1. Check if server is running: lsof -i :8000")
        print("2. Check server logs in the terminal where you ran 'make serve'")
        print("3. Try restarting the server: make serve-stop && make serve")
        sys.exit(1)
    
    # Get model info
    try:
        response = requests.get(
            f"{api_base}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10
        )
        response.raise_for_status()
        models = response.json()
        model_name = models.get("data", [{}])[0].get("id", "default")
        print(f"Using model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not get model info: {e}")
        model_name = "default"
    
    # Golden prompts for baseline capture
    golden_prompts = [
        {
            "type": "chat",
            "messages": [
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
            ],
            "description": "Fibonacci function request"
        },
        {
            "type": "chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "description": "Simple math question"
        },
        {
            "type": "completion",
            "prompt": "The capital of France is",
            "description": "Completion test"
        },
        {
            "type": "chat",
            "messages": [
                {"role": "user", "content": "Explain machine learning in one sentence."}
            ],
            "description": "ML explanation"
        }
    ]
    
    # Capture baseline outputs
    baseline_outputs = []
    
    for i, prompt_config in enumerate(golden_prompts, 1):
        print(f"\n[{i}/{len(golden_prompts)}] Testing: {prompt_config['description']}")
        
        try:
            if prompt_config["type"] == "chat":
                result = test_chat_completion(
                    api_base, API_KEY, prompt_config["messages"], model_name
                )
            else:
                result = test_completion(
                    api_base, API_KEY, prompt_config["prompt"], model_name
                )
            
            # Extract response text
            if prompt_config["type"] == "chat":
                response_text = result["choices"][0]["message"]["content"]
            else:
                response_text = result["choices"][0]["text"]
            
            baseline_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": prompt_config["type"],
                "description": prompt_config["description"],
                "input": prompt_config.get("messages") or prompt_config.get("prompt"),
                "output": response_text,
                "full_response": result
            }
            
            baseline_outputs.append(baseline_entry)
            print(f"  Response: {response_text[:100]}...")
            
        except Exception as e:
            print(f"  Error: {e}")
            baseline_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": prompt_config["type"],
                "description": prompt_config["description"],
                "input": prompt_config.get("messages") or prompt_config.get("prompt"),
                "error": str(e)
            }
            baseline_outputs.append(baseline_entry)
    
    # Save baseline outputs to JSONL file
    BASELINE_OUTPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_OUTPUTS_FILE, "w") as f:
        for entry in baseline_outputs:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"Baseline outputs saved to: {BASELINE_OUTPUTS_FILE}")
    print(f"Captured {len(baseline_outputs)} test cases")
    print("=" * 60)
    
    return baseline_outputs


def main():
    """Main function."""
    try:
        baseline_outputs = capture_baseline_outputs()
        
        # Summary
        successful = sum(1 for entry in baseline_outputs if "error" not in entry)
        failed = len(baseline_outputs) - successful
        
        print(f"\nSummary: {successful} successful, {failed} failed")
        
        if failed > 0:
            print("Warning: Some tests failed. Check baseline_outputs.jsonl for details.")
            sys.exit(1)
        else:
            print("All baseline tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
