.PHONY: setup setup-clear activate test serve serve-test serve-stop clean clean-temp help

help:
	@echo "Available targets:"
	@echo "  setup      - Set up the ML training environment (auto-clears existing venv)"
	@echo "  setup-clear - Set up with explicit UV_VENV_CLEAR flag"
	@echo "  activate   - Show activation instructions"
	@echo "  test       - Run inference sanity check"
	@echo "  serve      - Start vLLM OpenAI-compatible server (must run first)"
	@echo "  serve-test - Run vLLM baseline output capture test (requires serve to be running)"
	@echo "  serve-stop - Stop vLLM server and clean up processes"
	@echo "  clean-temp - Clean up temporary =* files"
	@echo "  clean      - Clean up generated files and caches"

setup:
	@./scripts/setup_env.sh

setup-clear:
	@UV_VENV_CLEAR=1 ./scripts/setup_env.sh

activate:
	@echo "To activate the environment, run:"
	@echo "  source scripts/activate_env.sh"

test:
	@bash -c "source scripts/activate_env.sh && python tests/test_inference.py"

serve:
	@./scripts/serve_planner.sh

serve-test:
	@bash -c "source scripts/activate_env.sh && python tests/test_vllm_baseline.py"

serve-stop:
	@./scripts/stop_server.sh

clean-temp:
	@./scripts/cleanup_temp_files.sh

clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -rf .cache
	@rm -rf __pycache__ *.pyc
	@rm -rf .pytest_cache
	@find . -maxdepth 1 -name "=*" -type f -delete 2>/dev/null || true
	@echo "Clean complete. Run 'make setup' to recreate environment."
