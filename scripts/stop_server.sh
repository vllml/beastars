#!/bin/bash
# Stop vLLM server script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${VLLM_PORT:-8000}"

echo "Stopping vLLM server on port $PORT..."

# Find processes using the port
PIDS=$(lsof -ti:$PORT 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo "No process found listening on port $PORT"
    exit 0
fi

# Also find vllm processes
VLLM_PIDS=$(pgrep -f "vllm serve" 2>/dev/null || true)

# Combine and kill
ALL_PIDS=$(echo "$PIDS $VLLM_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')

if [ -z "$ALL_PIDS" ]; then
    echo "No vLLM server processes found"
    exit 0
fi

echo "Found processes: $ALL_PIDS"
for PID in $ALL_PIDS; do
    echo "  Killing process $PID..."
    kill -TERM "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null || true
done

# Wait a bit for processes to terminate
sleep 2

# Check if any are still running
REMAINING=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "Force killing remaining processes..."
    for PID in $REMAINING; do
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo "Server stopped"
