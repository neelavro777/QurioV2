#!/usr/bin/env bash
# start_web.sh — Sequential startup script for Qurio Web Server
# Usage: bash backend/scripts/start_web.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════╗"
echo "║   Qurio Web Server Startup Sequence  ║"
echo "║   API is running at:                 ║"
echo "║   http://localhost:8080              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Validate sudo upfront
echo "This script requires sudo privileges to start the Docker models."
sudo -v

# Cleanup on exit
cleanup() {
    echo ""
    echo "----------------------------------------"
    echo "Initiating cleanup..."
    echo "Stopping Docker containers (qwen35-server, bge-embed)..."
    sudo docker stop qwen35-server bge-embed || true
    echo "✅ Cleanup complete. Goodbye!"
}
trap cleanup EXIT

# 1. Start Qwen3.5 server
echo "----------------------------------------"
echo "Starting Qwen3.5 Server..."
sudo docker start qwen35-server

echo "Waiting for Qwen3.5 Server to become ready (port 8000)..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 2
done
echo "✅ Qwen3.5 Server is ready."
echo ""

# 2. Start BGE Embedding server
echo "----------------------------------------"
echo "Starting BGE Embedding Server..."
sudo docker start bge-embed

echo "Waiting for BGE Embedding Server to become ready (port 8001)..."
while ! curl -s http://localhost:8001/v1/models > /dev/null; do
    sleep 2
done
echo "✅ BGE Embedding Server is ready."
echo ""

# 3. Start Web Server
echo "----------------------------------------"
echo "Starting Qurio Observability API Server..."
echo "👉 API Server will be available at: http://localhost:8080"
echo "----------------------------------------"
cd "$BACKEND_DIR"

if [ -f ".venv/bin/python" ]; then
  .venv/bin/python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8080 --reload --log-level warning
elif command -v uv &>/dev/null; then
  uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8080 --reload --log-level warning
else
  python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8080 --reload --log-level warning
fi

# Prevent bash exec optimization so the EXIT trap fires reliably
exit 0
