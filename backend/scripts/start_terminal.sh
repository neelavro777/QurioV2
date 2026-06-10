#!/usr/bin/env bash
# start_terminal.sh — Sequential startup script for Qurio Terminal Interface
# Usage: bash backend/scripts/start_terminal.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════╗"
echo "║   Qurio Terminal Startup Sequence    ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Validate sudo upfront
echo "This script requires sudo privileges to start the Docker models."
sudo -v

# cleanup on exit
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

# 3. Start Terminal Interface
echo "----------------------------------------"
echo "Starting Qurio Terminal Interface..."
cd "$BACKEND_DIR"

if [ -f ".venv/bin/python" ]; then
  .venv/bin/python scripts/main.py
elif command -v uv &>/dev/null; then
  uv run python scripts/main.py
else
  python scripts/main.py
fi

# Prevent bash exec optimization so the EXIT trap fires reliably
exit 0
