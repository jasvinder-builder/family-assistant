#!/bin/bash
set -e

# Start ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for server to accept requests (curl not available in ollama image)
echo "Waiting for Ollama server to start..."
until (echo > /dev/tcp/localhost/11434) 2>/dev/null; do
    sleep 2
done
echo "Ollama server ready."

# Pull the model only if not already present (skips the 9GB download on restarts)
MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
# ollama list output: "NAME   ID   SIZE   MODIFIED" — match anywhere in the name column
if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qF "${MODEL}"; then
    echo "${MODEL} already present, skipping pull."
else
    echo "Pulling ${MODEL} (this may take several minutes on first run)..."
    ollama pull "${MODEL}"
    echo "Pull complete."
fi

# Keep the ollama server process in foreground
wait $OLLAMA_PID
