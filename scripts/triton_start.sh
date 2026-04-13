#!/bin/bash
# Starts the management sidecar (port 8004) then tritonserver.
# Both share /trt_engines and /models via bind-mounts.

set -e

# Management sidecar — handles TRT re-export requests from deepstream_service
cd /models
uvicorn management:app --host 0.0.0.0 --port 8004 --log-level info &
MGMT_PID=$!
echo "[triton_start] Management sidecar started (pid $MGMT_PID, port 8004)"

# Hand off to tritonserver — exec so signals pass through correctly
exec tritonserver \
    --model-repository=/models \
    --http-port=8002 \
    --grpc-port=8001 \
    --metrics-port=8003 \
    --log-verbose=0 \
    --model-control-mode=explicit \
    --load-model=yoloworld
