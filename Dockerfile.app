FROM python:3.11-slim-bookworm

# Make apt-installed Python packages (python3-gi) visible to the python.org
# Python binary. python:slim uses a custom Python build whose sys.path does
# not include /usr/lib/python3/dist-packages by default.
#
# Use a .pth file (APPENDED to sys.path) instead of PYTHONPATH (PREPENDED).
# Prepending causes system packages (e.g. system matplotlib) to shadow pip
# packages of the same name, leading to pybind11 crashes at import time.
# GStreamer + Python bindings (python3-gi must come from apt — pip PyGObject
# requires build tools and frequently mismatches the system GStreamer version)
# gstreamer1.0-libav   -> avdec_h264 software H.264 decode
# gstreamer1.0-plugins-bad -> rtspsrc for RTSP streams
# python3-gi / gir1.2-gstreamer-1.0 -> gi.repository.Gst Python bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.app.txt .
RUN pip install --no-cache-dir -r requirements.app.txt

# Append system dist-packages so pip packages win over system ones (gi/gst fallback).
RUN echo "/usr/lib/python3/dist-packages" >> /usr/local/lib/python3.11/site-packages/system-gi.pth

# Application code (no model weights, no triton_models)
COPY config.py main.py ./
COPY services/ services/
COPY handlers/ handlers/
COPY models/ models/
COPY prompts/ prompts/
COPY templates/ templates/

# Seed file -- copied to /data/family.md by entrypoint on first run
COPY family.md /app/family.md.default

COPY scripts/app-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# /data is the named volume mount point for family.md
RUN mkdir -p /data /app/logs

ENV FAMILY_MD_PATH=/data/family.md

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
