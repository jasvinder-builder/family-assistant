# Scene Service — Production Improvement Ideas

## Current state (as of Step 2)

- **Decode:** PyAV (FFmpeg software decode, 4 threads)
- **Inference:** Grounding DINO Tiny fp16 on CUDA, synchronous, single-stream
- **Tracking:** supervision ByteTrack on CPU (~1–5ms, not a bottleneck)
- **Display cap:** 25fps — skips `to_ndarray` + JPEG encode on non-display frames
- **Analysis cap:** 5fps — GDINO runs at most 5x/sec per stream
- **Pre-resize:** frames downscaled to max 800px wide before GDINO (~15–20fps achievable)

---

## Step 3 — GStreamer NVDEC hardware decode

Replace PyAV reader with a GStreamer pipeline using `nvv4l2decoder`.

**Pipeline:**
```
rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv → appsink
```

**Benefits:**
- Zero-copy NVDEC: H.264/H.265 decoded directly on GPU, no CPU decode at all
- Frames land in CUDA memory — eliminates `to_ndarray` CPU copy for the decode path
- Frees ~1–2 CPU cores currently spent on PyAV software decode
- Handles 4K streams without CPU becoming the bottleneck

**Implementation sketch:**
```python
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

pipeline_str = (
    f"rtspsrc location={url} latency=100 protocols=tcp ! rtph264depay ! h264parse ! "
    "avdec_h264 ! videoconvert ! video/x-raw,format=BGRx ! "
    "appsink name=sink emit-signals=false max-buffers=2 drop=true sync=false"
)
```

Use `appsink.emit("pull-sample")` (blocking) rather than the `new-sample` signal to avoid GLib main loop complexity.

**Status (2026-04-03): partially implemented — BLOCKED by CUDA context conflict.**

The GStreamer reader (`_reader_loop_gst`) is written and working as a pipeline but causes a segfault (SIGABRT / core dump) when combined with PyTorch/GDINO on CUDA:

- `nvh264dec` (desktop NVDEC) outputs `video/x-raw(memory:CUDAMemory)` and creates its own CUDA context on the reader thread
- PyTorch creates its CUDA context on the analysis thread when GDINO loads
- Two concurrent CUDA context initialisations on different threads → crash
- Switching to `avdec_h264` (software decode in GStreamer) did not fully resolve it — crash still occurs, suggesting an underlying GStreamer + PyTorch CUDA interaction even without nvh264dec

**Required to unblock:**
- Eager-initialise PyTorch CUDA at app startup (before any GStreamer pipeline starts) so the CUDA context exists before GStreamer touches it
- OR run the GStreamer reader in a subprocess to isolate CUDA contexts entirely
- Currently falling back to PyAV unconditionally (one-line change in `_reader_loop`)

**Zero-copy path (advanced, post-fix):**
- Use `nvbufsurface` CUDA memory directly → torch tensor without touching CPU RAM
- Eliminates PIL conversion bottleneck entirely

---

## Step 4 — Triton Inference Server for GDINO

Run GDINO as a gRPC service via NVIDIA Triton, batching frames from multiple cameras.

**Why:**
- Current synchronous inference: GPU utilisation ~20–30% (inference + idle gaps)
- Triton dynamic batching: GPU utilisation ~80–90%
- Enables multi-camera: 4 streams each at 5fps = 20 frames/sec batched together

**Architecture:**
```
Camera 1 reader ──┐
Camera 2 reader ──┼──► shared frame queue ──► Triton gRPC client ──► GDINO batch inference
Camera 3 reader ──┘                                                       │
                                                                    ByteTrack per-stream
```

**Steps:**
1. Export GDINO Tiny to ONNX
2. Serve via `tritonserver` with TensorRT backend (fp16)
3. Replace direct model call in `_analysis_loop` with `tritonclient.grpc` async call

**Expected throughput:** ~50–100ms per batch of 4 frames → effectively ~40fps total across 4 cameras

---

## Step 5 — Multi-camera fan-out

Scale to multiple RTSP streams with a single shared GPU inference pipeline.

**Architecture:**
```
set_stream_urls(["rtsp://cam1", "rtsp://cam2", ...])
  → one reader thread + one ByteTrack instance per camera
  → all push frames to a shared inference queue
  → single analysis thread (or Triton) batches and runs GDINO
  → results routed back per-camera by stream_id tag
```

**State changes needed in `scene_service.py`:**
- `_latest_detections` → dict keyed by stream URL
- `_events` → tag each event with source stream
- `_shared_frame` → queue of `(stream_id, frame)` tuples
- `fired` and `score_buffer` → keyed by `(stream_id, track_id, q_idx)`

**UI changes needed in `cameras.html`:**
- Multi-stream URL input
- Per-stream video tile grid
- Events feed filtered by stream

---

## Step 6 — Shared-memory zero-copy frame passing

For multi-camera at high fps, the `push_frame(bgr)` NumPy copy becomes a bottleneck.

**Option A — `multiprocessing.shared_memory`:**
- Pre-allocate a fixed-size slot per camera
- Writer stamps a sequence number; reader checks before consuming
- No copy, no GIL

**Option B — ZeroMQ `inproc` transport:**
- Camera reader publishes raw frame bytes on `inproc://frames`
- Analysis worker subscribes; ZeroMQ handles buffer lifecycle
- Clean pub/sub; easy to add more consumers (recording, alerting)

---

---

## Step 7 — DALI decode + preprocessing (zero-copy GPU path)

Replace both PyAV (camera_service) and PIL/cv2 conversion (scene_service) with NVIDIA DALI.

**Why:**
- DALI `fn.readers.video` can open RTSP directly and output decoded tensors on GPU
- Frame never touches CPU RAM — no `to_ndarray`, no `PIL.fromarray`, no `cv2.cvtColor`
- Combines naturally with Triton (Step 4): GPU tensor → gRPC inference call

**Sketch:**
```python
# pip install nvidia-dali-cuda120
import nvidia.dali.fn as fn
import nvidia.dali.pipeline as pipeline

@pipeline.pipeline_def(batch_size=1, num_threads=2, device_id=0)
def build_pipeline(rtsp_url):
    video, _ = fn.readers.video(device="gpu", filenames=[rtsp_url], sequence_length=1)
    frames = fn.resize(video, resize_shorter=800)  # stays on GPU
    return frames
```

**Best combined with Step 4 (Triton)** so the GPU tensor feeds directly into the inference call without intermediate copies.

---

## Step 8 — DeepStream full pipeline (4+ camera scale)

Full hardware path using NVIDIA DeepStream with Triton as the inference backend.

**Architecture:**
```
DeepStream RTSP source → NVDEC → NvDsPreProcess → Triton gRPC → NvDsPostProcess → ByteTrack
```

**Prerequisite — model choice:** GDINO's dynamic text inputs make it awkward for DeepStream's `nvinfer`.
Two options:
- Write a custom `nvdsinfer_custom_impl` library for GDINO
- Replace GDINO with **YOLOv8-world** (open-vocabulary YOLO, fixed output format, DeepStream-native)

**Based on prior triton work** at github.com/jasvinder-builder/triton (YOLOv7+CLIP+DALI+DeepStream pattern):
- DeepStream container acts as Triton client for RTSP/face detection
- DALI handles mixed CPU+GPU video decode
- YOLO TRT model served via Triton dynamic batching
- CLIP TRT for embeddings; FAISS for identity search

**Only warranted for 4+ cameras or when full NVIDIA stack is desired.**

---

## Phased Roadmap

| Phase | Step | Effort | Impact |
|-------|------|--------|--------|
| 1 | GStreamer NVDEC (Step 3) | ~1 day | High — frees CPU, zero-copy decode |
| 2 | GDINO → TensorRT → Triton (Step 4) | 2–3 days | 2–3× inference speedup, GPU ~80% utilised |
| 3 | DALI preprocessing (Step 7) | 2–3 days | Eliminates PIL/cv2 CPU chain entirely |
| 4 | Multi-camera fan-out (Step 5) | 1–2 days | Requires phases 1+2 |
| 5 | DeepStream full pipeline (Step 8) | 1–2 weeks | Maximum throughput, 4+ cameras |

**VRAM budget (all phases):** Qwen 10GB + Whisper 1.5GB + GDINO TRT 0.3GB + Triton 0.5GB ≈ 3.7GB free — tight but fine.

---

## Notes

- Steps 3–6 are independent — can be done in any order
- Step 3 (GStreamer) is the highest-leverage single change for one-camera use
- Step 4 (Triton) only worth doing if going multi-camera or need >20fps analysis
- Step 5 (multi-camera) requires Steps 3+4 to be worthwhile
- Current VRAM headroom: ~4.2GB free (Qwen 10GB + Whisper 1.5GB + GDINO 0.3GB)
  — enough for TensorRT-optimised GDINO Tiny + Triton overhead
