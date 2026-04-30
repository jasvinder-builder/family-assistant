# Bianca — Family Assistant: Architecture

## Overview

Bianca is a family assistant that runs entirely on local hardware. It handles day-to-day task and event management, acts as a household knowledge base powered by a local LLM with web search, sends proactive reminders via WhatsApp, and runs a production-grade real-time home surveillance pipeline.

**Key characteristics:**
- All AI runs locally on GPU — no cloud AI services, no subscriptions
- Video decoding and ML inference happen entirely on the GPU (CPU never touches video frames)
- Surveillance alerts are defined in plain English — no retraining required
- Seven static containers on one machine; no per-camera dynamic containers (Phase 1 step B, 2026-04-29)

The stack is built on several GPU-capable containers:

| Container | Role | GPU | Key tech |
|---|---|---|---|
| `bianca-app` | FastAPI app, all routes, browser UI, HLS proxy → MediaMTX | No (CPU-only) | FastAPI, uvicorn, Jinja2 |
| `bianca-whisper` | Speech-to-text | Yes | faster-whisper large-v3 |
| `bianca-mediamtx` | Single RTSP/HLS hub: ingest + LL-HLS :8888 + WebRTC :8889 + RTSP :8554 + HTTP API :9997 | No | MediaMTX (latest-ffmpeg) |
| `bianca-inference` | Subprocess-per-camera inference + sibling ffmpeg clip recorder | Yes | savant-deepstream base, PyAV (h264_cuvid), Triton HTTP client, system ffmpeg |
| `bianca-deepstream` | Stream control plane + events deque + clip cutter + REST API | Yes | DeepStream 7.0, imageio-ffmpeg static binary (local-file cuts only) |
| `bianca-triton` | Object detection inference | Yes | Triton 25.03, YOLO-World M TRT |
| `bianca-ollama` | LLM for NLU and generation | Yes | Ollama, Qwen 2.5:14b |

Pre-Phase-1 components (`bianca-go2rtc`, `bianca-savant-module`, dynamic `bianca-rtsp-{cam_id}` and `bianca-sink-{cam_id}` containers) have been removed entirely (Phase 1 cleanup, 2026-04-29). `Dockerfile.savant`, `pipeline/`, `go2rtc.yaml`, `stub.jpg`, and `services/pipeline_worker.py` are gone too.

Inter-container communication:

```
Browser/Twilio ──HTTPS──► app:8000 ──HTTP──► whisper:8080           (STT)
                                    ──HTTP──► deepstream:8090        (events + clips API + stream CRUD)
                                    ──HTTP──► mediamtx:8888          (LL-HLS playlists/segments — proxied)
                                    ──HTTP──► ollama:11434           (LLM)
                                    ──HTTP──► triton:8002            (Triton model management)
                                    ──HTTP──► triton:8004            (TRT re-export trigger)
                    deepstream:8090 ──HTTP──► mediamtx:9997          (path add/remove on add_stream)
                    deepstream:8090 ──HTTP──► inference:8091         (subprocess add/remove on add_stream)
                                             camera ──RTSP──► mediamtx:8554/{cam_id}   (single source connection)
                                                              mediamtx:8554/{cam_id} ──RTSP──► inference (PyAV cuvid → Triton)
                                                              mediamtx:8554/{cam_id} ──RTSP──► inference (sibling ffmpeg → rolling segs)
                    inference (worker) ──HTTP──► deepstream:8090     (per-detection trigger + rate-limited events)
                    inference (worker) ──HTTP──► triton:8002         (YOLO-World inference)
```

---

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Phone calls | Twilio (inbound PSTN) | Handles carrier complexity, webhooks, TTS |
| Phone TTS | AWS Polly via Twilio (`Polly.Joanna`) | Natural voice, no extra integration |
| Browser mic + VAD | `@ricky0123/vad-web` (Silero VAD, ONNX Runtime Web) | ML-based voice detection, runs on device, no tap needed |
| Browser TTS | `speechSynthesis` API | Built-in, no server round-trip |
| Speech-to-text | faster-whisper `large-v3` on CUDA | Free, accurate, used for both phone and browser |
| LLM | Qwen 2.5:14b via Ollama | Strong reasoning, runs fully locally |
| Web search | Tavily API | Clean results API, image search support |
| Messaging | Twilio WhatsApp API | Async research + reminder delivery |
| Storage | Markdown file (`family.md`) | Human-readable, editable, no DB setup |
| File locking | `filelock` | Prevents concurrent write corruption |
| Video ingest | Savant DeepStream framework — per-camera RTSP adapter container → ZeroMQ → Savant module (`pyfunc`) → HTTP POST to deepstream service | Savant handles RTCP SR sync and resilient reconnect; pyfunc extracts NVMM frames via `get_nvds_buf_surface`, applies ROI crop, encodes JPEG, POSTs to `/internal/frame` at configurable FPS |
| Live display | Savant Always-On Sink (LL-HLS port 888) | LL-HLS served directly from pipeline output; browser uses hls.js; no go2rtc fMP4 issues |
| Scene detection | YOLO-World M TRT via Triton 25.03 | Open-vocabulary detection; 8ms median @ 1280×720; queries updated without TRT re-export |
| Object tracking | `_SimpleTracker` (pure-numpy IoU) | Persistent track IDs, deduplication; CPU-only; no pybind11/supervision |
| Inference serving | NVIDIA Triton Inference Server 25.03 (Python backend + TRT) | Model management API enables live query swaps; HTTP :8002 for inference + model control |
| Scheduler | APScheduler `AsyncIOScheduler` | Proactive reminders without Celery/Redis |
| Backend | FastAPI + uvicorn | Async, fast, minimal boilerplate |
| Templates | Jinja2 + Bootstrap 5 | No build step, zero JS framework needed |
| Logging | `TimedRotatingFileHandler` | Daily log files, 7-day retention |
| Tunnel (dev) | Cloudflare Tunnel (`cloudflared`) | Twilio webhooks + HTTPS for browser mic; no bandwidth limits |

---

## GPU Memory Layout (RTX 4070 Ti Super — 16GB)

| Allocation | Size | Notes |
|---|---|---|
| Qwen 2.5:14b Q4_K_M (Ollama) | ~9–10 GB | Always hot; loaded at startup |
| Whisper large-v3 int8_float16 | ~1.5 GB | Always loaded; not active during Qwen inference |
| YOLO-World M TRT engine (Triton) | ~0.3 GB | Loaded at triton startup; persistent |
| DeepStream pipeline buffers | ~0.3 GB | Per-camera nvv4l2decoder + NVMM buffers; scales with camera count |
| **Steady-state total** | **~12.3 GB** | ~3.7 GB headroom |

Note: Whisper and Qwen never run at the same time. TRT re-export (if needed) peaks at ~4GB extra — do not trigger while Qwen is running a long generation.

---

## Docker Compose Architecture

```mermaid
graph TD
    subgraph Stack["docker-compose stack — host: RTX 4070 Ti Super"]
        subgraph whisper["bianca-whisper :8080"]
            W1["faster-whisper large-v3 fp16\n~1.5 GB VRAM\nPOST /transcribe\nGET  /health"]
        end
        subgraph triton["bianca-triton :8001/8002/8003/8004"]
            T1["Triton Inference Server 25.03\nYOLO-World M TRT backend\n~0.3 GB VRAM\nHTTP :8002  gRPC :8001  metrics :8003\nManagement sidecar FastAPI :8004\n(TRT re-export API)"]
        end
        subgraph savant["bianca-savant-module + bianca-savant-sink"]
            SV1["savant-module: pyfunc frame forwarder\nRTSP adapter → ZMQ :5555\npyfunc extracts JPEG from NVMM\nPOST deepstream:8090/internal/frame\nAlways-On Sink subscribed via ZMQ :5556\nLL-HLS :888  WebRTC :8889  RTSP :554"]
        end
        subgraph deepstream["bianca-deepstream :8090"]
            DS1["DeepStream inference service\nFastAPI REST :8090\nPOST /internal/frame → inference slot\nTriton HTTP :8002 inference\nDocker SDK → per-camera RTSP adapter containers\nffmpeg → rolling .ts segments\nClip extraction + index"]
        end
        subgraph ollama["bianca-ollama :11434"]
            O1["Ollama\nQwen 2.5:14b\n~9-10 GB VRAM\nPOST /api/chat\nGET  /api/tags"]
        end
        subgraph app["bianca-app :8000"]
            A1["FastAPI + uvicorn\nNO GPU (CPU-only)\nProxies camera routes → deepstream\nCalls whisper + ollama directly"]
        end
    end

    W1 -- bianca-net --> A1
    DS1 -- bianca-net --> A1
    T1 -- bianca-net --> DS1
    O1 -- bianca-net --> A1
    A1 -->|:8000 host port| EXT["Browser / Twilio\ncloudflared tunnel"]
```

### Container responsibilities

| Container | Image | GPU | Purpose |
|---|---|---|---|
| `bianca-whisper` | `bianca-whisper` (Dockerfile.whisper) | Yes | faster-whisper STT; exposes REST `/transcribe` |
| `bianca-triton` | `bianca-triton` (Dockerfile.triton) | Yes | Triton 25.03 + YOLO-World M TRT; HTTP :8002, gRPC :8001; management sidecar FastAPI :8004 (TRT re-export) |
| `bianca-deepstream` | `bianca-deepstream` (Dockerfile.deepstream) | Yes | Inference service; FastAPI REST :8090; receives JPEG frames via `/internal/frame`; manages RTSP adapter + sink containers via Docker SDK; ffmpeg clip recorders |
| `bianca-savant-module` | `bianca-savant-module` (Dockerfile.savant) | Yes | Savant pyfunc pipeline; receives frames from per-camera RTSP adapters via ZMQ; extracts JPEG; POSTs to deepstream service |
| `bianca-go2rtc` | `alexxit/go2rtc:latest` | No | RTSP normalizing proxy; holds one connection per camera; re-streams DTS-clean H.264 on :8554; registered dynamically via HTTP API :1984 |
| `bianca-rtsp-{cam_id}` | `savant-adapters-gstreamer:latest` | No | Per-camera Savant RTSP adapter; started dynamically; connects to go2rtc :8554; forwards encoded H.264 over ZMQ to savant-module |
| `bianca-sink-{cam_id}` | `savant-adapters-deepstream:latest` | Yes | Per-camera Always-On Sink; started dynamically; subscribes to savant-module ZMQ PUB; serves LL-HLS :888 + RTSP :554 |
| `bianca-ollama` | `ollama/ollama:latest` | Yes | Qwen 2.5:14b LLM; entrypoint auto-pulls model |
| `bianca-app` | `bianca-app` (Dockerfile.app) | **No** | Main FastAPI app; explicitly CUDA-free; proxies camera routes |

### Key Docker Compose design decisions

- **App has no GPU reservation** — enforces CUDA-free constraint; any accidental PyTorch/CUDA import fails fast
- **All five on one bridge network** (`bianca-net`) — containers reach each other by service name
- **`depends_on` with `service_healthy`** — deepstream waits for triton; app waits for deepstream + whisper + ollama
- **Bind-mounts for AI weights** — `~/.cache/huggingface` into whisper; `./models` into triton and deepstream; weights downloaded once to host
- **Named volume for Ollama models** (`ollama-models`) — `qwen2.5:14b` (~9GB) persists across restarts
- **App proxies camera routes** — `main.py` forwards `/cameras/*` to `deepstream:8090` via httpx; app container never imports GStreamer or DeepStream

---

## System Architecture

```mermaid
graph TD
    subgraph Callers["External Callers"]
        PHONE["Caller's Phone (PSTN)"]
        BROWSER["Browser (phone / portal / tablet)"]
    end

    subgraph Cloud["Cloud Services"]
        TWILIO["Twilio Cloud\ninbound call / TTS playback"]
        TAVILY["Tavily API\nweb + image search"]
        WHATSAPP["Family WhatsApp\nresearch results / reminders"]
    end

    subgraph App["FastAPI Application (uvicorn :8000)"]
        PHONE_ROUTES["Phone routes\nPOST /voice/incoming\nPOST /voice/transcription\nPOST /voice/research-choice\nPOST /voice/research-whatsapp-choice\nGET  /voice/answer/{sid}"]
        BROWSER_ROUTES["Browser routes\nGET /  GET /talk  GET /dashboard\nPOST /dashboard/* CRUD\nPOST /transcribe  POST /chat\nGET /games/*  POST /games/*\nGET /cameras  POST /cameras/*\nGET /health"]
    end

    subgraph LocalMachine["Local Machine (RTX 4070 Ti Super, 16 GB VRAM)"]
        WHISPER["faster-whisper large-v3\nCUDA / int8_float16\n~1.5 GB VRAM — loaded at startup"]
        OLLAMA["Ollama (REST :11434)\nQwen 2.5:14b Q4_K_M\n~9-10 GB VRAM — warmed at startup"]
        FAMILYMD["family.md\npipe-delimited markdown\nfilelock protected\n## Todos  ## Events"]
        SCHEDULER["APScheduler\nscans events every 30 min\nsends WhatsApp at 24h and 4h"]
        LOGS["logs/app.log\ndaily rotation, 7 days retention"]
    end

    PHONE -->|PSTN call| TWILIO
    BROWSER -->|HTTPS via cloudflared\nMediaRecorder audio / JSON| BROWSER_ROUTES
    TWILIO -->|webhooks HTTP POST| PHONE_ROUTES
    PHONE_ROUTES --> WHISPER
    PHONE_ROUTES --> OLLAMA
    BROWSER_ROUTES --> WHISPER
    BROWSER_ROUTES --> OLLAMA
    PHONE_ROUTES --> FAMILYMD
    BROWSER_ROUTES --> FAMILYMD
    SCHEDULER --> WHATSAPP
    PHONE_ROUTES --> TAVILY
    BROWSER_ROUTES --> TAVILY
    TAVILY -->|search results| PHONE_ROUTES
    PHONE_ROUTES -->|WhatsApp delivery| WHATSAPP
```

---

## Call Flow Detail

### Flow 1 — Greeting & Speech Capture

```mermaid
flowchart TD
    A["Phone call arrives at Twilio"] --> B["POST /voice/incoming\ncall_handler.handle_incoming()"]
    B --> C{Phone number\nin PHONE_TO_NAME?}
    C -->|No| D["TwiML: 'Not registered. Goodbye.' → hangup"]
    C -->|Yes| E["TwiML: Say 'Hi {name}, this is Bianca...'\nRecord maxLength=15\naction=/voice/transcription"]
    E --> F["Twilio speaks greeting,\nrecords caller,\nPOSTs RecordingUrl"]
```

### Flow 2 — Transcription & Intent Routing (Filler + Async)

The transcription route returns a filler phrase **immediately** and computes the answer in parallel, eliminating perceived silence.

```mermaid
flowchart TD
    A["POST /voice/transcription\nRecordingUrl, RecordingSid, From"] --> B["Create Session in session_store\nkeyed by RecordingSid\nholds asyncio.Event + result slot"]
    B --> C["asyncio.create_task(_compute_answer)\nruns in background"]
    C --> D["Return immediately:\nTwiML: Say filler phrase\nRedirect GET /voice/answer/{sid}"]

    subgraph BG["_compute_answer() — background task"]
        E["httpx download recording"] --> F["asyncio.to_thread\nwhisper_service.transcribe\nconfidence = exp(avg_logprob)"]
        F --> G{confidence < 0.3\nor < 2 words?}
        G -->|Yes| H["result = 'Could you repeat?'"]
        G -->|No| I["asyncio.to_thread\nqwen.classify_intent"]
        I --> J["asyncio.to_thread\nintent_handler.route"]
        J --> K["session.result = TwiML\nsession.event.set()"]
    end

    D --> L["GET /voice/answer/{sid}\nawait session.event (timeout=30s)"]
    L --> M{timeout?}
    M -->|Yes| N["'Sorry, that took too long.' → hangup"]
    M -->|No| O["return TwiML answer\n→ continue conversation"]

    O --> P{intent}
    P -->|confidence < 0.6 or unknown| Q["help message + Record loop"]
    P -->|goodbye| R["farewell + hangup"]
    P -->|add_todo| S["Flow 3a"]
    P -->|complete_todo| T["Flow 3b"]
    P -->|add_event| U["Flow 3c"]
    P -->|query_tasks / query_events| V["Flow 3d"]
    P -->|research / research_images| W["Flow 3e"]
```

**New component:** `services/session_store.py` — thin dict-based store mapping `RecordingSid → Session(event, result)`.

### Flow 3a — Add Todo

```mermaid
flowchart TD
    A["qwen.extract_todo(transcript)\nprompts/todo_extract.txt\nReturns: {text, due (ISO date or null)}"]
    A --> B["markdown_service.append_todo(item)\nSanitize text (strip | and \\n)\nFileLock → write to family.md"]
    B --> C["TwiML: Say 'Done. I've added {text}...'\n+ Record (loop)"]
```

### Flow 3b — Complete Todo

```mermaid
flowchart TD
    A["markdown_service.read_todos()\n→ pending todos list"] --> B["qwen.match_todo(transcript, pending_todos)\nprompts/todo_match.txt (fuzzy semantic match)\nReturns: matched todo text or null"]
    B --> C{match found?}
    C -->|No| D["'I couldn't find that. Be more specific?' + Record"]
    C -->|Yes| E["markdown_service.complete_todo(matched_text)\nFileLock → rewrite - [ ] as - [x]\n+ completed_at timestamp"]
    E --> F["TwiML: Say 'Done. Marked {text} as complete.'\n+ Record (loop)"]
```

### Flow 3c — Add Event

```mermaid
flowchart TD
    A["qwen.extract_event(transcript)\nprompts/event_extract.txt\nResolves relative dates to absolute ISO datetime\nReturns: {title, event_datetime, human_readable}"]
    A --> B["markdown_service.append_event(item)\nFileLock → append to ## Events in family.md"]
    B --> C["TwiML: Say 'Done. Added {title} on {human_readable}.'\n+ Record (loop)"]
```

### Flow 3d — Query Todos / Events

```mermaid
flowchart TD
    A["markdown_service.read_todos() or read_events()\nParses all items from family.md"]
    A --> B["qwen.answer_family_query(transcript, items, item_type)\nprompts/family_query.txt\nKnows today's date and time\nAnswers specific question (not a list dump)"]
    B --> C["TwiML: Say {answer} + Record (loop)"]
```

### Flow 3e — Research

```mermaid
flowchart TD
    A{intent.query present\nand ≥ 2 words?}
    A -->|No| B["'Could you give me more detail?' + Record"]
    A -->|Yes| C{research_images intent?}
    C -->|Yes| D["asyncio.create_task(_deliver_images)\nTavily → WhatsApp\nTwiML: 'I'll send those images to WhatsApp.' + Record"]
    C -->|No| E["qwen.quick_answer(query)\nCan Qwen answer from knowledge alone?"]
    E -->|Yes| F["TwiML: Say answer + Record (loop)"]
    E -->|No| G["asyncio.create_task(_do_research(query))\nTavily search → asyncio.gather(\n  qwen.voice_summarize_research,\n  qwen.synthesize_research\n)"]
    G --> H["await asyncio.wait_for(shield(task), timeout=10s)"]
    H -->|Done in time| I["store whatsapp_text in _pending_whatsapp[wid]\nTwiML: Say voice_text (2-3 sentences)\n'Want full details on WhatsApp?'\nRecord action=/voice/research-whatsapp-choice/{wid}"]
    H -->|Timeout| J["store task in _pending[rid]\nTwiML: 'Still researching... say wait or WhatsApp'\nRecord action=/voice/research-choice/{rid}"]

    J --> K["POST /voice/research-choice/{rid}\nDownload audio → Whisper → keyword match"]
    K -->|WhatsApp| L["asyncio.create_task(_deliver_whatsapp_when_done)\nTwiML: 'I'll send to WhatsApp shortly.' + Record"]
    K -->|wait| M["await shield(task), timeout=15s"]
    M -->|Done| N["store whatsapp_text in _pending_whatsapp[wid]\nTwiML: Say voice_text\n'Want full details on WhatsApp?'\nRecord action=/voice/research-whatsapp-choice/{wid}"]
    M -->|Timeout| O["asyncio.create_task(_deliver_whatsapp_when_done)\nTwiML: 'Taking longer, sending to WhatsApp.' + Record"]

    I --> P["POST /voice/research-whatsapp-choice/{wid}\nDownload audio → Whisper → keyword match"]
    N --> P
    P -->|yes/sure/send| Q["twilio_service.send_whatsapp(full detail)\nTwiML: 'Sent! Anything else?' + Record"]
    P -->|no| R["TwiML: 'No problem. Anything else?' + Record"]
```

---

## Proactive Reminders

A background scheduler (`APScheduler AsyncIOScheduler`) starts at app startup and scans `family.md` every 30 minutes for upcoming events.

```mermaid
flowchart TD
    A["App startup (lifespan)"] --> B["reminder_service.start()\nSchedules _scan_and_schedule()\nto run immediately + every 30 min"]
    B --> C["_scan_and_schedule()\nmarkdown_service.read_events\n(after=now, before=now+48h)"]
    C --> D["For each event in window:\nFor each offset in 24h, 4h:\n  remind_at = event_datetime - offset"]
    D --> E{remind_at <= now?}
    E -->|Yes| F["skip (already past)"]
    E -->|No| G{job_id already\nin scheduler?}
    G -->|Yes| H["skip (deduplication)"]
    G -->|No| I["scheduler.add_job\nDateTrigger(run_date=remind_at)\n_send_reminder"]
    I --> J["_send_reminder(title, label, event_dt)\nFormats message\nSends WhatsApp to every number in PHONE_TO_NAME"]
```

**Deduplication:** APScheduler job IDs are deterministic (`reminder_{iso_datetime}_{minutes}`). The 30-minute scan simply skips any job ID that already exists. One-off jobs are removed by APScheduler after they fire, so there is no risk of double-sending within a single server run.

**Restart behaviour:** The in-memory job store is cleared on restart. The scanner runs immediately on startup and reschedules any reminders whose `remind_at` is still in the future. Reminders already sent (whose `remind_at` is in the past) are naturally skipped by the `remind_at <= now` guard.

---

## Web Dashboard Flow

```mermaid
flowchart TD
    A["Browser → GET /dashboard"] --> B["markdown_service.read_all_data()\n→ todos + events\nAnnotate events with is_past\nSort (pending/upcoming first)\nResolve family_names from PHONE_TO_NAME"]
    B --> C["Jinja2 renders dashboard.html\nBootstrap 5 two-column layout\nauto-refreshes every 30s"]
    C --> D["Dashboard is fully editable\nAll mutations are JSON POSTs\nReload on success"]
    D --> E["POST /dashboard/add-todo\n{text, due?, added_by}\n→ append_todo()"]
    D --> F["POST /dashboard/complete-todo\n{text}\n→ complete_todo()"]
    D --> G["POST /dashboard/delete-todo\n{text}\n→ delete_todo()"]
    D --> H["POST /dashboard/add-event\n{title, event_datetime, added_by}\n→ append_event()"]
    D --> I["POST /dashboard/delete-event\n{title, event_datetime}\n→ delete_event()"]
    D --> J["Edit (todo or event) — client-side:\ndelete-old → add-new\ntwo sequential API calls, single page reload"]
```

---

## Browser Interface Flow

Family members open the browser interface on any device on the local network. The Talk and Hangman pages require HTTPS (use the cloudflared URL) because mic access and WASM require a secure context.

```mermaid
flowchart TD
    A["Browser (Portal / phone / tablet)"] --> B["GET /\nhome.html\n4 nav cards: Games, Dashboard, Talk, Cameras"]

    B --> C["GET /games\ngames.html\ngames hub: Hangman, Multiply, Clock, Quiz"]

    B --> CAM["GET /cameras\ncameras.html\nproxied from app → deepstream:8090"]
    CAM --> CAM1["POST /cameras/streams {cam_id, uri}\nRegisters stream in deepstream_service\nStarts bianca-rtsp-{cam_id} container via Docker SDK\nStarts ffmpeg clip recorder from Savant RTSP :554"]
    CAM --> CAM3["Browser video via LL-HLS (hls.js)\nSavant Always-On Sink :888\n/stream/{cam_id}/index.m3u8\nWorks via cloudflared tunnel through FastAPI"]

    CAM --> CAM4["Savant pipeline (bianca-savant-module)\nRTSP adapter → ZMQ dealer → pyfunc\nget_nvds_buf_surface → RGBA numpy\nROI crop (polled from deepstream /streams)\ncv2.imencode JPEG → POST /internal/frame\nRate-limited to INFER_FPS (10fps default)"]

    CAM --> CAM6["GET /cameras/queries\nPOST /cameras/queries {text}\nDELETE /cameras/queries/{i}\nGET|POST /cameras/threshold\nGET /cameras/clips\nGET /cameras/clips/file/{cam_id}/{filename}"]

    CAM4 --> CAM7["deepstream_service.py _inference_loop\nReads JPEG from _infer_slots (leaky)\nSoftware motion gate: cv2.absdiff 320×180 grey\n(skips Triton if no motion — ~80-90% reduction)\nPOST triton:8002/v2/models/yoloworld/infer\nYOLO-World M TRT — 8ms median\nReturns {boxes, scores, labels}\n_SimpleTracker IoU dedup\nClip trigger → ffmpeg concat from rolling .ts segments"]

    B --> TALK["GET /talk\ntalk.html"]
    TALK --> T1["Page loads → Silero VAD initialises\nONNX model from CDN, cached, runs via WASM"]
    T1 --> T2["VAD listens continuously — no tap required\nUser speaks → onSpeechEnd(Float32Array @ 16kHz)\nVAD paused while processing"]
    T2 --> T3["Float32Array encoded to WAV in browser\nPOST /transcribe (audio/wav blob)\nasyncio.to_thread(whisper_service.transcribe)\n→ {transcript, confidence}"]
    T3 --> T4["POST /chat {transcript, caller_name}\nchat_handler.handle_chat()\nclassify_intent → route to handler\nresearch → Tavily + parallel Qwen summaries\nreturns {speech, display, intent}"]
    T4 --> T5["speechSynthesis.speak(speech)\nVAD resumes after TTS finishes (onend callback)\nDisplay text in Full Answer panel (research only)"]

    B --> DASH["GET /dashboard\ndashboard.html\nEditable todos and events\nAll mutations via JSON POST, reload on success\nAuto-refreshes every 30s"]

    B --> HM["GET /games/hangman\nhangman.html\nSame VAD auto-detection as /talk"]
    HM --> HM1["POST /games/hangman/new\nhangman_service.new_game()\nPre-reveals hint letters based on word length:\n≤4→0 hints, 5→1, 6→2, 7+→3 (random distinct)"]
    HM --> HM2["POST /games/hangman/guess {session_id, guess}\nhangman_service.guess()\nAccepts: 'letter A', 'word elephant', bare letter\n→ {display_word, wrong_letters, figure, speech, won, lost}"]

    B --> MUL["GET /games/multiply\nmultiply.html\nTimes Tables practice — all logic client-side\nRandom A×B (A,B ∈ 1-9), spoken via speechSynthesis\nVAD → /transcribe → parseSpokenNumber()\nTracks correct / answered score"]

    B --> CLK["GET /games/clock\nclock.html\nTell the Time — 4-option multiple choice\nAll logic client-side\nRandom time, 4 SVG analogue clock faces (pure JS)\nVAD → /transcribe → parseOption()\nTracks correct / answered score"]

    B --> QUIZ["GET /games/quiz\nquiz.html\nKnowledge Quiz — subject + grade selection\nPOST /games/quiz/generate {subject, grade}\nasyncio.to_thread(qwen.generate_quiz)\nprompts/quiz_generate.txt\nReturns JSON array of up to 10 questions\nVAD or tap for answers"]

    B --> BC["GET /games/bulls-cows\nbulls_cows.html\nBulls and Cows — 4-digit code-breaking (no LLM)\nPOST /games/bulls-cows/new\nPOST /games/bulls-cows/guess {session_id, guess}\nbulls_cows_service.parse_spoken_number()\n_score → (bulls, cows); max 10 attempts"]

    B --> WL["GET /games/word-ladder\nword_ladder.html\nWord Ladder — change one letter at a time\nPOST /games/word-ladder/new\n  qwen.generate_word_ladder + BFS validation\nPOST /games/word-ladder/step {session_id, word}\nPOST /games/word-ladder/hint {session_id}\nVertical chain visualiser; VAD + tap"]

    B --> TQ["GET /games/twenty-questions\ntwenty_questions.html\n20 Questions — Bianca asks yes/no; kid thinks of something\nPOST /games/twenty-questions/new\nPOST /games/twenty-questions/start {session_id}\n  qwen.ask_twenty_questions — multi-turn /api/chat\nPOST /games/twenty-questions/answer {session_id, answer}\nPOST /games/twenty-questions/confirm {session_id, answer}\nPhase flow: thinking → loading → playing → guessing → finished"]
```

---

## Component Map

```
family-assistant/
│
├── main.py                    FastAPI app, all routes, startup warmup, log setup
├── config.py                  Pydantic settings, .env loading, phone→name map
├── family.md                  Shared storage: todos + events (pipe-delimited markdown)
├── logs/                      Daily rotating logs (app.log, 7 days retention)
│
├── handlers/
│   ├── call_handler.py        Twilio webhooks → download audio → Whisper → classify
│   ├── chat_handler.py        Browser /chat → classify → route → return {speech, display}
│   ├── intent_handler.py      Routes IntentResult to correct sub-handler (phone path)
│   ├── todo_handler.py        add_todo, query_todos, complete_todo
│   ├── event_handler.py       add_event, query_events
│   ├── research_handler.py    quick_answer → Tavily → parallel voice+WhatsApp summaries
│   └── response_handler.py    TwiML builders: voice_gather, voice_say_then_gather, etc.
│
├── pipeline/
│   ├── module.yml             Savant pipeline YAML — pyfunc-only; declares BiancaFrameForwarder
│   └── pyfunc.py              NvDsPyFuncPlugin — NVMM frame extraction, ROI crop, JPEG encode, HTTP POST to /internal/frame
│
├── services/
│   ├── deepstream_service.py  FastAPI :8090; receives JPEG frames via POST /internal/frame; software motion gate; Triton YOLO-World inference; Docker SDK RTSP adapter management; ffmpeg rolling clip segments; clip extraction; camera persistence via cameras.json
│   ├── whisper_service.py     faster-whisper large-v3 CUDA, suffix param for webm/wav
│   ├── qwen.py                Ollama REST wrapper, all LLM calls, JSON extraction
│   ├── markdown_service.py    Read/write/parse/delete family.md with FileLock
│   ├── session_store.py       In-memory sessions (RecordingSid → asyncio.Event + result)
│   ├── tavily_service.py      Tavily web + image search with retry
│   ├── twilio_service.py      WhatsApp message + image sender
│   ├── reminder_service.py    APScheduler — 24h/4h WhatsApp reminders for events
│   ├── hangman_service.py     Hangman game state, word list, guess logic
│   ├── bulls_cows_service.py  Bulls and Cows game state, secret generation, spoken-digit parser
│   ├── word_ladder_service.py BFS-validated word ladder puzzles; /usr/share/dict/words word set
│   └── twenty_questions_service.py  20Q multi-turn Qwen session, phase management, answer normalisation
│
├── cameras.json               Persisted camera registry {cam_id: uri} — loaded by deepstream_service at startup
├── clips/                     Per-camera H.264 MP4 clips (gitignored); served via GET /cameras/clips/file/{cam_id}/{fn}
│
├── models/
│   └── schemas.py             Pydantic models: IntentResult, TodoItem, EventItem
│
├── prompts/
│   ├── intent_classify.txt    Few-shot intent classifier (9 intents)
│   ├── todo_extract.txt       Extract todo text + due date from transcript
│   ├── todo_match.txt         Fuzzy-match transcript to existing todo
│   ├── event_extract.txt      Extract event title + datetime from transcript
│   ├── family_query.txt       Answer natural language questions about todos/events
│   ├── quick_answer.txt       Decide if Qwen can answer from knowledge vs web search
│   ├── research_voice.txt     2-3 sentence spoken summary of search results
│   ├── research_synthesize.txt  Full Markdown summary for WhatsApp / browser display
│   └── quiz_generate.txt      Generate 10 kid-safe MCQ questions for subject + grade
│
└── templates/
    ├── home.html              Landing page — 4 nav cards: Games, Dashboard, Talk, Cameras
    ├── games.html             Games hub — links to all 7 games
    ├── cameras.html           RTSP stream viewer — URL form + live MJPEG feed + AI event placeholder
    ├── talk.html              Browser voice interface — Silero VAD + Whisper STT, no tap needed
    ├── hangman.html           Voice hangman — VAD, hint letters pre-revealed at start
    ├── multiply.html          Times Tables game — VAD, spoken number parsing, score tracking
    ├── clock.html             Tell the Time game — SVG clocks, 4-option MCQ, VAD
    ├── quiz.html              Knowledge Quiz — subject/grade setup, Qwen questions, VAD
    ├── bulls_cows.html        Bulls and Cows — history table, spoken digit VAD
    ├── word_ladder.html       Word Ladder — vertical chain visualiser, BFS hints, VAD
    ├── twenty_questions.html  20 Questions — 4-phase UI, multi-turn Qwen yes/no VAD
    └── dashboard.html         Editable family dashboard — Bootstrap 5, vanilla JS
```

---

## Data Schema (family.md)

```markdown
# Family Assistant

## Todos

- [ ] Buy groceries | due: 2026-04-03 | added_by: Alice | added_at: 2026-03-29T10:15:00
- [x] Renew passport | due: none | added_by: Bob | added_at: 2026-03-20T09:00:00 | completed_at: 2026-03-25T12:00:00

## Events

- 2026-04-10T14:00:00 | Dentist appointment | added_by: Alice | added_at: 2026-03-29T10:20:00
- 2026-05-01T00:00:00 | Family vacation starts | added_by: Bob | added_at: 2026-03-01T08:00:00
```

**Rules:**
- Todos: `- [ ]` pending, `- [x]` complete. Fields pipe-delimited. Text must not contain `|` (sanitized on write).
- Events: ISO datetime first field, then title + metadata. Sorted by insertion order.
- File is protected by `filelock` — one writer at a time, 5-second timeout.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `<Record>` + Whisper instead of `<Gather input="speech">` | Free, more accurate STT especially for names and natural speech |
| Two Qwen calls per request (classify → extract) | More reliable than one combined call; smaller focused prompts |
| `quick_answer` check before Tavily | Avoids paid API call for simple knowledge questions |
| FastAPI `BackgroundTasks` for research | No Redis/Celery needed; delivers WhatsApp while call is already ended |
| Markdown file instead of database | Human-readable, manually editable, zero infrastructure |
| `filelock` for writes | Prevents corruption from rare concurrent calls; family-scale volume is fine |
| Whisper loaded at startup | Eliminates cold-start delay on first call |
| Qwen warmed up at startup via dummy `_chat("hi")` | Ollama lazy-loads model; warmup ensures GPU is ready |
| Few-shot examples in intent classifier | Significantly reduces misclassifications vs zero-shot |
| DeepStream → Triton via HTTP not gRPC | gRPC TYPE_STRING/pybind11 crash in tritonserver ≤24.09; UINT8+FP32 inputs mean gRPC could work on 25.03, but HTTP is simpler to debug (curl, browser) and required anyway for the model management API (`/v2/repository/models/yoloworld/load`); at 10fps with ~8ms GPU inference the ~0.9ms HTTP overhead is negligible; gRPC worth revisiting only at 10+ cameras |
| TRT inference always; query changes trigger full re-export via management sidecar | TRT bakes CLIP text embeddings at export time — changing queries without re-exporting fires the engine on old class slots but maps to new labels (silent wrong results). PyTorch fallback is not used after initial engine creation; instead the management sidecar (`triton:8004`) re-exports a new TRT engine (~90s) in the background. The old engine keeps running until the new one is ready. |
| App-level cross-service orchestration for query changes | When the user changes detection queries, `main.py` owns the full lifecycle: (1) pause Qwen (`keep_alive=0` so VRAM is freed), (2) trigger TRT re-export via `POST triton:8004/reexport`, (3) poll status every 3s up to 180s, (4) reload Triton model via `triton:8002` management API, (5) commit updated queries to deepstream via `POST deepstream:8090/queries/commit`, (6) resume Qwen. During re-export all LLM routes return HTTP 503. `deepstream_service` has no knowledge of Ollama. |
| GStreamer subprocess isolation for RTSP | asyncio's epoll races with rtspsrc's GLib socket watcher in the same process → SIGABRT. Spawning `pipeline_worker.py` as a subprocess removes the interference entirely. All GLib/GStreamer code lives only in the subprocess. |
| `import cv2` must come after `Gst.init()` | cv2 (CUDA build) initialises NVIDIA codec libs on import, conflicting with DeepStream's CUDA init inside Gst.init(). Importing cv2 before Gst.init() → SIGABRT. Solution: remove cv2 from pipeline_worker.py entirely — use GStreamer's `nvjpegenc` for JPEG encoding instead. |
| Per-camera chains instead of nvstreammux+tiler | nvstreammux batches streams for nvinfer — unnecessary for display-only pipelines. Per-camera chains (each with its own nvv4l2decoder→nvvideoconvert→nvjpegenc→appsink) are simpler, remove the nvmultistreamtiler, and scale independently. nvstreammux is only needed when running nvinfer on the pipeline output. |
| nvjpegenc over jpegenc | nvjpegenc accepts `video/x-raw(memory:NVMM)` — the frame stays in GPU memory through encode. jpegenc (CPU) requires a GPU→CPU copy first. On an RTX 4070 Ti Super the difference is measurable in frame latency at 25fps. |
| Software motion gate before Triton | Frame differencing on a 320×180 grayscale thumbnail (~0.5ms CPU) skips Triton entirely when the scene is static. Reduces inference calls by ~80–90% in idle scenes. `MOTION_GATE` env var disables it if needed. Camera-side hardware detection (Thingino webhook) documented in `improvement_ideas.md` as a future zero-CPU alternative. |
| `imageio-ffmpeg` for clip encoding | System ffmpeg in the DeepStream 7.0 image is broken (libav `.so` files removed post-install despite dpkg listing them). `imageio-ffmpeg` pip package ships a bundled static ffmpeg binary with libx264; accessed via `imageio_ffmpeg.get_ffmpeg_exe()`. |
| Direct `open(..., "w")` for cameras.json | `os.replace(tmp, dest)` (atomic rename) fails with `EBUSY` when `dest` is a Docker bind-mounted single file — Linux `rename(2)` can't replace a bind-mount inode. Direct write truncates in place, which is safe for a small JSON config file. |
| `FileResponse` for clip serving (not httpx proxy) | FastAPI's `FileResponse` handles `Accept-Ranges` / `206 Partial Content` natively, which browsers require for video seeking. Forwarding through httpx would need manual Range header plumbing. The `clips/` directory is bind-mounted into both the deepstream and app containers so app can serve files directly. |
| go2rtc as RTSP normalizing proxy | The Thingino camera generates DTS discontinuities in its audio tracks. The Savant RTSP adapter's ffmpeg_src plugin stalls when audio DTS jumps, causing a pipeline restart every ~30s. go2rtc absorbs the raw camera RTSP connection and re-streams a clean copy. The adapter connects to `rtsp://bianca-go2rtc:8554/{cam_id}?video=h264` — the `?video=h264` consumer filter tells go2rtc to forward only the H.264 video track, dropping all audio tracks and their DTS issues entirely. |
| go2rtc stream registration via `?name=X&src=Y` query params | go2rtc 1.9.14's HTTP API has two registration formats. The JSON body format (`PUT /api/streams?name=X` with `{"url": "..."}`) returns 200 but creates no in-memory stream entry. The query-param format (`PUT /api/streams?name=X&src=Y`) registers correctly. The yaml write-back requires `streams:` (bare key, no `{}`) in go2rtc.yaml to avoid a parse error on write. |
| LL-HLS query params forwarded end-to-end | hls.js in `lowLatencyMode` sends blocking playlist requests with `_HLS_msn` and `_HLS_part` query params; MediaMTX holds the HTTP connection open until the requested segment/part is ready. Both the `app` proxy (`/cameras/hls/`) and the `deepstream` proxy (`/hls/`) must forward `request.query_params` to the upstream. Without forwarding, hls.js gets immediate responses without the expected content and stalls. |

---

## Surveillance Frame Flow (Phase 1 step B, 2026-04-29)

```
Camera ──RTSP──► bianca-mediamtx                 (single RTSP/HLS hub)
                  │ paths added/removed at runtime via HTTP API :9997
                  │ rtspTransport: tcp; sourceOnDemand: false
                  │ Re-streams on rtsp://bianca-mediamtx:8554/{cam_id}
                  ├──► LL-HLS  :8888 /{cam_id}/index.m3u8
                  │     └──► browser (hls.js, via app:8000/cameras/hls/* proxy → mediamtx:8888 directly)
                  └──► RTSP    :8554 /{cam_id}
                        ├──► bianca-inference (per-camera worker subprocess)
                        │     PyAV h264_cuvid (NVDEC) → motion gate → cv2.imencode JPEG
                        │     → tritonclient.http call to bianca-triton:8002
                        │     → _SimpleTracker IoU update → for each detection above threshold:
                        │       POST bianca-deepstream:8090/internal/trigger  (every frame, extends clip window)
                        │     → on RECHECK_INTERVAL_S cooldown:
                        │       POST bianca-deepstream:8090/internal/event   (rate-limited; carries img_b64 crop)
                        │
                        └──► bianca-inference (sibling ffmpeg process per camera)
                             ffmpeg -i rtsp://bianca-mediamtx:8554/{cam_id}
                               -f segment -segment_time 30 -segment_atclocktime 1
                               clips/{cam_id}/segs/seg_%05d.ts

         bianca-deepstream:8090          (Stream control plane + events + clip cutter)
           POST /streams              → _mediamtx_add_path + _inference_add_camera
           DELETE /streams/{cam_id}   → _inference_remove_camera + _mediamtx_remove_path
           POST /internal/trigger     → _trigger_clip_on_detection (extends window)
           POST /internal/event       → _events.append + _trigger_clip_on_detection
           _segment_watcher_loop polls each cam's segs/ every 5s → _seg_ring
           _clip_manager: on trigger expiry → _process_trigger
              waits ≤SEG_DURATION_S+5s for finalized seg covering clip_end_wall
              (checks both _seg_ring and disk fallback)
              imageio_ffmpeg static binary stream-copies concat → clips/{cam_id}/{ts}_{query}.mp4
              Disk-only ops here, so the deepstream image's broken libavcodec.so.58
              doesn't matter — RTSP reads happen in the inference container.
           GET /clips, GET /clips/file/{cam_id}/{filename}  (range-served)
```

Why two ffmpegs (and why not one in deepstream): the deepstream image (DeepStream 7.0 base) has a broken apt ffmpeg — `dpkg -L libavcodec58` lists `.so.58.134.100` but the file is missing on disk. The inference image is built on `savant-deepstream` which has a working system ffmpeg with cuvid + libx264. So the rolling-segment recorder lives next to the inference worker. Clip CUTTING (local file → local file) uses the imageio_ffmpeg static binary in deepstream — that binary's known segfault was specific to RTSP reads, which we no longer do there.

---

## Testing the Camera Pipeline

### Verify go2rtc stream registration

```bash
# Check registered streams and their status
curl -s http://localhost:1984/api/streams | python3 -m json.tool

# Expected: entry for each camera with "producers" (camera connection) and "consumers" (RTSP adapter)
```

### Check pipeline health after adding a camera

```bash
# 1. Verify go2rtc registered the stream
curl -s http://localhost:1984/api/streams | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.keys()))"

# 2. Verify RTSP adapter and sink containers are running
docker ps --filter "name=bianca-rtsp" --filter "name=bianca-sink" --format "{{.Names}}: {{.Status}}"

# 3. Verify HLS is serving live segments
curl -s http://localhost:8000/cameras/hls/{cam_id}/index.m3u8

# 4. Verify frames are reaching deepstream (check for POST /internal/frame)
docker logs bianca-deepstream --tail=20 | grep internal/frame
```

### Add/remove a stream

```bash
# Add
curl -s -X POST http://localhost:8090/streams \
  -H "Content-Type: application/json" \
  -d '{"cam_id": "cam0", "uri": "rtsp://camera-host/stream"}'

# Remove
curl -s -X DELETE http://localhost:8090/streams/cam0
```
