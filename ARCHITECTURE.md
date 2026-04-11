# Bianca — Family Assistant: Architecture

## Overview

Bianca is a family productivity assistant with two interfaces: phone calls (via Twilio) and a browser interface (any device on the local network). She handles todos, events, research, and games. Research results are delivered via WhatsApp. Proactive event reminders are sent automatically. All AI runs locally on GPU — no cloud AI services.

---

## System Architecture

```
                    ┌─────────────────┐        ┌──────────────────────┐
                    │  CALLER'S PHONE │        │  BROWSER (phone /    │
                    │  (PSTN call)    │        │  Portal / tablet)    │
                    └────────┬────────┘        └──────────┬───────────┘
                             │                            │ HTTPS (cloudflared)
                             ▼                            │ MediaRecorder audio
                    ┌────────────────┐                    │ JSON responses
                    │  TWILIO CLOUD  │                    │
                    │  inbound call  │                    │
                    │  TTS playback  │                    │
                    └────────┬───────┘                    │
                             │ webhooks (HTTP POST)       │
                             ▼                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION  (uvicorn :8000)               │
│                                                                        │
│  Phone routes:                    Browser routes:                      │
│  POST /voice/incoming             GET  /                               │
│  POST /voice/transcription        GET  /talk                           │
│  POST /voice/research-choice      GET  /dashboard                      │
│  POST /voice/research-whatsapp-   POST /dashboard/add-todo             │
│       choice                      POST /dashboard/complete-todo        │
│  GET  /voice/answer/{sid}         POST /dashboard/delete-todo          │
│                                   POST /dashboard/add-event            │
│                                   POST /dashboard/delete-event         │
│                                   POST /transcribe  (Whisper STT)      │
│                                   POST /chat        (text→JSON)        │
│                                   GET  /games        (games hub)       │
│                                   GET  /games/hangman                  │
│                                   POST /games/hangman/new              │
│                                   POST /games/hangman/guess            │
│                                   GET  /games/multiply                 │
│                                   GET  /games/clock                    │
│                                   GET  /games/quiz                     │
│                                   POST /games/quiz/generate            │
│                                   GET  /games/bulls-cows               │
│                                   POST /games/bulls-cows/new           │
│                                   POST /games/bulls-cows/guess         │
│                                   GET  /games/word-ladder              │
│                                   POST /games/word-ladder/new          │
│                                   POST /games/word-ladder/step         │
│                                   POST /games/word-ladder/hint         │
│                                   GET  /games/twenty-questions         │
│                                   POST /games/twenty-questions/new     │
│                                   POST /games/twenty-questions/start   │
│                                   POST /games/twenty-questions/answer  │
│                                   POST /games/twenty-questions/confirm │
│                                   GET  /cameras                        │
│                                   POST /cameras/set-stream             │
│                                   GET  /cameras/stream  (MJPEG)        │
│                                   GET  /health                         │
└───────────────┬────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                  LOCAL MACHINE  (RTX 4070 Ti Super, 16GB VRAM)        │
│                                                                       │
│  ┌──────────────────────┐    ┌─────────────────────────────────────┐  │
│  │  faster-whisper      │    │  Ollama  (REST on :11434)           │  │
│  │  large-v3 / CUDA     │    │  Qwen 2.5:14b  Q4_K_M              │  │
│  │  int8_float16        │    │  ~9-10GB VRAM                       │  │
│  │  ~1.5GB VRAM         │    │  Warmed up at startup               │  │
│  │  Loaded at startup   │    └─────────────────────────────────────┘  │
│  └──────────────────────┘                                             │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  family.md  (pipe-delimited markdown, filelock protected)        │ │
│  │  ## Todos    — [ ] / [x] items with due date, added_by, etc     │ │
│  │  ## Events   — ISO datetime | title | added_by                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  APScheduler — scans events every 30min                         │ │
│  │  Sends WhatsApp reminders at 24h and 4h before each event       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  logs/app.log  (daily rotation, 7 days retention)               │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐       ┌──────────────────────────────────┐
│  TAVILY API (cloud)       │       │  FAMILY WHATSAPP                 │
│  Web + image search       │       │  Research results, reminders,    │
│  (research intents only)  │       │  WhatsApp-choice deliveries      │
└───────────────────────────┘       └──────────────────────────────────┘
```
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE  (RTX 4070 Ti Super, 16GB VRAM)   │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  faster-whisper     │    │  Ollama                              │ │
│  │  large-v3           │    │  (separate process, REST on :11434)  │ │
│  │  CUDA               │    │                                      │ │
│  │  int8_float16       │    │  ┌────────────────────────────────┐  │ │
│  │  ~1.5GB VRAM        │    │  │  Qwen 2.5:14b  Q4_K_M         │  │ │
│  │                     │    │  │  ~9-10GB VRAM                  │  │ │
│  │  Loaded at startup  │    │  │  Warmed up at startup          │  │ │
│  │  ~1-2s per call     │    │  └────────────────────────────────┘  │ │
│  └─────────────────────┘    └──────────────────────────────────────┘ │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  family.md  (pipe-delimited markdown, filelock protected)     │   │
│  │  ## Todos    — [ ] / [x] items with due date, added_by, etc  │   │
│  │  ## Events   — ISO datetime | title | added_by               │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│     TAVILY API (cloud)       │
│  Web search, image search    │
│  (used only for research     │
│   intents)                   │
└──────────────────────────────┘
```

---

## Call Flow Detail

### Flow 1 — Greeting & Speech Capture

```
Phone call arrives at Twilio
        │
        ▼
POST /voice/incoming
  call_handler.handle_incoming()
        │
        ├─ Phone number in PHONE_TO_NAME? ──No──▶ TwiML: "Not registered. Goodbye." → hangup
        │
        └─ Yes ──▶ TwiML: <Say> "Hi {name}, this is Bianca..."
                          <Record maxLength=15 action=/voice/transcription>
                   Twilio speaks greeting, records caller, POSTs RecordingUrl
```

### Flow 2 — Transcription & Intent Routing (Filler + Async)

The transcription route returns a filler phrase **immediately** and computes the answer in parallel, eliminating perceived silence.

```
POST /voice/transcription  (RecordingUrl, RecordingSid, From)
  call_handler.handle_transcription()
        │
        ├─ Create Session (session_store) keyed by RecordingSid
        │   Session holds: asyncio.Event + result slot
        │
        ├─ asyncio.create_task(_compute_answer(...))  ← runs in background
        │
        └─ Return immediately:
           TwiML: <Say> "Sure, let me check that..." </Say>
                  <Redirect GET /voice/answer/{sid}>

                          ┌─────────────────────────────────┐
                          │  _compute_answer() (background) │
                          │                                 │
                          │  httpx download recording       │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    whisper_service.transcribe)  │
                          │  confidence = exp(avg_logprob)  │
                          │         ↓                       │
                          │  confidence < 0.3 or < 2 words? │
                          │    → "Could you repeat?" result │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    qwen.classify_intent)        │
                          │         ↓                       │
                          │  asyncio.to_thread(             │
                          │    intent_handler.route)        │
                          │         ↓                       │
                          │  session.result = TwiML         │
                          │  session.event.set()            │
                          └─────────────────────────────────┘

GET /voice/answer/{sid}                 ← Twilio calls this after filler plays
  await session.event (timeout=30s)
        │
        ├─ timeout ──▶ "Sorry, that took too long." → hangup
        └─ result ready ──▶ return TwiML answer → continue conversation

  intent_handler.route() returns:
        ├─ confidence < 0.6 or unknown ──▶ help message + <Record> (loop)
        ├─ goodbye ──────────────────────▶ farewell + hangup
        ├─ add_todo ─────────────────────▶ [Flow 3a]
        ├─ complete_todo ────────────────▶ [Flow 3b]
        ├─ add_event ────────────────────▶ [Flow 3c]
        ├─ query_tasks ──────────────────▶ [Flow 3d]
        ├─ query_events ─────────────────▶ [Flow 3d]
        └─ research / research_images ───▶ [Flow 3e]
```

**New component:** `services/session_store.py` — thin dict-based store mapping `RecordingSid → Session(event, result)`.

### Flow 3a — Add Todo

```
qwen.extract_todo(transcript)
  Qwen — prompts/todo_extract.txt
  Returns: { text, due (ISO date or null) }
        │
        ▼
markdown_service.append_todo(item)
  Sanitize text (strip | and \n)
  FileLock → write to family.md
        │
        ▼
TwiML: <Say> "Done. I've added '{text}'..." + <Record> (loop)
```

### Flow 3b — Complete Todo

```
markdown_service.read_todos()  →  pending todos list
        │
        ▼
qwen.match_todo(transcript, pending_todos)
  Qwen — prompts/todo_match.txt  (fuzzy semantic match)
  Returns: matched todo text or null
        │
        ├─ no match ──▶ "I couldn't find that. Be more specific?" + <Record>
        │
        ▼
markdown_service.complete_todo(matched_text)
  FileLock → rewrite - [ ] as - [x] + completed_at timestamp
        │
        ▼
TwiML: <Say> "Done. I've marked '{text}' as complete." + <Record> (loop)
```

### Flow 3c — Add Event

```
qwen.extract_event(transcript)
  Qwen — prompts/event_extract.txt
  Resolves relative dates ("next Friday") to absolute ISO datetime
  Returns: { title, event_datetime, human_readable }
        │
        ▼
markdown_service.append_event(item)
  FileLock → append to ## Events in family.md
        │
        ▼
TwiML: <Say> "Done. Added '{title}' on {human_readable}." + <Record> (loop)
```

### Flow 3d — Query Todos / Events

```
markdown_service.read_todos() or read_events()
  Parses all items from family.md (past + future for events)
        │
        ▼
qwen.answer_family_query(transcript, items, item_type)
  Qwen — prompts/family_query.txt
  Knows today's date and time
  Answers the specific question (not just a list dump)
  e.g. "when is the birthday?" → finds event, says "14 days from now"
        │
        ▼
TwiML: <Say> {answer} + <Record> (loop)
```

### Flow 3e — Research

```
intent.query present and ≥ 2 words?
        │
        ├─ No ──▶ "Could you give me more detail?" + <Record>
        │
        ▼
research_images intent?
        ├─ Yes ──▶ asyncio.create_task(_deliver_images) → Tavily → WhatsApp
        │           TwiML: "I'll send those images to your WhatsApp." + <Record>
        │
        ▼  (research intent)
qwen.quick_answer(query)   [asyncio.to_thread]
  Can Qwen answer from knowledge alone?
        │
        ├─ Yes ──▶ TwiML: <Say> answer + <Record> (loop)
        │
        └─ No ──▶ asyncio.create_task(_do_research(query))
                       _do_research returns (voice_text, whatsapp_text):
                         Tavily search → asyncio.gather(
                           qwen.voice_summarize_research,  ← 2-3 spoken sentences
                           qwen.synthesize_research,       ← full WhatsApp detail
                         )

                  await asyncio.wait_for(shield(task), timeout=10s)
                        │
                        ├─ Done in time ──▶ store whatsapp_text in _pending_whatsapp[wid]
                        │                   TwiML: <Say> voice_text (2-3 sentences)
                        │                          "Want the full details on WhatsApp?"
                        │                          <Record action=/voice/research-whatsapp-choice/{wid}>
                        │
                        └─ Timeout ──▶ store task in _pending[rid]
                                       TwiML: "Still researching... say 'wait' or 'WhatsApp'"
                                              <Record action=/voice/research-choice/{rid}>

POST /voice/research-choice/{rid}
  Download audio → Whisper → keyword match ("wait" / "WhatsApp")
        │
        ├─ "WhatsApp" ──▶ asyncio.create_task(_deliver_whatsapp_when_done)
        │                  TwiML: "I'll send to your WhatsApp shortly." + <Record>
        │
        └─ "wait" ──▶ await shield(task), timeout=15s
                            │
                            ├─ Done ──▶ store whatsapp_text in _pending_whatsapp[wid]
                            │           TwiML: <Say> voice_text + "Want full details on WhatsApp?"
                            │                  <Record action=/voice/research-whatsapp-choice/{wid}>
                            └─ Timeout ──▶ asyncio.create_task(_deliver_whatsapp_when_done)
                                           TwiML: "Taking longer, sending to WhatsApp." + <Record>

POST /voice/research-whatsapp-choice/{wid}
  Download audio → Whisper → keyword match ("yes" / "no")
        │
        ├─ "yes/sure/send" ──▶ twilio_service.send_whatsapp(full detail text)
        │                       TwiML: "Sent! Anything else?" + <Record>
        │
        └─ "no" ──▶ TwiML: "No problem. Anything else?" + <Record>
```

---

## Proactive Reminders

A background scheduler (`APScheduler AsyncIOScheduler`) starts at app startup and scans `family.md` every 30 minutes for upcoming events.

```
App startup (lifespan)
        │
        ▼
reminder_service.start()
  Schedules _scan_and_schedule() to run immediately + every 30 min
        │
        ▼
_scan_and_schedule()
  markdown_service.read_events(after=now, before=now+48h)
        │
        ▼
  For each event in window:
    For each offset in [24h, 4h]:
      remind_at = event_datetime - offset
      if remind_at <= now → skip (already past)
      job_id = f"reminder_{event_datetime}_{offset_minutes}"
      if job already exists in scheduler → skip (deduplication)
      scheduler.add_job(DateTrigger(run_date=remind_at), _send_reminder)

_send_reminder(title, label, event_dt)
  Formats message: "Reminder: *{title}* is in {label} ({human time})."
  Sends WhatsApp to every number in PHONE_TO_NAME
```

**Deduplication:** APScheduler job IDs are deterministic (`reminder_{iso_datetime}_{minutes}`). The 30-minute scan simply skips any job ID that already exists. One-off jobs are removed by APScheduler after they fire, so there is no risk of double-sending within a single server run.

**Restart behaviour:** The in-memory job store is cleared on restart. The scanner runs immediately on startup and reschedules any reminders whose `remind_at` is still in the future. Reminders already sent (whose `remind_at` is in the past) are naturally skipped by the `remind_at <= now` guard.

---

## Web Dashboard Flow

```
Browser → GET /dashboard
        markdown_service.read_all_data() → todos + events
        Annotate events with is_past, sort (pending/upcoming first)
        Resolve family_names from PHONE_TO_NAME for name dropdowns
        Jinja2 renders templates/dashboard.html
        Bootstrap 5 two-column layout, auto-refreshes every 30s

Dashboard is fully editable — all mutations are JSON POSTs, reload on success:

  POST /dashboard/add-todo        {text, due?, added_by}  → append_todo()
  POST /dashboard/complete-todo   {text}                  → complete_todo()
  POST /dashboard/delete-todo     {text}                  → delete_todo()

  POST /dashboard/add-event       {title, event_datetime, added_by}  → append_event()
  POST /dashboard/delete-event    {title, event_datetime}            → delete_event()

Edit (todo or event) is handled client-side:
  delete-old → add-new (two sequential API calls, single page reload)
```

---

## Browser Interface Flow

Family members open the browser interface on any device on the local network. The Talk and Hangman pages require HTTPS (use the cloudflared URL) because mic access and WASM require a secure context.

```
Browser (Portal / phone / tablet)
        │
        ├─ GET /          → home.html  (4 nav cards: Games, Dashboard, Talk, Cameras)
        │
        ├─ GET /games      → games.html  (games hub: Hangman, Multiply, Clock, Quiz)
        │
        ├─ GET /cameras    → cameras.html
        │   POST /cameras/set-stream  {url}
        │     camera_service.set_stream_url(url)
        │       → also calls scene_service.start_analysis(url) or stop_analysis()
        │   GET  /cameras/stream      → MJPEG StreamingResponse (local fallback)
        │   WS   /cameras/ws          → WebSocket stream (works through Cloudflare Tunnel)
        │
        │   Singleton reader (camera_service._reader_loop):
        │     One background thread per active stream URL
        │     Decode backend selected at first start:
        │       PyAV software decode (4 CPU threads) — current active backend
        │         GStreamer code present but disabled: Gst.init() called on reader
        │         thread violates GStreamer threading contract → SIGSEGV.
        │         Fix tracked in improvement_ideas.md Step 3.
        │     Reads frames → draws debug overlay if enabled → encodes JPEG
        │     Broadcasts JPEG bytes to all subscriber queues (_broadcast)
        │     Started/stopped by set_stream_url(); shared by all consumers
        │
        │   Consumers subscribe via subscribe_frames() → Queue, unsubscribe on disconnect
        │     mjpeg_generator: wraps frames in multipart/x-mixed-replace chunks
        │     ws_frame_generator: yields raw JPEG bytes → WebSocket sends as binary
        │     Browser renders frames on <canvas> via createImageBitmap()
        │   GET  /cameras/queries     → list of global scene queries
        │   POST /cameras/queries     {text} → scene_service.add_query()
        │   DELETE /cameras/queries/{i}      → scene_service.remove_query(i)
        │   GET  /cameras/events      → last 1-hour events (polled every 5s by browser)
        │
        │   GET  /cameras/threshold   → current CLIP similarity threshold
        │   POST /cameras/threshold   {value} → set threshold (0.0–1.0)
        │   GET  /cameras/pad         → current crop padding factor
        │   POST /cameras/pad         {value} → set pad factor (0.0–2.0)
        │   POST /cameras/debug-overlay {enabled} → toggle bounding-box overlay
        │
        │   Single-reader frame sharing:
        │     camera_service._reader_loop() decodes every frame (GStreamer or PyAV)
        │       → calls scene_service.push_frame(frame) on each decoded frame
        │       → encodes JPEG for MJPEG display (with optional debug overlay drawn)
        │
        │   Scene analysis pipeline (scene_service.py — background thread):
        │     Receives frames via push_frame() / _shared_frame — no separate VideoCapture
        │     Samples at 5 fps (waits on threading.Event, sleeps remainder of interval)
        │     No PyTorch in main process — all GPU inference runs in GDINO container
        │       → JPEG-encode frame → httpx.Client POST /infer to GDINO FastAPI service
        │           Returns JSON: {boxes: [[x1,y1,x2,y2],...], scores: [...], labels: [...]}
        │       → _SimpleTracker (pure-numpy IoU): assign persistent track IDs
        │             No supervision/ByteTrack — eliminates pybind11/matplotlib deps
        │       → _set_latest_detections() updates shared Detection list (used by overlay)
        │       → For each tracked object:
        │           if conf ≥ threshold for 30s recheck window → CameraEvent logged
        │           CameraEvent includes JPEG crop as base64 for event log thumbnails
        │       → Rolling event log: deque(maxlen=500), filtered to last 1h on read
        │     Skips inference when no queries defined and debug overlay is off
        │     GDINO connection: waits up to 60s on start_analysis() via GET /health poll
        │
        │   GDINO FastAPI Service (gdino_server.py — runs in "triton" Docker container):
        │     uvicorn on port 8082, script bind-mounted from ./triton_models/gdino_server.py
        │     Loads GDINO Tiny (IDEA-Research/grounding-dino-tiny) fp16 on CUDA at startup
        │     POST /infer: multipart JPEG + JSON queries → JSON {boxes, scores, labels}
        │       Resize to max 800px wide → run GDINO → post_process to full-res pixel coords
        │     GET /health: used by Docker Compose healthcheck and scene_service wait loop
        │     Note: container named "triton"; uses FastAPI not Triton Server — Triton's
        │       Python backend was abandoned due to a shared-memory IPC bug in 24.04 where
        │       all tensor elements are overwritten with the first element's value.
        │
        ├─ GET /talk       → talk.html
        │   Page loads → Silero VAD initialises (ONNX model downloaded from CDN,
        │                 cached after first load, runs on device CPU via WASM)
        │   VAD listens continuously — no tap required
        │   User speaks → onSpeechEnd(Float32Array @ 16kHz)
        │     VAD paused while processing (prevents feedback loop with TTS)
        │        │
        │        ▼
        │   Float32Array encoded to WAV in browser (float32ToWav)
        │   POST /transcribe  (audio/wav blob)
        │     asyncio.to_thread(whisper_service.transcribe, bytes, ".wav")
        │     → {transcript, confidence}
        │        │
        │        ▼
        │   POST /chat  {transcript, caller_name}
        │     chat_handler.handle_chat()
        │       classify_intent → route to handler
        │       research → Tavily + parallel Qwen summaries
        │       returns {speech, display, intent}
        │        │
        │        ▼
        │   speechSynthesis.speak(speech)     ← browser reads aloud
        │   VAD resumes after TTS finishes (onend callback)
        │   Show display text in Full Answer panel (research only)
        │
        ├─ GET /dashboard  → dashboard.html
        │   Editable todos and events (add / complete / edit / delete)
        │   All mutations via JSON POST, reload on success
        │   Auto-refreshes every 30s
        │
        ├─ GET /games/hangman → hangman.html
        │   Same VAD auto-detection as /talk
        │   POST /games/hangman/new    → new HangmanGame (random word)
        │     hangman_service.new_game() pre-reveals hint letters based on word length:
        │       ≤4 letters → 0 hints, 5→1, 6→2, 7+→3 (randomly chosen distinct letters)
        │   POST /games/hangman/guess  {session_id, guess}
        │     hangman_service.guess()
        │     Accepts: "letter A", "word elephant", bare single letter
        │     → {display_word, wrong_letters, figure, speech, won, lost}
        │   VAD stays active after game over — player can say "new game"
        │   minSpeechMs: 250ms
        │   Duration guard: audio > 4s rejected (TTS echo filter)
        │   Confidence check: transcripts with confidence < 0.30 silently ignored
        │   speak() has watchdog timer — forces resumeListening() if onend never fires
        │
        ├─ GET /games/multiply → multiply.html
        │   Times Tables practice game for kids
        │   All logic client-side — no new backend endpoints
        │   Generates random A×B (A,B ∈ 1–9), speaks question via speechSynthesis
        │   VAD captures spoken answer → /transcribe → parseSpokenNumber()
        │     Handles digits ("24") and English words ("twenty four", "eight")
        │   Confidence check: transcripts with confidence < 0.45 silently ignored
        │   minSpeechMs: 250ms (numbers are short; duration guard handles echo)
        │   Tracks correct / answered score; Fresh Start resets
        │
        ├─ GET /games/clock → clock.html
        │   Tell the Time game — 4-option multiple choice
        │   All logic client-side — no new backend endpoints
        │   Generates random time (hour 1–12, minute in multiples of 5)
        │   Renders 4 SVG analogue clock faces (pure JS, no images/libraries):
        │     white face, 12 tick marks, hour numbers at 12/3/6/9,
        │     short thick dark hour hand, long thin purple minute hand
        │   Distractors differ by ≥15 min or different hour (visually distinct)
        │   VAD captures spoken answer → /transcribe → parseOption()
        │     Strips filler phrases ("I think", "the answer is", "letter", etc.)
        │     Matches: bare letter (A/B/C/D), spoken names (ay/bee/see/dee)
        │     Conservative "A" guard: only accepts "a" when little other content
        │   Confidence check: transcripts with confidence < 0.30 silently ignored
        │   minSpeechMs: 250ms (single letters ~150ms; duration guard handles echo)
        │   Tapping a clock card also accepted as answer
        │   Speaks human-friendly time: "half past 3", "quarter to 6", "3 o'clock"
        │   Tracks correct / answered score; Fresh Start resets
        │
        └─ GET /games/quiz → quiz.html
            Knowledge Quiz — subject + grade selection → Qwen-generated questions
            POST /games/quiz/generate  {subject, grade}
              asyncio.to_thread(qwen.generate_quiz)
              Prompt: prompts/quiz_generate.txt — enforces kid-safe content,
                grade-appropriate difficulty, structured JSON output
              qwen.generate_quiz() validates each question (4 options, correct index 0-3)
              Requires ≥5 valid questions or raises error
              Returns JSON array of up to 10 questions
            Client flow:
              Setup screen → pick subject (8 options) + grade (1–8)
              Loading screen — cycling messages while Qwen generates (~15-25s)
              Question screen — progress bar, 4 option cards, VAD or tap
              Funny response phrases (8 correct, 8 wrong) picked at random
              Final score screen — message scaled to performance, Play Again or New Quiz
            VAD initialised after questions load (not during setup/loading)
            Voice answer matching — parseAnswer(transcript, options):
              1. Option text match: if transcript contains any option text, use it
              2. Filler stripping: removes "I think", "the answer is", "I choose", etc.
              3. Spoken letter names: ay→A, bee→B, see/cee→C, dee→D
              4. Single letter match with conservative "A" guard
            Confidence check: transcripts with confidence < 0.30 silently ignored
            minSpeechMs: 250ms (single letters ~150ms; duration guard handles echo)

        ├─ GET /games/bulls-cows → bulls_cows.html
            Bulls and Cows — 4-digit code-breaking game (no LLM)
            POST /games/bulls-cows/new   → new BullsCowsGame (4-digit secret, all unique, non-zero first)
            POST /games/bulls-cows/guess {session_id, guess: "two four one three"}
              bulls_cows_service.parse_spoken_number() — token-by-token word→digit mapping
                Handles: digit words, digit characters, homophones (to→2, for→4, ate→8)
                Returns None if not exactly 4 tokens or any token unrecognised
              _score(secret, guess) → (bulls, cows): O(n) comparison
              Max 10 attempts; win = 4 bulls; speech narrates result each turn
            VAD duration guard: 5s (digit sequences are short)

        ├─ GET /games/word-ladder → word_ladder.html
            Word Ladder — change one letter at a time from start→target word
            POST /games/word-ladder/new
              asyncio.to_thread(qwen.generate_word_ladder) → {"start": ..., "target": ...}
              Prompt: prompts/word_ladder_generate.txt — 4-letter common kid words, 2-5 step path
              word_ladder_service.new_game(start, target) validates pair with BFS
              Falls back to hardcoded pairs if Qwen fails or BFS returns None
            POST /games/word-ladder/step {session_id, word}
              Validates: same length, exactly 1 letter different, word in _WORD_SET
              _WORD_SET: /usr/share/dict/words filtered to lowercase alpha-only 3-5 letters
                         built as frozenset at module import time; pre-grouped by length
              5 wrong attempts allowed (bad word OR not in dict counts as wrong)
            POST /games/word-ladder/hint {session_id}
              BFS from current_word to target → next_word → reports which letter position to change
            Chain visualiser: JS renders vertical word→word chain with differing letter underlined
            VAD: says word aloud → Whisper → submitWord(); "hint" / "new game" also VAD-detected

        └─ GET /games/twenty-questions → twenty_questions.html
            20 Questions — Bianca asks yes/no questions; kid thinks of something; Bianca guesses
            POST /games/twenty-questions/new     → TwentyQGame (phase=thinking)
            POST /games/twenty-questions/start   {session_id}
              asyncio.to_thread(twenty_questions_service.start_questions)
              Primes Ollama messages list with system prompt + "I've thought of something"
              qwen.ask_twenty_questions(messages) → multi-turn /api/chat call (30s timeout)
              Returns {"type": "question"|"guess", "content": "..."} — JSON strict
                Fallback on non-JSON: treat raw text as question, log warning
            POST /games/twenty-questions/answer  {session_id, answer}
              asyncio.to_thread(twenty_questions_service.answer)
              Normalises answer (yes/no/maybe) via word-list matching before appending to messages
              Appends user turn to messages list; calls ask_twenty_questions with full history
              At MAX_QUESTIONS: calls qwen.force_twenty_questions_guess (appends override user msg)
              Phase transitions: playing → guessing when Qwen returns type=="guess"
            POST /games/twenty-questions/confirm {session_id, answer}
              twenty_questions_service.confirm() — no Qwen call; phase→finished
            VAD behaviour: paused during thinking/loading/finished phases;
              active during playing (yes/no/maybe) and guessing (yes/no) phases only
              Duration guard: 6s (questions are ~3-5s TTS; answers are short)
            Phase flow: thinking → loading → playing → guessing → finished
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
├── services/
│   ├── camera_service.py      RTSP URL storage; PyAV software decode (GStreamer present but disabled — thread-safety bug); WebSocket/MJPEG broadcast; triggers scene analysis
│   ├── scene_service.py       Grounding DINO Tiny via GDINO FastAPI (httpx POST) + _SimpleTracker (IoU, pure-numpy); query management; event log; no PyTorch in main process
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
| RTSP streaming | PyAV (FFmpeg software decode) + WebSocket canvas | GStreamer present but disabled (Gst.init thread-safety bug); browser receives JPEG frames over WebSocket |
| Scene detection | Grounding DINO Tiny (fp16, HuggingFace) via GDINO FastAPI | Open-vocabulary detection in one pass; ~0.3GB VRAM in gdino container |
| Object tracking | _SimpleTracker (pure-numpy IoU) | Persistent track IDs, deduplication; CPU-only; no pybind11/supervision |
| Inference serving | FastAPI/uvicorn (Docker, port 8082) | Isolated CUDA context; plain HTTP replaces Triton Python backend (bug) |
| Scheduler | APScheduler `AsyncIOScheduler` | Proactive reminders without Celery/Redis |
| Backend | FastAPI + uvicorn | Async, fast, minimal boilerplate |
| Templates | Jinja2 + Bootstrap 5 | No build step, zero JS framework needed |
| Logging | `TimedRotatingFileHandler` | Daily log files, 7-day retention |
| Tunnel (dev) | Cloudflare Tunnel (`cloudflared`) | Twilio webhooks + HTTPS for browser mic; no bandwidth limits |

---

## GPU Memory Layout (RTX 4070 Ti Super — 16GB)

```
┌──────────────────────────────────────────────────┐
│                  16 GB VRAM                      │
│                                                  │
│  ┌────────────────────────┐                      │
│  │  Qwen 2.5:14b Q4_K_M  │  ~9-10 GB            │
│  │  (Ollama, always hot)  │                      │
│  └────────────────────────┘                      │
│  ┌──────────────┐                                │
│  │  Whisper     │  ~1.5 GB                       │
│  │  large-v3    │  (always loaded, not active    │
│  │  int8_float16│   during Qwen inference)       │
│  └──────────────┘                                │
│  ┌──────────────────────────────┐                │
│  │  Free headroom  ~4.5 GB     │                │
│  └──────────────────────────────┘                │
└──────────────────────────────────────────────────┘

GDINO Docker container (separate CUDA context — no conflict with main process):
┌──────────────────────────────────────────────────┐
│  ┌──────────────┐                                │
│  │  GDINO Tiny  │  ~0.3 GB  (fp16)               │
│  │  FastAPI svc │                                │
│  └──────────────┘                                │
└──────────────────────────────────────────────────┘

Note: Whisper and Qwen never run at the same time —
Whisper transcribes first, then Qwen processes.
GDINO runs continuously at 5 fps via HTTP POST when a
camera stream is active. _SimpleTracker runs in the main
process on CPU (no VRAM needed).
```

---

## Docker Compose Architecture

Bianca runs as four containers on a single host, sharing the GPU via Nvidia Container Toolkit.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           docker-compose stack                                  │
│                           host: RTX 4070 Ti Super                               │
│                                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │  bianca-whisper  │  │  bianca-triton   │  │  bianca-ollama   │              │
│  │  :8080           │  │  :8082           │  │  :11434          │              │
│  │                  │  │                  │  │                  │              │
│  │  faster-whisper  │  │  gdino_server.py │  │  Ollama          │              │
│  │  large-v3  fp16  │  │  GDINO Tiny fp16 │  │  Qwen 2.5:14b   │              │
│  │  ~1.5GB VRAM     │  │  ~0.3GB VRAM     │  │  ~9-10GB VRAM   │              │
│  │                  │  │                  │  │                  │              │
│  │  POST /transcribe│  │  POST /infer     │  │  POST /api/chat  │              │
│  │  GET  /health    │  │  GET  /health    │  │  GET  /api/tags  │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
│           │                     │                     │                         │
│           └─────────────────────┴─────────────────────┘                         │
│                                         │  bianca-net (bridge)                  │
│                                         │                                       │
│                          ┌──────────────▼──────────────┐                        │
│                          │       bianca-app  :8000      │                        │
│                          │                              │                        │
│                          │  FastAPI + uvicorn           │                        │
│                          │  NO GPU  (CPU-only)          │                        │
│                          │                              │                        │
│                          │  → Whisper for STT           │                        │
│                          │  → GDINO for scene detection │                        │
│                          │  → Ollama/Qwen for LLM       │                        │
│                          └──────────────────────────────┘                        │
│                                         │                                        │
└─────────────────────────────────────────┼────────────────────────────────────────┘
                                          │ :8000 (host port)
                                          ▼
                               ┌─────────────────────┐
                               │  Browser / Twilio   │
                               │  cloudflared tunnel │
                               └─────────────────────┘
```

### Container responsibilities

| Container | Image | GPU | Purpose |
|---|---|---|---|
| `bianca-whisper` | `bianca-whisper` (Dockerfile.whisper) | Yes | faster-whisper STT; exposes REST `/transcribe` |
| `bianca-triton` | `bianca-triton` (Dockerfile.triton) | Yes | GDINO FastAPI (gdino_server.py); object detection `/infer` |
| `bianca-ollama` | `ollama/ollama:latest` | Yes | Qwen 2.5:14b LLM; entrypoint auto-pulls model |
| `bianca-app` | `bianca-app` (Dockerfile.app) | **No** | Main FastAPI app; explicitly CUDA-free |

### Key Docker Compose design decisions

- **App has no GPU reservation** — enforces CUDA-free constraint on the main process; any accidental PyTorch/CUDA import will fail fast rather than silently consuming VRAM
- **All four on one bridge network** (`bianca-net`) — containers reach each other by service name (e.g. `http://whisper:8080`); no host networking needed
- **`depends_on` with `service_healthy`** — app waits for all three AI services to pass their healthchecks before starting; avoids race conditions on startup
- **Bind-mounts for AI weights** — `~/.cache/huggingface` bind-mounted into whisper and triton containers; weights downloaded once to host, reused across rebuilds
- **Named volume for Ollama models** (`ollama-models`) — `qwen2.5:14b` (~9GB) persists across container restarts and rebuilds
- **Named volumes for app data** — `family-data` (family.md) and `app-logs` survive app container rebuilds

### Camera inference pipeline

```
camera_service (app container, CPU)
        │  push_frame(bgr)
        ▼
scene_service (app container, CPU)
        │  JPEG-encode → httpx POST /infer
        ▼
┌─────────────────────────────────┐
│  bianca-triton container (GPU)  │
│                                 │
│  gdino_server.py (FastAPI)      │
│  ┌───────────────────────────┐  │
│  │  GDINO Tiny  fp16 CUDA    │  │
│  │  AutoProcessor             │  │
│  │  resize → max 800px wide  │  │
│  │  post_process → pixel     │  │
│  │  coords (original dims)   │  │
│  └───────────────────────────┘  │
│  returns JSON {boxes,scores,    │
│                labels}          │
└─────────────┬───────────────────┘
              │  JSON response
              ▼
scene_service
        │  _SimpleTracker (IoU, pure-numpy)
        │  → persistent track IDs
        │  → CameraEvent on match (threshold + 30s dedup)
        ▼
cameras.html (browser, polling /cameras/events every 5s)
```

---

## Camera Inference — Future Roadmap

Current stack works well for a home assistant at 5 fps. The following phases would bring it to
production-quality, near-zero-latency inference:

```
CURRENT (Phase 1)                   FUTURE (Phase 2-4)
─────────────────────               ───────────────────────────────
PyAV software decode                GStreamer NVDEC hardware decode
  (CPU, ~1-2 cores)      ──────▶      (GPU DMA, ~0 CPU overhead)

GDINO Tiny (fp16)                   TensorRT engine (.plan file)
  HuggingFace model      ──────▶      engine = trt.Runtime.deserialize()
  ~80ms / frame                        ~10-20ms / frame  (4-8× faster)

httpx multipart POST                NVIDIA Triton Inference Server
  plain HTTP/1.1         ──────▶      gRPC / shared-memory IPC
                                       batch inference, multi-model

_SimpleTracker (IoU)                NVIDIA DeepStream / DALI pipeline
  pure Python, CPU       ──────▶      GPU-resident pipeline, cuDNN tracking
                                       DALI image decoding stays on GPU
```

### Phase 2 — GStreamer NVDEC
- Replace PyAV in `camera_service.py` with GStreamer NVDEC pipeline
- Pipeline: `uridecodebin → nvv4l2decoder → nvvidconv → appsink`
- For local files use `decodebin` (auto-selects nvv4l2decoder); for RTSP use `rtspsrc`
- `appsink sync=true` for local files (correct playback speed); `sync=false` for RTSP (live, no buffer)
- GStreamer must be initialised from the main thread — not from a reader thread (SIGSEGV otherwise)

### Phase 3 — TensorRT for GDINO / replacement model
- GDINO Tiny cannot currently be exported to ONNX/TRT — data-dependent Python loops in the
  text encoder block all export paths (investigated 2026-04-03, see `scripts/export_gdino_onnx.py`)
- **Option A:** Wait for HuggingFace/IDEA-Research to add TRT support upstream
- **Option B:** Replace GDINO with `YOLOWorld` (YOLOv8 + open-vocab) — fully ONNX-exportable,
  TRT-compatible, similar open-vocabulary detection capability
- Once a TRT engine is available: load in `gdino_server.py` with `tensorrt` Python bindings;
  keep the same `/infer` HTTP API — app container unchanged

### Phase 4 — NVIDIA Triton + DALI/DeepStream
- Move TRT engine into Triton model repository (proper `config.pbtxt`, no Python backend)
- Use Triton's gRPC shared-memory API for zero-copy frame transfer from app → Triton
- DALI pipeline: GPU-resident decode → resize → normalise → GDINO/YOLOWorld → NMS
- DeepStream: full pipeline from RTSP ingest to analytics on GPU with no CPU round-trips
- Expected: sub-5ms per frame end-to-end, 30+ fps capable

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
