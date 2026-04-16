# Improvement Ideas

## Camera-side motion detection gate (Thingino webhook)

**Context**: Currently a software frame-differencing gate runs before Triton inference
(Option A, already implemented). Option B uses the camera itself to signal motion.

**Idea**: Thingino firmware has built-in hardware-assisted motion detection that runs on
the camera's own SoC — zero load on the server. Configure it to POST a webhook to Bianca
when motion starts and stops. The server enables/disables Triton inference per camera
based on these signals.

**Why it's better than software frame differencing**:
- Completely zero GPU/CPU cost when idle — inference thread sleeps until a webhook arrives
- Motion detection tuned on the camera: sensitivity zones, scheduling, ignore masks
- Camera hardware can detect motion before the first frame even arrives over RTSP
- Software frame diff can be fooled by lighting changes (clouds, lamps); camera-side is smarter

**Rough implementation**:
1. Add a `POST /cameras/motion/{cam_id}` endpoint that sets `_motion_active[cam_id] = True/False`
2. In `_inference_loop`, check `_motion_active.get(cam_id, True)` before calling Triton
   (default True so cameras without Thingino still work)
3. On the camera: configure Thingino's HTTP action to POST to
   `http://<server>/cameras/motion/{cam_id}` with `{"state": "start"}` / `{"state": "stop"}`
4. Add a timeout fallback: if no "stop" webhook arrives within N seconds, auto-clear the flag

**Thingino config path**: Settings → Alarm → Motion Detection → HTTP action
