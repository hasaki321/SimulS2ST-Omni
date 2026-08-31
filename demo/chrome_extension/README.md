# Chrome extension: OmniTalker tab S2ST bridge

Capture **current tab audio** (Bilibili / YouTube / …) → stream to
[`demo/server.py`](../server.py) over WebSocket → play translated speech + show captions.

## Install (unpacked)

1. Start the OmniTalker WS server on the GPU machine (and FRP/`wss` if remote):

```bash
DEVICE=cuda:0 bash demo/run_server.sh --default-latency 2 --port 8765
```

2. Chrome → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this folder:

`demo/chrome_extension`

3. Open a normal tab (e.g. Bilibili video), click the extension icon:
   - WebSocket URL: `ws://127.0.0.1:8765` (local) or `wss://your-frp-host/...`
   - Direction / Latency L
   - **Start**

Translated audio plays from the extension’s offscreen AudioContext (~L seconds late).
Captions appear as a **floating bar at the bottom of the tab** (and still in the popup if open).

On **Start** you should hear a short **beep**. If no beep, click **Test beep / resume audio**.
Status should later show `audio_recv` when PCM arrives from the server.

## Notes

- Cannot capture `chrome://` pages; use a normal https tab.
- Tab capture often **mutes the tab** in Chrome; translation plays via the extension.
  Check “Also hear original tab” if you want a quiet mix of the source.
- Reload the extension after updates (`chrome://extensions` → ↻).
- Protocol matches [`demo/protocol.py`](../protocol.py): PCM16 @16 kHz up /
  @24 kHz down, 1 s frames.
- For remote access, expose the server through a trusted VPN or TLS-terminated
  `wss://` reverse proxy. Do not expose the unauthenticated demo server directly
  to the public internet.

## Layout

| File | Role |
|------|------|
| `manifest.json` | MV3, `tabCapture` + `offscreen` + `scripting` |
| `background.js` | Start/stop, get stream id, forward captions to tab |
| `offscreen.js` | Capture, resample, WS, playback |
| `capture-processor.js` | AudioWorklet capture (replaces ScriptProcessor) |
| `content_overlay.js` | Bottom floating caption bar (injected on Start) |
| `popup.*` | Controls + captions |
| `protocol.js` | Shared binary framing |
