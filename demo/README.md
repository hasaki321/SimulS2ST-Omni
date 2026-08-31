# OmniTalker streaming S2ST WebSocket demo

Minimal realtime demo: **WebSocket server** + **Moshi-style file client**.

Browser path: **Chrome extension** (tab audio capture) in [`chrome_extension/`](chrome_extension/).

## Protocol

Binary frames: `[type:u8][payload...]`

| type | Direction | Payload |
|------|-----------|---------|
| `0x00` | C↔S | UTF-8 JSON (`config` / `ready` / `error` / `done`) |
| `0x01` | C→S | PCM int16 LE mono @ **16 kHz** |
| `0x01` | S→C | PCM int16 LE mono @ **24 kHz** |
| `0x02` | S→C | UTF-8 translation text chunk |
| `0x03` | C→S | EOS (empty) |

Config example:

```json
{"direction":"en2zh","latency_multiplier":2,"sample_rate":16000}
```

## Quick start

Terminal 1 — server (loads OmniTalker once; pick a free GPU):

```bash
DEVICE=cuda:0 bash demo/run_server.sh
```

Terminal 2 — file client (paces a 16 kHz mono WAV at 1 s per frame):

```bash
INPUT=/path/to/input.wav bash demo/run_file_client.sh
```

Outputs:

- Translated audio: `outputs/demo/file_client_out.wav`
- Server text sidecars: `outputs/demo/server_sidecar/`

## Defaults

Aligned with RealSI eval scripts:

- `--thinker-num-beams 4`, thinker KV cache on, greedy decode
- `latency_multiplier=2`, `history-window-turns=28`, `history-overlap-turns=16`
- Model package: `models/SimulS2ST-Omni/offline` + `models/SimulS2ST-Omni/simuls2st_adapter`

## Timeline log (L3 realtime)

```bash
# server (example: free GPU, L3 default)
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash demo/run_server.sh --default-latency 3 --port 8766

# client with wall-clock send/recv JSON
python -m demo.file_client --url ws://127.0.0.1:8766 --latency 3 \
  --timeline outputs/demo/l3_timeline/json/utt_timeline.json
```

## Chrome extension (Bilibili / any tab)

See [`chrome_extension/README.md`](chrome_extension/README.md).

```text
chrome://extensions → Load unpacked → demo/chrome_extension
```

Captures the **active tab** audio, streams to `demo/server.py`, plays translation + captions.
