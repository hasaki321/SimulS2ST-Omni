"""Binary framing for OmniTalker streaming S2ST WebSocket demo.

Frame layout: ``[type:u8][payload...]``

| type | Direction | Payload |
|------|-----------|---------|
| 0x00 | C↔S       | UTF-8 JSON (config / ready / error / done) |
| 0x01 | C→S       | PCM int16 LE mono @ 16 kHz |
| 0x01 | S→C       | PCM int16 LE mono @ 24 kHz |
| 0x02 | S→C       | UTF-8 translation text chunk |
| 0x03 | C→S       | EOS (empty payload) |
"""

from __future__ import annotations

import json
from typing import Any

TYPE_JSON = 0x00
TYPE_PCM = 0x01
TYPE_TEXT = 0x02
TYPE_EOS = 0x03

SAMPLE_RATE_IN = 16_000
SAMPLE_RATE_OUT = 24_000


def pack_frame(frame_type: int, payload: bytes = b"") -> bytes:
    if not (0 <= frame_type <= 255):
        raise ValueError(f"invalid frame_type={frame_type}")
    return bytes((frame_type,)) + payload


def unpack_frame(message: bytes) -> tuple[int, bytes]:
    if not message:
        raise ValueError("empty websocket message")
    return message[0], message[1:]


def pack_json(obj: dict[str, Any]) -> bytes:
    return pack_frame(TYPE_JSON, json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def unpack_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


def pack_pcm_i16(pcm: bytes) -> bytes:
    return pack_frame(TYPE_PCM, pcm)


def pack_text(text: str) -> bytes:
    return pack_frame(TYPE_TEXT, text.encode("utf-8"))


def pack_eos() -> bytes:
    return pack_frame(TYPE_EOS)
