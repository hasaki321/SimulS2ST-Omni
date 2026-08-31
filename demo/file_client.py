#!/usr/bin/env python3
"""Moshi-style paced file client for OmniTalker streaming S2ST WebSocket demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import websockets
import websockets.exceptions as wsex

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.protocol import (  # noqa: E402
    SAMPLE_RATE_IN,
    SAMPLE_RATE_OUT,
    TYPE_JSON,
    TYPE_PCM,
    TYPE_TEXT,
    pack_eos,
    pack_json,
    pack_pcm_i16,
    unpack_frame,
    unpack_json,
)

logger = logging.getLogger("demo.file_client")


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _to_float32(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32)
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    raise TypeError(f"unsupported wav dtype {x.dtype}")


def _resample_float(x: np.ndarray, sr: int, tgt: int) -> np.ndarray:
    if sr == tgt:
        return x.astype(np.float32)
    y = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    y = AF.resample(y, sr, tgt)[0].numpy()
    return y.astype(np.float32)


def _chunk_float(sig: np.ndarray, frame_smp: int) -> list[np.ndarray]:
    pad = (-len(sig)) % frame_smp
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=np.float32)])
    return [sig[i : i + frame_smp] for i in range(0, len(sig), frame_smp)]


def _float_to_pcm16_bytes(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class OmniTalkerFileClient:
    def __init__(
        self,
        ws_url: str,
        inp: Path,
        out: Path,
        direction: str,
        latency_multiplier: int,
        frame_ms: int,
        timeline_path: Optional[Path] = None,
    ):
        self.url = ws_url
        self.inp = inp
        self.out = out
        self.direction = direction
        self.latency_multiplier = latency_multiplier
        self.frame_ms = frame_ms
        self.timeline_path = timeline_path
        self.frame_smp = int(SAMPLE_RATE_IN * frame_ms / 1000)
        self.frame_sec = self.frame_smp / SAMPLE_RATE_IN

        wav, sr = sf.read(str(inp), dtype="float32", always_2d=False)
        self.sig16 = _resample_float(_mono(_to_float32(np.asarray(wav))), int(sr), SAMPLE_RATE_IN)
        self.source_dur_sec = float(len(self.sig16) / SAMPLE_RATE_IN)
        self._recv_pcm: list[np.ndarray] = []
        self._texts: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._t0: float = 0.0
        self._src_sent_sec: float = 0.0
        self._out_recv_sec: float = 0.0

    def _t(self) -> float:
        return time.perf_counter() - self._t0

    def _log(self, event: str, **kwargs: Any) -> None:
        row = {"t_wall_sec": round(self._t(), 4), "event": event, **kwargs}
        self._events.append(row)
        logger.info(
            "timeline t=%.3f %s %s",
            row["t_wall_sec"],
            event,
            {k: v for k, v in kwargs.items() if k != "text"},
        )

    async def _handshake(self, ws) -> None:
        self._t0 = time.perf_counter()
        self._log(
            "session_start",
            input=str(self.inp),
            direction=self.direction,
            latency_multiplier=self.latency_multiplier,
            frame_ms=self.frame_ms,
            source_dur_sec=round(self.source_dur_sec, 4),
        )
        await ws.send(
            pack_json(
                {
                    "direction": self.direction,
                    "latency_multiplier": self.latency_multiplier,
                    "sample_rate": SAMPLE_RATE_IN,
                }
            )
        )
        self._log("config_sent")
        ready_msg = await ws.recv()
        if isinstance(ready_msg, str):
            ready_msg = ready_msg.encode("utf-8")
        ftype, payload = unpack_frame(ready_msg)
        if ftype != TYPE_JSON:
            raise RuntimeError(f"expected ready JSON, got type={ftype}")
        ready = unpack_json(payload)
        if not ready.get("ok"):
            raise RuntimeError(f"server rejected config: {ready}")
        self._log("ready", **{k: ready[k] for k in ready if k != "ok"})
        logger.info("ready: %s", ready)

    async def _send(self, ws) -> None:
        for i, frame in enumerate(_chunk_float(self.sig16, self.frame_smp)):
            await ws.send(pack_pcm_i16(_float_to_pcm16_bytes(frame)))
            self._src_sent_sec += len(frame) / SAMPLE_RATE_IN
            self._log(
                "input_pcm_sent",
                frame_index=i,
                frame_sec=round(len(frame) / SAMPLE_RATE_IN, 4),
                src_sent_sec=round(self._src_sent_sec, 4),
            )
            await asyncio.sleep(self.frame_sec)

        await ws.send(pack_eos())
        self._log("eos_sent", src_sent_sec=round(self._src_sent_sec, 4))
        logger.info("sent EOS (%d samples @ %d Hz)", len(self.sig16), SAMPLE_RATE_IN)

    async def _recv(self, ws) -> None:
        audio_chunk_index = 0
        text_chunk_index = 0
        try:
            async for message in ws:
                if isinstance(message, str):
                    message = message.encode("utf-8")
                ftype, payload = unpack_frame(message)
                if ftype == TYPE_TEXT:
                    text = payload.decode("utf-8")
                    self._texts.append(text)
                    self._log(
                        "text_recv",
                        chunk_index=text_chunk_index,
                        text=text,
                        src_sent_sec=round(self._src_sent_sec, 4),
                        out_recv_sec=round(self._out_recv_sec, 4),
                    )
                    text_chunk_index += 1
                    print(f"[TEXT] {text}", flush=True)
                elif ftype == TYPE_PCM:
                    pcm = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
                    self._recv_pcm.append(pcm)
                    out_sec = float(len(pcm) / SAMPLE_RATE_OUT)
                    self._out_recv_sec += out_sec
                    lag = self._t() - self._src_sent_sec
                    self._log(
                        "audio_recv",
                        chunk_index=audio_chunk_index,
                        out_chunk_sec=round(out_sec, 4),
                        out_recv_sec=round(self._out_recv_sec, 4),
                        src_sent_sec=round(self._src_sent_sec, 4),
                        wall_minus_src_sent=round(lag, 4),
                    )
                    audio_chunk_index += 1
                elif ftype == TYPE_JSON:
                    obj = unpack_json(payload)
                    # Avoid clashing with _log(event=...); keep payload nested.
                    self._log("json_recv", payload=obj)
                    logger.info("json: %s", obj)
                    if obj.get("done") or obj.get("ok") is False:
                        break
                else:
                    logger.warning("unknown frame type %s", ftype)
        except wsex.ConnectionClosed as exc:
            self._log("connection_closed", reason=str(exc))
            logger.info("connection closed: %s", exc)

    def _summary(self, audio: np.ndarray) -> dict[str, Any]:
        first_audio = next((e for e in self._events if e["event"] == "audio_recv"), None)
        first_text = next((e for e in self._events if e["event"] == "text_recv"), None)
        done = next(
            (
                e
                for e in self._events
                if e["event"] == "json_recv" and isinstance(e.get("payload"), dict) and e["payload"].get("done")
            ),
            None,
        )
        first_chunk_ready_src = float(self.latency_multiplier)
        first_audio_t = float(first_audio["t_wall_sec"]) if first_audio else None
        return {
            "input": str(self.inp),
            "direction": self.direction,
            "latency_multiplier": self.latency_multiplier,
            "source_dur_sec": round(self.source_dur_sec, 4),
            "out_dur_sec": round(float(len(audio) / SAMPLE_RATE_OUT), 4) if len(audio) else 0.0,
            "pred_text": "".join(self._texts),
            "n_input_frames": sum(1 for e in self._events if e["event"] == "input_pcm_sent"),
            "n_audio_chunks": sum(1 for e in self._events if e["event"] == "audio_recv"),
            "n_text_chunks": sum(1 for e in self._events if e["event"] == "text_recv"),
            "t_first_audio_sec": first_audio_t,
            "t_first_text_sec": float(first_text["t_wall_sec"]) if first_text else None,
            "t_done_sec": float(done["t_wall_sec"]) if done else (
                self._events[-1]["t_wall_sec"] if self._events else None
            ),
            "first_chunk_policy_src_sec": first_chunk_ready_src,
            "first_audio_lag_after_chunk_ready": (
                round(first_audio_t - first_chunk_ready_src, 4) if first_audio_t is not None else None
            ),
            "session_rtf": (
                round(float(done["t_wall_sec"]) / self.source_dur_sec, 4)
                if done and self.source_dur_sec > 0
                else None
            ),
            "server_compute_secs": [
                e["payload"]["compute_sec"]
                for e in self._events
                if e["event"] == "json_recv"
                and isinstance(e.get("payload"), dict)
                and e["payload"].get("event") == "server_emit"
                and "compute_sec" in e["payload"]
            ],
        }

    async def _run(self) -> dict[str, Any]:
        async with websockets.connect(self.url, max_size=None) as ws:
            await self._handshake(ws)
            await asyncio.gather(self._send(ws), self._recv(ws))

        self.out.parent.mkdir(parents=True, exist_ok=True)
        if self._recv_pcm:
            audio = np.concatenate(self._recv_pcm)
        else:
            audio = np.zeros(0, dtype=np.float32)
        sf.write(str(self.out), audio, SAMPLE_RATE_OUT, subtype="PCM_16")
        summary = self._summary(audio)
        self._log("session_done", **{k: v for k, v in summary.items() if k != "pred_text"})
        if self.timeline_path is not None:
            self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"summary": summary, "events": self._events}
            self.timeline_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[TIMELINE] {self.timeline_path}", flush=True)
        print(f"[DONE] text={summary['pred_text']!r}", flush=True)
        print(
            f"[DONE] wrote {self.out} ({summary['out_dur_sec']:.2f}s @ {SAMPLE_RATE_OUT} Hz) "
            f"first_audio={summary['t_first_audio_sec']} "
            f"lag_after_L={summary['first_audio_lag_after_chunk_ready']} "
            f"session_rtf={summary['session_rtf']}",
            flush=True,
        )
        return summary

    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run())


def parse_args() -> argparse.Namespace:
    repo = REPO_ROOT
    default_wav = (
        repo
        / "external"
        / "RealSI"
        / "data"
        / "en2zh"
        / "simuleval_sentence_level"
        / "wavs"
        / "en2zh-01-tech__seg0000.wav"
    )
    parser = argparse.ArgumentParser(description="OmniTalker S2ST WebSocket file client")
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765")
    parser.add_argument("--input", type=Path, default=default_wav)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "outputs" / "demo" / "file_client_out.wav",
    )
    parser.add_argument("--direction", choices=["en2zh", "zh2en"], default="en2zh")
    parser.add_argument("--latency", type=int, default=2)
    parser.add_argument("--frame-ms", type=int, default=1000)
    parser.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="Write JSON timeline (send/recv wall-clock events) to this path",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    client = OmniTalkerFileClient(
        ws_url=args.url,
        inp=args.input,
        out=args.output,
        direction=args.direction,
        latency_multiplier=args.latency,
        frame_ms=args.frame_ms,
        timeline_path=args.timeline,
    )
    client.run()


if __name__ == "__main__":
    main()
