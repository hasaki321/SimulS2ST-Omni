#!/usr/bin/env python3
"""WebSocket server wrapping the release OmniTalker streaming S2ST agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from argparse import Namespace
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pedalboard import time_stretch as pedalboard_time_stretch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "external" / "SimulEval",
    REPO_ROOT / "src" / "train",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from simuleval.data.segments import EmptySegment, SpeechSegment  # noqa: E402

from demo.protocol import (  # noqa: E402
    SAMPLE_RATE_IN,
    SAMPLE_RATE_OUT,
    TYPE_EOS,
    TYPE_JSON,
    TYPE_PCM,
    pack_json,
    pack_pcm_i16,
    pack_text,
    unpack_frame,
    unpack_json,
)
logger = logging.getLogger("demo.server")

DIRECTION_LANGS = {
    "en2zh": ("English", "Chinese"),
    "zh2en": ("Chinese", "English"),
}


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _build_agent_args(
    cli: argparse.Namespace,
    output_dir: Path,
) -> Namespace:
    source_lang, target_lang = DIRECTION_LANGS[cli.default_direction]
    return Namespace(
        model_name_or_path=cli.model,
        checkpoint_path=cli.checkpoint,
        adapter_name_or_path=None,
        device=cli.device,
        dtype=cli.dtype,
        source_lang=source_lang,
        target_lang=target_lang,
        latency_multiplier=cli.default_latency,
        min_start_sec=1.0,
        min_final_chunk_sec=0.32,
        thinker_max_new_tokens=256,
        thinker_do_sample=False,
        thinker_temperature=1.0,
        thinker_top_p=0.8,
        thinker_top_k=20,
        thinker_num_beams=cli.thinker_num_beams,
        thinker_repetition_penalty=1.2,
        thinker_no_repeat_ngram_size=5,
        no_thinker_kv_cache=False,
        talker_max_new_tokens=500,
        talker_do_sample=False,
        talker_temperature=1.0,
        talker_top_p=0.8,
        talker_top_k=20,
        talker_repetition_penalty=1.4,
        talker_no_repeat_ngram_size=5,
        no_talker_kv_cache=False,
        enable_wait_silence_decode=False,
        history_window_turns=28,
        history_overlap_turns=16,
        prompt_source_chunks=2,
        prompt_generated_chunks=1,
        prompt_source_sec=4.0,
        prompt_generated_sec=2.0,
        voicebox_path=None,
        vocos_path=None,
        voicebox_config_path=None,
        codec_model_path=None,
        codec_stats_path=None,
        use_prompt_mel=True,
        enable_history_window=True,
        output=str(output_dir),
        source="",
    )


def _pcm16_bytes_to_float_list(payload: bytes) -> list[float]:
    pcm = np.frombuffer(payload, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).tolist()


def _float_audio_to_pcm16_bytes(samples: list[float] | np.ndarray) -> bytes:
    arr = np.asarray(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


# Send-window length alignment: never shorten more than this (duration floor).
MIN_DURATION_RATIO = 0.7
# Sliding average window for applied duration ratio (smooths per-chunk jumps).
SPEED_SMOOTH_WINDOW = 5


def _time_stretch_pcm16_duration(pcm: bytes, ratio: float, sample_rate: int) -> bytes:
    """Pitch-preserving time-stretch: duration' = duration * ratio, ratio in [0.7, 1].

    Uses Spotify Pedalboard (Signalsmith / high_quality stretcher). Much cleaner on
    speech than librosa phase-vocoder or torchaudio.speed (which shifts pitch).

    Note: pedalboard's stretch_factor behaves as a *speed* factor here:
    duration_out ≈ duration_in / stretch_factor, so we pass 1/ratio.
    """
    if ratio >= 0.999:
        return pcm
    if ratio < MIN_DURATION_RATIO:
        raise ValueError(f"duration ratio {ratio} < floor {MIN_DURATION_RATIO}")
    wav = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    # Pedalboard expects (num_channels, num_samples).
    stretched = pedalboard_time_stretch(
        wav[None, :],
        float(sample_rate),
        stretch_factor=1.0 / ratio,
        high_quality=True,
        transient_mode="crisp",
    )
    arr = np.clip(np.asarray(stretched[0], dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


class SessionTimingLog:
    """Wall-clock event log for one WebSocket session (console + jsonl)."""

    def __init__(self, session_id: int, log_path: Path):
        self.session_id = session_id
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self._fh = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": _utc_ts(),
            "t_rel_ms": round((time.perf_counter() - self.t0) * 1000.0, 1),
            "session_id": self.session_id,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        # Compact console line for live eyeballing.
        extras = " ".join(
            f"{k}={v}"
            for k, v in fields.items()
            if k not in {"modules_ms"} and not isinstance(v, (dict, list))
        )
        logger.info("[timing] sid=%s +%.0fms %s %s", self.session_id, record["t_rel_ms"], event, extras)
        if "modules_ms" in fields:
            mods = fields["modules_ms"]
            ranked = sorted(mods.items(), key=lambda kv: kv[1], reverse=True)
            top = " | ".join(f"{k}={v:.0f}ms" for k, v in ranked[:6])
            logger.info("[timing] sid=%s modules: %s", self.session_id, top)
        return record


class OmniTalkerSession:
    """One WebSocket connection → one agent reset + streaming policy drain."""

    def __init__(self, agent: Any, session_id: int, timing: SessionTimingLog):
        self.agent = agent
        self.session_id = session_id
        self.timing = timing
        self._text_cursor = 0
        self._configured = False
        self._pcm_frames = 0
        self._pcm_samples_total = 0
        self._last_policy_step = -1
        # Send-window A/V length controller (seconds).
        self._out_emitted_sec = 0.0
        self._wait_credit_sec = 0.0
        self._ratio_hist: deque[float] = deque(maxlen=SPEED_SMOOTH_WINDOW)

    def configure(self, config: dict[str, Any]) -> dict[str, Any]:
        direction = str(config["direction"] if "direction" in config else "en2zh")
        if direction not in DIRECTION_LANGS:
            raise ValueError(f"unsupported direction={direction}")
        latency = int(
            config["latency_multiplier"]
            if "latency_multiplier" in config
            else self.agent.latency_multiplier
        )
        if latency < 1:
            raise ValueError(f"latency_multiplier must be >= 1, got {latency}")
        sample_rate = int(config["sample_rate"] if "sample_rate" in config else SAMPLE_RATE_IN)
        if sample_rate != SAMPLE_RATE_IN:
            raise ValueError(f"only sample_rate={SAMPLE_RATE_IN} supported, got {sample_rate}")

        source_lang, target_lang = DIRECTION_LANGS[direction]
        self.agent.source_lang = source_lang
        self.agent.target_lang = target_lang
        self.agent.latency_multiplier = latency
        self.agent.chunk_duration = float(latency)
        self.agent.history_window_turns = max(0, 28 // max(1, latency))
        self.agent.history_overlap_turns = max(0, 16 // max(1, latency))
        self.agent.system_instruction = self.agent._build_system_instruction()
        self.agent._system_token_ids = self.agent.tokenizer(
            self.agent.system_instruction, add_special_tokens=False
        ).input_ids
        self.agent.reset()
        self.agent.last_policy_timing = None
        self._text_cursor = 0
        self._configured = True
        self._pcm_frames = 0
        self._pcm_samples_total = 0
        self._last_policy_step = -1
        self._out_emitted_sec = 0.0
        self._wait_credit_sec = 0.0
        self._ratio_hist.clear()
        ready = {
            "ok": True,
            "direction": direction,
            "latency_multiplier": latency,
            "chunk_sec": float(latency),
            "in_sample_rate": SAMPLE_RATE_IN,
            "out_sample_rate": SAMPLE_RATE_OUT,
            "session_id": self.session_id,
            "min_duration_ratio": MIN_DURATION_RATIO,
            "speed_smooth_window": SPEED_SMOOTH_WINDOW,
            "time_stretch": "pedalboard.time_stretch",
        }
        self.timing.emit("session_configured", **ready)
        return ready

    def _speech_segment(self, samples: list[float], finished: bool) -> SpeechSegment:
        return SpeechSegment(
            content=samples,
            finished=finished,
            sample_rate=SAMPLE_RATE_IN,
            config={"instance_index": self.session_id},
        )

    def _source_sec(self) -> float:
        states = self.agent.states
        if states.source_sample_rate == 0:
            return 0.0
        return len(states.source) / float(states.source_sample_rate)

    def _known_input_sec(self) -> float:
        """Source seconds already consumed by policy (known input timeline)."""
        states = self.agent.states
        if states.source_sample_rate == 0:
            return 0.0
        return float(states.processed_samples) / float(states.source_sample_rate)

    def _note_policy_wait(self, policy: dict[str, Any] | None) -> None:
        if policy is None:
            return
        if policy["is_wait"]:
            self._wait_credit_sec += float(policy["chunk_sec"])

    def _target_duration_ratio(self, raw_sec: float) -> float:
        known_in = self._known_input_sec()
        wait_credit = self._wait_credit_sec
        budget = known_in + wait_credit
        projected = self._out_emitted_sec + raw_sec
        ratio = 1.0
        if raw_sec > 0.0 and projected > budget:
            allowed_new = known_in - self._out_emitted_sec
            if allowed_new <= 0.0:
                ratio = MIN_DURATION_RATIO
            else:
                ratio = allowed_new / raw_sec
                if ratio > 1.0:
                    ratio = 1.0
                if ratio < MIN_DURATION_RATIO:
                    ratio = MIN_DURATION_RATIO
        return ratio

    def _smooth_duration_ratio(self, target_ratio: float) -> float:
        """Sliding-window mean of recent target ratios → applied tempo."""
        self._ratio_hist.append(float(target_ratio))
        applied = float(sum(self._ratio_hist) / len(self._ratio_hist))
        if applied > 1.0:
            applied = 1.0
        if applied < MIN_DURATION_RATIO:
            applied = MIN_DURATION_RATIO
        return applied

    def align_outgoing_pcm(self, pcm: bytes) -> tuple[bytes, dict[str, Any]]:
        """Pitch-preserving tempo align if cumulative out would exceed known_in + wait.

        Target: keep known output duration as close as possible to known input.
        Duration ratio floor: 0.7x. Applied ratio is a sliding average over recent windows.
        """
        raw_sec = len(pcm) / (2.0 * SAMPLE_RATE_OUT)
        known_in = self._known_input_sec()
        wait_credit = self._wait_credit_sec
        budget = known_in + wait_credit
        projected = self._out_emitted_sec + raw_sec

        target_ratio = self._target_duration_ratio(raw_sec)
        applied_ratio = self._smooth_duration_ratio(target_ratio)

        out_pcm = (
            _time_stretch_pcm16_duration(pcm, applied_ratio, SAMPLE_RATE_OUT)
            if applied_ratio < 0.999
            else pcm
        )
        out_sec = len(out_pcm) / (2.0 * SAMPLE_RATE_OUT)
        self._out_emitted_sec += out_sec
        meta = {
            "raw_out_sec": round(raw_sec, 4),
            "sent_out_sec": round(out_sec, 4),
            "target_duration_ratio": round(target_ratio, 4),
            "duration_ratio": round(applied_ratio, 4),
            "ratio_hist": [round(x, 4) for x in self._ratio_hist],
            "known_input_sec": round(known_in, 4),
            "wait_credit_sec": round(wait_credit, 4),
            "budget_sec": round(budget, 4),
            "out_emitted_sec": round(self._out_emitted_sec, 4),
            "projected_raw_sec": round(projected, 4),
            "compressed": bool(applied_ratio < 0.999),
            "stretch": "pedalboard_hq",
        }
        return out_pcm, meta

    def _drain(self) -> list[tuple[str | None, bytes | None, bool]]:
        """Run policy until ReadAction (or finished). Returns (text, pcm24, finished) events."""
        events: list[tuple[str | None, bytes | None, bool]] = []
        while True:
            out = self.agent.pop()
            states = self.agent.states
            new_texts = states.pred_text_chunks[self._text_cursor :]
            self._text_cursor = len(states.pred_text_chunks)
            text_joined = "".join(new_texts) if new_texts else None

            if isinstance(out, EmptySegment) or out.is_empty:
                if text_joined:
                    events.append((text_joined, None, False))
                break

            pcm = None
            if out.content:
                pcm = _float_audio_to_pcm16_bytes(out.content)
            if text_joined or pcm is not None or out.finished:
                events.append((text_joined, pcm, bool(out.finished)))
            if out.finished:
                break
        return events

    def push_pcm(self, payload: bytes) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("session not configured; send JSON config first")
        samples = _pcm16_bytes_to_float_list(payload)
        frame_sec = len(samples) / float(SAMPLE_RATE_IN)
        self._pcm_frames += 1
        self._pcm_samples_total += len(samples)
        source_before = self._source_sec()
        self.timing.emit(
            "pcm_recv",
            frame_idx=self._pcm_frames,
            frame_sec=round(frame_sec, 4),
            frame_bytes=len(payload),
            source_buffered_sec_before=round(source_before, 4),
        )

        t0 = time.perf_counter()
        self.agent.last_policy_timing = None
        self.agent.push(self._speech_segment(samples, finished=False))
        events = self._drain()
        compute_sec = time.perf_counter() - t0
        source_after = self._source_sec()

        policy = self.agent.last_policy_timing
        ran_policy = policy is not None and int(policy["step_index"]) > self._last_policy_step
        if ran_policy:
            self._last_policy_step = int(policy["step_index"])
            self._note_policy_wait(policy)
            self.timing.emit(
                "policy_chunk",
                step_index=policy["step_index"],
                chunk_sec=policy["chunk_sec"],
                accounted_ms=policy["accounted_ms"],
                total_ms=policy["total_ms"],
                rtf=policy["rtf"],
                wall_rtf=policy["wall_rtf"],
                is_wait=policy["is_wait"],
                talker_new_codes=policy["talker_new_codes"],
                out_audio_sec=policy["out_audio_sec"],
                pred_text=policy["pred_text"],
                modules_ms=policy["modules_ms"],
                push_drain_ms=round(compute_sec * 1000.0, 1),
                source_buffered_sec=round(source_after, 4),
                known_input_sec=round(self._known_input_sec(), 4),
                wait_credit_sec=round(self._wait_credit_sec, 4),
            )
        else:
            self.timing.emit(
                "policy_read",
                push_drain_ms=round(compute_sec * 1000.0, 1),
                source_buffered_sec=round(source_after, 4),
                emitted_events=len(events),
            )

        return {
            "events": events,
            "compute_sec": compute_sec,
            "source_sec": source_after,
            "policy": policy if ran_policy else None,
        }

    def push_eos(self) -> dict[str, Any]:
        if not self._configured:
            raise RuntimeError("session not configured; send JSON config first")
        self.timing.emit("eos_recv", source_buffered_sec=round(self._source_sec(), 4))
        t0 = time.perf_counter()
        self.agent.last_policy_timing = None
        self.agent.push(self._speech_segment([], finished=True))
        events = self._drain()
        compute_sec = time.perf_counter() - t0
        source_after = self._source_sec()
        policy = self.agent.last_policy_timing
        ran_policy = policy is not None and int(policy["step_index"]) > self._last_policy_step
        if ran_policy:
            self._last_policy_step = int(policy["step_index"])
            self._note_policy_wait(policy)
            self.timing.emit(
                "policy_chunk",
                step_index=policy["step_index"],
                chunk_sec=policy["chunk_sec"],
                accounted_ms=policy["accounted_ms"],
                total_ms=policy["total_ms"],
                rtf=policy["rtf"],
                wall_rtf=policy["wall_rtf"],
                is_wait=policy["is_wait"],
                talker_new_codes=policy["talker_new_codes"],
                out_audio_sec=policy["out_audio_sec"],
                pred_text=policy["pred_text"],
                modules_ms=policy["modules_ms"],
                push_drain_ms=round(compute_sec * 1000.0, 1),
                source_buffered_sec=round(source_after, 4),
                known_input_sec=round(self._known_input_sec(), 4),
                wait_credit_sec=round(self._wait_credit_sec, 4),
                from_eos=True,
            )
        self.timing.emit(
            "eos_done",
            push_drain_ms=round(compute_sec * 1000.0, 1),
            source_buffered_sec=round(source_after, 4),
            emitted_events=len(events),
            out_emitted_sec=round(self._out_emitted_sec, 4),
        )
        return {
            "events": events,
            "compute_sec": compute_sec,
            "source_sec": source_after,
            "policy": policy if ran_policy else None,
        }


async def _send_events(
    websocket,
    session: OmniTalkerSession,
    result: dict[str, Any],
) -> bool:
    """Send drain events. Returns True if session finished."""
    finished = False
    events = result["events"]
    compute_sec = float(result["compute_sec"])
    source_sec = float(result["source_sec"])
    policy = result["policy"]
    timing = session.timing

    for text, pcm, done in events:
        align_meta: dict[str, Any] | None = None
        if pcm:
            pcm, align_meta = session.align_outgoing_pcm(pcm)
        out_chunk_sec = round(len(pcm) / (2 * SAMPLE_RATE_OUT), 4) if pcm else 0.0
        meta: dict[str, Any] = {
            "event": "server_emit",
            "compute_sec": round(compute_sec, 4),
            "source_buffered_sec": round(source_sec, 4),
            "known_input_sec": round(session._known_input_sec(), 4),
            "wait_credit_sec": round(session._wait_credit_sec, 4),
            "out_emitted_sec": round(session._out_emitted_sec, 4),
        }
        if text:
            meta["text"] = text
        if pcm:
            meta["out_chunk_sec"] = out_chunk_sec
        if align_meta is not None:
            meta["align"] = align_meta
        if policy is not None:
            meta["policy_rtf"] = policy["rtf"]
            meta["policy_accounted_ms"] = policy["accounted_ms"]
            meta["policy_modules_ms"] = policy["modules_ms"]
            meta["policy_step_index"] = policy["step_index"]
        if text or pcm:
            await websocket.send(pack_json(meta))
            timing.emit(
                "server_emit",
                has_text=bool(text),
                has_pcm=bool(pcm),
                text=(text[:80] if text else ""),
                out_chunk_sec=out_chunk_sec,
                compute_sec=round(compute_sec, 4),
                source_buffered_sec=round(source_sec, 4),
                policy_rtf=(policy["rtf"] if policy is not None else None),
                policy_accounted_ms=(policy["accounted_ms"] if policy is not None else None),
                duration_ratio=(align_meta["duration_ratio"] if align_meta is not None else None),
                target_duration_ratio=(
                    align_meta["target_duration_ratio"] if align_meta is not None else None
                ),
                raw_out_sec=(align_meta["raw_out_sec"] if align_meta is not None else None),
                compressed=(align_meta["compressed"] if align_meta is not None else False),
                stretch=(align_meta["stretch"] if align_meta is not None else None),
                out_emitted_sec=round(session._out_emitted_sec, 4),
                known_input_sec=round(session._known_input_sec(), 4),
                wait_credit_sec=round(session._wait_credit_sec, 4),
            )
        if text:
            await websocket.send(pack_text(text))
        if pcm:
            await websocket.send(pack_pcm_i16(pcm))
        finished = finished or done
    return finished


async def handle_connection(
    websocket,
    agent: Any,
    session_counter: list[int],
    gpu_lock: asyncio.Lock,
    timing_dir: Path,
):
    session_counter[0] += 1
    session_id = session_counter[0]
    peer = getattr(websocket, "remote_address", None)
    timing_path = timing_dir / f"session_{session_id}_{int(time.time())}.jsonl"
    timing = SessionTimingLog(session_id, timing_path)
    timing.emit("client_connected", peer=str(peer), timing_log=str(timing_path))
    logger.info("client connected session=%s peer=%s timing=%s", session_id, peer, timing_path)
    session = OmniTalkerSession(agent, session_id, timing)

    try:
        async for message in websocket:
            if isinstance(message, str):
                message = message.encode("utf-8")
            frame_type, payload = unpack_frame(message)

            if frame_type == TYPE_JSON:
                config = unpack_json(payload)
                async with gpu_lock:
                    ready = session.configure(config)
                await websocket.send(pack_json(ready))
                logger.info("session=%s configured %s", session_id, ready)
                continue

            if frame_type == TYPE_PCM:
                async with gpu_lock:
                    result = await asyncio.to_thread(session.push_pcm, payload)
                if await _send_events(websocket, session, result):
                    await websocket.send(pack_json({"done": True}))
                    timing.emit("session_done", reason="finished_after_pcm")
                    break
                continue

            if frame_type == TYPE_EOS:
                async with gpu_lock:
                    result = await asyncio.to_thread(session.push_eos)
                await _send_events(websocket, session, result)
                await websocket.send(pack_json({"done": True}))
                timing.emit("session_done", reason="eos")
                break

            await websocket.send(pack_json({"ok": False, "error": f"unknown frame type {frame_type}"}))
    except Exception as exc:
        logger.exception("session=%s error", session_id)
        timing.emit("session_error", error=str(exc))
        try:
            await websocket.send(pack_json({"ok": False, "error": str(exc)}))
        except Exception:
            pass
    finally:
        timing.emit("client_disconnected")
        timing.close()
        logger.info("client disconnected session=%s", session_id)


async def run_server(cli: argparse.Namespace) -> None:
    from src.agents.simuleval_omni_talker_s2st_agent import OmniTalkerStreamingS2STAgent
    from websockets.asyncio.server import serve

    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_dir = Path(cli.timing_dir)
    timing_dir.mkdir(parents=True, exist_ok=True)
    args = _build_agent_args(cli, output_dir)
    logger.info(
        "Loading OmniTalker agent (device=%s, default_L=%s) ...",
        cli.device,
        cli.default_latency,
    )
    agent = OmniTalkerStreamingS2STAgent(args)
    logger.info(
        "Agent ready. Listening on ws://%s:%s | timing_dir=%s",
        cli.host,
        cli.port,
        timing_dir,
    )

    session_counter = [0]
    gpu_lock = asyncio.Lock()

    async def handler(websocket):
        await handle_connection(websocket, agent, session_counter, gpu_lock, timing_dir)

    async with serve(handler, cli.host, cli.port, max_size=None, ping_interval=20):
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    repo = REPO_ROOT
    model_root = Path(
        os.environ.get("SIMULS2ST_MODEL_ROOT", repo / "models" / "SimulS2ST-Omni")
    ).expanduser()
    parser = argparse.ArgumentParser(description="OmniTalker streaming S2ST WebSocket server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--model",
        type=str,
        default=str(model_root / "offline"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(model_root / "simuls2st_adapter"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--default-direction", choices=sorted(DIRECTION_LANGS), default="en2zh")
    parser.add_argument("--default-latency", type=int, default=2)
    parser.add_argument("--thinker-num-beams", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo / "outputs" / "demo" / "server_sidecar"),
    )
    parser.add_argument(
        "--timing-dir",
        type=str,
        default=str(repo / "outputs" / "demo" / "timing"),
        help="Per-session wall-clock event jsonl directory",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    logging.basicConfig(
        level=getattr(logging, cli.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    asyncio.run(run_server(cli))


if __name__ == "__main__":
    main()
