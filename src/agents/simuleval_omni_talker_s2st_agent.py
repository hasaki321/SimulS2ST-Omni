#!/usr/bin/env python3
"""SimulEval agent for direct OmniTalker streaming S2ST inference."""

from __future__ import annotations

import json
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.data.segments import SpeechSegment
from simuleval.utils import entrypoint


def _prepare_import_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src" / "train"):
        path_str = str(path)
        if path_str not in sys.path and os.path.isdir(path_str):
            sys.path.insert(0, path_str)
    return repo_root


REPO_ROOT = _prepare_import_paths()

from src.train.lang_utils import normalize_lang
from src.train.modeling_dual_head import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IDLE_TOKEN,
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
)
from src.train.prompt_formats import TASK_STREAMING_S2S_TRANSLATE, build_audio_span, build_system_prompt


SAMPLE_RATE_16K = 16_000
SAMPLE_RATE_24K = 24_000
SPEECH_SEGMENT_SIZE = 25
SEMANTIC_CODE_VOCAB_SIZE = 16384


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor) -> torch.LongTensor:
    input_lengths = (input_lengths - 1) // 2 + 1
    input_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths.to(torch.long)


def _strip(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "")).strip()


def _normalize_pred(text: str) -> str:
    text = text or ""
    for token in (DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_IDLE_TOKEN):
        text = text.replace(token, "")
    text = re.sub(r"<\|code_\d+\|>", "", text)
    return _strip(text)


def _is_wait(text: str) -> bool:
    raw = (text or "").strip()
    return raw == DEFAULT_IDLE_TOKEN or _normalize_pred(raw) == ""


def _maybe_add_streaming_word_boundary(previous_text: str, current_text: str, lang: str) -> str:
    if not previous_text or not current_text:
        return current_text
    if normalize_lang(lang) != "English":
        return current_text
    if previous_text[-1].isalnum() and current_text[0].isalnum():
        return " " + current_text
    return current_text


def _word_latency_tokens(text: str) -> list[str]:
    return text.split()


class OmniTalkerS2STStates(AgentStates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_history: list[dict] = []
        self.accumulated_audio: list[np.ndarray] = []
        self.processed_samples: int = 0
        self.total_audio_seconds: float = 0.0
        self.current_instance_index: int = -1
        self.accumulated_output: list[str] = []
        self.pred_text_chunks: list[str] = []
        self.pred_text_saved: bool = False
        self.step_event_index: int = 0
        self.thinker_past_key_values: Any = None
        self.talker_past_key_values: Any = None

    def reset(self):
        super().reset()
        self.turn_history = []
        self.accumulated_audio = []
        self.processed_samples = 0
        self.total_audio_seconds = 0.0
        self.current_instance_index = -1
        self.accumulated_output = []
        self.pred_text_chunks = []
        self.pred_text_saved = False
        self.step_event_index = 0
        self.thinker_past_key_values = None
        self.talker_past_key_values = None

    def update_source(self, segment):
        super().update_source(segment)
        # Some internal SimulEval forks attach an instance index to each segment;
        # the official upstream does not. The index is only sidecar metadata, so
        # keep a stable fallback instead of making inference depend on that fork.
        self.current_instance_index = int(self.config.get("instance_index", 0))


@entrypoint
class OmniTalkerStreamingS2STAgent(SpeechToSpeechAgent):
    """Direct thinker+talker streaming S2ST agent for the simultaneous adapter."""

    def __init__(self, args: Namespace):
        super().__init__(args)
        model_root = Path(
            os.environ.get("SIMULS2ST_MODEL_ROOT", REPO_ROOT / "models" / "SimulS2ST-Omni")
        ).expanduser()
        self.model_name_or_path = args.model_name_or_path
        self.checkpoint_path = (
            args.checkpoint_path
            or getattr(args, "adapter_name_or_path", None)
            or args.model_name_or_path
        )
        self.device = args.device
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        self.source_lang = normalize_lang(args.source_lang)
        self.target_lang = normalize_lang(args.target_lang)
        self.latency_multiplier = int(args.latency_multiplier)
        self.chunk_duration = float(args.latency_multiplier)
        self.min_start_sec = float(args.min_start_sec)
        self.min_final_chunk_sec = float(args.min_final_chunk_sec)
        # Thinker generation params (separate from talker; defaults match cvss eval)
        self.thinker_max_new_tokens = int(args.thinker_max_new_tokens)
        self.thinker_do_sample = bool(args.thinker_do_sample)
        self.thinker_temperature = float(args.thinker_temperature)
        self.thinker_top_k = int(args.thinker_top_k)
        self.thinker_top_p = float(args.thinker_top_p)
        self.thinker_num_beams = int(args.thinker_num_beams)
        self.thinker_repetition_penalty = float(args.thinker_repetition_penalty)
        self.thinker_no_repeat_ngram_size = int(args.thinker_no_repeat_ngram_size)
        self.no_thinker_kv_cache = bool(args.no_thinker_kv_cache)
        # Talker generation params
        self.talker_max_new_tokens = int(args.talker_max_new_tokens)
        self.talker_do_sample = bool(args.talker_do_sample)
        self.talker_temperature = float(args.talker_temperature)
        self.talker_top_k = int(args.talker_top_k)
        self.talker_top_p = float(args.talker_top_p)
        self.talker_repetition_penalty = float(args.talker_repetition_penalty)
        self.talker_no_repeat_ngram_size = int(args.talker_no_repeat_ngram_size)
        self.no_talker_kv_cache = bool(args.no_talker_kv_cache)
        self.enable_wait_silence_decode = bool(args.enable_wait_silence_decode)
        self.use_prompt_mel = bool(args.use_prompt_mel)
        self.prompt_source_chunks = int(args.prompt_source_chunks)
        self.prompt_generated_chunks = int(args.prompt_generated_chunks)
        self.prompt_source_sec = float(args.prompt_source_sec)
        self.prompt_generated_sec = float(args.prompt_generated_sec)
        self.enable_history_window = bool(args.enable_history_window)
        self.history_window_turns = max(0, int(args.history_window_turns) // max(1, self.latency_multiplier))
        self.history_overlap_turns = max(0, int(args.history_overlap_turns) // max(1, self.latency_multiplier))

        voicebox_path = (
            args.voicebox_path
            or os.environ.get("SIMULS2ST_VOICEBOX_PATH")
            or str(model_root / "voicebox" / "voicebox.safetensors")
        )
        vocos_path = (
            args.vocos_path
            or os.environ.get("SIMULS2ST_VOCOS_PATH")
            or str(model_root / "voicebox" / "vocos.safetensors")
        )
        voicebox_config_path = (
            args.voicebox_config_path
            or os.environ.get("SIMULS2ST_VOICEBOX_CONFIG")
            or str(model_root / "voicebox" / "voicebox_config.json")
        )
        codec_model_path = (
            args.codec_model_path
            or os.environ.get("SIMULS2ST_CODEC_MODEL_PATH")
            or str(model_root / "dualcodec" / "dualcodec.safetensors")
        )
        codec_stats_path = (
            args.codec_stats_path
            or os.environ.get("SIMULS2ST_CODEC_STATS_PATH")
            or str(model_root / "dualcodec" / "w2v_bert_stats.pt")
        )

        os.environ.setdefault("SIMULS2ST_W2V_BERT_PATH", str(model_root / "w2v"))

        from src.inference.infer_omni_talker_tts import OmniTalkerSemanticTTSInference

        self.inference = OmniTalkerSemanticTTSInference.from_pretrained(
            checkpoint_path=self.checkpoint_path,
            omni_model_path=self.model_name_or_path,
            voicebox_path=voicebox_path,
            vocos_path=vocos_path,
            voicebox_config_path=voicebox_config_path,
            codec_model_path=codec_model_path,
            codec_stats_path=codec_stats_path,
            device=self.device,
            dtype=self.dtype,
            n_audio_codes=SEMANTIC_CODE_VOCAB_SIZE,
        )
        self.processor = self.inference.processor
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.model_device = self.inference.device
        self.model_dtype = self.inference.dtype
        self._silence_codes_cache: dict[float, torch.Tensor] = {}
        self.system_instruction = self._build_system_instruction()
        self._system_token_ids: list[int] = self.tokenizer(
            self.system_instruction, add_special_tokens=False
        ).input_ids
        self._asst_suffix_token_ids: list[int] = self.tokenizer(
            f"{DEFAULT_EOS_TOKEN}\n", add_special_tokens=False
        ).input_ids
        self._asst_prefix_text = f"{DEFAULT_BOS_TOKEN}assistant\n"
        self.text_sidecar_path = Path(args.output) / "text_predictions.jsonl"
        if self.text_sidecar_path.exists():
            self.text_sidecar_path.unlink()
        self.text_step_events_path = Path(args.output) / "text_step_events.jsonl"
        if self.text_step_events_path.exists():
            self.text_step_events_path.unlink()
        self._sidecar_record_id = 0
        self._source_entries = self._load_source_entries(getattr(args, "source", ""))

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--checkpoint-path", type=str, default=None)
        parser.add_argument("--adapter-name-or-path", type=str, default=None)
        parser.add_argument("--model-name-or-path", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
        parser.add_argument("--source-lang", type=str, default="English")
        parser.add_argument("--target-lang", type=str, default="Chinese")
        parser.add_argument("--latency-multiplier", type=int, default=1)
        parser.add_argument("--min-start-sec", type=float, default=1.0)
        parser.add_argument("--min-final-chunk-sec", type=float, default=0.32)
        # Thinker generation (defaults align with cvss eval: greedy, mild rep penalty)
        parser.add_argument("--thinker-max-new-tokens", type=int, default=50)
        parser.add_argument(
            "--thinker-do-sample", action="store_true", default=False
        )
        parser.add_argument(
            "--thinker-no-sample", action="store_false", dest="thinker_do_sample"
        )
        parser.add_argument("--thinker-temperature", type=float, default=1.0)
        parser.add_argument("--thinker-top-p", type=float, default=0.8)
        parser.add_argument("--thinker-top-k", type=int, default=20)
        parser.add_argument("--thinker-num-beams", type=int, default=1)
        parser.add_argument("--thinker-repetition-penalty", type=float, default=1.1)
        parser.add_argument("--thinker-no-repeat-ngram-size", type=int, default=0)
        parser.add_argument("--no-thinker-kv-cache", action="store_true")
        # Talker generation (defaults align with cvss eval: greedy, stronger rep penalty + ngram block)
        parser.add_argument("--talker-max-new-tokens", type=int, default=125)
        parser.add_argument(
            "--talker-do-sample", action="store_true", default=False
        )
        parser.add_argument(
            "--talker-no-sample", action="store_false", dest="talker_do_sample"
        )
        parser.add_argument("--talker-temperature", type=float, default=1.0)
        parser.add_argument("--talker-top-p", type=float, default=0.8)
        parser.add_argument("--talker-top-k", type=int, default=20)
        parser.add_argument("--talker-repetition-penalty", type=float, default=1.4)
        parser.add_argument("--talker-no-repeat-ngram-size", type=int, default=0)
        # Default to the full-prefix path, which exactly mirrors the training
        # collator. Cross-chunk talker cache is still experimental.
        # parser.add_argument("--enable-talker-kv-cache", action="store_false", dest="no_talker_kv_cache", default=True)
        parser.add_argument("--no-talker-kv-cache", action="store_true", dest="no_talker_kv_cache")
        parser.add_argument(
            "--enable-wait-silence-decode",
            action="store_true",
            default=False,
            help="On wait/idle chunks, synthesize cached silence codes through VoiceBox instead of emitting no audio.",
        )
        parser.add_argument("--history-window-turns", type=int, default=30)
        parser.add_argument("--history-overlap-turns", type=int, default=8)
        parser.add_argument("--prompt-source-chunks", type=int, default=2)
        parser.add_argument("--prompt-generated-chunks", type=int, default=1)
        parser.add_argument("--prompt-source-sec", type=float, default=4.0)
        parser.add_argument("--prompt-generated-sec", type=float, default=2.0)
        parser.add_argument("--voicebox-path", type=str, default=None)
        parser.add_argument("--vocos-path", type=str, default=None)
        parser.add_argument("--voicebox-config-path", type=str, default=None)
        parser.add_argument("--codec-model-path", type=str, default=None)
        parser.add_argument("--codec-stats-path", type=str, default=None)
        parser.add_argument("--disable-prompt-mel", action="store_false", default=True, dest="use_prompt_mel")
        parser.add_argument(
            "--disable-history-window",
            action="store_false",
            default=True,
            dest="enable_history_window",
        )

    def build_states(self) -> OmniTalkerS2STStates:
        return OmniTalkerS2STStates()

    def _load_source_entries(self, source_manifest: str) -> list[str]:
        if not source_manifest or not os.path.exists(source_manifest):
            return []
        with open(source_manifest, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _build_system_instruction(self) -> str:
        system_prompt = build_system_prompt(
            task_type=TASK_STREAMING_S2S_TRANSLATE,
            base_system_prompt=DEFAULT_SYSTEM_PROMPT,
            src_lang=self.source_lang,
            tgt_lang=self.target_lang,
            latency=self.latency_multiplier,
        )
        return f"{DEFAULT_BOS_TOKEN}system\n{system_prompt}{DEFAULT_EOS_TOKEN}\n"

    def _extract_audio_features(self, audio_np: np.ndarray):
        feats = self.feature_extractor(
            [audio_np],
            sampling_rate=SAMPLE_RATE_16K,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = torch.from_numpy(feats["input_features"]).to(self.model_device, self.model_dtype)
        feature_attention_mask = torch.from_numpy(feats["attention_mask"]).to(self.model_device, torch.long)
        n_frames = feature_attention_mask.sum(dim=1)
        speech_lens = _get_feat_extract_output_lengths(n_frames)
        n_audio_tokens = min(int(speech_lens[0].item()), SPEECH_SEGMENT_SIZE * self.latency_multiplier)
        return input_features, feature_attention_mask, max(1, n_audio_tokens)

    def _build_user_turn(self, n_audio_tokens: int) -> str:
        user_payload = self.source_lang + build_audio_span(int(n_audio_tokens))
        return f"{DEFAULT_BOS_TOKEN}user\n{user_payload}{DEFAULT_EOS_TOKEN}\n"

    def _build_assistant_turn(self, text: str) -> str:
        assistant_text = text if text else DEFAULT_IDLE_TOKEN
        return f"{DEFAULT_BOS_TOKEN}assistant\n{assistant_text}{DEFAULT_EOS_TOKEN}\n"

    def _build_prompt(self, states: OmniTalkerS2STStates, n_audio_tokens: int) -> str:
        prompt = self.system_instruction
        for turn in states.turn_history:
            prompt += self._build_user_turn(turn["n_audio_tokens"])
            prompt += self._build_assistant_turn(turn["text"])
        prompt += self._build_user_turn(n_audio_tokens)
        prompt += self._asst_prefix_text
        return prompt

    def _tokenize_history_chunk(self, n_audio_tokens: int, asst_text: str) -> tuple[list[int], tuple[int, int]]:
        """Tokenize a past chunk into training-format thinker tokens.

        Mirrors `DataCollatorForStreamingTTSTalkerDataset._tokenize_chunk`:
        chunk_ids = user_ids + asst_prefix_ids + asst_text_ids + asst_suffix_ids
        reply_local_span = [len(user_ids) + len(asst_prefix_ids), len(chunk_ids)]
        """
        user_text = self._build_user_turn(n_audio_tokens)
        asst_text = asst_text if asst_text else DEFAULT_IDLE_TOKEN
        user_ids = self.tokenizer(user_text, add_special_tokens=False).input_ids
        pre_ids = self.tokenizer(self._asst_prefix_text, add_special_tokens=False).input_ids
        full_asst_text = self._asst_prefix_text + asst_text + DEFAULT_EOS_TOKEN + "\n"
        full_asst_ids = self.tokenizer(full_asst_text, add_special_tokens=False).input_ids
        chunk_ids = list(user_ids) + list(full_asst_ids)
        sup_start = len(user_ids) + len(pre_ids)
        sup_end = len(user_ids) + len(full_asst_ids)
        return chunk_ids, (sup_start, sup_end)

    def _build_streaming_inputs(
        self,
        states: OmniTalkerS2STStates,
        n_audio_tokens_curr: int,
    ) -> tuple[list[dict], list[int]]:
        """Build (history_chunks, current_chunk_thinker_prefix_ids).

        - history_chunks: per-chunk dicts {thinker_token_ids, reply_local_span, codes}.
          Chunk 0's thinker_token_ids include the system prefix.
        - current_chunk_thinker_prefix_ids: chunk-N user turn + asst_prefix tokens
          (system prepended only when there is no history).
        """
        history_chunks: list[dict] = []
        for idx, turn in enumerate(states.turn_history):
            assert "codes" in turn, "turn_history entry missing codes"
            thinker_token_ids = list(turn["thinker_token_ids"])
            reply_local_span = turn["reply_local_span"]
            if idx == 0 and thinker_token_ids[: len(self._system_token_ids)] != self._system_token_ids:
                thinker_token_ids = list(self._system_token_ids) + thinker_token_ids
                reply_local_span = (
                    reply_local_span[0] + len(self._system_token_ids),
                    reply_local_span[1] + len(self._system_token_ids),
                )
            history_chunks.append(
                {
                    "thinker_token_ids": thinker_token_ids,
                    "reply_local_span": reply_local_span,
                    "codes": turn["codes"],
                }
            )

        user_text = self._build_user_turn(n_audio_tokens_curr)
        user_ids = self.tokenizer(user_text, add_special_tokens=False).input_ids
        pre_ids = self.tokenizer(self._asst_prefix_text, add_special_tokens=False).input_ids
        current_prefix_ids = list(user_ids) + list(pre_ids)
        if not history_chunks:
            current_prefix_ids = list(self._system_token_ids) + current_prefix_ids
        return history_chunks, current_prefix_ids

    def _collect_all_audio_features(self, states, cur_input_features, cur_feature_attention_mask):
        feat_list = [turn["input_features"] for turn in states.turn_history]
        mask_list = [turn["feature_attention_mask"] for turn in states.turn_history]
        feat_list.append(cur_input_features)
        mask_list.append(cur_feature_attention_mask)

        max_feat_len = max(feat.shape[1] for feat in feat_list)
        max_mask_len = max(mask.shape[1] for mask in mask_list)
        padded_feats, padded_masks = [], []
        for feat, mask in zip(feat_list, mask_list):
            if feat.shape[1] < max_feat_len:
                feat = torch.nn.functional.pad(feat, (0, 0, 0, max_feat_len - feat.shape[1]))
            if mask.shape[1] < max_mask_len:
                mask = torch.nn.functional.pad(mask, (0, max_mask_len - mask.shape[1]))
            padded_feats.append(feat)
            padded_masks.append(mask)
        return torch.cat(padded_feats, dim=0), torch.cat(padded_masks, dim=0)

    def _resample_np(self, audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
        if src_sr == tgt_sr:
            return np.asarray(audio, dtype=np.float32)
        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        return torchaudio.functional.resample(audio_tensor, src_sr, tgt_sr).squeeze(0).numpy()

    def _select_prompt_audio(
        self,
        audio_parts: list[np.ndarray],
        *,
        max_chunks: int,
        max_seconds: float,
        sample_rate: int,
    ) -> list[np.ndarray]:
        if max_chunks <= 0 and max_seconds <= 0:
            return []

        selected_parts = list(audio_parts[-max_chunks:]) if max_chunks > 0 else list(audio_parts)
        selected_parts = [np.asarray(part, dtype=np.float32) for part in selected_parts if part.shape[0] > 0]
        if not selected_parts:
            return []
        if max_seconds <= 0:
            return selected_parts

        max_samples = max(1, int(round(max_seconds * sample_rate)))
        total_samples = sum(int(part.shape[0]) for part in selected_parts)
        if total_samples <= max_samples:
            return selected_parts
        trimmed = np.concatenate(selected_parts, axis=0)[-max_samples:]
        return [trimmed.astype(np.float32, copy=False)]

    def _build_flow_matching_prompt_audio(
        self,
        states: OmniTalkerS2STStates,
        current_audio_16k: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_parts_16k = self._select_prompt_audio(
            [turn["audio_16k"] for turn in states.turn_history],
            max_chunks=self.prompt_source_chunks,
            max_seconds=self.prompt_source_sec,
            sample_rate=SAMPLE_RATE_16K,
        )
        generated_parts_24k = self._select_prompt_audio(
            [turn["output_audio_24k"] for turn in states.turn_history],
            max_chunks=self.prompt_generated_chunks,
            max_seconds=self.prompt_generated_sec,
            sample_rate=SAMPLE_RATE_24K,
        )

        if not source_parts_16k and not generated_parts_24k:
            source_parts_16k = [current_audio_16k]

        prompt_parts_16k = list(source_parts_16k) + [
            self._resample_np(audio, SAMPLE_RATE_24K, SAMPLE_RATE_16K)
            for audio in generated_parts_24k
        ]
        prompt_parts_24k = [
            self._resample_np(audio, SAMPLE_RATE_16K, SAMPLE_RATE_24K)
            for audio in source_parts_16k
        ] + list(generated_parts_24k)

        prompt_audio_16k = np.concatenate(prompt_parts_16k).astype(np.float32)
        prompt_audio_24k = np.concatenate(prompt_parts_24k).astype(np.float32)
        return torch.from_numpy(prompt_audio_16k).float(), torch.from_numpy(prompt_audio_24k).float()

    def _get_silence_codes(self, duration_sec: float) -> torch.Tensor:
        duration_sec = float(duration_sec)
        if duration_sec not in self._silence_codes_cache:
            n_samples = int(round(duration_sec * SAMPLE_RATE_16K))
            silence = np.zeros(n_samples, dtype=np.float32)
            codes = self.inference.audio_tokenizer(silence)
            if isinstance(codes, np.ndarray):
                codes = torch.from_numpy(codes)
            codes = codes.detach().cpu().long().reshape(1, -1)
            assert codes.shape[1] > 0, f"silence tokenizer produced empty codes for {duration_sec}s"
            self._silence_codes_cache[duration_sec] = codes
        return self._silence_codes_cache[duration_sec]

    @torch.inference_mode()
    def _synthesize_codes(
        self,
        generated_codes: torch.Tensor,
        states: OmniTalkerS2STStates,
        current_audio_16k: np.ndarray,
    ) -> np.ndarray:
        if generated_codes.shape[1] == 0:
            return np.array([], dtype=np.float32)

        ref_audio_16k, ref_audio_24k = self._build_flow_matching_prompt_audio(
            states=states,
            current_audio_16k=current_audio_16k,
        )
        ref_codes = self.inference.audio_tokenizer(ref_audio_16k.numpy())
        if isinstance(ref_codes, np.ndarray):
            ref_codes = torch.from_numpy(ref_codes)
        if ref_codes.dim() == 1:
            ref_codes = ref_codes.unsqueeze(0)
        ref_codes = ref_codes.to(device=self.model_device, dtype=torch.long)

        prompt_mel = None
        if self.use_prompt_mel:
            prompt_mel = self.inference.extract_mel_feature(ref_audio_24k.unsqueeze(0).to(self.model_device))

        generated_codes = generated_codes.to(device=self.model_device, dtype=torch.long)

        # VoiceBox's reverse_diffusion requires
        # `target_len = cond.shape[1] - prompt_mel.shape[1] > 0`,
        # where `cond.shape[1] = (prompt_codes_len + generated_codes_len) * cond_scale_factor`.
        # Mel framing vs audio_tokenizer framing can drift by a few frames, so when the
        # talker emits very few codes (e.g. 1-2 on tail chunks) target_len can collapse to
        # <= 0 and crash inside `diff_estimator` (zero-length view).
        # Pad `generated_codes` with silence codes (in-distribution, audio_tokenizer of
        # zero-padded waveform) so we keep a stable margin.
        if prompt_mel is not None:
            cond_scale_factor = int(getattr(self.inference.voicebox, "cond_scale_factor", 1))
            prompt_mel_len = int(prompt_mel.shape[1])
            prompt_codes_len = int(ref_codes.shape[1])
            gen_len = int(generated_codes.shape[1])
            cond_len = (prompt_codes_len + gen_len) * cond_scale_factor
            min_margin = max(4, cond_scale_factor)
            if cond_len <= prompt_mel_len + min_margin:
                deficit_codes = (
                    (prompt_mel_len + min_margin - cond_len + cond_scale_factor - 1)
                    // cond_scale_factor
                )
                # Pull from a 1s silence cache (~25 codes) and slice. Tokenizer can
                # fail on too-short waveforms, so always tokenize a fixed safe length.
                silence_pool = self._get_silence_codes(1.0).to(
                    device=self.model_device, dtype=torch.long
                )
                pool_len = int(silence_pool.shape[1])
                assert pool_len > 0, "silence pool unexpectedly empty"
                if deficit_codes <= pool_len:
                    pad_codes = silence_pool[:, :deficit_codes]
                else:
                    repeats = (deficit_codes + pool_len - 1) // pool_len
                    pad_codes = silence_pool.repeat(1, repeats)[:, :deficit_codes]
                generated_codes = torch.cat([generated_codes, pad_codes], dim=1)

        mel = self.inference.codes_to_mel(
            generated_codes=generated_codes,
            prompt_codes=ref_codes,
            prompt_mel=prompt_mel,
        )
        return self.inference.mel_to_audio(mel)

    def _final_pred_text(self, states: OmniTalkerS2STStates) -> str:
        return "".join(states.accumulated_output).strip()

    def _source_entry_for_state(self, states: OmniTalkerS2STStates) -> str:
        idx = int(states.current_instance_index)
        if 0 <= idx < len(self._source_entries):
            return self._source_entries[idx]
        return ""

    def _flush_pred_text_sidecar(self, states: OmniTalkerS2STStates) -> None:
        if states.pred_text_saved:
            return
        source_length_ms = 0
        if states.source_sample_rate > 0:
            source_length_ms = int(round(len(states.source) * 1000.0 / states.source_sample_rate))
        record = {
            "record_id": int(self._sidecar_record_id),
            "index": int(states.current_instance_index),
            "source": self._source_entry_for_state(states),
            "source_length_ms": source_length_ms,
            "pred_text": self._final_pred_text(states),
            "pred_text_chunks": list(states.pred_text_chunks),
        }
        with self.text_sidecar_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._sidecar_record_id += 1
        states.pred_text_saved = True

    def _append_text_step_event(
        self,
        states: OmniTalkerS2STStates,
        *,
        raw_thinker_text: str,
        normalized_text: str,
        emitted_text: str,
        latency_tokens: list[str],
        is_wait: bool,
        has_write_action: bool,
        output_audio_num_samples: int,
        source_consumed_ms: float,
        source_total_ms: float,
        finished: bool,
    ) -> None:
        event = {
            "instance_index": int(states.current_instance_index),
            "step_index": int(states.step_event_index),
            "source": self._source_entry_for_state(states),
            "source_consumed_ms": float(source_consumed_ms),
            "source_total_ms": float(source_total_ms),
            "raw_thinker_text": raw_thinker_text,
            "normalized_text": normalized_text,
            "pred_text_chunk": emitted_text,
            "latency_tokens": list(latency_tokens),
            "latency_text": " ".join(latency_tokens),
            "is_wait": bool(is_wait),
            "has_write_action": bool(has_write_action),
            "output_audio_ms": float(output_audio_num_samples * 1000.0 / SAMPLE_RATE_24K),
            "finished": bool(finished),
        }
        with self.text_step_events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        states.step_event_index += 1

    def _finalize_and_finish(self, states: OmniTalkerS2STStates) -> WriteAction:
        self._flush_pred_text_sidecar(states)
        return self._finish()

    def policy(self, states: Optional[OmniTalkerS2STStates] = None) -> WriteAction | ReadAction:
        if states is None:
            states = self.states

        length_in_seconds = 0.0 if states.source_sample_rate == 0 else len(states.source) / states.source_sample_rate
        states.total_audio_seconds = length_in_seconds
        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()

        total_samples = len(states.source)
        new_samples = total_samples - states.processed_samples
        if new_samples <= 0 and not states.source_finished:
            return ReadAction()
        if new_samples <= 0 and states.source_finished:
            return self._finalize_and_finish(states)

        sample_rate = states.source_sample_rate
        chunk_samples = max(1, int(self.chunk_duration * sample_rate))
        min_final_samples = max(1, int(self.min_final_chunk_sec * sample_rate))
        start_idx = states.processed_samples
        available_samples = total_samples - start_idx
        has_full_chunk = available_samples >= chunk_samples
        has_final_chunk = states.source_finished and available_samples > 0
        if not has_full_chunk and not has_final_chunk:
            return ReadAction()

        if has_full_chunk:
            end_idx = start_idx + chunk_samples
        else:
            if available_samples < min_final_samples:
                states.processed_samples = total_samples
                return self._finalize_and_finish(states)
            end_idx = total_samples

        current_audio = np.array(states.source[start_idx:end_idx], dtype=np.float32)
        if current_audio.ndim > 1:
            current_audio = current_audio.mean(axis=-1, dtype=np.float32)
        if current_audio.shape[0] < chunk_samples:
            current_audio = np.pad(current_audio, (0, chunk_samples - current_audio.shape[0]))
        if sample_rate != SAMPLE_RATE_16K:
            current_audio_tensor = torch.from_numpy(current_audio).unsqueeze(0)
            current_audio_16k = torchaudio.functional.resample(
                current_audio_tensor, sample_rate, SAMPLE_RATE_16K
            ).squeeze(0).numpy()
        else:
            current_audio_16k = current_audio

        input_features, feature_attention_mask, n_audio_tokens = self._extract_audio_features(current_audio_16k)
        start_new_context = False
        if (
            self.enable_history_window
            and self.history_window_turns > 0
            and len(states.turn_history) >= self.history_window_turns
        ):
            overlap = max(0, self.history_overlap_turns)
            states.turn_history = states.turn_history[-overlap:] if overlap > 0 else []
            states.thinker_past_key_values = None
            states.talker_past_key_values = None
            start_new_context = True

        all_features, all_masks = self._collect_all_audio_features(states, input_features, feature_attention_mask)
        history_chunks, current_prefix_ids = self._build_streaming_inputs(states, n_audio_tokens)

        thinker_generation_kwargs = {
            "max_new_tokens": self.thinker_max_new_tokens,
            "do_sample": self.thinker_do_sample,
            "temperature": self.thinker_temperature,
            "top_k": self.thinker_top_k,
            "top_p": self.thinker_top_p,
            "num_beams": self.thinker_num_beams,
            "repetition_penalty": self.thinker_repetition_penalty,
            "no_repeat_ngram_size": self.thinker_no_repeat_ngram_size,
        }
        talker_generation_kwargs = {
            "max_new_tokens": self.talker_max_new_tokens,
            "do_sample": self.talker_do_sample,
            "temperature": self.talker_temperature,
            "top_k": self.talker_top_k,
            "top_p": self.talker_top_p,
            "repetition_penalty": self.talker_repetition_penalty,
            "no_repeat_ngram_size": self.talker_no_repeat_ngram_size,
        }
        (
            generated_codes,
            thinker_text,
            current_chunk_record,
            states.thinker_past_key_values,
            states.talker_past_key_values,
        ) = (
            self.inference.generate_streaming_semantic_codes(
                history_chunks=history_chunks,
                current_chunk_thinker_prefix_ids=current_prefix_ids,
                asst_suffix_token_ids=self._asst_suffix_token_ids,
                input_features=all_features,
                feature_attention_mask=all_masks,
                thinker_step_input_features=input_features,
                thinker_step_feature_attention_mask=feature_attention_mask,
                max_text_tokens=self.thinker_max_new_tokens,
                max_audio_tokens=self.talker_max_new_tokens,
                thinker_generation_kwargs=thinker_generation_kwargs,
                talker_generation_kwargs=talker_generation_kwargs,
                allow_empty_codes=True,
                thinker_past_key_values=None if start_new_context else states.thinker_past_key_values,
                use_thinker_kv_cache=not self.no_thinker_kv_cache,
                talker_past_key_values=None if start_new_context else states.talker_past_key_values,
                use_talker_kv_cache=not self.no_talker_kv_cache,
            )
        )
        translated_text = _normalize_pred(thinker_text)
        is_wait = _is_wait(thinker_text) and not states.source_finished
        emit_text = translated_text
        if translated_text and not is_wait:
            previous_text = states.accumulated_output[-1] if states.accumulated_output else ""
            emit_text = _maybe_add_streaming_word_boundary(previous_text, translated_text, self.target_lang)
            states.accumulated_output.append(emit_text)
            states.pred_text_chunks.append(emit_text)

        # Talker context always uses the codes the talker actually produced (so
        # subsequent chunks see a consistent autoregressive history). Optionally
        # substitute cached silence codes at the VoiceBox stage for wait/idle
        # chunks; otherwise wait chunks emit no audio unless the talker produced
        # real codes.
        if is_wait and self.enable_wait_silence_decode:
            voicebox_codes = self._get_silence_codes(self.chunk_duration).to(
                device=self.model_device, dtype=torch.long
            )
        else:
            voicebox_codes = generated_codes
        output_audio = self._synthesize_codes(voicebox_codes, states, current_audio_16k)

        print(f"[{states.total_audio_seconds}] pred: {emit_text}, audio: {output_audio.shape[0] / SAMPLE_RATE_24K}s")

        states.processed_samples = end_idx
        states.turn_history.append(
            {
                "n_audio_tokens": n_audio_tokens,
                "text": DEFAULT_IDLE_TOKEN if is_wait else translated_text,
                "thinker_token_ids": current_chunk_record["thinker_token_ids"],
                "reply_local_span": current_chunk_record["reply_local_span"],
                "codes": current_chunk_record["codes"],
                "audio_16k": current_audio_16k,
                "output_audio_24k": output_audio,
                "input_features": input_features,
                "feature_attention_mask": feature_attention_mask,
            }
        )
        if output_audio.shape[0] > 0:
            states.accumulated_audio.append(output_audio)

        finished_now = states.source_finished and states.processed_samples >= total_samples
        source_consumed_ms = states.processed_samples * 1000.0 / sample_rate
        event_text = "" if is_wait else emit_text
        self._append_text_step_event(
            states,
            raw_thinker_text=thinker_text,
            normalized_text=translated_text,
            emitted_text=event_text,
            latency_tokens=_word_latency_tokens(event_text),
            is_wait=is_wait,
            has_write_action=bool(output_audio.shape[0] > 0),
            output_audio_num_samples=int(output_audio.shape[0]),
            source_consumed_ms=source_consumed_ms,
            source_total_ms=states.total_audio_seconds * 1000.0,
            finished=finished_now,
        )
        if output_audio.shape[0] > 0:
            segment = SpeechSegment(
                content=output_audio.tolist(),
                finished=finished_now,
                sample_rate=SAMPLE_RATE_24K,
            )
            if finished_now:
                self._flush_pred_text_sidecar(states)
            return WriteAction(content=segment, finished=finished_now)
        if finished_now:
            return self._finalize_and_finish(states)
        return ReadAction()

    def _finish(self) -> WriteAction:
        segment = SpeechSegment(content=[], finished=True, sample_rate=SAMPLE_RATE_24K)
        return WriteAction(content=segment, finished=True)
