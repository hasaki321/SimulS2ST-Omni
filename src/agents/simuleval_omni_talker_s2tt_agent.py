#!/usr/bin/env python3
"""SimulEval agent for OmniTalker streaming S2TT/ASR inference."""

from __future__ import annotations

import os
import re
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint


def _prepare_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src" / "train"):
        path_str = str(path)
        if path_str not in sys.path and os.path.isdir(path_str):
            sys.path.insert(0, path_str)


_prepare_import_paths()

from src.inference.omni_talker_s2t_inference import OmniTalkerS2TInference
from src.train.lang_utils import normalize_lang
from src.train.modeling_dual_head import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IDLE_TOKEN,
    DEFAULT_SPEECH_END_TOKEN,
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
)
from src.train.prompt_formats import (
    TASK_STREAMING_S2TT_ASR,
    TASK_STREAMING_S2TT_TRANSLATE,
    build_audio_span,
    build_system_prompt,
)


SAMPLE_RATE = 16_000
SPEECH_SEGMENT_SIZE = 25


def _strip(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "")).strip()


def _normalize_pred(text: str) -> str:
    text = text or ""
    for token in (DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_IDLE_TOKEN):
        text = text.replace(token, "")
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


class OmniTalkerS2TTStates(AgentStates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_history: list[dict] = []
        self.accumulated_output: list[str] = []
        self.processed_samples: int = 0
        self.total_audio_seconds: float = 0.0
        self.past_key_values: Any = None

    def reset(self):
        super().reset()
        self.turn_history = []
        self.accumulated_output = []
        self.processed_samples = 0
        self.total_audio_seconds = 0.0
        self.past_key_values = None


@entrypoint
class OmniTalkerStreamingS2TTAgent(SpeechToTextAgent):
    """Streaming S2TT agent that mirrors the Stage-2 streaming collator prompt."""

    def __init__(self, args: Namespace):
        super().__init__(args)
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
        self.max_new_tokens = int(args.max_new_tokens)
        self.do_sample = bool(args.do_sample)
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.top_k = int(args.top_k)
        self.num_beams = int(args.num_beams)
        self.repetition_penalty = float(args.repetition_penalty)
        self.no_repeat_ngram_size = int(args.no_repeat_ngram_size)
        self.no_kv_cache = bool(args.no_kv_cache)
        self.enable_history_window = bool(args.enable_history_window)
        self.history_window_turns = max(0, int(args.history_window_turns) // max(1, self.latency_multiplier))
        self.history_overlap_turns = max(0, int(args.history_overlap_turns) // max(1, self.latency_multiplier))

        self.inference = OmniTalkerS2TInference.from_pretrained(
            checkpoint_path=self.checkpoint_path,
            omni_model_path=self.model_name_or_path,
            device=self.device,
            dtype=self.dtype,
        )
        self.processor = self.inference.processor
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.model_device = self.inference.device
        self.model_dtype = self.inference.dtype
        self.system_instruction = self._build_system_instruction()
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.im_end_id

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
        parser.add_argument("--max-new-tokens", type=int, default=64)
        parser.add_argument("--do-sample", action="store_true")
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top-p", type=float, default=1.0)
        parser.add_argument("--top-k", type=int, default=50)
        parser.add_argument("--num-beams", type=int, default=1)
        parser.add_argument("--repetition-penalty", type=float, default=1.1)
        parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
        parser.add_argument("--no-kv-cache", action="store_true")
        parser.add_argument("--history-window-turns", type=int, default=30)
        parser.add_argument("--history-overlap-turns", type=int, default=8)
        parser.add_argument(
            "--disable-history-window",
            action="store_false",
            default=True,
            dest="enable_history_window",
        )

    def build_states(self) -> OmniTalkerS2TTStates:
        return OmniTalkerS2TTStates()

    def _task_type(self) -> str:
        if normalize_lang(self.source_lang) == normalize_lang(self.target_lang):
            return TASK_STREAMING_S2TT_ASR
        return TASK_STREAMING_S2TT_TRANSLATE

    def _build_system_instruction(self) -> str:
        system_prompt = build_system_prompt(
            task_type=self._task_type(),
            base_system_prompt=DEFAULT_SYSTEM_PROMPT,
            src_lang=self.source_lang,
            tgt_lang=self.target_lang,
            latency=self.latency_multiplier,
        )
        return f"{DEFAULT_BOS_TOKEN}system\n{system_prompt}{DEFAULT_EOS_TOKEN}\n"

    def _extract_audio_features(self, audio_np: np.ndarray):
        feats = self.feature_extractor(
            [audio_np],
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            return_attention_mask=True,
        )
        input_features = torch.from_numpy(feats["input_features"]).to(self.model_device, self.model_dtype)
        feature_attention_mask = torch.from_numpy(feats["attention_mask"]).to(self.model_device, torch.long)
        n_frames = feature_attention_mask.sum(dim=1)
        speech_lens = self.inference.length_shrink_func(n_frames)
        n_audio_tokens = min(int(speech_lens[0].item()), SPEECH_SEGMENT_SIZE * self.latency_multiplier)
        return input_features, feature_attention_mask, max(1, n_audio_tokens)

    def _build_user_turn(self, n_audio_tokens: int) -> str:
        user_payload = self.source_lang + build_audio_span(int(n_audio_tokens))
        return f"{DEFAULT_BOS_TOKEN}user\n{user_payload}{DEFAULT_EOS_TOKEN}\n"

    def _build_assistant_turn(self, text: str) -> str:
        assistant_text = text if text else DEFAULT_IDLE_TOKEN
        return f"{DEFAULT_BOS_TOKEN}assistant\n{assistant_text}{DEFAULT_EOS_TOKEN}\n"

    def _build_prompt(self, states: OmniTalkerS2TTStates, n_audio_tokens: int) -> str:
        prompt = self.system_instruction
        for turn in states.turn_history:
            prompt += self._build_user_turn(turn["n_audio_tokens"])
            prompt += self._build_assistant_turn(turn["pred_raw"])
        prompt += self._build_user_turn(n_audio_tokens)
        prompt += f"{DEFAULT_BOS_TOKEN}assistant\n"
        return prompt

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

    def _build_delta_prompt(self, n_audio_tokens: int) -> str:
        return self._build_user_turn(n_audio_tokens) + f"{DEFAULT_BOS_TOKEN}assistant\n"

    def _decode_response(self, response_ids: torch.LongTensor) -> str:
        text = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)
        for token in ("<|text_eos|>", DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN):
            text = text.replace(token, "")
        return text.strip()

    def _text_generation_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "eos_token_id": self.im_end_id,
            "pad_token_id": self.pad_token_id,
            "return_dict_in_generate": True,
            "num_beams": self.num_beams,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["top_k"] = self.top_k
        return kwargs

    def _generate_text_no_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> torch.LongTensor:
        out = self.inference.thinker.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_cache=False,
            **self._text_generation_kwargs(),
        )
        return out.sequences[:, input_ids.shape[1]:]

    def _generate_text_with_kv_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> tuple[torch.LongTensor, Any]:
        prefill_input_ids = input_ids[:, :-1]
        current_past_key_values = past_key_values

        if prefill_input_ids.shape[1] > 0:
            if current_past_key_values is not None:
                past_seen = current_past_key_values.get_seq_length()
                prefill_len = prefill_input_ids.shape[1]
                cache_position = torch.arange(
                    past_seen,
                    past_seen + prefill_len,
                    device=self.model_device,
                    dtype=torch.long,
                )
                prefill_attention_mask = torch.ones(
                    (1, past_seen + prefill_len),
                    device=self.model_device,
                    dtype=torch.long,
                )
            else:
                cache_position = None
                prefill_attention_mask = torch.ones_like(prefill_input_ids)

            outputs = self.inference.thinker(
                input_ids=prefill_input_ids,
                attention_mask=prefill_attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                past_key_values=current_past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            current_past_key_values = outputs.past_key_values

        if current_past_key_values is not None:
            last_cache_pos = torch.arange(
                current_past_key_values.get_seq_length(),
                current_past_key_values.get_seq_length() + 1,
                device=self.model_device,
                dtype=torch.long,
            )
            generation_attention_mask = torch.ones(
                (1, current_past_key_values.get_seq_length() + 1),
                device=self.model_device,
                dtype=torch.long,
            )
        else:
            last_cache_pos = None
            generation_attention_mask = attention_mask

        if current_past_key_values is not None and self.num_beams > 1 and not self.do_sample:
            current_past_key_values.batch_repeat_interleave(self.num_beams)

        out = self.inference.thinker.generate(
            input_ids=input_ids,
            attention_mask=generation_attention_mask,
            past_key_values=current_past_key_values,
            cache_position=last_cache_pos,
            use_cache=True,
            **self._text_generation_kwargs(),
        )
        next_past_key_values = out.past_key_values if self.num_beams == 1 else None
        return out.sequences[:, input_ids.shape[1]:], next_past_key_values

    def policy(self, states: Optional[OmniTalkerS2TTStates] = None) -> WriteAction | ReadAction:
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
            return WriteAction(content="", finished=True)

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
                return WriteAction(content="", finished=True)
            end_idx = total_samples

        current_audio = np.array(states.source[start_idx:end_idx], dtype=np.float32)
        if current_audio.ndim > 1:
            current_audio = current_audio.mean(axis=-1, dtype=np.float32)
        if current_audio.shape[0] < chunk_samples:
            current_audio = np.pad(current_audio, (0, chunk_samples - current_audio.shape[0]))
        if sample_rate != SAMPLE_RATE:
            current_audio_tensor = torch.from_numpy(current_audio).unsqueeze(0)
            current_audio = torchaudio.functional.resample(
                current_audio_tensor, sample_rate, SAMPLE_RATE
            ).squeeze(0).numpy()

        input_features, feature_attention_mask, n_audio_tokens = self._extract_audio_features(current_audio)
        start_new_context = False
        if (
            self.enable_history_window
            and self.history_window_turns > 0
            and len(states.turn_history) >= self.history_window_turns
        ):
            overlap = max(0, self.history_overlap_turns)
            states.turn_history = states.turn_history[-overlap:] if overlap > 0 else []
            states.past_key_values = None
            start_new_context = True

        if self.no_kv_cache:
            prompt_text = self._build_prompt(states, n_audio_tokens)
            gen_input_features, gen_feature_attention_mask = self._collect_all_audio_features(
                states,
                input_features,
                feature_attention_mask,
            )
        elif states.past_key_values is None or start_new_context:
            prompt_text = self._build_prompt(states, n_audio_tokens)
            gen_input_features, gen_feature_attention_mask = self._collect_all_audio_features(
                states,
                input_features,
                feature_attention_mask,
            )
        else:
            prompt_text = self._build_delta_prompt(n_audio_tokens)
            gen_input_features = input_features
            gen_feature_attention_mask = feature_attention_mask

        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids.to(self.model_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            if self.no_kv_cache:
                response_ids = self._generate_text_no_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=gen_input_features,
                    feature_attention_mask=gen_feature_attention_mask,
                )
            else:
                response_ids, states.past_key_values = self._generate_text_with_kv_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=gen_input_features,
                    feature_attention_mask=gen_feature_attention_mask,
                    past_key_values=None if start_new_context else states.past_key_values,
                )
        pred_raw = self._decode_response(response_ids)
        pred_norm = _normalize_pred(pred_raw)

        print(f"[{states.total_audio_seconds}] pred: {pred_norm}")
        states.processed_samples = end_idx

        turn_record = {
            "n_audio_tokens": n_audio_tokens,
            "pred_raw": DEFAULT_IDLE_TOKEN if _is_wait(pred_raw) else pred_norm,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
        }
        states.turn_history.append(turn_record)

        if _is_wait(pred_raw) and not states.source_finished:
            return ReadAction()

        emit_text = pred_norm
        if pred_norm:
            previous_text = states.accumulated_output[-1] if states.accumulated_output else ""
            emit_text = _maybe_add_streaming_word_boundary(previous_text, pred_norm, self.target_lang)
            states.accumulated_output.append(emit_text)

        finished_now = states.source_finished and states.processed_samples >= total_samples
        if emit_text or finished_now:
            return WriteAction(content=emit_text, finished=finished_now)
        return ReadAction()
