#!/usr/bin/env python3
"""
Inference for Qwen-Omni talker checkpoints with VoiceBox decoding.

Pipeline:
1. thinker.generate() produces assistant text hidden states
2. talker.generate() autoregressively produces semantic codec tokens
3. VoiceBox + Vocos decode semantic tokens into waveform
"""

import argparse
import gc
import json
import logging
import math
import types
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import LogitsProcessor, LogitsProcessorList

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    os.environ.get("SIMULS2ST_MODEL_ROOT", PROJECT_ROOT / "models" / "SimulS2ST-Omni")
).expanduser()
os.environ.setdefault("SIMULS2ST_W2V_BERT_PATH", str(MODEL_ROOT / "w2v"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generate_tts_offline import SEMANTIC_CODE_VOCAB_SIZE, SemanticCodeTTSInference
from src.train.modeling_omni_talker import (
    TALKER_COMPONENTS_NAME,
    TALKER_LORA_DIR_NAME,
    THINKER_LORA_DIR_NAME,
    load_talker_components,
)
from src.train.prompt_formats import TASK_S2S_TRANSLATE, build_prompt_bundle, build_prompt_texts, resolve_tts_task_type

logger = logging.getLogger(__name__)

LANG_CODE_TO_NAME = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "chinese": "Chinese",
    "cn": "Chinese",
    "zhs": "Chinese",
    "zht": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    "eng": "English",
    "de": "German",
    "german": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "ja": "Japanese",
    "japanese": "Japanese",
    "fr": "French",
    "french": "French",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "ko": "Korean",
    "korean": "Korean",
}


def normalize_lang(lang: str) -> str:
    lang_lower = lang.lower().strip()
    return LANG_CODE_TO_NAME.get(lang_lower, lang)


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor) -> torch.LongTensor:
    input_lengths = (input_lengths - 1) // 2 + 1
    input_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths.to(torch.long)


def _resize_talker_vocab(talker, n_audio_codes: int) -> None:
    import torch.nn as nn

    new_vocab_size = n_audio_codes + 4
    codec_pad_token_id = n_audio_codes
    codec_bos_token_id = n_audio_codes + 1
    codec_eos_token_id = n_audio_codes + 2
    codec_mask_token_id = n_audio_codes + 3

    old_embed = talker.model.embed_tokens
    old_head = talker.codec_head
    current_vocab_size = old_embed.num_embeddings
    current_head_size = old_head.out_features

    if current_vocab_size == new_vocab_size and current_head_size == new_vocab_size:
        talker.config.vocab_size = new_vocab_size
        talker.vocab_size = new_vocab_size
        talker.codebook_size = new_vocab_size
        talker.config.tts_codec_pad_token_id = codec_pad_token_id
        talker.config.tts_codec_start_token_id = codec_bos_token_id
        talker.config.tts_codec_end_token_id = codec_eos_token_id
        talker.config.tts_codec_mask_token_id = codec_mask_token_id
        talker.codec_pad_token = codec_pad_token_id
        talker.codec_bos_token = codec_bos_token_id
        talker.codec_eos_token = codec_eos_token_id
        talker.codec_mask_token = codec_mask_token_id
        logger.info("Keeping loaded talker codec embedding/head at vocab size %s", new_vocab_size)
        return

    device = old_embed.weight.device
    dtype = old_embed.weight.dtype

    talker.model.embed_tokens = nn.Embedding(
        new_vocab_size,
        old_embed.embedding_dim,
        padding_idx=codec_pad_token_id,
        device=device,
        dtype=dtype,
    )
    talker.codec_head = nn.Linear(
        old_head.in_features,
        new_vocab_size,
        bias=False,
        device=device,
        dtype=old_head.weight.dtype,
    )
    nn.init.normal_(talker.model.embed_tokens.weight, mean=0.0, std=0.02)
    nn.init.normal_(talker.codec_head.weight, mean=0.0, std=0.02)

    talker.config.vocab_size = new_vocab_size
    talker.vocab_size = new_vocab_size
    talker.codebook_size = new_vocab_size
    talker.config.tts_codec_pad_token_id = codec_pad_token_id
    talker.config.tts_codec_start_token_id = codec_bos_token_id
    talker.config.tts_codec_end_token_id = codec_eos_token_id
    talker.config.tts_codec_mask_token_id = codec_mask_token_id
    talker.codec_pad_token = codec_pad_token_id
    talker.codec_bos_token = codec_bos_token_id
    talker.codec_eos_token = codec_eos_token_id
    talker.codec_mask_token = codec_mask_token_id


def _parse_audio_spec(audio_spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    parts = audio_spec.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit() and os.path.exists(parts[0]):
        start = int(parts[1])
        end = int(parts[2])
        return parts[0], start, end
    return audio_spec, None, None


def load_audio_mono(audio_spec: str, target_sr: int) -> torch.Tensor:
    wav_path, start, end = _parse_audio_spec(audio_spec)
    if start is None:
        wav, sr = torchaudio.load(wav_path)
    else:
        num_frames = end - start
        wav, sr = torchaudio.load(wav_path, frame_offset=start, num_frames=num_frames)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav[0].contiguous()


def normalize_device(device: str) -> str:
    device = device.strip()
    if device.isdigit():
        return f"cuda:{device}"
    return device


class OmniTalkerSemanticTTSInference(SemanticCodeTTSInference):
    def __init__(
        self,
        base_model,
        processor,
        voicebox,
        vocoder,
        mel_model,
        voicebox_cfg,
        audio_tokenizer=None,
    ):
        super().__init__(
            thinker=base_model.thinker,
            processor=processor,
            code_token_offset=-1,
            voicebox=voicebox,
            vocoder=vocoder,
            mel_model=mel_model,
            voicebox_cfg=voicebox_cfg,
            audio_tokenizer=audio_tokenizer,
        )
        self.base_model = base_model
        self.talker = base_model.talker
        self._patch_talker_generation_for_cached_inputs_embeds()
        self.last_talker_generation_diagnostics: Dict[str, Any] = {}

    def _patch_talker_generation_for_cached_inputs_embeds(self) -> None:
        """Allow Qwen-Omni talker generate() to use inputs_embeds with a non-empty cache."""
        if getattr(self.talker, "_omnitalker_cached_inputs_patch", False):
            return

        original_forward = self.talker.forward
        original_get_initial_cache_position = self.talker._get_initial_cache_position

        def patched_get_initial_cache_position(talker_self, seq_length, device, model_kwargs):
            if "inputs_embeds" in model_kwargs:
                return original_get_initial_cache_position(seq_length, device, model_kwargs)
            return super(type(talker_self), talker_self)._get_initial_cache_position(
                seq_length,
                device,
                model_kwargs,
            )

        def patched_forward(talker_self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            cache_position = kwargs.get("cache_position")
            position_ids = kwargs.get("position_ids")
            attention_mask = kwargs.get("attention_mask")
            if (
                input_ids is None
                and inputs_embeds is not None
                and attention_mask is not None
                and position_ids is None
                and cache_position is not None
                and cache_position[0] != 0
                and getattr(talker_self, "rope_deltas", None) is not None
            ):
                kwargs["input_ids"] = torch.empty(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
            return original_forward(*args, **kwargs)

        self.talker._get_initial_cache_position = types.MethodType(patched_get_initial_cache_position, self.talker)
        self.talker.forward = types.MethodType(patched_forward, self.talker)
        self.talker._omnitalker_cached_inputs_patch = True

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        omni_model_path: str,
        voicebox_path: str,
        vocos_path: str,
        voicebox_config_path: str,
        codec_model_path: str = None,
        codec_stats_path: str = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        n_audio_codes: int = SEMANTIC_CODE_VOCAB_SIZE,
    ) -> "OmniTalkerSemanticTTSInference":
        import accelerate
        import safetensors.torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from src.voicebox.melspec import MelSpectrogram
        from src.voicebox.tokenizer import VoiceBoxAudioTokenizer
        from src.voicebox.util import load_config
        from src.voicebox.vocos import Vocos
        from src.voicebox.voicebox_model import VoiceBox

        device = normalize_device(device)
        # transformers<=4.52 evaluates f"Model config {config}" even when logging is
        # disabled; config.__repr__ then JSON-dumps a torch.dtype and crashes.
        from transformers import PretrainedConfig

        _orig_config_repr = PretrainedConfig.__repr__

        def _safe_config_repr(self):
            try:
                return _orig_config_repr(self)
            except TypeError:
                return f"{self.__class__.__name__}(<unprintable>)"

        PretrainedConfig.__repr__ = _safe_config_repr

        processor = Qwen2_5OmniProcessor.from_pretrained(omni_model_path)

        ckpt_tokenizer_path = os.path.join(checkpoint_path, "tokenizer_config.json")
        if os.path.exists(ckpt_tokenizer_path):
            logger.info("Loading tokenizer from checkpoint: %s", checkpoint_path)
            processor.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        logger.info("Loading base Qwen-Omni model from: %s", omni_model_path)
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            omni_model_path,
            torch_dtype=dtype_name,
            attn_implementation="sdpa",
            device_map=device,
            trust_remote_code=True,
        )
        if not hasattr(base_model, "talker"):
            base_model.enable_talker()
        if hasattr(base_model, "token2wav"):
            del base_model.token2wav
        if getattr(base_model.thinker, "visual", None) is not None:
            logger.info("Removing visual module to save memory...")
            del base_model.thinker.visual
            base_model.thinker.visual = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ckpt_vocab_size = len(processor.tokenizer)
        thinker_vocab_size = base_model.thinker.get_input_embeddings().weight.shape[0]
        min_required_vocab_size = max(
            ckpt_vocab_size,
            base_model.talker.config.tts_text_pad_token_id + 1,
            base_model.talker.config.tts_text_start_token_id + 1,
            base_model.talker.config.tts_text_end_token_id + 1,
            base_model.config.thinker_config.audio_token_index + 1,
        )
        if thinker_vocab_size < min_required_vocab_size:
            logger.info(
                "Expanding thinker embeddings: %s -> %s (checkpoint tokenizer=%s)",
                thinker_vocab_size,
                min_required_vocab_size,
                ckpt_vocab_size,
            )
            base_model.thinker.resize_token_embeddings(min_required_vocab_size)
            thinker_vocab_size = min_required_vocab_size
        else:
            logger.info(
                "Keeping thinker embeddings at %s (checkpoint tokenizer=%s, min_required=%s)",
                thinker_vocab_size,
                ckpt_vocab_size,
                min_required_vocab_size,
            )
        assert thinker_vocab_size > base_model.talker.config.tts_text_end_token_id, (
            "thinker embedding vocab is smaller than talker text special token ids"
        )

        _resize_talker_vocab(base_model.talker, n_audio_codes)

        talker_loaded = load_talker_components(base_model, checkpoint_path, merge_lora=True)
        if not talker_loaded:
            talker_path = os.path.join(checkpoint_path, TALKER_COMPONENTS_NAME)
            talker_lora_dir = os.path.join(checkpoint_path, TALKER_LORA_DIR_NAME)
            logger.warning(
                "Missing talker checkpoint artifacts under %s (expected %s or %s); keeping talker weights from omni_model_path",
                checkpoint_path,
                talker_path,
                talker_lora_dir,
            )

        thinker_lora_dir = os.path.join(checkpoint_path, THINKER_LORA_DIR_NAME)
        if os.path.isdir(thinker_lora_dir):
            logger.info("Loading thinker LoRA from: %s", thinker_lora_dir)
            base_model.thinker = PeftModel.from_pretrained(base_model.thinker, thinker_lora_dir)
            base_model.thinker = base_model.thinker.merge_and_unload()

        logger.info("Loading VoiceBox config from %s", voicebox_config_path)
        cfg = load_config(voicebox_config_path)

        logger.info("Loading VoiceBox from %s", voicebox_path)
        voicebox = VoiceBox(cfg=cfg.model.voicebox)
        voicebox.eval()
        voicebox.to(device)
        safetensors.torch.load_model(voicebox, voicebox_path)

        logger.info("Loading Vocos from %s", vocos_path)
        vocoder = Vocos(cfg=cfg.model.vocos)
        vocoder.eval()
        vocoder.to(device)
        accelerate.load_checkpoint_and_dispatch(vocoder, vocos_path)

        mel_model = MelSpectrogram(
            sampling_rate=cfg.preprocess.sample_rate,
            n_fft=cfg.preprocess.n_fft,
            num_mels=cfg.preprocess.num_mels,
            hop_size=cfg.preprocess.hop_size,
            win_size=cfg.preprocess.win_size,
            fmin=cfg.preprocess.fmin,
            fmax=cfg.preprocess.fmax,
        )
        mel_model.eval()
        mel_model.to(device)

        logger.info("Loading audio tokenizer for prompt code extraction...")
        if codec_model_path is None:
            codec_model_path = str(MODEL_ROOT / "dualcodec" / "dualcodec.safetensors")
        if codec_stats_path is None:
            codec_stats_path = str(MODEL_ROOT / "dualcodec" / "w2v_bert_stats.pt")
        cfg.model.dual_codec.pretrained_path = codec_model_path
        cfg.model.kmeans.stat_mean_var_path = codec_stats_path
        audio_tokenizer = VoiceBoxAudioTokenizer(cfg, device)

        base_model.eval()
        return cls(
            base_model=base_model,
            processor=processor,
            voicebox=voicebox,
            vocoder=vocoder,
            mel_model=mel_model,
            voicebox_cfg=cfg,
            audio_tokenizer=audio_tokenizer,
        )

    def _preprocess_speech(self, wav_16k: torch.Tensor):
        speech_feature_batch = self.processor.feature_extractor(
            [wav_16k.numpy()],
            sampling_rate=16000,
            padding="max_length",
            return_attention_mask=True,
        )
        audio_mask = torch.from_numpy(speech_feature_batch["attention_mask"])
        audio_features = torch.from_numpy(speech_feature_batch["input_features"])
        n_frames = audio_mask.sum(dim=1)
        speech_lens = _get_feat_extract_output_lengths(n_frames)
        return audio_features, audio_mask, speech_lens

    def _build_unpaired_icl_text(self, ref_text: str, tgt_text: str) -> str:
        ref_text_clean = ref_text.strip()
        tgt_text_clean = tgt_text.strip()
        assert ref_text_clean, "unpaired ICL mode requires non-empty ref_text"
        assert tgt_text_clean, "tgt_text must be non-empty"
        return f"{ref_text_clean}\n{tgt_text_clean}"

    def build_unpaired_prompt(self, ref_text: str, tgt_text: str, tgt_lang: str) -> Dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        icl_text = self._build_unpaired_icl_text(ref_text=ref_text, tgt_text=tgt_text)
        prompt_bundle = build_prompt_bundle(
            task_type=resolve_tts_task_type(is_paired=False, ref_lang=tgt_lang, tgt_lang=tgt_lang),
            text=icl_text,
            target_text=icl_text,
            src_lang=tgt_lang,
            tgt_lang=tgt_lang,
        )
        prompt, _ = build_prompt_texts(prompt_bundle)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def build_unpaired_prefill_prompt(self, ref_text: str, tgt_text: str, tgt_lang: str) -> Dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        icl_text = self._build_unpaired_icl_text(ref_text=ref_text, tgt_text=tgt_text)
        prompt_bundle = build_prompt_bundle(
            task_type=resolve_tts_task_type(is_paired=False, ref_lang=tgt_lang, tgt_lang=tgt_lang),
            text=icl_text,
            target_text=icl_text,
            src_lang=tgt_lang,
            tgt_lang=tgt_lang,
        )
        prefix_prompt, full_prompt = build_prompt_texts(prompt_bundle)
        prefix_encoded = tokenizer(prefix_prompt, return_tensors="pt", add_special_tokens=False)
        full_encoded = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        prefix_len = prefix_encoded["input_ids"].shape[1]
        reply_len = full_encoded["input_ids"].shape[1] - prefix_len
        assert reply_len > 0, "prefill-only prompt must contain non-empty assistant reply"
        return {
            "input_ids": full_encoded["input_ids"],
            "attention_mask": full_encoded["attention_mask"],
            "prefix_len": prefix_len,
            "reply_len": reply_len,
        }

    def build_paired_prompt(
        self,
        tgt_text: str,
        ref_audio_16k: torch.Tensor,
        ref_lang: str,
        tgt_lang: str,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        audio_features, audio_mask, speech_lens = self._preprocess_speech(ref_audio_16k)
        n_sp_token = max(1, int(speech_lens[0].item()))
        prompt_bundle = build_prompt_bundle(
            task_type=resolve_tts_task_type(is_paired=True, ref_lang=ref_lang, tgt_lang=tgt_lang),
            text=tgt_text,
            target_text=tgt_text,
            src_lang=ref_lang,
            tgt_lang=tgt_lang,
            n_audio_tokens=n_sp_token,
        )
        prompt, _ = build_prompt_texts(prompt_bundle)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "input_features": audio_features,
            "feature_attention_mask": audio_mask,
        }

    def build_paired_prefill_prompt(
        self,
        tgt_text: str,
        ref_audio_16k: torch.Tensor,
        ref_lang: str,
        tgt_lang: str,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        audio_features, audio_mask, speech_lens = self._preprocess_speech(ref_audio_16k)
        n_sp_token = max(1, int(speech_lens[0].item()))
        prompt_bundle = build_prompt_bundle(
            task_type=resolve_tts_task_type(is_paired=True, ref_lang=ref_lang, tgt_lang=tgt_lang),
            text=tgt_text,
            target_text=tgt_text,
            src_lang=ref_lang,
            tgt_lang=tgt_lang,
            n_audio_tokens=n_sp_token,
        )
        prefix_prompt, full_prompt = build_prompt_texts(prompt_bundle)
        prefix_encoded = tokenizer(prefix_prompt, return_tensors="pt", add_special_tokens=False)
        full_encoded = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        prefix_len = prefix_encoded["input_ids"].shape[1]
        reply_len = full_encoded["input_ids"].shape[1] - prefix_len
        assert reply_len > 0, "prefill-only prompt must contain non-empty assistant reply"
        return {
            "input_ids": full_encoded["input_ids"],
            "attention_mask": full_encoded["attention_mask"],
            "input_features": audio_features,
            "feature_attention_mask": audio_mask,
            "prefix_len": prefix_len,
            "reply_len": reply_len,
        }

    def build_s2s_prompt(
        self,
        ref_audio_16k: torch.Tensor,
        ref_lang: str,
        tgt_lang: str,
    ) -> Dict[str, torch.Tensor]:
        tokenizer = self.processor.tokenizer
        audio_features, audio_mask, speech_lens = self._preprocess_speech(ref_audio_16k)
        n_sp_token = max(1, int(speech_lens[0].item()))
        prompt_bundle = build_prompt_bundle(
            task_type=TASK_S2S_TRANSLATE,
            target_text="",
            src_lang=ref_lang,
            tgt_lang=tgt_lang,
            n_audio_tokens=n_sp_token,
        )
        prefix_prompt, _ = build_prompt_texts(prompt_bundle)
        encoded = tokenizer(prefix_prompt, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "input_features": audio_features,
            "feature_attention_mask": audio_mask,
        }

    def _decode_generated_text(self, full_sequences: torch.Tensor, prompt_len: int) -> str:
        tokenizer = self.processor.tokenizer
        generated_ids = full_sequences[:, prompt_len:]
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        for token in ["<|im_end|>", "<|im_start|>assistant\n", "<|im_start|>", "<|text_eos|>", "<|text_bos|>"]:
            text = text.replace(token, "")
        return text.strip()

    def _decode_prefill_reply_text(self, input_ids: torch.Tensor, prefix_len: int, reply_len: int) -> str:
        tokenizer = self.processor.tokenizer
        reply_ids = input_ids[0, prefix_len : prefix_len + reply_len]
        text = tokenizer.decode(reply_ids, skip_special_tokens=False)
        for token in ["<|im_end|>", "<|im_start|>assistant\n", "<|im_start|>", "<|text_eos|>", "<|text_bos|>"]:
            text = text.replace(token, "")
        return text.strip()

    def _scatter_audio_features(
        self,
        input_ids: torch.Tensor,
        thinker_inputs_embeds: torch.Tensor,
        input_features: Optional[torch.Tensor],
        feature_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_features is None:
            return thinker_inputs_embeds

        audio_features = self.thinker.get_audio_features(
            input_features.to(device=self.device, dtype=self.dtype),
            feature_attention_mask=feature_attention_mask.to(device=self.device, dtype=torch.bool),
        )
        audio_features = audio_features.to(thinker_inputs_embeds.device, thinker_inputs_embeds.dtype)
        audio_input_mask = input_ids == self.base_model.config.thinker_config.audio_token_index
        audio_input_mask = audio_input_mask.unsqueeze(-1).expand_as(thinker_inputs_embeds)
        return thinker_inputs_embeds.masked_scatter(audio_input_mask, audio_features)

    def _build_force_prefix_logits_processor(
        self,
        force_prefix_codes: Optional[torch.Tensor],
        initial_input_len: int,
        logits_processor: Optional[LogitsProcessorList] = None,
    ) -> Optional[LogitsProcessorList]:
        processors = LogitsProcessorList(logits_processor or [])
        if force_prefix_codes is None:
            return processors if len(processors) > 0 else None
        assert force_prefix_codes.dim() == 1, "force_prefix_codes must be a 1D tensor"
        assert torch.all((force_prefix_codes >= 0) & (force_prefix_codes < SEMANTIC_CODE_VOCAB_SIZE)), (
            "force_prefix_codes must be semantic code ids in valid range"
        )

        class _ForcePrefixCodecTokens(LogitsProcessor):
            def __init__(self, prefix_tokens: torch.Tensor, prompt_len: int):
                self.prefix_tokens = prefix_tokens.to(dtype=torch.long)
                self.prompt_len = int(prompt_len)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                step = input_ids.shape[1] - self.prompt_len
                if 0 <= step < self.prefix_tokens.shape[0]:
                    forced_token = int(self.prefix_tokens[step].item())
                    forced_scores = torch.full_like(scores, float("-inf"))
                    forced_scores[:, forced_token] = 0.0
                    return forced_scores
                return scores

        processors.append(_ForcePrefixCodecTokens(force_prefix_codes.to(device=self.device), initial_input_len))
        return processors

    def _build_legacy_generation_kwargs(
        self,
        max_text_tokens: int,
        max_audio_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        talker_no_repeat_ngram_size: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        thinker_generation_kwargs = {
            "max_new_tokens": max_text_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        talker_generation_kwargs = {
            "max_new_tokens": max_audio_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": talker_no_repeat_ngram_size,
        }
        return thinker_generation_kwargs, talker_generation_kwargs

    def _extract_talker_generation_diagnostics(
        self,
        talker_result,
        prompt_len: int,
        forced_prefix_len: int = 0,
    ) -> Dict[str, Any]:
        if not hasattr(talker_result, "scores") or talker_result.scores is None:
            return {}

        sequences = talker_result.sequences
        generated = sequences[:, prompt_len:]
        n_steps = min(len(talker_result.scores), generated.shape[1])
        step_records = []
        all_nll = []
        continuation_nll = []

        for step_idx in range(n_steps):
            token_id = int(generated[0, step_idx].item())
            if token_id < 0 or token_id >= SEMANTIC_CODE_VOCAB_SIZE:
                continue
            score = talker_result.scores[step_idx].float()
            log_prob = score.log_softmax(dim=-1)[0, token_id]
            nll = float((-log_prob).item())
            ppl = float(math.exp(min(nll, 50.0)))
            is_forced_prefix = step_idx < forced_prefix_len
            step_records.append(
                {
                    "step": step_idx,
                    "code": token_id,
                    "nll": nll,
                    "ppl": ppl,
                    "forced_prefix": is_forced_prefix,
                }
            )
            all_nll.append(nll)
            if not is_forced_prefix:
                continuation_nll.append(nll)

        diagnostics: Dict[str, Any] = {
            "num_scored_steps": len(step_records),
            "forced_prefix_len": int(forced_prefix_len),
            "steps": step_records,
        }
        if all_nll:
            mean_nll = float(np.mean(all_nll))
            diagnostics["mean_nll"] = mean_nll
            diagnostics["mean_ppl"] = float(math.exp(min(mean_nll, 50.0)))
        if continuation_nll:
            continuation_mean_nll = float(np.mean(continuation_nll))
            diagnostics["continuation_mean_nll"] = continuation_mean_nll
            diagnostics["continuation_mean_ppl"] = float(math.exp(min(continuation_mean_nll, 50.0)))
        return diagnostics

    def _trim_prompt_prefix_audio(
        self,
        audio: np.ndarray,
        full_mel: torch.Tensor,
        prompt_mel: Optional[torch.Tensor],
    ) -> np.ndarray:
        if prompt_mel is None:
            return audio
        prompt_mel_len = int(prompt_mel.shape[1])
        full_mel_len = int(full_mel.shape[1])
        assert full_mel_len > prompt_mel_len > 0, "invalid mel lengths for trimming prompt prefix"
        trim_samples = int(round(len(audio) * (prompt_mel_len / full_mel_len)))
        assert trim_samples < len(audio), "trim would remove all audio samples"
        return audio[trim_samples:]

    @torch.inference_mode()
    def generate_semantic_codes_prefill_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_len: int,
        reply_len: int,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        max_audio_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        talker_no_repeat_ngram_size: int = 0,
        force_prefix_codes: Optional[torch.Tensor] = None,
        talker_generation_kwargs: Optional[Dict[str, Any]] = None,
        allow_empty_codes: bool = False,
    ) -> Tuple[torch.Tensor, str]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        thinker_inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        thinker_inputs_embeds = self._scatter_audio_features(
            input_ids=input_ids,
            thinker_inputs_embeds=thinker_inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        thinker_outputs = self.thinker.model(
            inputs_embeds=thinker_inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        thinker_hidden_states = thinker_outputs.last_hidden_state
        thinker_cond = thinker_hidden_states + thinker_inputs_embeds

        prefill_cond = thinker_cond[:, :prefix_len, :]
        reply_cond = thinker_cond[:, prefix_len : prefix_len + reply_len, :]
        assert reply_cond.shape[1] > 0, "prefill-only thinker reply span is empty"

        thinker_text = self._decode_prefill_reply_text(input_ids, prefix_len, reply_len)

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        talker_text_bos_token = self.talker.text_bos_token

        talker_input_text_ids = torch.cat(
            [
                input_ids[:, :prefix_len],
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.device),
                input_ids[:, prefix_len : prefix_len + 1],
            ],
            dim=1,
        )
        talker_input_ids = torch.cat(
            [
                torch.full((1, prefix_len), self.talker.codec_mask_token, dtype=torch.long, device=self.device),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        text_bos_embed = thinker_embed_tokens(
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        text_eos_embed = thinker_embed_tokens(
            torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        text_pad_embed = thinker_embed_tokens(
            torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        talker_inputs_embeds = torch.cat(
            [
                prefill_cond,
                text_bos_embed,
                reply_cond[:, :1, :],
            ],
            dim=1,
        )

        thinker_reply_part = reply_cond[:, 1:, :]
        if thinker_reply_part.shape[1] == 0:
            thinker_reply_part = text_eos_embed
        thinker_reply_part = torch.cat([thinker_reply_part, text_eos_embed, text_pad_embed], dim=1)

        talker_attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long, device=self.device)
        talker_generation_kwargs = dict(talker_generation_kwargs or {})
        user_logits_processor = talker_generation_kwargs.pop("logits_processor", None)
        logits_processor = self._build_force_prefix_logits_processor(
            force_prefix_codes=force_prefix_codes,
            initial_input_len=talker_input_ids.shape[1],
            logits_processor=user_logits_processor,
        )
        talker_generation_kwargs.setdefault("suppress_tokens", [self.talker.codec_bos_token])
        talker_generation_kwargs.setdefault("max_new_tokens", max_audio_tokens)
        talker_generation_kwargs.setdefault("eos_token_id", self.talker.codec_eos_token)
        talker_generation_kwargs.setdefault("pad_token_id", self.talker.codec_pad_token)
        talker_generation_kwargs.setdefault("do_sample", do_sample)
        talker_generation_kwargs.setdefault("temperature", temperature)
        talker_generation_kwargs.setdefault("top_k", top_k)
        talker_generation_kwargs.setdefault("top_p", top_p)
        talker_generation_kwargs.setdefault("repetition_penalty", repetition_penalty)
        talker_generation_kwargs.setdefault("no_repeat_ngram_size", talker_no_repeat_ngram_size)
        talker_generation_kwargs.setdefault("return_dict_in_generate", True)
        talker_generation_kwargs.setdefault("output_scores", True)
        if logits_processor is not None:
            talker_generation_kwargs["logits_processor"] = logits_processor

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            **talker_generation_kwargs,
        )

        talker_sequences = talker_result.sequences if hasattr(talker_result, "sequences") else talker_result
        self.last_talker_generation_diagnostics = self._extract_talker_generation_diagnostics(
            talker_result=talker_result,
            prompt_len=talker_input_ids.shape[1],
            forced_prefix_len=force_prefix_codes.numel() if force_prefix_codes is not None else 0,
        )
        generated = talker_sequences[:, talker_input_ids.shape[1] :]
        if generated.shape[1] > 0 and generated[0, -1].item() == self.talker.codec_eos_token:
            generated = generated[:, :-1]
        valid_mask = (generated >= 0) & (generated < SEMANTIC_CODE_VOCAB_SIZE)
        generated_codes = generated[valid_mask]
        if generated_codes.numel() == 0 and allow_empty_codes:
            return generated_codes.new_empty((1, 0)), thinker_text
        assert generated_codes.numel() > 0, "talker generated zero semantic codes"
        return generated_codes.unsqueeze(0), thinker_text

    @torch.inference_mode()
    def generate_semantic_codes(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        prefill_only: bool = False,
        prefix_len: int = 0,
        reply_len: int = 0,
        max_text_tokens: int = 256,
        max_audio_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        talker_no_repeat_ngram_size: int = 0,
        force_prefix_codes: Optional[torch.Tensor] = None,
        thinker_generation_kwargs: Optional[Dict[str, Any]] = None,
        talker_generation_kwargs: Optional[Dict[str, Any]] = None,
        allow_empty_codes: bool = False,
    ) -> Tuple[torch.Tensor, str]:
        self.last_talker_generation_diagnostics = {}
        if thinker_generation_kwargs is None and talker_generation_kwargs is None:
            thinker_generation_kwargs, talker_generation_kwargs = self._build_legacy_generation_kwargs(
                max_text_tokens=max_text_tokens,
                max_audio_tokens=max_audio_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
            )
        thinker_generation_kwargs = dict(thinker_generation_kwargs or {})
        talker_generation_kwargs = dict(talker_generation_kwargs or {})

        if prefill_only:
            assert prefix_len > 0 and reply_len > 0, "prefill_only requires prefix_len and reply_len"
            return self.generate_semantic_codes_prefill_only(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prefix_len=prefix_len,
                reply_len=reply_len,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                max_audio_tokens=max_audio_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
                force_prefix_codes=force_prefix_codes,
                talker_generation_kwargs=talker_generation_kwargs,
                allow_empty_codes=allow_empty_codes,
            )

        tokenizer = self.processor.tokenizer
        thinker_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        }
        thinker_generation_kwargs.setdefault("max_new_tokens", max_text_tokens)
        thinker_generation_kwargs.setdefault("do_sample", do_sample)
        thinker_generation_kwargs.setdefault("temperature", temperature)
        thinker_generation_kwargs.setdefault("top_k", top_k)
        thinker_generation_kwargs.setdefault("top_p", top_p)
        thinker_generation_kwargs.setdefault("repetition_penalty", repetition_penalty)
        thinker_generation_kwargs.setdefault("no_repeat_ngram_size", no_repeat_ngram_size)
        thinker_generation_kwargs.setdefault("eos_token_id", tokenizer.convert_tokens_to_ids("<|im_end|>"))
        thinker_generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        thinker_kwargs.update(thinker_generation_kwargs)
        thinker_kwargs["output_hidden_states"] = True
        thinker_kwargs["return_dict_in_generate"] = True
        if input_features is not None:
            thinker_kwargs["input_features"] = input_features.to(device=self.device, dtype=self.dtype)
        if feature_attention_mask is not None:
            thinker_kwargs["feature_attention_mask"] = feature_attention_mask.to(
                device=self.device, dtype=torch.bool
            )

        thinker_result = self.thinker.generate(**thinker_kwargs)
        thinker_text = self._decode_generated_text(thinker_result.sequences, input_ids.shape[1])
        thinker_generate_ids = thinker_result.sequences[:, input_ids.shape[1] :].to(self.device)
        assert thinker_generate_ids.shape[1] > 0, "thinker generated zero text tokens"

        embeds_to_talker = thinker_result.hidden_states[0][0].clone().to(self.device)
        if input_features is not None:
            audio_ids_mask = input_ids.to(self.device) == self.base_model.config.thinker_config.audio_token_index
            audio_mask = audio_ids_mask.unsqueeze(-1).expand_as(embeds_to_talker)
            audio_mask_tensor = torch.zeros(
                [audio_ids_mask.sum(), embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=self.device,
            )
            embeds_to_talker.masked_scatter_(audio_mask, audio_mask_tensor)

        processed_thinker_hidden = ((embeds_to_talker,) + thinker_result.hidden_states[0][1:],) + thinker_result.hidden_states[1:]
        thinker_token_embeds = [hidden[0].to(self.device) for hidden in processed_thinker_hidden]
        thinker_hidden_states = [hidden[-1].to(self.device) for hidden in processed_thinker_hidden]

        talker_text_bos_token = self.talker.text_bos_token
        talker_input_text_ids = torch.cat(
            [
                input_ids.to(self.device),
                torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )
        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids.to(self.device), fill_value=self.talker.codec_mask_token),
                torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.device),
                torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.device),
            ],
            dim=1,
        )

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_text_bos_embed = thinker_embed_tokens(
            torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        talker_inputs_embeds = torch.cat(
            [talker_inputs_embeds, talker_text_bos_embed, thinker_reply_part[:, :1, :]],
            dim=1,
        )

        eos_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        pad_embedding = thinker_embed_tokens(
            torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.device)
        ).to(self.device)
        thinker_reply_part = torch.cat(
            [thinker_reply_part[:, 1:, :], eos_embedding, pad_embedding],
            dim=1,
        )

        talker_attention_mask = torch.ones_like(talker_input_ids, dtype=torch.long, device=self.device)
        user_logits_processor = talker_generation_kwargs.pop("logits_processor", None)
        logits_processor = self._build_force_prefix_logits_processor(
            force_prefix_codes=force_prefix_codes,
            initial_input_len=talker_input_ids.shape[1],
            logits_processor=user_logits_processor,
        )
        talker_generation_kwargs.setdefault("suppress_tokens", [self.talker.codec_bos_token])
        talker_generation_kwargs.setdefault("max_new_tokens", max_audio_tokens)
        talker_generation_kwargs.setdefault("eos_token_id", self.talker.codec_eos_token)
        talker_generation_kwargs.setdefault("pad_token_id", self.talker.codec_pad_token)
        talker_generation_kwargs.setdefault("do_sample", do_sample)
        talker_generation_kwargs.setdefault("temperature", temperature)
        talker_generation_kwargs.setdefault("top_k", top_k)
        talker_generation_kwargs.setdefault("top_p", top_p)
        talker_generation_kwargs.setdefault("repetition_penalty", repetition_penalty)
        talker_generation_kwargs.setdefault("no_repeat_ngram_size", talker_no_repeat_ngram_size)
        talker_generation_kwargs.setdefault("return_dict_in_generate", True)
        talker_generation_kwargs.setdefault("output_scores", True)
        if logits_processor is not None:
            talker_generation_kwargs["logits_processor"] = logits_processor

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=talker_inputs_embeds,
            attention_mask=talker_attention_mask,
            **talker_generation_kwargs,
        )

        talker_sequences = talker_result.sequences if hasattr(talker_result, "sequences") else talker_result
        self.last_talker_generation_diagnostics = self._extract_talker_generation_diagnostics(
            talker_result=talker_result,
            prompt_len=talker_input_ids.shape[1],
            forced_prefix_len=force_prefix_codes.numel() if force_prefix_codes is not None else 0,
        )
        generated = talker_sequences[:, talker_input_ids.shape[1] :]
        if generated.shape[1] > 0 and generated[0, -1].item() == self.talker.codec_eos_token:
            generated = generated[:, :-1]
        valid_mask = (generated >= 0) & (generated < SEMANTIC_CODE_VOCAB_SIZE)
        generated_codes = generated[valid_mask]
        if generated_codes.numel() == 0 and allow_empty_codes:
            return generated_codes.new_empty((1, 0)), thinker_text
        assert generated_codes.numel() > 0, "talker generated zero semantic codes"
        return generated_codes.unsqueeze(0), thinker_text

    @torch.inference_mode()
    def generate_streaming_semantic_codes(
        self,
        *,
        history_chunks: list,
        current_chunk_thinker_prefix_ids: list,
        asst_suffix_token_ids: list,
        input_features: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        thinker_step_input_features: Optional[torch.Tensor] = None,
        thinker_step_feature_attention_mask: Optional[torch.Tensor] = None,
        max_text_tokens: int = 256,
        max_audio_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        talker_no_repeat_ngram_size: int = 0,
        thinker_generation_kwargs: Optional[Dict[str, Any]] = None,
        talker_generation_kwargs: Optional[Dict[str, Any]] = None,
        allow_empty_codes: bool = True,
        min_audio_tokens: int = 0,
        thinker_past_key_values: Optional[Any] = None,
        use_thinker_kv_cache: bool = False,
        talker_past_key_values: Optional[Any] = None,
        use_talker_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, str, Dict[str, Any], Optional[Any], Optional[Any]]:
        """Streaming, chunk-aligned semantic code generation.

        Mirrors the training collator (`DataCollatorForStreamingTTSTalkerDataset`):
        the talker is fed a chunk-concatenated sequence

            [mask_1][pad][bos][codes_1][eos] ... [mask_{N-1}][pad][bos][codes_{N-1}][eos]
            [mask_N][pad][bos]

        for chunk N's code generation. Each chunk's mask region scatters from the
        corresponding thinker hidden span; codec_pad / codec_bos / codes / codec_eos
        positions get codec embeddings plus additive text conditioning that mirrors
        `_build_talker_inputs_embeds_scatter`. The trailing `[pad][bos]` of chunk N
        is left without codec_pad / codec_bos additive contributions, so that the
        underlying Qwen-Omni talker model adds them itself on the first forward
        (`inputs_embeds[:, -2/-1, :] += codec_pad/codec_bos`), matching the
        offline path.

        Args:
            history_chunks: list of dicts for chunks 0..N-2, each with
                - "thinker_token_ids": list[int] (chunk i's full training-format
                  thinker tokens; chunk 0 already includes the system prefix)
                - "reply_local_span": (sup_start, sup_end) within thinker_token_ids
                  marking the assistant text+suffix range
                - "codes": LongTensor of shape [L_i] (audio codes for chunk i)
            current_chunk_thinker_prefix_ids: list[int]. Chunk N's thinker tokens
                up to (and including) the assistant prefix (so the next position
                is where the model starts generating). For chunk 0 this should
                already include system tokens.
            asst_suffix_token_ids: list[int] for the chunk-end suffix
                "<|im_end|>\\n" (typically two tokens).
            input_features / feature_attention_mask: cumulative audio features
                covering chunks 0..N-1 for full thinker re-forward + talker conditioning.
            thinker_step_input_features / thinker_step_feature_attention_mask:
                current chunk audio features for thinker incremental generation when
                `use_thinker_kv_cache=True`.
        Returns:
            (codes_N [1, L_N], thinker_text, current_chunk_record) where
            current_chunk_record carries the thinker_token_ids + reply_local_span
            for the just-finished chunk N so the caller can append to history,
            plus the next thinker/talker past_key_values when cache reuse is enabled.
        """
        device = self.device
        dtype = self.dtype
        tokenizer = self.processor.tokenizer

        # ---- Step 1: thinker.generate to obtain chunk-N text ----
        prompt_ids: List[int] = []
        for ch in history_chunks:
            prompt_ids.extend(ch["thinker_token_ids"])
        chunk_N_start = len(prompt_ids)
        chunk_N_reply_local_start = len(current_chunk_thinker_prefix_ids)
        prompt_ids.extend(current_chunk_thinker_prefix_ids)
        prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_attention_mask = torch.ones_like(prompt_input_ids)

        if thinker_generation_kwargs is None and talker_generation_kwargs is None:
            thinker_generation_kwargs, talker_generation_kwargs = self._build_legacy_generation_kwargs(
                max_text_tokens=max_text_tokens,
                max_audio_tokens=max_audio_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
            )
        thinker_generation_kwargs = dict(thinker_generation_kwargs or {})
        talker_generation_kwargs = dict(talker_generation_kwargs or {})

        thinker_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
        }
        thinker_generation_kwargs.setdefault("max_new_tokens", max_text_tokens)
        thinker_generation_kwargs.setdefault("do_sample", do_sample)
        thinker_generation_kwargs.setdefault("temperature", temperature)
        thinker_generation_kwargs.setdefault("top_k", top_k)
        thinker_generation_kwargs.setdefault("top_p", top_p)
        thinker_generation_kwargs.setdefault("repetition_penalty", repetition_penalty)
        thinker_generation_kwargs.setdefault("no_repeat_ngram_size", no_repeat_ngram_size)
        thinker_generation_kwargs.setdefault("eos_token_id", tokenizer.convert_tokens_to_ids("<|im_end|>"))
        thinker_generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        thinker_generation_kwargs.setdefault("num_beams", 1)
        thinker_generation_kwargs.setdefault("return_dict_in_generate", True)
        thinker_next_past_key_values = None

        if not use_thinker_kv_cache:
            thinker_kwargs.update(thinker_generation_kwargs)
            if input_features is not None:
                thinker_kwargs["input_features"] = input_features.to(device=device, dtype=dtype)
            if feature_attention_mask is not None:
                thinker_kwargs["feature_attention_mask"] = feature_attention_mask.to(device=device, dtype=torch.bool)

            thinker_result = self.thinker.generate(**thinker_kwargs)
            gen_seq = thinker_result.sequences if hasattr(thinker_result, "sequences") else thinker_result
            gen_ids = gen_seq[:, prompt_input_ids.shape[1] :].to(device)
            thinker_text = self._decode_generated_text(gen_seq, prompt_input_ids.shape[1])
        else:
            step_input_ids_list = (
                prompt_ids if thinker_past_key_values is None else current_chunk_thinker_prefix_ids
            )
            step_input_ids = torch.tensor([step_input_ids_list], dtype=torch.long, device=device)
            step_attention_mask = torch.ones_like(step_input_ids)
            step_input_features = input_features
            step_feature_attention_mask = feature_attention_mask
            if thinker_past_key_values is not None:
                step_input_features = thinker_step_input_features
                step_feature_attention_mask = thinker_step_feature_attention_mask

            prefill_input_ids = step_input_ids[:, :-1]
            current_thinker_past_key_values = thinker_past_key_values
            if prefill_input_ids.shape[1] > 0:
                if current_thinker_past_key_values is not None:
                    past_seen = current_thinker_past_key_values.get_seq_length()
                    prefill_len = prefill_input_ids.shape[1]
                    cache_position = torch.arange(
                        past_seen,
                        past_seen + prefill_len,
                        device=device,
                        dtype=torch.long,
                    )
                    prefill_attention_mask = torch.ones(
                        (1, past_seen + prefill_len),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    cache_position = None
                    prefill_attention_mask = torch.ones_like(prefill_input_ids)

                thinker_forward_kwargs = {
                    "input_ids": prefill_input_ids,
                    "attention_mask": prefill_attention_mask,
                    "past_key_values": current_thinker_past_key_values,
                    "cache_position": cache_position,
                    "use_cache": True,
                    "return_dict": True,
                }
                if step_input_features is not None:
                    thinker_forward_kwargs["input_features"] = step_input_features.to(device=device, dtype=dtype)
                if step_feature_attention_mask is not None:
                    thinker_forward_kwargs["feature_attention_mask"] = step_feature_attention_mask.to(
                        device=device, dtype=torch.bool
                    )
                thinker_prefill_outputs = self.thinker(**thinker_forward_kwargs)
                current_thinker_past_key_values = thinker_prefill_outputs.past_key_values

            if current_thinker_past_key_values is not None:
                generation_cache_position = torch.arange(
                    current_thinker_past_key_values.get_seq_length(),
                    current_thinker_past_key_values.get_seq_length() + 1,
                    device=device,
                    dtype=torch.long,
                )
                generation_attention_mask = torch.ones(
                    (1, current_thinker_past_key_values.get_seq_length() + 1),
                    device=device,
                    dtype=torch.long,
                )
            else:
                generation_cache_position = None
                generation_attention_mask = step_attention_mask

            thinker_num_beams = int(thinker_generation_kwargs.get("num_beams", 1))
            thinker_do_sample = bool(thinker_generation_kwargs.get("do_sample", False))
            if current_thinker_past_key_values is not None and thinker_num_beams > 1 and not thinker_do_sample:
                current_thinker_past_key_values.batch_repeat_interleave(thinker_num_beams)

            thinker_result = self.thinker.generate(
                input_ids=step_input_ids,
                attention_mask=generation_attention_mask,
                past_key_values=current_thinker_past_key_values,
                cache_position=generation_cache_position,
                use_cache=True,
                **thinker_generation_kwargs,
            )
            gen_seq = thinker_result.sequences if hasattr(thinker_result, "sequences") else thinker_result
            gen_ids = gen_seq[:, step_input_ids.shape[1] :].to(device)
            thinker_text = self._decode_generated_text(gen_seq, step_input_ids.shape[1])
            thinker_next_past_key_values = thinker_result.past_key_values if thinker_num_beams == 1 else None

        # Trim trailing <|im_end|> if present (we re-append exactly via asst_suffix).
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen_list = gen_ids[0].tolist()
        if gen_list and gen_list[-1] == im_end_id:
            gen_list = gen_list[:-1]

        # ---- Step 2: build full thinker token sequence (training-format) ----
        full_thinker_ids_list = list(prompt_ids) + gen_list + list(asst_suffix_token_ids)
        chunk_N_end = len(full_thinker_ids_list)
        chunk_N_reply_local_end = chunk_N_end - chunk_N_start
        assert chunk_N_reply_local_end > chunk_N_reply_local_start, (
            "Streaming chunk N produced no assistant reply tokens"
        )
        chunk_N_reply_span = (chunk_N_reply_local_start, chunk_N_reply_local_end)
        full_thinker_ids = torch.tensor([full_thinker_ids_list], dtype=torch.long, device=device)
        full_attention_mask = torch.ones_like(full_thinker_ids)

        # ---- Step 3: re-run thinker forward on full sequence to get hidden states ----
        full_thinker_inputs_embeds = self.thinker.get_input_embeddings()(full_thinker_ids)
        full_thinker_inputs_embeds = self._scatter_audio_features(
            input_ids=full_thinker_ids,
            thinker_inputs_embeds=full_thinker_inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        thinker_outputs = self.thinker.model(
            inputs_embeds=full_thinker_inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        full_thinker_hidden = thinker_outputs.last_hidden_state
        thinker_cond = (full_thinker_hidden + full_thinker_inputs_embeds).to(dtype)

        # ---- Step 4: compute absolute spans for all N chunks ----
        thinker_spans: List[Tuple[int, int]] = []
        reply_spans: List[Tuple[int, int]] = []
        offset = 0
        for ch in history_chunks:
            length = len(ch["thinker_token_ids"])
            thinker_spans.append((offset, offset + length))
            r0, r1 = ch["reply_local_span"]
            reply_spans.append((offset + r0, offset + r1))
            offset += length
        thinker_spans.append((chunk_N_start, chunk_N_end))
        reply_spans.append((chunk_N_start + chunk_N_reply_span[0], chunk_N_start + chunk_N_reply_span[1]))
        assert thinker_spans[-1][1] == full_thinker_ids.shape[1], (
            f"thinker span end {thinker_spans[-1][1]} != full thinker length {full_thinker_ids.shape[1]}"
        )

        # ---- Step 5: build talker prefix (input_ids, inputs_embeds, input_text_ids) ----
        codec_pad = self.talker.codec_pad_token
        codec_bos = self.talker.codec_bos_token
        codec_eos = self.talker.codec_eos_token
        codec_mask = self.talker.codec_mask_token
        text_bos = self.talker.text_bos_token
        text_eos = self.talker.text_eos_token
        text_pad = self.talker.text_pad_token

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        talker_embed_tokens = self.talker.get_input_embeddings()
        text_bos_embed = thinker_embed_tokens(
            torch.tensor([text_bos], dtype=torch.long, device=device)
        ).to(dtype)[0]  # [H]
        text_eos_embed = thinker_embed_tokens(
            torch.tensor([text_eos], dtype=torch.long, device=device)
        ).to(dtype)[0]
        text_pad_embed = thinker_embed_tokens(
            torch.tensor([text_pad], dtype=torch.long, device=device)
        ).to(dtype)[0]
        codec_pad_embed = talker_embed_tokens(
            torch.tensor([codec_pad], dtype=torch.long, device=device)
        ).to(dtype)[0]
        codec_bos_embed = talker_embed_tokens(
            torch.tensor([codec_bos], dtype=torch.long, device=device)
        ).to(dtype)[0]
        codec_eos_embed = talker_embed_tokens(
            torch.tensor([codec_eos], dtype=torch.long, device=device)
        ).to(dtype)[0]

        H = full_thinker_hidden.shape[-1]

        seg_input_ids: List[int] = []
        seg_input_text_ids: List[int] = []
        seg_embeds_list: List[torch.Tensor] = []  # each [seg_len, H]

        N = len(history_chunks) + 1
        for i in range(N):
            t_start, t_end = thinker_spans[i]
            r_start, r_end = reply_spans[i]
            chunk_thinker_len = t_end - t_start
            assert chunk_thinker_len > 0
            assert r_end > r_start

            # mask region: codec_mask token, embed = thinker_cond[t_start:t_end]
            seg_input_ids.extend([codec_mask] * chunk_thinker_len)
            seg_input_text_ids.extend(full_thinker_ids_list[t_start:t_end])
            seg_embeds_list.append(thinker_cond[0, t_start:t_end])

            is_last = i == N - 1
            if not is_last:
                # full chunk: pad, bos, codes, eos with codec+text-cond additives.
                # L == 0 is the legitimate idle/wait pattern from training (code
                # ranges like [k, k] on `<|idle|>` chunks); we emit
                # [mask][pad][bos][eos] with no codes between, matching
                # `_build_talker_for_sample` behaviour for code_len == 0.
                codes_i = history_chunks[i]["codes"].to(device=device, dtype=torch.long).reshape(-1)
                L = codes_i.shape[0]

                # pad position: codec_pad + text_bos
                seg_input_ids.append(codec_pad)
                seg_input_text_ids.append(text_bos)
                seg_embeds_list.append((codec_pad_embed + text_bos_embed).unsqueeze(0))

                # bos position: codec_bos + thinker_cond[r_start]
                seg_input_ids.append(codec_bos)
                seg_input_text_ids.append(full_thinker_ids_list[r_start])
                seg_embeds_list.append((codec_bos_embed + thinker_cond[0, r_start]).unsqueeze(0))

                # codes positions: codec_embed(code_j) + tail_cond[j]
                # tail_cond order matches training: tail_positions[k] for k in [0, L]
                # k=0..r_end-r_start-2 -> thinker_cond[r_start+1+k]
                # k=r_end-r_start-1    -> text_eos_embed
                # k>=r_end-r_start     -> text_pad_embed
                # We need L+1 entries (covers L code positions + 1 eos position).
                tail_len_supervised = L + 1
                tail_embeds = torch.zeros((tail_len_supervised, H), device=device, dtype=dtype)
                tail_text_ids: List[int] = []
                base_tail_len = r_end - r_start - 1
                for k in range(tail_len_supervised):
                    if k < base_tail_len:
                        idx = r_start + 1 + k
                        tail_embeds[k] = thinker_cond[0, idx]
                        tail_text_ids.append(full_thinker_ids_list[idx])
                    elif k == base_tail_len:
                        tail_embeds[k] = text_eos_embed
                        tail_text_ids.append(text_eos)
                    else:
                        tail_embeds[k] = text_pad_embed
                        tail_text_ids.append(text_pad)

                if L > 0:
                    code_emb = talker_embed_tokens(codes_i).to(dtype)  # [L, H]
                    code_seg_embeds = code_emb + tail_embeds[:L]
                    seg_input_ids.extend(codes_i.tolist())
                    seg_input_text_ids.extend(tail_text_ids[:L])
                    seg_embeds_list.append(code_seg_embeds)

                # eos position: codec_eos + tail_cond[L]
                seg_input_ids.append(codec_eos)
                seg_input_text_ids.append(tail_text_ids[L])
                seg_embeds_list.append((codec_eos_embed + tail_embeds[L]).unsqueeze(0))
            else:
                # current chunk N: only [pad][bos]; do NOT add codec_pad_embed /
                # codec_bos_embed because the talker model itself adds them at
                # inputs_embeds[:, -2/-1, :] on the first forward call (matches
                # offline `generate_semantic_codes`).
                seg_input_ids.append(codec_pad)
                seg_input_text_ids.append(text_bos)
                seg_embeds_list.append(text_bos_embed.unsqueeze(0))

                seg_input_ids.append(codec_bos)
                seg_input_text_ids.append(full_thinker_ids_list[r_start])
                seg_embeds_list.append(thinker_cond[0, r_start].unsqueeze(0))

        talker_input_ids = torch.tensor([seg_input_ids], dtype=torch.long, device=device)
        talker_input_text_ids = torch.tensor([seg_input_text_ids], dtype=torch.long, device=device)
        talker_inputs_embeds = torch.cat(seg_embeds_list, dim=0).unsqueeze(0)  # [1, T, H]
        assert talker_inputs_embeds.shape[1] == talker_input_ids.shape[1], (
            f"talker embeds length {talker_inputs_embeds.shape[1]} != input_ids length "
            f"{talker_input_ids.shape[1]}"
        )
        talker_attention_mask = torch.ones_like(talker_input_ids)

        # ---- Step 6: build thinker_reply_part for chunk-N tail ----
        r_start_N, r_end_N = reply_spans[-1]
        base_tail_N = thinker_cond[0, r_start_N + 1 : r_end_N]  # [T1, H]
        thinker_reply_part_pieces = []
        if base_tail_N.shape[0] > 0:
            thinker_reply_part_pieces.append(base_tail_N)
        thinker_reply_part_pieces.append(text_eos_embed.unsqueeze(0))
        thinker_reply_part_pieces.append(text_pad_embed.unsqueeze(0))
        thinker_reply_part = torch.cat(thinker_reply_part_pieces, dim=0).unsqueeze(0)

        # ---- Step 7: talker.generate ----
        user_logits_processor = talker_generation_kwargs.pop("logits_processor", None)
        talker_generation_kwargs.setdefault("suppress_tokens", [self.talker.codec_bos_token])
        talker_generation_kwargs.setdefault("max_new_tokens", max_audio_tokens)
        talker_generation_kwargs.setdefault("eos_token_id", self.talker.codec_eos_token)
        talker_generation_kwargs.setdefault("pad_token_id", self.talker.codec_pad_token)
        talker_generation_kwargs.setdefault("do_sample", do_sample)
        talker_generation_kwargs.setdefault("temperature", temperature)
        talker_generation_kwargs.setdefault("top_k", top_k)
        talker_generation_kwargs.setdefault("top_p", top_p)
        talker_generation_kwargs.setdefault("repetition_penalty", repetition_penalty)
        talker_generation_kwargs.setdefault("no_repeat_ngram_size", talker_no_repeat_ngram_size)
        talker_generation_kwargs.setdefault("return_dict_in_generate", True)
        talker_generation_kwargs.setdefault("output_scores", True)
        talker_generation_kwargs.setdefault("num_beams", 1)
        if use_talker_kv_cache:
            talker_generation_kwargs.setdefault("use_cache", True)

        talker_num_beams = int(talker_generation_kwargs.get("num_beams", 1))
        use_talker_cache_for_chunk = use_talker_kv_cache and talker_num_beams == 1
        cached_talker_continuation = use_talker_cache_for_chunk and talker_past_key_values is not None
        generation_prompt_len = 1 if cached_talker_continuation else talker_input_ids.shape[1]
        logits_processor = self._build_force_prefix_logits_processor(
            force_prefix_codes=None,
            initial_input_len=generation_prompt_len,
            logits_processor=user_logits_processor,
        )
        # Optional: force >= min_audio_tokens semantic codes per chunk by
        # suppressing codec_eos for the first `min_audio_tokens` generation
        # steps. Default is 0 (disabled) because training data DOES contain
        # legitimate L=0 idle chunks (code_range=[k, k] with `<|idle|>` text)
        # and the talker is supervised to emit eos directly after bos in that
        # case. Only set min_audio_tokens > 0 for diagnostic / fallback runs
        # where you want to suppress idle-chunk eos.
        if min_audio_tokens > 0:
            class _MinCodesSuppressEos(LogitsProcessor):
                def __init__(self, eos_token: int, prompt_len: int, min_codes: int):
                    self.eos_token = int(eos_token)
                    self.prompt_len = int(prompt_len)
                    self.min_codes = int(min_codes)

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                    step = input_ids.shape[1] - self.prompt_len
                    if step < self.min_codes:
                        scores = scores.clone()
                        scores[:, self.eos_token] = float("-inf")
                    return scores

            min_codes_processor = _MinCodesSuppressEos(
                eos_token=self.talker.codec_eos_token,
                prompt_len=generation_prompt_len,
                min_codes=int(min_audio_tokens),
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([min_codes_processor])
            else:
                logits_processor.append(min_codes_processor)
        if logits_processor is not None:
            talker_generation_kwargs["logits_processor"] = logits_processor

        full_talker_position_ids, full_talker_rope_deltas = self.talker.get_rope_index(
            talker_input_text_ids,
            None,
            None,
            talker_attention_mask,
            None,
            None,
            None,
        )

        def _talker_forward_prefill(
            *,
            input_ids: torch.LongTensor,
            input_text_ids: torch.LongTensor,
            inputs_embeds: torch.Tensor,
            past_key_values: Any,
            position_ids: Optional[torch.LongTensor],
        ) -> Any:
            seq_len = int(input_ids.shape[1])
            assert seq_len > 0, "talker prefill requires at least one token"
            past_seen = int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen,
                past_seen + seq_len,
                device=device,
                dtype=torch.long,
            )
            attention_mask = torch.ones((1, past_seen + seq_len), device=device, dtype=torch.long)
            outputs = self.talker(
                input_ids=input_ids,
                input_text_ids=input_text_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            return outputs.past_key_values

        def _talker_forward_step(
            *,
            input_ids: torch.LongTensor,
            input_text_ids: torch.LongTensor,
            inputs_embeds: torch.Tensor,
            past_key_values: Any,
            position_ids: Optional[torch.LongTensor],
        ):
            seq_len = int(input_ids.shape[1])
            past_seen = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + seq_len,
                device=device,
                dtype=torch.long,
            )
            attention_mask = torch.ones((1, past_seen + seq_len), device=device, dtype=torch.long)
            return self.talker(
                input_ids=input_ids,
                input_text_ids=input_text_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )

        def _tail_cond_for_current_chunk(tail_idx: int):
            r_start_cur, r_end_cur = reply_spans[-1]
            base_tail_len_cur = r_end_cur - r_start_cur - 1
            if tail_idx < base_tail_len_cur:
                text_idx = r_start_cur + 1 + tail_idx
                return thinker_cond[0, text_idx], full_thinker_ids_list[text_idx]
            if tail_idx == base_tail_len_cur:
                return text_eos_embed, text_eos
            return text_pad_embed, text_pad

        def _manual_talker_decode(
            *,
            next_scores: torch.Tensor,
            past_key_values: Any,
            manual_sequence: torch.LongTensor,
        ) -> Tuple[torch.Tensor, Any]:
            generated_code_ids: List[int] = []
            do_sample_talker = bool(talker_generation_kwargs.get("do_sample", False))
            temperature_talker = float(talker_generation_kwargs.get("temperature", 1.0))
            top_k_talker = int(talker_generation_kwargs.get("top_k", 0) or 0)
            top_p_talker = float(talker_generation_kwargs.get("top_p", 1.0))
            repetition_penalty_talker = float(talker_generation_kwargs.get("repetition_penalty", 1.0))
            no_repeat_ngram_talker = int(talker_generation_kwargs.get("no_repeat_ngram_size", 0) or 0)
            max_new_tokens_talker = int(talker_generation_kwargs.get("max_new_tokens", max_audio_tokens))
            current_talker_past_key_values = past_key_values

            for _ in range(max_new_tokens_talker):
                scores = next_scores.float()
                eos_score = scores[:, codec_eos].clone()
                scores[:, SEMANTIC_CODE_VOCAB_SIZE:] = float("-inf")
                scores[:, codec_eos] = eos_score
                scores[:, codec_bos] = float("-inf")
                if repetition_penalty_talker != 1.0:
                    for prev_token in set(int(x) for x in manual_sequence[0].tolist()):
                        if scores[0, prev_token] < 0:
                            scores[0, prev_token] *= repetition_penalty_talker
                        else:
                            scores[0, prev_token] /= repetition_penalty_talker
                if no_repeat_ngram_talker > 0 and manual_sequence.shape[1] + 1 >= no_repeat_ngram_talker:
                    prefix = tuple(manual_sequence[0, -(no_repeat_ngram_talker - 1) :].tolist())
                    seq = manual_sequence[0].tolist()
                    banned = []
                    for i in range(len(seq) - no_repeat_ngram_talker + 1):
                        if tuple(seq[i : i + no_repeat_ngram_talker - 1]) == prefix:
                            banned.append(seq[i + no_repeat_ngram_talker - 1])
                    if banned:
                        scores[0, banned] = float("-inf")
                if logits_processor is not None:
                    scores = logits_processor(manual_sequence, scores)

                if do_sample_talker:
                    sample_scores = scores
                    if temperature_talker != 1.0:
                        sample_scores = sample_scores / temperature_talker
                    if top_k_talker > 0:
                        topk_values, _ = torch.topk(sample_scores, min(top_k_talker, sample_scores.shape[-1]))
                        sample_scores = sample_scores.masked_fill(sample_scores < topk_values[:, [-1]], float("-inf"))
                    if top_p_talker < 1.0:
                        sorted_logits, sorted_indices = torch.sort(sample_scores, descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p_talker
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        remove_indices = sorted_indices[sorted_indices_to_remove]
                        sample_scores[:, remove_indices] = float("-inf")
                    probs = torch.softmax(sample_scores, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(scores, dim=-1, keepdim=True)

                next_token_id = int(next_token[0, 0].item())
                if next_token_id == codec_eos:
                    break
                if 0 <= next_token_id < SEMANTIC_CODE_VOCAB_SIZE:
                    generated_code_ids.append(next_token_id)

                tail_embed, tail_text_id = _tail_cond_for_current_chunk(len(generated_code_ids) - 1)
                code_input_ids = next_token.to(device=device, dtype=torch.long)
                code_input_text_ids = torch.tensor([[tail_text_id]], dtype=torch.long, device=device)
                code_inputs_embeds = talker_embed_tokens(code_input_ids).to(dtype) + tail_embed.view(1, 1, -1)
                step_outputs = _talker_forward_step(
                    input_ids=code_input_ids,
                    input_text_ids=code_input_text_ids,
                    inputs_embeds=code_inputs_embeds,
                    past_key_values=current_talker_past_key_values,
                    position_ids=None,
                )
                current_talker_past_key_values = step_outputs.past_key_values
                manual_sequence = torch.cat([manual_sequence, code_input_ids], dim=1)
                next_scores = step_outputs.logits[:, -1, :]

            eos_tail_embed, eos_text_id = _tail_cond_for_current_chunk(len(generated_code_ids))
            eos_input_ids = torch.tensor([[codec_eos]], dtype=torch.long, device=device)
            eos_input_text_ids = torch.tensor([[eos_text_id]], dtype=torch.long, device=device)
            eos_inputs_embeds = (codec_eos_embed + eos_tail_embed).view(1, 1, -1)
            next_past_key_values = _talker_forward_prefill(
                input_ids=eos_input_ids,
                input_text_ids=eos_input_text_ids,
                inputs_embeds=eos_inputs_embeds,
                past_key_values=current_talker_past_key_values,
                position_ids=None,
            )
            generated_codes_tensor = torch.tensor(generated_code_ids, dtype=torch.long, device=device)
            print(
                "[talker-cache-debug] manual_decode "
                f"prompt_len={manual_sequence.shape[1]} generated_codes={generated_codes_tensor.numel()} "
                f"next_cache_len={next_past_key_values.get_seq_length()}",
                flush=True,
            )
            return generated_codes_tensor, next_past_key_values

        if cached_talker_continuation:
            # Keep the RoPE offset that created the cached keys. Recomputing
            # rope_deltas from the longer full prefix would assign new tokens a
            # different coordinate system from the historical cached keys.
            assert getattr(self.talker, "rope_deltas", None) is not None, "cached talker continuation lost rope_deltas"
            # Continue from the cached history. The cached state already includes
            # the previous chunk boundary eos, so prefill only current mask + pad.
            t_start_N, t_end_N = thinker_spans[-1]
            r_start_N, _ = reply_spans[-1]
            current_mask_len = t_end_N - t_start_N
            current_talker_start = talker_input_ids.shape[1] - (current_mask_len + 2)
            talker_cache_len = int(talker_past_key_values.get_seq_length())
            if talker_cache_len != current_talker_start:
                history_block_lens = []
                for history_idx, history_chunk in enumerate(history_chunks):
                    history_thinker_len = len(history_chunk["thinker_token_ids"])
                    history_code_len = int(history_chunk["codes"].reshape(-1).numel())
                    history_block_lens.append(
                        {
                            "i": history_idx,
                            "thinker_len": history_thinker_len,
                            "code_len": history_code_len,
                            "block_len": history_thinker_len + 3 + history_code_len,
                        }
                    )
                print(
                    "[talker-cache-debug] cache/prefix mismatch "
                    f"cache_len={talker_cache_len} expected_history_prefix={current_talker_start} "
                    f"talker_input_len={talker_input_ids.shape[1]} current_mask_len={current_mask_len} "
                    f"history_blocks={history_block_lens}",
                    flush=True,
                )
            current_prefill_input_ids = torch.tensor(
                [[*([codec_mask] * current_mask_len), codec_pad]],
                dtype=torch.long,
                device=device,
            )
            current_prefill_input_text_ids = torch.tensor(
                [full_thinker_ids_list[t_start_N:t_end_N] + [text_bos]],
                dtype=torch.long,
                device=device,
            )
            current_prefill_inputs_embeds = torch.cat(
                [
                    thinker_cond[0, t_start_N:t_end_N],
                    (codec_pad_embed + text_bos_embed).unsqueeze(0),
                ],
                dim=0,
            ).unsqueeze(0)
            current_talker_past_key_values = _talker_forward_prefill(
                input_ids=current_prefill_input_ids,
                input_text_ids=current_prefill_input_text_ids,
                inputs_embeds=current_prefill_inputs_embeds,
                past_key_values=talker_past_key_values,
                position_ids=None,
            )
            seed_input_ids = torch.tensor([[codec_bos]], dtype=torch.long, device=device)
            seed_input_text_ids = torch.tensor([[full_thinker_ids_list[r_start_N]]], dtype=torch.long, device=device)
            seed_inputs_embeds = (codec_bos_embed + thinker_cond[0, r_start_N]).view(1, 1, -1)
            step_outputs = _talker_forward_step(
                input_ids=seed_input_ids,
                input_text_ids=seed_input_text_ids,
                inputs_embeds=seed_inputs_embeds,
                past_key_values=current_talker_past_key_values,
                position_ids=None,
            )
            current_talker_past_key_values = step_outputs.past_key_values
            next_scores = step_outputs.logits[:, -1, :]
            generated_codes, talker_next_past_key_values = _manual_talker_decode(
                next_scores=next_scores,
                past_key_values=current_talker_past_key_values,
                manual_sequence=seed_input_ids.clone(),
            )
            self.last_talker_generation_diagnostics = {}
        elif use_talker_cache_for_chunk:
            if hasattr(self.talker, "rope_deltas"):
                self.talker.rope_deltas = None
            step_outputs = self.talker(
                input_ids=talker_input_ids,
                input_text_ids=talker_input_text_ids,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                use_cache=True,
                return_dict=True,
            )
            generated_codes, talker_next_past_key_values = _manual_talker_decode(
                next_scores=step_outputs.logits[:, -1, :],
                past_key_values=step_outputs.past_key_values,
                manual_sequence=talker_input_ids,
            )
            self.last_talker_generation_diagnostics = {}
        else:
            # Reset rope_deltas so get_rope_index is computed from input_text_ids fresh.
            if hasattr(self.talker, "rope_deltas"):
                self.talker.rope_deltas = None
            talker_result = self.talker.generate(
                input_ids=talker_input_ids,
                input_text_ids=talker_input_text_ids,
                thinker_reply_part=thinker_reply_part,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                **talker_generation_kwargs,
            )
            cache_eos_base_len = talker_input_ids.shape[1]
            talker_sequences = talker_result.sequences if hasattr(talker_result, "sequences") else talker_result
            self.last_talker_generation_diagnostics = self._extract_talker_generation_diagnostics(
                talker_result=talker_result,
                prompt_len=generation_prompt_len,
                forced_prefix_len=0,
            )
            generated = talker_sequences[:, generation_prompt_len:]
            if generated.shape[1] > 0 and generated[0, -1].item() == self.talker.codec_eos_token:
                generated = generated[:, :-1]
            valid_mask = (generated >= 0) & (generated < SEMANTIC_CODE_VOCAB_SIZE)
            generated_codes = generated[valid_mask]
            talker_next_past_key_values = None

            if use_talker_cache_for_chunk:
                raw_talker_next_past_key_values = talker_result.past_key_values
                assert raw_talker_next_past_key_values is not None, "talker.generate did not return past_key_values"
                generated_code_len = int(generated_codes.numel())
                raw_cache_len = int(raw_talker_next_past_key_values.get_seq_length())
                expected_without_eos = int(cache_eos_base_len + generated_code_len)
                expected_with_eos = expected_without_eos + 1
                assert raw_cache_len in {expected_without_eos, expected_with_eos}, (
                    f"Unexpected talker cache length {raw_cache_len}; expected {expected_without_eos} "
                    f"or {expected_with_eos}"
                )
                if raw_cache_len == expected_with_eos:
                    talker_next_past_key_values = raw_talker_next_past_key_values
                else:
                    eos_tail_embed, eos_text_id = _tail_cond_for_current_chunk(int(generated_codes.numel()))
                    eos_input_ids = torch.tensor([[codec_eos]], dtype=torch.long, device=device)
                    eos_input_text_ids = torch.tensor([[eos_text_id]], dtype=torch.long, device=device)
                    eos_inputs_embeds = (codec_eos_embed + eos_tail_embed).view(1, 1, -1)
                    talker_next_past_key_values = _talker_forward_prefill(
                        input_ids=eos_input_ids,
                        input_text_ids=eos_input_text_ids,
                        inputs_embeds=eos_inputs_embeds,
                        past_key_values=raw_talker_next_past_key_values,
                        position_ids=None,
                    )

        current_chunk_record = {
            "thinker_token_ids": full_thinker_ids_list[chunk_N_start:chunk_N_end],
            "reply_local_span": chunk_N_reply_span,
        }

        if generated_codes.numel() == 0:
            # Zero-code chunks are an in-distribution training pattern (idle
            # chunks have code_range=[k, k] and the talker is supervised to
            # emit eos right after bos). Carry an empty codes tensor so the
            # next chunk sees [mask][pad][bos][eos] in its talker prefix,
            # exactly matching training.
            assert allow_empty_codes, (
                "talker generated zero semantic codes for streaming chunk; "
                "pass allow_empty_codes=True (default) to allow idle chunks"
            )
            current_chunk_record["codes"] = generated_codes.new_empty(0).long()
            return (
                generated_codes.new_empty((1, 0)),
                thinker_text,
                current_chunk_record,
                thinker_next_past_key_values,
                talker_next_past_key_values,
            )

        codes_out = generated_codes.unsqueeze(0)  # [1, L_N]
        current_chunk_record["codes"] = codes_out[0].detach().cpu().long()
        return codes_out, thinker_text, current_chunk_record, thinker_next_past_key_values, talker_next_past_key_values

    @torch.inference_mode()
    def synthesize(
        self,
        tgt_text: str,
        ref_text: Optional[str],
        ref_audio_16k: torch.Tensor,
        ref_audio_24k: torch.Tensor,
        prompt_format: str = "unpaired",
        prefill_only: bool = False,
        ref_lang: str = "Chinese",
        tgt_lang: str = "English",
        max_text_tokens: int = 256,
        max_audio_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        talker_no_repeat_ngram_size: int = 0,
        use_prompt_mel: bool = True,
        thinker_generation_kwargs: Optional[Dict[str, Any]] = None,
        talker_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, torch.Tensor, str]:
        assert ref_audio_16k is not None and ref_audio_24k is not None, "reference audio is required"
        assert prompt_format in {"unpaired", "paired"}, f"Unsupported prompt_format: {prompt_format}"
        assert self.audio_tokenizer is not None, "audio_tokenizer required for extracting ref_codes"

        if prompt_format == "paired":
            if prefill_only:
                prompt = self.build_paired_prefill_prompt(
                    tgt_text=tgt_text,
                    ref_audio_16k=ref_audio_16k,
                    ref_lang=ref_lang,
                    tgt_lang=tgt_lang,
                )
            else:
                prompt = self.build_paired_prompt(
                    tgt_text=tgt_text,
                    ref_audio_16k=ref_audio_16k,
                    ref_lang=ref_lang,
                    tgt_lang=tgt_lang,
                )
        else:
            assert ref_text is not None and ref_text.strip(), "unpaired ICL mode requires ref_text"
            if prefill_only:
                prompt = self.build_unpaired_prefill_prompt(
                    ref_text=ref_text,
                    tgt_text=tgt_text,
                    tgt_lang=tgt_lang,
                )
            else:
                prompt = self.build_unpaired_prompt(
                    ref_text=ref_text,
                    tgt_text=tgt_text,
                    tgt_lang=tgt_lang,
                )

        ref_codes = self.audio_tokenizer(ref_audio_16k.cpu().numpy())
        if isinstance(ref_codes, np.ndarray):
            ref_codes = torch.from_numpy(ref_codes)
        if ref_codes.dim() == 1:
            ref_codes = ref_codes.unsqueeze(0)
        ref_codes = ref_codes.to(device=self.device, dtype=torch.long)
        force_prefix_codes = ref_codes[0] if prompt_format == "unpaired" else None

        generated_codes, thinker_text = self.generate_semantic_codes(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            input_features=prompt.get("input_features"),
            feature_attention_mask=prompt.get("feature_attention_mask"),
            prefill_only=prefill_only,
            prefix_len=prompt.get("prefix_len", 0),
            reply_len=prompt.get("reply_len", 0),
            max_text_tokens=max_text_tokens,
            max_audio_tokens=max_audio_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
            force_prefix_codes=force_prefix_codes,
            thinker_generation_kwargs=thinker_generation_kwargs,
            talker_generation_kwargs=talker_generation_kwargs,
        )

        prompt_mel = None
        if use_prompt_mel:
            prompt_mel = self.extract_mel_feature(ref_audio_24k.unsqueeze(0).to(self.device))

        mel = self.codes_to_mel(
            generated_codes=generated_codes,
            prompt_codes=ref_codes,
            prompt_mel=prompt_mel,
        )
        audio = self.mel_to_audio(mel)
        if prompt_format == "unpaired":
            audio = self._trim_prompt_prefix_audio(audio=audio, full_mel=mel, prompt_mel=prompt_mel)
        return audio, generated_codes, thinker_text

    @torch.inference_mode()
    def e2e(
        self,
        audio_16k: torch.Tensor,
        audio_24k: torch.Tensor,
        ref_text: str = "",
        source_lang: str = "Chinese",
        target_lang: str = "English",
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_s2tt_new_tokens: int = 256,
        max_tts_new_tokens: int = 500,
        talker_no_repeat_ngram_size: int = 0,
        no_repeat_ngram_size: int = 0,
        use_prompt_mel: bool = True,
        thinker_generation_kwargs: Optional[Dict[str, Any]] = None,
        talker_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, str, torch.Tensor]:
        del ref_text
        ref_audio_16k = audio_16k.detach().cpu() if torch.is_tensor(audio_16k) else torch.as_tensor(audio_16k)
        ref_audio_24k = audio_24k.detach().cpu() if torch.is_tensor(audio_24k) else torch.as_tensor(audio_24k)

        prompt = self.build_s2s_prompt(
            ref_audio_16k=ref_audio_16k,
            ref_lang=source_lang,
            tgt_lang=target_lang,
        )
        generated_codes, thinker_text = self.generate_semantic_codes(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            input_features=prompt["input_features"],
            feature_attention_mask=prompt["feature_attention_mask"],
            max_text_tokens=max_s2tt_new_tokens,
            max_audio_tokens=max_tts_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
            thinker_generation_kwargs=thinker_generation_kwargs,
            talker_generation_kwargs=talker_generation_kwargs,
        )

        ref_codes = self.audio_tokenizer(ref_audio_16k.numpy())
        if isinstance(ref_codes, np.ndarray):
            ref_codes = torch.from_numpy(ref_codes)
        if ref_codes.dim() == 1:
            ref_codes = ref_codes.unsqueeze(0)
        ref_codes = ref_codes.to(device=self.device, dtype=torch.long)

        prompt_mel = None
        if use_prompt_mel:
            prompt_mel = self.extract_mel_feature(ref_audio_24k.unsqueeze(0).to(self.device))
        mel = self.codes_to_mel(
            generated_codes=generated_codes,
            prompt_codes=ref_codes,
            prompt_mel=prompt_mel,
        )
        audio = self.mel_to_audio(mel)
        return audio, thinker_text, generated_codes


def single_sample_inference(
    checkpoint_path: str,
    omni_model_path: str,
    voicebox_path: str,
    vocos_path: str,
    voicebox_config: str,
    ref_audio_path: str,
    ref_text: str,
    tgt_text: str,
    output_path: str,
    prompt_format: str = "unpaired",
    prefill_only: bool = False,
    ref_lang: str = "Chinese",
    tgt_lang: str = "English",
    device: str = "cuda:0",
    max_text_tokens: int = 256,
    max_audio_tokens: int = 500,
    do_sample: bool = True,
    talker_no_repeat_ngram_size: int = 0,
    thinker_generation_kwargs: Optional[Dict[str, Any]] = None,
    talker_generation_kwargs: Optional[Dict[str, Any]] = None,
):
    model = OmniTalkerSemanticTTSInference.from_pretrained(
        checkpoint_path=checkpoint_path,
        omni_model_path=omni_model_path,
        voicebox_path=voicebox_path,
        vocos_path=vocos_path,
        voicebox_config_path=voicebox_config,
        device=device,
    )

    ref_wav_16k = load_audio_mono(ref_audio_path, 16000)
    ref_wav_24k = load_audio_mono(ref_audio_path, 24000)
    audio, codes, thinker_text = model.synthesize(
        tgt_text=tgt_text,
        ref_text=ref_text,
        ref_audio_16k=ref_wav_16k,
        ref_audio_24k=ref_wav_24k,
        prompt_format=prompt_format,
        prefill_only=prefill_only,
        ref_lang=ref_lang,
        tgt_lang=tgt_lang,
        max_text_tokens=max_text_tokens,
        max_audio_tokens=max_audio_tokens,
        do_sample=do_sample,
        talker_no_repeat_ngram_size=talker_no_repeat_ngram_size,
        thinker_generation_kwargs=thinker_generation_kwargs,
        talker_generation_kwargs=talker_generation_kwargs,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, 24000)
    np.save(output_path.with_suffix(".codes.npy"), codes.squeeze(0).cpu().numpy())
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ref_audio_path": ref_audio_path,
                "ref_text": ref_text,
                "tgt_text": tgt_text,
                "thinker_text": thinker_text,
                "prompt_format": prompt_format,
                "prefill_only": prefill_only,
                "ref_lang": ref_lang,
                "tgt_lang": tgt_lang,
                "num_codes": int(codes.shape[1]),
                "output_wav": str(output_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return audio, codes, thinker_text


def main():
    parser = argparse.ArgumentParser(description="Infer semantic TTS with OmniTalker + VoiceBox")
    parser.add_argument("--checkpoint_path", type=str, default=str(MODEL_ROOT / "offline"))
    parser.add_argument("--omni_model_path", type=str, default=str(MODEL_ROOT / "offline"))
    parser.add_argument("--voicebox_path", type=str, default=str(MODEL_ROOT / "voicebox" / "voicebox.safetensors"))
    parser.add_argument("--vocos_path", type=str, default=str(MODEL_ROOT / "voicebox" / "vocos.safetensors"))
    parser.add_argument(
        "--voicebox_config",
        type=str,
        default=str(MODEL_ROOT / "voicebox" / "voicebox_config.json"),
    )
    parser.add_argument("--ref_audio_path", type=str, required=True)
    parser.add_argument("--ref_text", type=str, default="")
    parser.add_argument("--tgt_text", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_format", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--prefill_only", action="store_true")
    parser.add_argument("--ref_lang", type=str, default="Chinese")
    parser.add_argument("--tgt_lang", type=str, default="English")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_text_tokens", type=int, default=256)
    parser.add_argument("--max_audio_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--talker_no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--thinker_generation_kwargs", type=str, default="")
    parser.add_argument("--talker_generation_kwargs", type=str, default="")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    model = OmniTalkerSemanticTTSInference.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        omni_model_path=args.omni_model_path,
        voicebox_path=args.voicebox_path,
        vocos_path=args.vocos_path,
        voicebox_config_path=args.voicebox_config,
        device=args.device,
    )

    ref_wav_16k = load_audio_mono(args.ref_audio_path, 16000)
    ref_wav_24k = load_audio_mono(args.ref_audio_path, 24000)
    thinker_generation_kwargs = json.loads(args.thinker_generation_kwargs) if args.thinker_generation_kwargs else None
    talker_generation_kwargs = json.loads(args.talker_generation_kwargs) if args.talker_generation_kwargs else None
    audio, codes, thinker_text = model.synthesize(
        tgt_text=args.tgt_text,
        ref_text=args.ref_text,
        ref_audio_16k=ref_wav_16k,
        ref_audio_24k=ref_wav_24k,
        prompt_format=args.prompt_format,
        prefill_only=args.prefill_only,
        ref_lang=args.ref_lang,
        tgt_lang=args.tgt_lang,
        max_text_tokens=args.max_text_tokens,
        max_audio_tokens=args.max_audio_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        talker_no_repeat_ngram_size=args.talker_no_repeat_ngram_size,
        thinker_generation_kwargs=thinker_generation_kwargs,
        talker_generation_kwargs=talker_generation_kwargs,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, 24000)
    np.save(output_path.with_suffix(".codes.npy"), codes.squeeze(0).cpu().numpy())
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ref_audio_path": args.ref_audio_path,
                "ref_text": args.ref_text,
                "tgt_text": args.tgt_text,
                "thinker_text": thinker_text,
                "prompt_format": args.prompt_format,
                "prefill_only": args.prefill_only,
                "ref_lang": args.ref_lang,
                "tgt_lang": args.tgt_lang,
                "num_codes": int(codes.shape[1]),
                "output_wav": str(output_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved waveform to %s", output_path)
    logger.info("Saved semantic codes to %s", output_path.with_suffix(".codes.npy"))
    logger.info("Thinker text: %s", thinker_text)


if __name__ == "__main__":
    main()
