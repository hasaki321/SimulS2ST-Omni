#!/usr/bin/env python3
"""Thinker-only S2T/ASR inference for OmniTalker checkpoints."""

import gc
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.train.modeling_omni_talker import THINKER_LORA_DIR_NAME
from src.train.prompt_formats import (
    TASK_S2TT_ASR,
    TASK_S2TT_TRANSLATE,
    build_prompt_bundle,
    build_prompt_texts,
)

logger = logging.getLogger(__name__)


class OmniTalkerS2TInference:
    """Speech-to-text inference using the OmniTalker checkpoint loading contract.

    This class intentionally does not load talker/VoiceBox/Vocos. It reuses the
    thinker tokenizer, LoRA loading, prompt format, and generation style used by
    `infer_omni_talker_tts.py`.
    """

    def __init__(self, model, processor, device: str = "cuda:0"):
        self.model = model
        self.processor = processor
        self._device = device
        self.dtype = next(model.parameters()).dtype

    @property
    def device(self):
        return self._device

    @property
    def thinker(self):
        return self.model.thinker

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        omni_model_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        processor = Qwen2_5OmniProcessor.from_pretrained(omni_model_path)
        ckpt_tokenizer_path = os.path.join(checkpoint_path, "tokenizer_config.json")
        if os.path.exists(ckpt_tokenizer_path):
            logger.info("Loading tokenizer from checkpoint: %s", checkpoint_path)
            processor.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        logger.info("Loading base Qwen-Omni model from: %s", omni_model_path)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            omni_model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=device,
            trust_remote_code=True,
        )

        if hasattr(model, "token2wav"):
            del model.token2wav
        if getattr(model.thinker, "visual", None) is not None:
            logger.info("Removing visual module to save memory...")
            del model.thinker.visual
            model.thinker.visual = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ckpt_vocab_size = len(processor.tokenizer)
        thinker_vocab_size = model.thinker.get_input_embeddings().weight.shape[0]
        min_required_vocab_size = max(
            ckpt_vocab_size,
            model.config.thinker_config.audio_token_index + 1,
        )
        if thinker_vocab_size < min_required_vocab_size:
            logger.info(
                "Expanding thinker embeddings: %s -> %s (checkpoint tokenizer=%s)",
                thinker_vocab_size,
                min_required_vocab_size,
                ckpt_vocab_size,
            )
            model.thinker.resize_token_embeddings(min_required_vocab_size)
        else:
            logger.info(
                "Keeping thinker embeddings at %s (checkpoint tokenizer=%s, min_required=%s)",
                thinker_vocab_size,
                ckpt_vocab_size,
                min_required_vocab_size,
            )

        thinker_lora_dir = os.path.join(checkpoint_path, THINKER_LORA_DIR_NAME)
        if os.path.isdir(thinker_lora_dir):
            logger.info("Loading thinker LoRA from: %s", thinker_lora_dir)
            model.thinker = PeftModel.from_pretrained(model.thinker, thinker_lora_dir)
            model.thinker = model.thinker.merge_and_unload()

        model.eval()
        return cls(model=model, processor=processor, device=device)

    def length_shrink_func(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        _, output_lengths = self.thinker.audio_tower._get_feat_extract_output_lengths(input_lengths)
        return output_lengths.to(torch.long)

    def _preprocess_speech(self, audio_16k: np.ndarray):
        speech_inputs = self.processor.feature_extractor(
            [audio_16k],
            sampling_rate=16000,
            padding="max_length",
            return_attention_mask=True,
        )
        audio_features = torch.from_numpy(speech_inputs["input_features"])
        audio_mask = torch.from_numpy(speech_inputs["attention_mask"])
        n_frames = audio_mask.sum(dim=1)
        speech_lens = self.length_shrink_func(n_frames)
        return audio_features, audio_mask, speech_lens

    def build_s2tt_prompt(
        self,
        audio_16k: np.ndarray,
        source_lang: str = "English",
        target_lang: str = "Chinese",
    ) -> Dict:
        tokenizer = self.processor.tokenizer
        audio_features, audio_mask, speech_lens = self._preprocess_speech(audio_16k)
        n_audio_tokens = max(1, int(speech_lens[0].item()))
        task_type = TASK_S2TT_ASR if source_lang == target_lang else TASK_S2TT_TRANSLATE
        prompt_bundle = build_prompt_bundle(
            task_type=task_type,
            target_text="",
            src_lang=source_lang,
            tgt_lang=target_lang,
            n_audio_tokens=n_audio_tokens,
        )
        prefix_prompt, _ = build_prompt_texts(prompt_bundle)
        encoded = tokenizer(prefix_prompt, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "input_features": audio_features,
            "feature_attention_mask": audio_mask,
        }

    @torch.inference_mode()
    def generate_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_features: torch.FloatTensor = None,
        feature_attention_mask: torch.Tensor = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        tokenizer = self.processor.tokenizer
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert eos_token_id is not None, "Tokenizer must contain <|im_end|>"
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "top_k": top_k if do_sample else 50,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if input_features is not None:
            gen_kwargs["input_features"] = input_features.to(device=self.device, dtype=self.dtype)
        if feature_attention_mask is not None:
            gen_kwargs["feature_attention_mask"] = feature_attention_mask.to(device=self.device, dtype=torch.bool)

        outputs = self.thinker.generate(**gen_kwargs)
        generated_ids = outputs[0, input_ids.shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        for token in ["<|text_eos|>", "<|im_end|>", "<|im_start|>"]:
            text = text.replace(token, "")
        return text.strip()

    def translate_audio(
        self,
        audio_16k: np.ndarray,
        source_lang: str = "English",
        target_lang: str = "Chinese",
        **gen_kwargs,
    ) -> str:
        prompt = self.build_s2tt_prompt(audio_16k, source_lang, target_lang)
        return self.generate_text(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            input_features=prompt["input_features"],
            feature_attention_mask=prompt["feature_attention_mask"],
            **gen_kwargs,
        )

    def transcribe(self, audio_16k: np.ndarray, lang: str = "English", **gen_kwargs) -> str:
        return self.translate_audio(audio_16k, source_lang=lang, target_lang=lang, **gen_kwargs)
