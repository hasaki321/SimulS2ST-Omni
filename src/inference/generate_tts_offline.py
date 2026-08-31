#!/usr/bin/env python3
"""
Offline batch TTS inference with Semantic Code + VoiceBox + Vocos pipeline.

This script:
1. Uses Qwen-Omni thinker to generate semantic codes (16384 discrete tokens)
2. Uses VoiceBox (flow matching) to convert semantic codes to mel spectrogram
3. Uses Vocos (vocoder) to convert mel to audio

Usage:
    python generate_tts_offline.py \
        --checkpoint_path /path/to/checkpoint \
        --omni_model_path /path/to/Qwen2.5-Omni-7B \
        --voicebox_path /path/to/diffusion.safetensors \
        --vocos_path /path/to/vocos.safetensors \
        --test_pairs_path /path/to/eval_pairs.pkl \
        --audio_root /path/to/Emilia-101k \
        --output_dir /path/to/output \
        --max_samples 100
"""

import os
import sys
import math
import logging
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_ROOT = os.path.expanduser(
    os.environ.get("SIMULS2ST_MODEL_ROOT", os.path.join(PROJECT_ROOT, "models", "SimulS2ST-Omni"))
)
os.environ.setdefault("SIMULS2ST_W2V_BERT_PATH", os.path.join(MODEL_ROOT, "w2v"))
logger = logging.getLogger(__name__)

# Default special tokens
DEFAULT_SPECIAL_TOKENS = (
    "<|text_bos|>,<|text_eos|>,<|tts_bos|>,<|tts_eos|>,"
    "<|CODE|>,<|wait|>,<|idle|>,"
    "<|latency_1|>,<|latency_2|>,<|latency_3|>,<|latency_4|>,"
    "<|latency_5|>,<|latency_6|>,<|latency_7|>,<|latency_8|>,"
    "<|latency_9|>,<|latency_10|>,<|latency_11|>,<|latency_12|>"
)

THINKER_LORA_DIR_NAME = "thinker_lora"
SEMANTIC_CODE_VOCAB_SIZE = 16384

# Prompt tokens
DEFAULT_EOS_TOKEN = "<|im_end|>"
DEFAULT_BOS_TOKEN = "<|im_start|>"
DEFAULT_SPEECH_PATCH_TOKEN = "<|AUDIO|>"
DEFAULT_SPEECH_START_TOKEN = "<|audio_bos|>"
DEFAULT_SPEECH_END_TOKEN = "<|audio_eos|>"
DEFAULT_TEXT_END_TOKEN = "<|text_eos|>"
DEFAULT_TEXT_START_TOKEN = "<|text_bos|>"
DEFAULT_TTS_BOS_TOKEN = "<|tts_bos|>"
DEFAULT_TTS_EOS_TOKEN = "<|tts_eos|>"


class SemanticCodeTTSInference(nn.Module):
    """
    Inference wrapper for Semantic Code TTS.

    Pipeline:
    1. Qwen-Omni thinker generates semantic codes (next-token prediction)
    2. VoiceBox converts semantic codes to mel spectrogram (flow matching)
    3. Vocos converts mel to audio (vocoder)
    """

    def __init__(
        self,
        thinker: nn.Module,
        processor,
        code_token_offset: int,
        voicebox: nn.Module,
        vocoder: nn.Module,
        mel_model: nn.Module,
        voicebox_cfg: Any,
        audio_tokenizer: Any = None,
    ):
        super().__init__()
        self.thinker = thinker
        self.processor = processor
        self.code_token_offset = code_token_offset
        self.voicebox = voicebox
        self.vocoder = vocoder
        self.mel_model = mel_model
        self.voicebox_cfg = voicebox_cfg
        self.audio_tokenizer = audio_tokenizer

    @property
    def device(self):
        return next(self.thinker.parameters()).device

    @property
    def dtype(self):
        return next(self.thinker.parameters()).dtype

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
        attn_implementation: str = "sdpa",
        expected_special_tokens: str = DEFAULT_SPECIAL_TOKENS,
        semantic_code_vocab_size: int = SEMANTIC_CODE_VOCAB_SIZE,
    ) -> "SemanticCodeTTSInference":
        """Load inference model from a training checkpoint."""
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniForConditionalGeneration,
            AutoTokenizer,
        )
        from peft import PeftModel
        import safetensors.torch
        import accelerate
        from src.voicebox.util import load_config
        from src.voicebox.voicebox_model import VoiceBox
        from src.voicebox.vocos import Vocos
        from src.voicebox.melspec import MelSpectrogram

# Load processor from base model (has all components)
        processor = Qwen2_5OmniProcessor.from_pretrained(omni_model_path)

        # Check if checkpoint has tokenizer with additional tokens
        ckpt_tokenizer_path = os.path.join(checkpoint_path, "tokenizer_config.json")
        if os.path.exists(ckpt_tokenizer_path):
            logger.info(f"Loading tokenizer from checkpoint: {checkpoint_path}")
            # Only load tokenizer, keep other processor components from base
            processor.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        code_token_offset = processor.tokenizer.convert_tokens_to_ids("<|code_0|>")
        # Check if full fine-tuned or LoRA
        has_safetensors = any(
            f.endswith(".safetensors")
            for f in os.listdir(checkpoint_path)
            if os.path.isfile(os.path.join(checkpoint_path, f))
        )
        has_lora = os.path.isdir(os.path.join(checkpoint_path, "thinker_lora"))

        if has_safetensors and not has_lora:
            # Full fine-tuned model
            logger.info(f"Loading full fine-tuned model from: {checkpoint_path}")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map=device,
                trust_remote_code=True,
            )
        else:
            # Base model + optional LoRA
            logger.info(f"Loading base model from: {omni_model_path}")
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                omni_model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map=device,
                trust_remote_code=True,
            )

            # Resize embeddings if needed
            ckpt_vocab_size = len(processor.tokenizer)
            if model.thinker.get_input_embeddings().weight.shape[0] != ckpt_vocab_size:
                logger.info(f"Resizing embeddings: {model.thinker.get_input_embeddings().weight.shape[0]} → {ckpt_vocab_size}")
                model.thinker.resize_token_embeddings(ckpt_vocab_size)

            # Remove vision tower BEFORE loading LoRA to avoid missing keys warnings
            # (LoRA checkpoint may contain visual module params that we don't need)
            if getattr(model.thinker, "visual", None) is not None:
                import gc
                logger.info("Removing visual module before LoRA loading...")
                del model.thinker.visual
                model.thinker.visual = None
                gc.collect()
                torch.cuda.empty_cache()

            # Load LoRA if exists
            if has_lora:
                lora_path = os.path.join(checkpoint_path, "thinker_lora")
                logger.info(f"Loading LoRA from: {lora_path}")
                model.thinker = PeftModel.from_pretrained(model.thinker, lora_path)
                model.thinker = model.thinker.merge_and_unload()

        # Remove visual module for full fine-tuned model as well
        if getattr(model.thinker, "visual", None) is not None:
            import gc
            logger.info("Removing visual module to save memory...")
            del model.thinker.visual
            model.thinker.visual = None
            gc.collect()
            torch.cuda.empty_cache()

        model.eval()

        # --- 4. Load VoiceBox config ---
        logger.info(f"Loading VoiceBox config from {voicebox_config_path}")
        cfg = load_config(voicebox_config_path)

        # --- 5. Build VoiceBox ---
        logger.info(f"Loading VoiceBox from {voicebox_path}")
        voicebox = VoiceBox(cfg=cfg.model.voicebox)
        voicebox.eval()
        voicebox.to(device)
        safetensors.torch.load_model(voicebox, voicebox_path)

        # --- 6. Build Vocos ---
        logger.info(f"Loading Vocos from {vocos_path}")
        vocoder = Vocos(cfg=cfg.model.vocos)
        vocoder.eval()
        vocoder.to(device)
        accelerate.load_checkpoint_and_dispatch(vocoder, vocos_path)

        # --- 7. Build MelSpectrogram ---
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

        # --- 8. Build Audio Tokenizer (for extracting prompt codes from ref audio) ---
        logger.info("Loading Audio Tokenizer for prompt code extraction...")
        from src.voicebox.tokenizer import VoiceBoxAudioTokenizer
        if codec_model_path is None:
            codec_model_path = os.path.join(MODEL_ROOT, "dualcodec", "dualcodec.safetensors")
        if codec_stats_path is None:
            codec_stats_path = os.path.join(MODEL_ROOT, "dualcodec", "w2v_bert_stats.pt")
        cfg.model.dual_codec.pretrained_path = codec_model_path
        cfg.model.kmeans.stat_mean_var_path = codec_stats_path
        audio_tokenizer = VoiceBoxAudioTokenizer(cfg, device)

        model.eval()

        return cls(
            thinker=model.thinker,
            processor=processor,
            code_token_offset=code_token_offset,
            voicebox=voicebox,
            vocoder=vocoder,
            mel_model=mel_model,
            voicebox_cfg=cfg,
            audio_tokenizer=audio_tokenizer,
        )

    def length_shrink_func(self, input_lengths: torch.LongTensor):
        input_lengths = (input_lengths - 1) // 2 + 1
        input_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths.to(torch.long)

    def _preprocess_speech(self, wav_16k: torch.Tensor):
        """Convert 16kHz waveform to audio features."""
        speech_feature_batch = self.processor.feature_extractor(
            [wav_16k.numpy()],
            sampling_rate=16000,
            padding="longest",
            return_attention_mask=True,
        )
        audio_mask = torch.from_numpy(speech_feature_batch["attention_mask"])
        audio_features = torch.from_numpy(speech_feature_batch["input_features"])
        n_frames = audio_mask.sum(dim=1)
        speech_lens = self.length_shrink_func(n_frames)
        return audio_features, audio_mask, n_frames, speech_lens

    @torch.no_grad()
    def extract_mel_feature(self, speech_24k: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from 24kHz speech."""
        mel_feature = self.mel_model(speech_24k)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.voicebox_cfg.preprocess.mel_mean) / math.sqrt(
            self.voicebox_cfg.preprocess.mel_var
        )
        return mel_feature

    def build_tts_prompt_icl(
        self,
        ref_text: str,
        ref_codes: torch.Tensor,
        tgt_text: str,
        include_instruction: bool = True,
    ) -> dict:
        """
        Build TTS prompt using In-Context Learning (ICL) format.

        Format: instruction + text_turn(ref_text + tgt_text) + code_prefix + ref_codes
        Model continues to generate tgt_codes after ref_codes.

        Args:
            ref_text: Text content of the reference audio
            ref_codes: Semantic codes from reference audio, shape (T,) or (1, T)
            tgt_text: Target text to synthesize
            include_instruction: Whether to include system instruction

        Returns:
            dict with input_ids, attention_mask, and ref_code_len for downstream FM processing
        """
        tokenizer = self.processor.tokenizer

        instruction = ""
        if include_instruction:
            instruction = (
                f"<|im_start|>system\n"
                f"Generate target language audio or text given source language audio."
                f"<|im_end|>"
            )

        # Text turn: <bos><text_bos>ref_text + tgt_text<text_eos><eos>
        text_turn = (
            DEFAULT_BOS_TOKEN
            + DEFAULT_TEXT_START_TOKEN + ref_text + tgt_text + DEFAULT_TEXT_END_TOKEN
            + DEFAULT_EOS_TOKEN
        )

        # Code prefix: <bos><tts_bos>
        code_prefix = DEFAULT_BOS_TOKEN + DEFAULT_TTS_BOS_TOKEN

        # Full text prompt before codes
        text_prompt = instruction + text_turn + code_prefix
        prefix_ids = tokenizer.encode(text_prompt, add_special_tokens=False)

        # Convert ref_codes to token IDs
        if ref_codes.dim() == 2:
            ref_codes = ref_codes.squeeze(0)
        ref_codes = ref_codes.clamp(0, SEMANTIC_CODE_VOCAB_SIZE - 1)
        ref_code_token_ids = ref_codes.long() + self.code_token_offset

        # Assemble: text_prompt + ref_codes (model will continue generating tgt_codes)
        input_ids = torch.cat([
            torch.tensor(prefix_ids, dtype=torch.long),
            ref_code_token_ids.cpu(),
        ]).unsqueeze(0)  # (1, seq_len)

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ref_code_len": len(ref_code_token_ids),
        }

    @torch.inference_mode()
    def generate_semantic_codes(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        do_sample: bool = True,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """
        Generate semantic codes using the thinker model (text-only, no audio features).

        For ICL mode, the input_ids already contain the ref_codes as context.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        tokenizer = self.processor.tokenizer
        tts_eos_id = tokenizer.convert_tokens_to_ids(DEFAULT_TTS_EOS_TOKEN)
        eos_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)

        # Generate tokens (no audio features needed for ICL mode)
        outputs = self.thinker.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=[tts_eos_id, eos_id],
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        # Extract generated tokens (remove input)
        generated = outputs[0, input_ids.shape[1]:]

        # Filter to only code tokens and convert to code indices
        code_mask = (generated >= self.code_token_offset) & (generated < self.code_token_offset + SEMANTIC_CODE_VOCAB_SIZE)
        code_tokens = generated[code_mask]
        semantic_codes = code_tokens - self.code_token_offset

        # Warn if no codes generated
        if len(semantic_codes) == 0:
            logger.warning(f"No semantic codes generated! Generated {len(generated)} tokens")

        return semantic_codes.unsqueeze(0)  # (1, T)

    @torch.inference_mode()
    def codes_to_mel(
        self,
        generated_codes: torch.Tensor,
        prompt_codes: Optional[torch.Tensor] = None,
        prompt_mel: Optional[torch.Tensor] = None,
        n_timesteps: int = 16,
        cfg_scale: float = 2.0,
        rescale_cfg: float = 0.75,
    ) -> torch.Tensor:
        """
        Convert semantic codes to mel spectrogram using VoiceBox.

        IMPORTANT: VoiceBox expects:
        - cond_feature: embedding of (prompt_codes + generated_codes) concatenated
        - prompt_mel: mel spectrogram of prompt audio
        - target_len = cond.shape[1] - prompt_mel.shape[1] > 0

        So we must concatenate prompt_codes and generated_codes before embedding.
        """
        # Ensure codes are in valid range [0, 16384)
        generated_codes = generated_codes.clamp(0, SEMANTIC_CODE_VOCAB_SIZE - 1)

        # Concatenate prompt_codes + generated_codes to form combined_codes
        if prompt_codes is not None:
            prompt_codes = prompt_codes.clamp(0, SEMANTIC_CODE_VOCAB_SIZE - 1)
            combined_codes = torch.cat([prompt_codes, generated_codes], dim=1)
        else:
            combined_codes = generated_codes

        # Embed combined codes
        cond_feature = self.voicebox.cond_emb(combined_codes.to(self.device))
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=self.voicebox.cond_scale_factor,
        ).transpose(1, 2)

        # Generate mel using reverse diffusion
        predict_mel = self.voicebox.reverse_diffusion(
            cond_feature,
            prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg_scale,
            rescale_cfg=rescale_cfg,
        )

        return predict_mel

    @torch.inference_mode()
    def mel_to_audio(self, mel: torch.Tensor) -> np.ndarray:
        """Convert mel spectrogram to audio using Vocos."""
        audio = self.vocoder(mel.transpose(1, 2)).detach().cpu().numpy()[0][0]
        return audio

    @torch.inference_mode()
    def synthesize(
        self,
        tgt_text: str,
        ref_text: str,
        ref_audio_16k: torch.Tensor,
        ref_audio_24k: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.8,
        do_sample: bool = True,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 0,
        use_prompt_mel: bool = True,
        ref_codes: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Full TTS pipeline using ICL (In-Context Learning) approach.

        Pipeline:
        1. Extract ref_codes from ref_audio (or use provided ref_codes)
        2. Build ICL prompt: ref_text + ref_codes + tgt_text -> tgt_codes
        3. Generate tgt_codes using LLM
        4. VoiceBox: (ref_codes + tgt_codes) -> mel, with ref_codes as prompt (not denoised)
        5. Vocos: mel -> audio

        Args:
            tgt_text: Target text to synthesize
            ref_text: Text content of the reference audio (for ICL)
            ref_audio_16k: Reference audio at 16kHz (for extracting ref_codes)
            ref_audio_24k: Reference audio at 24kHz (for extracting prompt_mel)
            max_new_tokens: Maximum codes to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            no_repeat_ngram_size: N-gram blocking size
            use_prompt_mel: Whether to use prompt mel for voice cloning
            ref_codes: Pre-extracted reference codes (optional, will extract if not provided)

        Returns:
            audio: Generated audio as numpy array (24kHz)
            generated_codes: Generated semantic codes
        """
        # 1. Extract ref_codes from ref_audio if not provided
        if ref_codes is None:
            assert self.audio_tokenizer is not None, "audio_tokenizer required for extracting ref_codes"
            ref_audio_16k_np = ref_audio_16k.cpu().numpy()
            ref_codes = self.audio_tokenizer(ref_audio_16k_np)  # (1, T) tensor
            if isinstance(ref_codes, np.ndarray):
                ref_codes = torch.from_numpy(ref_codes)

        # Ensure ref_codes is on CPU for prompt building
        ref_codes_cpu = ref_codes.cpu()
        if ref_codes_cpu.dim() == 2:
            ref_codes_1d = ref_codes_cpu.squeeze(0)
        else:
            ref_codes_1d = ref_codes_cpu

        logger.info(f"ICL synthesis: ref_codes={len(ref_codes_1d)}, ref_text='{ref_text[:50]}...', tgt_text='{tgt_text[:50]}...'")

        # 2. Build ICL prompt: ref_text + ref_codes + tgt_text
        prompt = self.build_tts_prompt_icl(
            ref_text=ref_text,
            ref_codes=ref_codes_1d,
            tgt_text=tgt_text,
        )

        logger.info(f"ICL prompt length: {prompt['input_ids'].shape[1]} tokens")

        # 3. Generate tgt_codes
        generated_codes = self.generate_semantic_codes(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        logger.info(f"Generated {generated_codes.shape[1]} semantic codes")

        # 4. Extract prompt_mel for VoiceBox (if using prompt mel)
        prompt_mel = None
        if use_prompt_mel and ref_audio_24k is not None:
            prompt_mel = self.extract_mel_feature(
                ref_audio_24k.unsqueeze(0).to(self.device)
            )
            logger.info(f"Extracted prompt_mel: {prompt_mel.shape}")

        # 5. Convert codes to mel using VoiceBox
        # IMPORTANT: Pass ref_codes as prompt_codes - VoiceBox will concatenate them
        # and only denoise the tgt_codes part (ref_codes serve as prompt/condition)
        ref_codes_for_fm = ref_codes_1d.unsqueeze(0).to(self.device)
        mel = self.codes_to_mel(
            generated_codes=generated_codes,
            prompt_codes=ref_codes_for_fm,
            prompt_mel=prompt_mel,
        )

        # 6. Convert mel to audio
        audio = self.mel_to_audio(mel)

        return audio, generated_codes


class SemanticCodeTestDataset:
    """Load test data for offline evaluation with ICL format support."""

    def __init__(
        self,
        pairs_path: str,
        audio_root: str,
        max_samples: int = -1,
    ):
        self.audio_root = audio_root

        with open(pairs_path, "rb") as f:
            self.pairs = pickle.load(f)

        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]

        logger.info(f"Loaded {len(self.pairs)} test samples from {pairs_path}")

    def __len__(self):
        return len(self.pairs)

    def _code_path_to_audio_path(self, code_path: str, lang: str) -> str:
        filename = os.path.basename(code_path)
        audio_filename = filename.replace(".npz", ".mp3")
        return os.path.join(self.audio_root, lang, audio_filename)

    def __getitem__(self, idx: int) -> dict:
        ref_info, tgt_info = self.pairs[idx]

        ref_audio_path = self._code_path_to_audio_path(ref_info["code_path"], ref_info["lang"])
        tgt_code_path = tgt_info["code_path"]

        # Load reference audio
        ref_wav_16k, sr = torchaudio.load(ref_audio_path)
        if sr != 16000:
            ref_wav_16k = torchaudio.functional.resample(ref_wav_16k, sr, 16000)
        ref_wav_16k = ref_wav_16k[0]  # Take first channel

        ref_wav_24k, sr = torchaudio.load(ref_audio_path)
        if sr != 24000:
            ref_wav_24k = torchaudio.functional.resample(ref_wav_24k, sr, 24000)
        ref_wav_24k = ref_wav_24k[0]

        # Load ground truth semantic codes
        gt_codes = np.load(tgt_code_path)["data"]

        # Load reference codes for ICL (from ref_info)
        ref_code_path = ref_info["code_path"]
        ref_codes = np.load(ref_code_path)["data"]

        return {
            "id": f"sample_{idx}",
            "ref_wav_16k": ref_wav_16k,
            "ref_wav_24k": ref_wav_24k,
            "ref_text": ref_info["text"],  # Reference text for ICL
            "ref_codes": ref_codes,  # Reference codes for ICL (pre-extracted)
            "tgt_text": tgt_info["text"],  # Target text to synthesize
            "lang": tgt_info["lang_name"],
            "gt_codes": gt_codes,
            "duration": tgt_info["duration"],
        }


def main():
    parser = argparse.ArgumentParser(description="Semantic Code TTS Inference")
    parser.add_argument("--checkpoint_path", type=str, default=os.path.join(MODEL_ROOT, "offline"),
                        help="Path to the offline merged model")
    parser.add_argument("--omni_model_path", type=str, default=os.path.join(MODEL_ROOT, "offline"),
                        help="Path to the offline merged model")
    parser.add_argument("--voicebox_path", type=str, default=os.path.join(MODEL_ROOT, "voicebox", "voicebox.safetensors"),
                        help="Path to VoiceBox weights")
    parser.add_argument("--vocos_path", type=str, default=os.path.join(MODEL_ROOT, "voicebox", "vocos.safetensors"),
                        help="Path to Vocos weights (vocos.safetensors)")
    parser.add_argument("--voicebox_config", type=str,
                        default=os.path.join(MODEL_ROOT, "voicebox", "voicebox_config.json"),
                        help="Path to VoiceBox config")
    parser.add_argument("--test_pairs_path", type=str, required=True,
                        help="Path to test pairs pkl file")
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Root directory for audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated audio")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Enable sampling (vs greedy decoding)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty for generation")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="No repeat n-gram size (0 to disable)")
    parser.add_argument("--use_prompt_mel", action="store_true", default=True,
                        help="Use prompt mel for voice cloning")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model = SemanticCodeTTSInference.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        omni_model_path=args.omni_model_path,
        voicebox_path=args.voicebox_path,
        vocos_path=args.vocos_path,
        voicebox_config_path=args.voicebox_config,
        device=args.device,
    )

    # Load test data
    logger.info("Loading test data...")
    dataset = SemanticCodeTestDataset(
        pairs_path=args.test_pairs_path,
        audio_root=args.audio_root,
        max_samples=args.max_samples,
    )

    # Run inference
    logger.info(f"Running inference on {len(dataset)} samples...")
    results = []

    for idx in tqdm(range(len(dataset)), desc="Generating"):
        sample = dataset[idx]

        try:
            # ICL mode: use pre-extracted ref_codes if available
            ref_codes = torch.from_numpy(sample["ref_codes"]) if "ref_codes" in sample else None

            audio, codes = model.synthesize(
                tgt_text=sample["tgt_text"],
                ref_text=sample["ref_text"],
                ref_audio_16k=sample["ref_wav_16k"],
                ref_audio_24k=sample["ref_wav_24k"] if args.use_prompt_mel else None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                use_prompt_mel=args.use_prompt_mel,
                ref_codes=ref_codes,
            )

            # Save audio
            output_path = os.path.join(args.output_dir, f"{sample['id']}.wav")
            sf.write(output_path, audio, 24000)

            results.append({
                "id": sample["id"],
                "ref_text": sample["ref_text"],
                "tgt_text": sample["tgt_text"],
                "lang": sample["lang"],
                "gt_code_len": len(sample["gt_codes"]),
                "gen_code_len": codes.shape[1],
                "output_path": output_path,
            })

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": sample["id"],
                "error": str(e),
            })

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Generated {len([r for r in results if 'error' not in r])} / {len(dataset)} samples")


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
    device: str = "cuda:0",
):
    """
    Simple single-sample inference for quick testing (ICL mode).

    Example:
        from generate_tts_offline import single_sample_inference

        single_sample_inference(
            checkpoint_path="/path/to/checkpoint",
            omni_model_path="/path/to/Qwen2.5-Omni",
            voicebox_path="/path/to/diffusion.safetensors",
            vocos_path="/path/to/vocos.safetensors",
            voicebox_config="/path/to/voicebox_config.json",
            ref_audio_path="/path/to/prompt.wav",
            ref_text="这是参考音频的文本内容。",
            tgt_text="你好，世界！",
            output_path="/path/to/output.wav",
            device="cuda:0",
        )
    """
    logging.basicConfig(level=logging.INFO)

    # Load model
    model = SemanticCodeTTSInference.from_pretrained(
        checkpoint_path=checkpoint_path,
        omni_model_path=omni_model_path,
        voicebox_path=voicebox_path,
        vocos_path=vocos_path,
        voicebox_config_path=voicebox_config,
        device=device,
    )

    # Load reference audio
    ref_wav_16k, sr = torchaudio.load(ref_audio_path)
    if sr != 16000:
        ref_wav_16k = torchaudio.functional.resample(ref_wav_16k, sr, 16000)
    ref_wav_16k = ref_wav_16k[0]

    ref_wav_24k, sr = torchaudio.load(ref_audio_path)
    if sr != 24000:
        ref_wav_24k = torchaudio.functional.resample(ref_wav_24k, sr, 24000)
    ref_wav_24k = ref_wav_24k[0]

    # Synthesize (ICL mode)
    audio, codes = model.synthesize(
        tgt_text=tgt_text,
        ref_text=ref_text,
        ref_audio_16k=ref_wav_16k,
        ref_audio_24k=ref_wav_24k,
        use_prompt_mel=True,
    )

    # Save
    sf.write(output_path, audio, 24000)
    logger.info(f"Saved to {output_path}")
    logger.info(f"Generated {codes.shape[1]} semantic codes")

    return audio, codes


if __name__ == "__main__":
    main()
