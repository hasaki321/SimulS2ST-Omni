import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import math
import accelerate
import safetensors
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.voicebox.chat_template import (
    gen_chat_prompt_for_tts,
)
from src.voicebox.tokenizer import VoiceBoxAudioTokenizer

from src.voicebox.melspec import MelSpectrogram
from src.voicebox.vocos import Vocos
from hyperpyyaml import load_hyperpyyaml

try:
    from vllm import LLM, SamplingParams
except:
    pass


def build_vocoder_model(cfg, device):
    vocoder_model = Vocos(cfg=cfg.model.vocos)
    vocoder_model.eval()
    vocoder_model.to(device)
    return vocoder_model

def build_mel_model(cfg, device):
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
    return mel_model


class VoiceBoxRuntime:
    def __init__(self, llm_path, cfg, device, voicebox_path, build_llm=True, use_vllm=False, log_norm=False):
        self.cfg = cfg
        self.device = torch.device(device)
        self.llm_path = llm_path

        if log_norm:
            from src.voicebox.voicebox_model_lognorm import VoiceBox
        else:
            from src.voicebox.voicebox_model import VoiceBox

        # use vllm or not
        self.use_vllm = use_vllm

        print(f"use_vllm: {self.use_vllm}")

        if build_llm:
            if use_vllm:
                pass
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model.pretrained_model_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=False,
                    attn_implementation="flash_attention_2",
                )
                if os.path.isdir(self.llm_path):
                    accelerate.load_checkpoint_and_dispatch(
                        self.llm,
                        self.llm_path,
                    )
                elif os.path.isfile(self.llm_path):
                    safetensors.torch.load_model(
                        self.llm,
                        self.llm_path,
                    )
                else:
                    raise ValueError(f"Invalid llm_path: {self.llm_path}")
        else:
            self.llm = None

        # text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_path,
        )

        # audio tokenizer
        self.audio_tokenizer = VoiceBoxAudioTokenizer(cfg, device)

        # vocoder
        self.vocoder = build_vocoder_model(cfg, device)
        vocoder_path = "./models/vocos.safetensors"
        accelerate.load_checkpoint_and_dispatch(self.vocoder, vocoder_path)

        # voicebox
        self.voicebox = VoiceBox(cfg=cfg.model.voicebox)
        self.voicebox.eval()
        self.voicebox.to(device)
        # voicebox_path = "./models/diffusion.safetensors"
        safetensors.torch.load_model(self.voicebox, voicebox_path)

        # mel model
        self.mel_model = build_mel_model(cfg, device)

    def __call__(
        self,
        target_text: str,
        prompt_speech_path: str = None,
        prompt_text: str = None,
        caption=None,
        top_k: int = 20,
        top_p: float = 0.8,
        temp: float = 1.0,
    ):
        if prompt_speech_path is not None:
            prompt_semantic_code, prompt_speech = self.preprocess_prompt_wav(
                prompt_speech_path
            )
        else:
            prompt_semantic_code = torch.zeros(1, 0, device=self.device).long()
            prompt_speech = None

        if prompt_text is None:
            prompt_text = ""

        if not self.use_vllm:

            input_ids = self.tokenize(
                prompt_text,
                target_text,
                prompt_semantic_code,
                caption,
            )

            generate_ids = self.llm.generate(
                input_ids=input_ids,
                min_new_tokens=3,
                max_new_tokens=400,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temp,
            )

            (
                prompt_semantic_code,
                generate_semantic_code,
                combine_semantic_code,
            ) = self.postprocess(prompt_semantic_code, generate_ids, input_ids.size(1))

        else:
            print("vllm not supported")
            pass

        predict_mel = self.code2mel(combine_semantic_code, prompt_speech)
        audio = self.mel2audio(predict_mel)
        return audio

    def preprocess_prompt_wav(self, prompt_speech_path: str):
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]
        speech = librosa.load(prompt_speech_path, sr=24000)[0]
        semantic_code = self.audio_tokenizer(speech_16k)
        return semantic_code, speech

    def tokenize(
        self,
        prompt_text: str,
        target_text: str,
        prompt_semantic_code: torch.Tensor,
        caption: None,
    ):
        text = gen_chat_prompt_for_tts(
            prompt_text.strip() + target_text.strip(), caption=caption
        )
        print(text)

        text_token_ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        # shift speech ids
        prompt_semantic_code = (
            prompt_semantic_code + self.cfg.preprocess.audio_token_shift
        )
        # add bos audio token
        prompt_semantic_code = torch.cat(
            [torch.tensor([[self.cfg.preprocess.bos_audio_token_id]], device=self.device), prompt_semantic_code], dim=1
        )

        input_ids = torch.cat([text_token_ids.input_ids, prompt_semantic_code], dim=1)

        return input_ids

    def postprocess(
        self,
        prompt_semantic_code: torch.Tensor,
        generate_ids: torch.Tensor,
        input_ids_size: int,
    ):
        generate_semantic_code = generate_ids[:, input_ids_size:]
        generate_semantic_code = generate_semantic_code[:, :-2]
        generate_semantic_code = generate_semantic_code - self.cfg.preprocess.audio_token_shift
        combine_semantic_code = torch.cat(
            [prompt_semantic_code, generate_semantic_code], dim=1
        )

        # assert combine_semantic_code in [0, 16384), else random replace the token not in [0, 16384)
        for i in range(combine_semantic_code.shape[0]):
            for j in range(combine_semantic_code.shape[1]):
                if (
                    combine_semantic_code[i, j] < 0
                    or combine_semantic_code[i, j] >= 16384
                ):
                    combine_semantic_code[i, j] = torch.randint(0, 16384, (1,))

        return prompt_semantic_code, generate_semantic_code, combine_semantic_code

    def code2mel(self, combine_semantic_code: torch.Tensor, prompt_speech):
        # 使用原始的voicebox模型
        cond_feature = self.voicebox.cond_emb(combine_semantic_code)
        cond_feature = F.interpolate(
            cond_feature.transpose(1, 2),
            scale_factor=self.voicebox.cond_scale_factor,
        ).transpose(1, 2)

        if prompt_speech is not None:
            prompt_mel_feat = self.extract_mel_feature(
                torch.tensor(prompt_speech).to(self.device).unsqueeze(0),
            )
        else:
            prompt_mel_feat = None

        predict_mel = self.voicebox.reverse_diffusion(
            cond_feature,
            prompt_mel_feat,
            n_timesteps=16,
            cfg=2.0,
            rescale_cfg=0.75,
        )

        return predict_mel

    def mel2audio(self, mel_feature: torch.Tensor):
        audio = self.vocoder(mel_feature.transpose(1, 2)).detach().cpu().numpy()[0][0]
        return audio

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.cfg.preprocess.mel_mean) / math.sqrt(
            self.cfg.preprocess.mel_var
        )
        return mel_feature
