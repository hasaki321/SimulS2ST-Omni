import torch
import numpy as np
from transformers import (
    Wav2Vec2BertModel,
    SeamlessM4TFeatureExtractor,
)
from src.voicebox.dual_model import DualCodec
import safetensors
import os

# Prefer a local checkpoint / HF cache if present; otherwise download facebook/w2v-bert-2.0.
_local_w2v_candidates = [
    os.environ.get("SIMULS2ST_W2V_BERT_PATH", ""),
]
w2v_path = "facebook/w2v-bert-2.0"
for _cand in _local_w2v_candidates:
    if _cand and os.path.isdir(_cand) and os.path.exists(os.path.join(_cand, "config.json")):
        w2v_path = _cand
        break


def build_codec_model(cfg, device):
    dual_codec_model = DualCodec(
        sample_rate=24000,
        encoder_rates=[4, 5, 6, 8, 2],
        decoder_rates=[2, 8, 6, 5, 4],
        encoder_dim=32,
        decoder_dim=1536,
        n_codebooks=cfg.model.dual_codec.n_codebooks,
        quantizer_dropout=0.5,
        codebook_size=cfg.model.dual_codec.codebook_size,
        semantic_codebook_size=cfg.model.dual_codec.semantic_codebook_size,
        is_causal=True,
        semantic_downsample_factor=cfg.model.dual_codec.semantic_downsample_factor,
    )

    dual_codec_model.eval()
    dual_codec_model.to(device)
    safetensors.torch.load_model(dual_codec_model, cfg.model.dual_codec.pretrained_path)
    return dual_codec_model


def build_semantic_model(cfg, device):
    semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_path)
    semantic_model.eval()
    semantic_model.to(device)
    layer_idx = 15
    output_idx = 17
    stat_mean_var = torch.load(cfg.model.kmeans.stat_mean_var_path)
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)
    return semantic_model, semantic_mean, semantic_std


class VoiceBoxAudioTokenizer:
    def __init__(self, cfg, device):
        self.device = device
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(w2v_path)
        self.dual_codec = build_codec_model(cfg, device)

        (
            self.semantic_model,
            self.semantic_mean,
            self.semantic_std,
        ) = build_semantic_model(cfg, device)

    def __call__(
        self,
        speech: np.ndarray,
    ):
        return self.wav2token(speech)

    def wav2token(self, speech: np.ndarray):
        input_features, attention_mask = self._extract_features(speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code = self._extract_semantic_code(input_features, attention_mask)
        return semantic_code

    def _extract_features(
        self,
        speech: np.ndarray,
    ):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    def _extract_semantic_code(
        self,
        input_features,
        attention_mask,
    ):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        feat = torch.nn.functional.avg_pool1d(
            feat.transpose(1, 2),
            self.dual_codec.semantic_downsample_factor,
            self.dual_codec.semantic_downsample_factor,
        )

        semantic_code = self.dual_codec.semantic_quantize(feat)  # [B, T]

        return semantic_code
