"""
Qwen-Omni style thinker->talker training wrapper.

This module mirrors the structure of `modeling_dual_head.py`, but instead of
using a custom audio head on top of thinker hidden states, it reuses the
original Qwen2.5-Omni talker decoder and reconstructs its teacher-forced
training path from the open-source generation logic.
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
THINKER_LORA_DIR_NAME = "thinker_lora"
TALKER_LORA_DIR_NAME = "talker_lora"
TALKER_COMPONENTS_NAME = "talker_components.pt"


@dataclass
class OmniTalkerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    thinker_hidden_states: Optional[torch.FloatTensor] = None
    talker_inputs_embeds: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


def _load_adapter_weights(lora_dir: str):
    adapter_bin = os.path.join(lora_dir, "adapter_model.bin")
    adapter_safetensors = os.path.join(lora_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_bin):
        return torch.load(adapter_bin, map_location="cpu", weights_only=True)
    if os.path.exists(adapter_safetensors):
        from safetensors.torch import load_file

        return load_file(adapter_safetensors)
    raise FileNotFoundError(f"No adapter weights found in {lora_dir}")


def _looks_like_peft_talker_state_dict(talker_state: Dict[str, torch.Tensor]) -> bool:
    return any(".lora_A." in key or ".base_layer." in key for key in talker_state.keys())


def _extract_talker_lora_config_from_state_dict(
    talker_state: Dict[str, torch.Tensor],
    saved_config: Optional[Dict] = None,
) -> Dict:
    target_modules = []
    seen_modules = set()
    ranks = []
    for key, value in talker_state.items():
        if ".lora_A." not in key:
            continue
        module_path = key.split(".lora_A.")[0]
        module_name = module_path.split(".")[-1]
        if module_name not in seen_modules:
            seen_modules.add(module_name)
            target_modules.append(module_name)
        ranks.append(int(value.shape[0]))

    assert target_modules, "Failed to infer talker LoRA target modules from checkpoint"
    assert ranks, "Failed to infer talker LoRA rank from checkpoint"
    rank = ranks[0]
    assert all(r == rank for r in ranks), f"Inconsistent talker LoRA ranks found: {ranks}"

    if saved_config is not None:
        saved_target_modules = list(saved_config.get("target_modules") or target_modules)
        return {
            "r": int(saved_config.get("r", rank)),
            "lora_alpha": int(saved_config.get("lora_alpha", rank)),
            "lora_dropout": float(saved_config.get("lora_dropout", 0.0)),
            "target_modules": saved_target_modules,
        }

    return {
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
    }


def _load_legacy_talker_lora_config_from_repo_logs(checkpoint_dir: str) -> Optional[Dict]:
    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "logs"
    if not logs_dir.is_dir():
        return None

    ckpt_path = Path(checkpoint_dir).resolve()
    run_name = ckpt_path.parent.name if ckpt_path.name.startswith("checkpoint-") else ckpt_path.name
    log_candidates = sorted(logs_dir.glob(f"{run_name}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_candidates:
        return None

    for log_path in log_candidates:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
        use_matches = re.findall(r"use_talker_lora:\s*(True|False|true|false)", content)
        r_matches = re.findall(r"talker_lora_r:\s*(\d+)", content)
        alpha_matches = re.findall(r"talker_lora_alpha:\s*(\d+)", content)
        if not use_matches or not r_matches or not alpha_matches:
            continue
        if use_matches[-1].lower() != "true":
            continue
        return {
            "r": int(r_matches[-1]),
            "lora_alpha": int(alpha_matches[-1]),
            "lora_dropout": 0.0,
            "target_modules": [],
        }
    return None


def _wrap_talker_with_lora_from_config(talker: nn.Module, talker_lora_config: Dict) -> PeftModel:
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=int(talker_lora_config["r"]),
        lora_alpha=int(talker_lora_config["lora_alpha"]),
        lora_dropout=float(talker_lora_config["lora_dropout"]),
        target_modules=list(talker_lora_config["target_modules"]),
        modules_to_save=None,
        bias="none",
    )
    return get_peft_model(talker, lora_config)


def _maybe_assign_talker(container, talker: nn.Module) -> None:
    if hasattr(container, "talker"):
        container.talker = talker


def load_talker_components(container, checkpoint_dir: str, merge_lora: bool = False) -> bool:
    talker = container.talker if hasattr(container, "talker") else container

    talker_lora_dir = os.path.join(checkpoint_dir, TALKER_LORA_DIR_NAME)
    if os.path.isdir(talker_lora_dir):
        logger.info(f"Loading talker LoRA from {talker_lora_dir}")
        if isinstance(talker, PeftModel):
            adapter_weights = _load_adapter_weights(talker_lora_dir)
            set_peft_model_state_dict(talker, adapter_weights)
            loaded_talker = talker
        else:
            loaded_talker = PeftModel.from_pretrained(
                talker,
                talker_lora_dir,
                is_trainable=not merge_lora,
            )
        if merge_lora and isinstance(loaded_talker, PeftModel):
            loaded_talker = loaded_talker.merge_and_unload()
        _maybe_assign_talker(container, loaded_talker)
        return True

    talker_path = os.path.join(checkpoint_dir, TALKER_COMPONENTS_NAME)
    if not os.path.exists(talker_path):
        return False

    logger.info(f"Loading talker components from {talker_path}")
    target_device = next(talker.parameters()).device
    talker_blob = torch.load(talker_path, map_location=target_device, weights_only=True)
    talker_state = talker_blob["talker"]

    if _looks_like_peft_talker_state_dict(talker_state):
        logger.info("Detected legacy PEFT-wrapped talker checkpoint format")
        if isinstance(talker, PeftModel):
            loaded_talker = talker
        else:
            legacy_cfg = _load_legacy_talker_lora_config_from_repo_logs(checkpoint_dir)
            if legacy_cfg is not None:
                logger.info(
                    "Recovered legacy talker LoRA config from logs for %s: r=%s alpha=%s",
                    checkpoint_dir,
                    legacy_cfg["r"],
                    legacy_cfg["lora_alpha"],
                )
            talker_lora_config = _extract_talker_lora_config_from_state_dict(
                talker_state,
                saved_config=talker_blob.get("talker_lora_config") or legacy_cfg,
            )
            loaded_talker = _wrap_talker_with_lora_from_config(talker, talker_lora_config)
        loaded_talker.load_state_dict(talker_state)
        if merge_lora and isinstance(loaded_talker, PeftModel):
            loaded_talker = loaded_talker.merge_and_unload()
        _maybe_assign_talker(container, loaded_talker)
        return True

    talker.load_state_dict(talker_state)
    return True


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)

    if not (flat_labels != ignore_index).any():
        return flat_logits.sum() * 0.0

    return F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="mean")


class OmniTalkerQwenOmni(nn.Module):
    """
    Minimal wrapper for Qwen-Omni talker training.

    Expected batch fields:
      - thinker text stream: `input_ids`, `attention_mask`
      - optional paired prompt audio: `audio_features`, `audio_mask`
      - talker codec stream: `talker_input_ids`, `talker_labels`, `talker_attention_mask`
      - alignment metadata: `reply_start_positions`, `reply_lengths`
    """

    def __init__(
        self,
        base_model,
        freeze_thinker: bool = False,
        freeze_talker: bool = False,
    ):
        super().__init__()
        self.thinker = base_model.thinker
        if not hasattr(base_model, "talker"):
            raise ValueError("base_model must contain an initialized Qwen-Omni talker module")
        self.talker = base_model.talker
        self.freeze_thinker = freeze_thinker

        object.__setattr__(self, "_thinker_backbone", base_model.thinker.model)
        object.__setattr__(self, "_audio_token_id", base_model.thinker.config.audio_token_id)

        if freeze_thinker:
            for param in self.thinker.parameters():
                param.requires_grad = False
            logger.info("Frozen thinker parameters")

        if freeze_talker:
            for param in self.talker.parameters():
                param.requires_grad = False
            logger.info("Frozen talker parameters")

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def _scatter_audio_features(
        self,
        input_ids: torch.Tensor,
        thinker_inputs_embeds: torch.Tensor,
        input_features: Optional[torch.FloatTensor],
        feature_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_features is None:
            return thinker_inputs_embeds

        audio_features = self.thinker.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
        )
        audio_features = audio_features.to(thinker_inputs_embeds.device, thinker_inputs_embeds.dtype)

        audio_input_mask = input_ids == self._audio_token_id
        audio_input_mask = audio_input_mask.unsqueeze(-1).expand_as(thinker_inputs_embeds)
        return thinker_inputs_embeds.masked_scatter(audio_input_mask, audio_features)

    def _build_talker_inputs_embeds_scatter(
        self,
        thinker_hidden_states: torch.Tensor,
        thinker_inputs_embeds: torch.Tensor,
        talker_input_ids: torch.Tensor,
        talker_thinker_scatter_indices: torch.Tensor,
        talker_text_cond_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Streaming, chunk-level talker input embeddings.

        For each talker position k:
          - if scatter_indices[b, k] >= 0: use thinker_cond[b, scatter_indices[b, k]]
            (the talker mask region inside chunk i, scattered from thinker hidden).
          - otherwise (codec_pad / codec_bos / codes / codec_eos): use the talker
            embed_tokens lookup of talker_input_ids[b, k], optionally plus
            talker_text_cond_indices text conditioning.

        thinker_cond = thinker_hidden + thinker_inputs_embeds, matching the additive
        contract used by the original Qwen-Omni talker prefix path.
        """
        assert talker_thinker_scatter_indices.shape == talker_input_ids.shape, (
            f"scatter shape {tuple(talker_thinker_scatter_indices.shape)} != "
            f"talker shape {tuple(talker_input_ids.shape)}"
        )
        H = thinker_hidden_states.shape[-1]
        dtype = thinker_hidden_states.dtype

        thinker_cond = (thinker_hidden_states + thinker_inputs_embeds).to(dtype)

        thinker_embed_tokens = self.thinker.get_input_embeddings()
        talker_embed_tokens = self.talker.get_input_embeddings()
        codec_emb = talker_embed_tokens(talker_input_ids).to(dtype)

        valid_mask = talker_thinker_scatter_indices >= 0  # [B, T_talker]
        # gather requires non-negative indices; clamp the masked-out slots to 0,
        # then drop them via where().
        safe_indices = talker_thinker_scatter_indices.clamp(min=0)
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, H)
        scattered = torch.gather(thinker_cond, dim=1, index=gather_idx)

        if talker_text_cond_indices is None:
            conditioned_codec_emb = codec_emb
        else:
            assert talker_text_cond_indices.shape == talker_input_ids.shape, (
                f"text-cond shape {tuple(talker_text_cond_indices.shape)} != "
                f"talker shape {tuple(talker_input_ids.shape)}"
            )
            text_cond = torch.zeros_like(codec_emb)

            gather_text_mask = talker_text_cond_indices >= 0
            if gather_text_mask.any():
                safe_text_indices = talker_text_cond_indices.clamp(min=0)
                text_gather_idx = safe_text_indices.unsqueeze(-1).expand(-1, -1, H)
                gathered_text = torch.gather(thinker_cond, dim=1, index=text_gather_idx)
                text_cond = torch.where(gather_text_mask.unsqueeze(-1), gathered_text, text_cond)

            special_tokens = {
                -2: self.talker.text_bos_token,
                -3: self.talker.text_eos_token,
                -4: self.talker.text_pad_token,
            }
            for sentinel, token_id in special_tokens.items():
                sentinel_mask = talker_text_cond_indices == sentinel
                if sentinel_mask.any():
                    token_embed = thinker_embed_tokens(
                        torch.tensor([token_id], device=talker_input_ids.device, dtype=torch.long)
                    )[0].to(dtype)
                    text_cond = torch.where(sentinel_mask.unsqueeze(-1), token_embed, text_cond)

            conditioned_codec_emb = codec_emb + text_cond

        return torch.where(valid_mask.unsqueeze(-1), scattered, conditioned_codec_emb)

    def _build_talker_inputs_embeds(
        self,
        thinker_hidden_states: torch.Tensor,
        thinker_inputs_embeds: torch.Tensor,
        talker_input_ids: torch.Tensor,
        talker_attention_mask: torch.Tensor,
        reply_start_positions: torch.Tensor,
        reply_lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, max_talker_len = talker_input_ids.shape
        H = thinker_hidden_states.shape[-1]
        device = thinker_hidden_states.device
        dtype = thinker_hidden_states.dtype

        thinker_cond = thinker_hidden_states + thinker_inputs_embeds
        thinker_embed_tokens = self.thinker.get_input_embeddings()
        talker_embed_tokens = self.talker.get_input_embeddings()

        text_bos_embed = thinker_embed_tokens(
            torch.tensor([self.talker.text_bos_token], device=device, dtype=torch.long)
        )[0].to(dtype)
        text_eos_embed = thinker_embed_tokens(
            torch.tensor([self.talker.text_eos_token], device=device, dtype=torch.long)
        )[0].to(dtype)
        text_pad_embed = thinker_embed_tokens(
            torch.tensor([self.talker.text_pad_token], device=device, dtype=torch.long)
        )[0].to(dtype)
        codec_pad_embed = talker_embed_tokens(
            torch.tensor([self.talker.codec_pad_token], device=device, dtype=torch.long)
        )[0].to(dtype)
        codec_bos_embed = talker_embed_tokens(
            torch.tensor([self.talker.codec_bos_token], device=device, dtype=torch.long)
        )[0].to(dtype)

        talker_inputs_embeds = torch.zeros((B, max_talker_len, H), device=device, dtype=dtype)

        for batch_idx in range(B):
            valid_len = int(talker_attention_mask[batch_idx].sum().item())
            prefix_len = int(reply_start_positions[batch_idx].item())
            reply_len = int(reply_lengths[batch_idx].item())

            assert valid_len >= prefix_len + 3, "Talker sequence is shorter than required prefix"
            assert reply_len > 0, "Reply length must be positive"

            prefill_cond = thinker_cond[batch_idx, :prefix_len]
            reply_cond = thinker_cond[batch_idx, prefix_len : prefix_len + reply_len]
            first_reply_cond = reply_cond[:1]

            codec_teacher_ids = talker_input_ids[batch_idx, prefix_len + 2 : valid_len]
            codec_teacher_embeds = talker_embed_tokens(codec_teacher_ids).to(dtype)

            tail_cond = reply_cond[1:]
            if tail_cond.shape[0] == 0:
                tail_cond = text_eos_embed.unsqueeze(0)
            tail_cond = torch.cat(
                [
                    tail_cond,
                    text_eos_embed.unsqueeze(0),
                    text_pad_embed.unsqueeze(0),
                ],
                dim=0,
            )
            if tail_cond.shape[0] < codec_teacher_embeds.shape[0]:
                pad_count = codec_teacher_embeds.shape[0] - tail_cond.shape[0]
                pad_tail = text_pad_embed.unsqueeze(0).expand(pad_count, -1)
                tail_cond = torch.cat([tail_cond, pad_tail], dim=0)
            else:
                tail_cond = tail_cond[: codec_teacher_embeds.shape[0]]

            prefix_embeds = torch.cat(
                [
                    prefill_cond,
                    (text_bos_embed + codec_pad_embed).unsqueeze(0),
                    (first_reply_cond[0] + codec_bos_embed).unsqueeze(0),
                ],
                dim=0,
            )
            full_embeds = torch.cat([prefix_embeds, codec_teacher_embeds + tail_cond], dim=0)
            assert full_embeds.shape[0] == valid_len
            talker_inputs_embeds[batch_idx, :valid_len] = full_embeds

        return talker_inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        talker_input_ids: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        talker_labels: Optional[torch.LongTensor] = None,
        talker_attention_mask: Optional[torch.Tensor] = None,
        reply_start_positions: Optional[torch.LongTensor] = None,
        reply_lengths: Optional[torch.LongTensor] = None,
        talker_thinker_scatter_indices: Optional[torch.LongTensor] = None,
        talker_text_cond_indices: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> OmniTalkerOutput:
        thinker_inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        thinker_inputs_embeds = self._scatter_audio_features(
            input_ids=input_ids,
            thinker_inputs_embeds=thinker_inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        thinker_outputs = self._thinker_backbone(
            inputs_embeds=thinker_inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        thinker_hidden_states = thinker_outputs.last_hidden_state

        text_logits = None
        text_loss = None
        if text_labels is not None and not self.freeze_thinker:
            text_head = self.thinker.get_output_embeddings()
            text_logits = text_head(thinker_hidden_states)
            text_loss = compute_causal_lm_loss(text_logits, text_labels, ignore_index=IGNORE_INDEX)

        talker_outputs = None
        talker_inputs_embeds = None
        audio_logits = None
        audio_loss = None
        if talker_labels is not None:
            if talker_input_ids is None:
                raise ValueError("talker_input_ids is required when talker_labels is provided")
            if talker_attention_mask is None:
                talker_attention_mask = torch.ones_like(talker_input_ids, dtype=attention_mask.dtype)

            if talker_thinker_scatter_indices is not None:
                # Streaming, chunk-level scatter alignment path.
                talker_inputs_embeds = self._build_talker_inputs_embeds_scatter(
                    thinker_hidden_states=thinker_hidden_states,
                    thinker_inputs_embeds=thinker_inputs_embeds,
                    talker_input_ids=talker_input_ids,
                    talker_thinker_scatter_indices=talker_thinker_scatter_indices,
                    talker_text_cond_indices=talker_text_cond_indices,
                )
            else:
                if reply_start_positions is None or reply_lengths is None:
                    raise ValueError(
                        "reply_start_positions and reply_lengths are required when talker_labels is provided "
                        "without talker_thinker_scatter_indices"
                    )
                talker_inputs_embeds = self._build_talker_inputs_embeds(
                    thinker_hidden_states=thinker_hidden_states,
                    thinker_inputs_embeds=thinker_inputs_embeds,
                    talker_input_ids=talker_input_ids,
                    talker_attention_mask=talker_attention_mask,
                    reply_start_positions=reply_start_positions,
                    reply_lengths=reply_lengths,
                )

            position_ids = torch.arange(
                talker_input_ids.shape[1],
                device=talker_input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(talker_input_ids.shape[0], -1)

            talker_outputs = self.talker(
                input_ids=None,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                return_dict=True,
            )
            audio_logits = talker_outputs.logits
            audio_loss = compute_causal_lm_loss(audio_logits, talker_labels, ignore_index=IGNORE_INDEX)

        losses = [loss for loss in (text_loss, audio_loss) if loss is not None]
        if not losses:
            loss = thinker_hidden_states.sum() * 0.0
        else:
            loss = sum(losses)

        return OmniTalkerOutput(
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            text_logits=text_logits,
            audio_logits=audio_logits,
            thinker_hidden_states=thinker_hidden_states,
            talker_inputs_embeds=talker_inputs_embeds,
            past_key_values=talker_outputs.past_key_values if use_cache and talker_outputs is not None else None,
        )


class _SyncStateCallback(TrainerCallback):
    """Force current args to override stale values loaded from trainer_state.json on resume."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        state.eval_steps = args.eval_steps
        return control


class QwenOmniTalkerTrainer(Trainer):
    """Trainer for Qwen-Omni thinker->talker TTS training."""

    def __init__(
        self,
        *args,
        eval_collator=None,
        train_sampler=None,
        eval_sampler=None,
        use_thinker_lora: bool = False,
        use_talker_lora: bool = False,
        eval_max_steps: int = -1,
        max_seq_len: int = 0,
        max_talker_seq_len: int = 0,
        max_talker_tokens_per_batch: int = 0,
        debug_batch_stats: bool = False,
        debug_batch_stats_steps: int = 1,
        debug_batch_stats_max_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_callback(_SyncStateCallback())
        self.eval_collator = eval_collator
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.use_thinker_lora = use_thinker_lora
        self.use_talker_lora = use_talker_lora
        self.eval_max_steps = eval_max_steps
        self.model_accepts_loss_kwargs = False
        self.max_seq_len = max_seq_len
        self.max_talker_seq_len = max_talker_seq_len
        self.max_talker_tokens_per_batch = int(max_talker_tokens_per_batch)
        self.debug_batch_stats = debug_batch_stats
        self.debug_batch_stats_steps = max(1, int(debug_batch_stats_steps))
        self.debug_batch_stats_max_steps = int(debug_batch_stats_max_steps)

        self._task_loss_sum: Dict[str, float] = {}
        self._task_loss_count: Dict[str, int] = {}
        self._text_loss_sum: float = 0.0
        self._text_loss_count: int = 0
        self._audio_loss_sum: float = 0.0
        self._audio_loss_count: int = 0
        self._skip_count: int = 0

    def _should_log_batch_stats(self) -> bool:
        if not self.debug_batch_stats:
            return False
        step = int(self.state.global_step)
        if self.debug_batch_stats_max_steps > 0 and step >= self.debug_batch_stats_max_steps:
            return False
        return step % self.debug_batch_stats_steps == 0

    @staticmethod
    def _tensor_shape(x) -> Optional[Tuple[int, ...]]:
        return tuple(x.shape) if torch.is_tensor(x) else None

    @staticmethod
    def _length_summary(mask: Optional[torch.Tensor]) -> str:
        if mask is None:
            return "None"
        lens = mask.detach().to(torch.long).sum(dim=1).cpu().tolist()
        return f"min={min(lens)}, max={max(lens)}, vals={lens}"

    @staticmethod
    def _valid_label_summary(labels: Optional[torch.Tensor]) -> str:
        if labels is None:
            return "None"
        counts = (labels.detach() != IGNORE_INDEX).sum(dim=1).cpu().tolist()
        return f"min={min(counts)}, max={max(counts)}, vals={counts}"

    def _log_batch_stats_before_forward(self, task_type: str, inputs: Dict, input_ids: torch.Tensor, talker_input_ids: Optional[torch.Tensor]):
        rank = int(os.environ.get("RANK", "-1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        cuda_stats = "cuda=unavailable"
        if torch.cuda.is_available():
            device = input_ids.device
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            cuda_stats = (
                f"cuda_device={device}, free_GB={free_bytes / 1e9:.3f}, total_GB={total_bytes / 1e9:.3f}, "
                f"allocated_GB={allocated / 1e9:.3f}, reserved_GB={reserved / 1e9:.3f}"
            )

        reply_starts = inputs.get("reply_start_positions")
        reply_lens = inputs.get("reply_lengths")
        scatter_idx = inputs.get("talker_thinker_scatter_indices")
        if torch.is_tensor(scatter_idx):
            scatter_valid = (scatter_idx >= 0).sum(dim=1).cpu().tolist()
            scatter_summary = f"shape={tuple(scatter_idx.shape)}, valid_per_row={scatter_valid}"
        else:
            scatter_summary = "None"
        text_cond_idx = inputs.get("talker_text_cond_indices")
        if torch.is_tensor(text_cond_idx):
            text_cond_valid = (text_cond_idx != -1).sum(dim=1).cpu().tolist()
            text_cond_summary = f"shape={tuple(text_cond_idx.shape)}, valid_per_row={text_cond_valid}"
        else:
            text_cond_summary = "None"
        ids = inputs.get("ids")
        logger.warning(
            "[BATCH_BEFORE_FORWARD] "
            f"step={int(self.state.global_step)} rank={rank} local_rank={local_rank} task={task_type} "
            f"input_shape={tuple(input_ids.shape)} attn_lens=({self._length_summary(inputs.get('attention_mask'))}) "
            f"text_valid=({self._valid_label_summary(inputs.get('text_labels', inputs.get('labels')))}) "
            f"talker_shape={self._tensor_shape(talker_input_ids)} "
            f"talker_attn_lens=({self._length_summary(inputs.get('talker_attention_mask'))}) "
            f"talker_valid=({self._valid_label_summary(inputs.get('talker_labels'))}) "
            f"reply_start={reply_starts.detach().cpu().tolist() if torch.is_tensor(reply_starts) else None} "
            f"reply_len={reply_lens.detach().cpu().tolist() if torch.is_tensor(reply_lens) else None} "
            f"talker_scatter=({scatter_summary}) "
            f"talker_text_cond=({text_cond_summary}) "
            f"audio_shape={self._tensor_shape(inputs.get('audio_features'))} "
            f"audio_lens=({self._length_summary(inputs.get('audio_mask'))}) "
            f"after_lens={inputs.get('after_lens').detach().cpu().tolist() if torch.is_tensor(inputs.get('after_lens')) else None} "
            f"ids={ids.detach().cpu().tolist()[:8] if torch.is_tensor(ids) else ids} "
            f"{cuda_stats}"
        )

    def save_model(self, output_dir=None, _internal_call=False):
        del _internal_call
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            base_model = self.model.module if hasattr(self.model, "module") else self.model
            talker = base_model.talker
            talker_state = {"talker": talker.state_dict()}

            if self.use_talker_lora and isinstance(talker, PeftModel):
                talker_lora_dir = os.path.join(output_dir, TALKER_LORA_DIR_NAME)
                os.makedirs(talker_lora_dir, exist_ok=True)
                talker.save_pretrained(talker_lora_dir)
                logger.info(f"Saved talker LoRA adapter to {talker_lora_dir}")
                peft_cfg = talker.peft_config["default"]
                talker_state["format"] = "peft_state_dict"
                talker_state["talker_lora_config"] = {
                    "r": int(peft_cfg.r),
                    "lora_alpha": int(peft_cfg.lora_alpha),
                    "lora_dropout": float(peft_cfg.lora_dropout),
                    "target_modules": list(peft_cfg.target_modules),
                }

            torch.save(talker_state, os.path.join(output_dir, TALKER_COMPONENTS_NAME))
            logger.info(f"Saved talker components to {output_dir}/{TALKER_COMPONENTS_NAME}")

            if self.use_thinker_lora:
                thinker = base_model.thinker
                if isinstance(thinker, PeftModel):
                    lora_dir = os.path.join(output_dir, THINKER_LORA_DIR_NAME)
                    os.makedirs(lora_dir, exist_ok=True)
                    thinker.save_pretrained(lora_dir)
                    logger.info(f"Saved thinker LoRA adapter to {lora_dir}")

            self.tokenizer.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        del model
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        load_talker_components(base_model, resume_from_checkpoint, merge_lora=False)

        if not self.use_thinker_lora:
            return

        lora_dir = os.path.join(resume_from_checkpoint, THINKER_LORA_DIR_NAME)
        if not os.path.isdir(lora_dir):
            logger.warning(f"No thinker_lora/ found in {resume_from_checkpoint}")
            return

        logger.info(f"Loading thinker LoRA weights from {lora_dir}")
        thinker = base_model.thinker
        if not isinstance(thinker, PeftModel):
            logger.warning("Thinker is not a PeftModel, skipping LoRA load")
            return

        adapter_bin = os.path.join(lora_dir, "adapter_model.bin")
        adapter_safetensors = os.path.join(lora_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_bin):
            adapters_weights = torch.load(adapter_bin, map_location="cpu", weights_only=True)
        elif os.path.exists(adapter_safetensors):
            from safetensors.torch import load_file

            adapters_weights = load_file(adapter_safetensors)
        else:
            raise FileNotFoundError(f"No adapter weights found in {lora_dir}")

        set_peft_model_state_dict(thinker, adapters_weights)
        logger.info(f"Thinker LoRA adapter loaded from checkpoint: {lora_dir}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        del ignore_keys, prediction_loss_only
        has_labels = "talker_labels" in inputs or "text_labels" in inputs or "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                if isinstance(loss, tuple):
                    loss = loss[0]
                loss = loss.detach()
            else:
                loss = None

        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        task_type = inputs.pop("task_type", "unknown")

        input_ids = inputs["input_ids"]
        B, T = input_ids.shape
        if self.max_seq_len > 0 and T > self.max_seq_len:
            self._skip_count += 1
            if self._skip_count % 10 == 1:
                logger.warning(
                    f"Skipping batch: seq_len={T} > max_seq_len={self.max_seq_len}, "
                    f"task={task_type}, total_skipped={self._skip_count}"
                )
            zero = torch.zeros(1, device=input_ids.device, dtype=torch.float32, requires_grad=True)
            return (zero.squeeze(), None) if return_outputs else zero.squeeze()

        talker_input_ids = inputs.get("talker_input_ids")
        if self.max_talker_seq_len > 0 and talker_input_ids is not None and talker_input_ids.shape[1] > self.max_talker_seq_len:
            self._skip_count += 1
            if self._skip_count % 10 == 1:
                logger.warning(
                    f"Skipping batch: talker_len={talker_input_ids.shape[1]} > max_talker_seq_len={self.max_talker_seq_len}, "
                    f"thinker_len={T}, task={task_type}, total_skipped={self._skip_count}"
                )
            zero = torch.zeros(1, device=input_ids.device, dtype=torch.float32, requires_grad=True)
            return (zero.squeeze(), None) if return_outputs else zero.squeeze()

        if self.max_talker_tokens_per_batch > 0 and talker_input_ids is not None:
            talker_tokens = talker_input_ids.shape[0] * talker_input_ids.shape[1]
            if talker_tokens > self.max_talker_tokens_per_batch:
                self._skip_count += 1
                if self._skip_count % 10 == 1:
                    logger.warning(
                        f"Skipping batch: talker_tokens={talker_tokens} > "
                        f"max_talker_tokens_per_batch={self.max_talker_tokens_per_batch}, "
                        f"talker_shape={tuple(talker_input_ids.shape)}, thinker_len={T}, "
                        f"task={task_type}, total_skipped={self._skip_count}"
                    )
                zero = torch.zeros(1, device=input_ids.device, dtype=torch.float32, requires_grad=True)
                return (zero.squeeze(), None) if return_outputs else zero.squeeze()

        if self._should_log_batch_stats():
            self._log_batch_stats_before_forward(task_type, inputs, input_ids, talker_input_ids)

        base_model = model.module if hasattr(model, "module") else model
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            talker_input_ids=talker_input_ids,
            text_labels=inputs.get("text_labels", inputs.get("labels")),
            talker_labels=inputs.get("talker_labels"),
            talker_attention_mask=inputs.get("talker_attention_mask"),
            reply_start_positions=inputs.get("reply_start_positions"),
            reply_lengths=inputs.get("reply_lengths"),
            talker_thinker_scatter_indices=inputs.get("talker_thinker_scatter_indices"),
            talker_text_cond_indices=inputs.get("talker_text_cond_indices"),
            input_features=inputs.get("audio_features"),
            feature_attention_mask=inputs.get("audio_mask"),
            return_dict=True,
        )
        loss = outputs.loss

        if outputs.text_loss is not None:
            self._text_loss_sum += outputs.text_loss.detach().item()
            self._text_loss_count += 1
        if outputs.audio_loss is not None:
            self._audio_loss_sum += outputs.audio_loss.detach().item()
            self._audio_loss_count += 1

        loss_val = loss.detach().item()
        self._task_loss_sum[task_type] = self._task_loss_sum.get(task_type, 0.0) + loss_val
        self._task_loss_count[task_type] = self._task_loss_count.get(task_type, 0) + 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float = None):
        if self._task_loss_count:
            for task, count in self._task_loss_count.items():
                if count > 0:
                    logs[f"loss/{task}"] = round(self._task_loss_sum[task] / count, 4)
                    logs[f"count/{task}"] = count
            self._task_loss_sum.clear()
            self._task_loss_count.clear()

        if self._text_loss_count > 0:
            logs["loss/text"] = round(self._text_loss_sum / self._text_loss_count, 4)
            self._text_loss_sum = 0.0
            self._text_loss_count = 0

        if self._audio_loss_count > 0:
            logs["loss/audio"] = round(self._audio_loss_sum / self._audio_loss_count, 4)
            self._audio_loss_sum = 0.0
            self._audio_loss_count = 0

        if self._skip_count > 0:
            logs["skipped_long_seq"] = self._skip_count

        super().log(logs, start_time=start_time)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        del eval_dataset
        return DataLoader(
            self.eval_dataset,
            batch_sampler=self.eval_sampler,
            collate_fn=self.eval_collator,
            num_workers=0,
            pin_memory=False,
        )


def log_trainable_parameters(module: nn.Module, module_name: str):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    logger.info(
        f"{module_name} trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_pct:.2f}%)"
    )
