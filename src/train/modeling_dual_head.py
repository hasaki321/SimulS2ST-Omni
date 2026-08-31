"""
Dual-Head Audio Architecture for Qwen-Omni.

This module provides:
1. Centralized token constants for all datasets
2. AudioHead module for audio token prediction
3. DualHeadQwenOmni wrapper that separates text/audio output heads
4. QwenOmniTrainer with dual-head loss computation
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, GenerationConfig, GenerationMixin
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from peft import PeftModel, set_peft_model_state_dict

logger = logging.getLogger(__name__)

# ============================================================
# Token Constants (centralized for all datasets)
# ============================================================

# Standard Qwen tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Audio input tokens (for ASR/S2TT)
AUDIO_BOS = "<|audio_bos|>"
AUDIO_EOS = "<|audio_eos|>"
AUDIO_PLACEHOLDER = "<|AUDIO|>"

# TTS output tokens

# Mode prefixes (non-special tokens, stable tokenization with leading space)
TEXT_MODE_PREFIX = "<tool_call>"
ENDOFTEXT = "</tool_call>"
TTS_MODE_PREFIX = "<|quad_start|>"
TTS_PLACEHOLDER = "<|IMAGE|>"  # Reuse IMAGE token since vision tower is removed
TTS_EOS_TOKEN = "<|quad_end|>"  # Index n_codes in audio_embed/audio_head

# Streaming control tokens
WAIT_TOKEN = ""
IDLE_TOKEN = ""
# WAIT_TOKEN = "@"
# IDLE_TOKEN = "#"
LATENCY_TOKEN_TEMPLATE = "{}"

# Legacy tokens (for backward compatibility with existing datasets)
DEFAULT_EOS_TOKEN = IM_END
DEFAULT_BOS_TOKEN = IM_START
DEFAULT_SPEECH_PATCH_TOKEN = AUDIO_PLACEHOLDER
DEFAULT_SPEECH_START_TOKEN = AUDIO_BOS
DEFAULT_SPEECH_END_TOKEN = AUDIO_EOS
DEFAULT_TEXT_START_TOKEN = TEXT_MODE_PREFIX
DEFAULT_TEXT_END_TOKEN = ENDOFTEXT
DEFAULT_TTS_BOS_TOKEN = TTS_MODE_PREFIX
DEFAULT_TTS_EOS_TOKEN = TTS_EOS_TOKEN
DEFAULT_WAIT_TOKEN = WAIT_TOKEN
DEFAULT_IDLE_TOKEN = IDLE_TOKEN
DEFAULT_LATENCY_TOKEN = LATENCY_TOKEN_TEMPLATE
DEFAULT_SYSTEM_PROMPT = "You are a multilingual speech-text translation model."

# Loss mask value
IGNORE_INDEX = -100

# Thinker LoRA directory name
THINKER_LORA_DIR_NAME = "thinker_lora"


# ============================================================
# Output Dataclass
# ============================================================

@dataclass
class DualHeadOutput(ModelOutput):
    """Output from DualHeadQwenOmni forward pass."""
    loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    text_logits: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# ============================================================
# Audio Head Module
# ============================================================

class AudioHead(nn.Module):
    """Configurable audio token head.

    Output dimension is n_codes + 1 to include tts_eos token.
    Supported head types:
    - linear: hidden -> vocab
    - mlp: hidden -> 4*hidden -> GELU -> vocab
    """

    def __init__(
        self,
        hidden_size: int,
        n_codes: int,
        head_type: str = "linear",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.head_type = head_type

        if head_type == "linear":
            self.head = nn.Linear(hidden_size, n_codes + 1, device=device, dtype=dtype)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden_size * 4, n_codes + 1, device=device, dtype=dtype),
            )
        else:
            raise ValueError(f"Unsupported audio head type: {head_type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_size] or [B, hidden_size]
        Returns:
            logits: [B, T, n_codes+1] or [B, n_codes+1]
        """
        return self.head(hidden_states)


# ============================================================
# Dual Head Wrapper
# ============================================================

def scatter_audio_embeds(
    text_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    audio_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter audio embeddings into text embedding positions.

    Args:
        text_embeds: [B, T, hidden_size] - base embeddings from text_embed
        audio_embeds: [B, code_len, hidden_size] - audio code embeddings
        audio_mask: [B, T] - bool mask, True where audio codes should be placed

    Returns:
        merged_embeds: [B, T, hidden_size] - text_embeds with audio positions replaced
    """
    B, T, H = text_embeds.shape
    result = text_embeds.clone().type_as(audio_embeds)

    for b in range(B):
        mask_positions = audio_mask[b].nonzero(as_tuple=True)[0]
        n_positions = mask_positions.shape[0]
        n_codes = audio_embeds.shape[1]

        if n_positions > 0 and n_codes > 0:
            n_to_fill = min(n_positions, n_codes)
            result[b, mask_positions[:n_to_fill]] = audio_embeds[b, :n_to_fill]

    return result


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute causal LM loss with proper shift.

    For causal LM: logits[t] predicts labels[t+1]
    So we use logits[:, :-1] to predict labels[:, 1:]

    When all labels are ignore_index (e.g. TTS batch for text head),
    returns zero loss connected to logits for DDP gradient flow.
    F.cross_entropy returns nan when all labels are ignored (0/0),
    so we must handle this case explicitly.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)

    if not (flat_labels != ignore_index).any():
        return flat_logits.sum() * 0.0

    return F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="mean")


def apply_repetition_penalty_(
    logits: torch.Tensor,
    generated_codes: list[torch.Tensor],
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty in-place to previously generated tokens."""
    if repetition_penalty == 1.0 or len(generated_codes) == 0:
        return logits

    history = torch.stack(generated_codes, dim=1)
    for batch_idx in range(logits.shape[0]):
        prev_tokens = torch.unique(history[batch_idx])
        prev_token_logits = logits[batch_idx, prev_tokens]
        prev_token_logits = torch.where(
            prev_token_logits < 0,
            prev_token_logits * repetition_penalty,
            prev_token_logits / repetition_penalty,
        )
        logits[batch_idx, prev_tokens] = prev_token_logits
    return logits


def apply_no_repeat_ngram_(
    logits: torch.Tensor,
    generated_codes: list[torch.Tensor],
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    """Ban tokens that would complete an already seen n-gram."""
    if no_repeat_ngram_size <= 0 or len(generated_codes) < no_repeat_ngram_size - 1:
        return logits

    history = torch.stack(generated_codes, dim=1)
    prefix_len = no_repeat_ngram_size - 1

    for batch_idx in range(logits.shape[0]):
        tokens = history[batch_idx].tolist()
        if len(tokens) < prefix_len:
            continue

        current_prefix = tuple(tokens[-prefix_len:]) if prefix_len > 0 else tuple()
        banned_tokens = set()
        max_start = len(tokens) - no_repeat_ngram_size + 1

        for start_idx in range(max_start):
            ngram_prefix = tuple(tokens[start_idx:start_idx + prefix_len]) if prefix_len > 0 else tuple()
            if ngram_prefix == current_prefix:
                banned_tokens.add(tokens[start_idx + prefix_len])

        if banned_tokens:
            banned_tensor = torch.tensor(list(banned_tokens), device=logits.device, dtype=torch.long)
            logits[batch_idx, banned_tensor] = float("-inf")

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> torch.LongTensor:
    """Sample or greedy-pick the next token from logits."""
    if do_sample:
        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    return torch.argmax(logits, dim=-1)


class _AudioGenerationAdapter(nn.Module, GenerationMixin):
    """HF GenerationMixin adapter for dual-head audio-code decoding."""

    main_input_name = "input_ids"
    _is_stateful = False

    def __init__(self, dual_model: "DualHeadQwenOmni"):
        super().__init__()
        object.__setattr__(self, "_dual_model", dual_model)
        self.config = dual_model.thinker.config
        self.generation_config = GenerationConfig(
            use_cache=True,
            pad_token_id=dual_model.n_codes,
            eos_token_id=dual_model.n_codes,
        )
        self._supports_cache_class = getattr(dual_model.thinker, "_supports_cache_class", False)

    @property
    def dual_model(self) -> "DualHeadQwenOmni":
        return object.__getattribute__(self, "_dual_model")

    @property
    def device(self) -> torch.device:
        return self.dual_model.audio_embed.weight.device

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def get_output_embeddings(self):
        head = self.dual_model.audio_head.head
        if isinstance(head, nn.Sequential):
            return head[-1]
        return head

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.LongTensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_audio_codes: Optional[torch.LongTensor] = None,
        prompt_audio_codes_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        cache_position=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ) -> Dict[str, Any]:
        del cache_position, position_ids, inputs_embeds, kwargs
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_audio_codes": prompt_audio_codes,
            "prompt_audio_codes_mask": prompt_audio_codes_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        if isinstance(past_key_values, tuple):
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )
        return past_key_values

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        prompt_input_ids: Optional[torch.LongTensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_audio_codes: Optional[torch.LongTensor] = None,
        prompt_audio_codes_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        cache_position=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del cache_position, position_ids, inputs_embeds, kwargs
        if not return_dict:
            raise ValueError("_AudioGenerationAdapter only supports return_dict=True")

        dual_model = self.dual_model
        if past_key_values is None:
            hidden_states, past_key_values = dual_model.forward_hidden(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                audio_codes=prompt_audio_codes,
                audio_codes_mask=prompt_audio_codes_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                past_key_values=None,
                use_cache=use_cache,
            )
        else:
            last_code = input_ids[:, -1:]
            last_embed = dual_model.audio_embed(last_code)
            full_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            outputs = dual_model._backbone(
                inputs_embeds=last_embed,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        logits = dual_model.audio_head(hidden_states[:, -1:, :])
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )


class DualHeadQwenOmni(nn.Module):
    """Wrapper around Qwen-Omni with separate text and audio output heads.

    Key features:
    - text_head: points to original lm_head (can be frozen)
    - audio_embed: independent embedding for audio codes
    - audio_head: MLP for audio token prediction
    - Supports stage-1 training where LLM is frozen
    """

    def __init__(
        self,
        base_model,
        n_codes: int,
        hidden_size: int,
        audio_head_type: str = "linear",
        freeze_llm: bool = False,
        freeze_audio_components: bool = False,
    ):
        super().__init__()
        self.thinker = base_model.thinker
        self.n_codes = n_codes
        self.hidden_size = hidden_size

        # Non-submodule references — avoids duplicate registration in module tree.
        # base_model.thinker is already registered as self.thinker; registering
        # base_model too would put the same parameters under two module-tree paths,
        # breaking DDP and inflating named_parameters().
        # PEFT injects LoRA in-place, so _backbone layers stay valid after wrapping.
        object.__setattr__(self, '_backbone', base_model.thinker.model)
        object.__setattr__(self, '_text_head', base_model.thinker.lm_head)
        object.__setattr__(self, '_base_generate', base_model.generate)
        object.__setattr__(self, '_audio_token_id', base_model.thinker.config.audio_token_id)

        # Create audio components on the same device/dtype as the base model.
        # With n_codes=16384 and hidden_size=3584 the audio head alone is ~286M params.
        # Leaving them on CPU in float32 wastes ~1.4 GB per process, which causes
        # os.fork() OOM when DataLoader workers are spawned under multi-GPU training.
        ref_param = next(base_model.thinker.lm_head.parameters())
        target_device = ref_param.device
        target_dtype = ref_param.dtype

        self.audio_embed = nn.Embedding(n_codes + 1, hidden_size, device=target_device, dtype=target_dtype)
        self.audio_head = AudioHead(
            hidden_size,
            n_codes,
            head_type=audio_head_type,
            device=target_device,
            dtype=target_dtype,
        )
        self.audio_head_type = audio_head_type

        self._init_audio_weights()

        self.tts_placeholder_id = None
        object.__setattr__(self, "_audio_generation_adapter", _AudioGenerationAdapter(self))

        self.freeze_llm = freeze_llm
        self.freeze_audio_components = freeze_audio_components
        if freeze_llm:
            self._freeze_llm()
        if freeze_audio_components:
            self._freeze_audio_components()

    def _init_audio_weights(self):
        """Initialize audio embedding and head weights."""
        nn.init.normal_(self.audio_embed.weight, mean=0.0, std=0.02)
        for module in self.audio_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _freeze_llm(self):
        """Freeze all LLM parameters (thinker) for stage-1 training."""
        for param in self.thinker.parameters():
            param.requires_grad = False
        logger.info("Frozen all thinker parameters for stage-1 training")

    def unfreeze_llm(self):
        """Unfreeze LLM parameters for stage-2 training."""
        for param in self.thinker.parameters():
            param.requires_grad = True
        logger.info("Unfrozen thinker parameters for stage-2 training")

    def _freeze_audio_components(self):
        """Freeze audio embedding and audio head parameters."""
        for param in self.audio_embed.parameters():
            param.requires_grad = False
        for param in self.audio_head.parameters():
            param.requires_grad = False
        logger.info("Frozen audio_embed and audio_head parameters")

    def set_placeholder_id(self, tokenizer):
        """Set the TTS placeholder token ID from tokenizer."""
        self.tts_placeholder_id = tokenizer.convert_tokens_to_ids(TTS_PLACEHOLDER)
        logger.info(f"TTS placeholder ID: {self.tts_placeholder_id}")

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        audio_codes: Optional[torch.LongTensor] = None,
        audio_codes_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        loss_weight_text: float = 1.0,
        loss_weight_audio: float = 1.0,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> DualHeadOutput:
        """
        Forward pass with dual-head architecture.

        Audio INPUT (S2TT/ASR): input_features → audio_tower → scatter to <|AUDIO|> positions.
        Audio OUTPUT (TTS):     audio_codes → audio_embed → scatter to <|IMAGE|> positions.
        These two paths use different placeholder tokens and can coexist.

        Loss computation uses causal LM shift: logits[:, t] predicts labels[:, t+1].
        """
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)

        # 1. Process audio INPUT features through audio tower (S2TT/ASR tasks).
        #    The original code called self.thinker.model(..., input_features=...) which
        #    is the text backbone — it silently ignores input_features via **kwargs.
        #    We must run the audio tower here, exactly like the thinker's own forward.
        if input_features is not None:
            audio_features = self.thinker.get_audio_features(
                input_features, feature_attention_mask=feature_attention_mask,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_input_mask = (input_ids == self._audio_token_id)
            audio_input_mask = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_input_mask, audio_features)

        # 2. Process audio OUTPUT codes through audio_embed (TTS tasks).
        if audio_codes is not None and audio_codes_mask is not None:
            audio_code_embeds = self.audio_embed(audio_codes)
            inputs_embeds = scatter_audio_embeds(inputs_embeds, audio_code_embeds, audio_codes_mask)

        # 3. Run backbone (text transformer, no lm_head).
        #    Uses _backbone (stored before PEFT wrapping) to avoid PeftModel blocking
        #    .model access in some PEFT versions.
        outputs = self._backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state

        # 4. Compute logits and losses.
        #    Only compute logits for the head that has valid labels to avoid
        #    wasting VRAM on [B, T, 152k] text logits for TTS-only batches.
        text_logits = None
        audio_logits = None
        text_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        audio_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        if text_labels is not None and not self.freeze_llm:
            text_logits = self._text_head(hidden_states)
            text_loss = compute_causal_lm_loss(text_logits, text_labels, ignore_index=IGNORE_INDEX)

        if audio_labels is not None:
            audio_logits = self.audio_head(hidden_states)
            audio_loss = compute_causal_lm_loss(audio_logits, audio_labels, ignore_index=IGNORE_INDEX)

        loss = loss_weight_text * text_loss + loss_weight_audio * audio_loss

        return DualHeadOutput(
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            text_logits=text_logits,
            audio_logits=audio_logits,
            hidden_states=hidden_states,
            past_key_values=outputs.past_key_values if use_cache else None,
        )

    def forward_hidden(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        audio_codes: Optional[torch.LongTensor] = None,
        audio_codes_mask: Optional[torch.BoolTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        """Forward pass returning only hidden states (for generation)."""
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_features = self.thinker.get_audio_features(
                input_features, feature_attention_mask=feature_attention_mask,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_input_mask = (input_ids == self._audio_token_id)
            audio_input_mask = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_input_mask, audio_features)

        if audio_codes is not None and audio_codes_mask is not None:
            audio_embeds = self.audio_embed(audio_codes)
            inputs_embeds = scatter_audio_embeds(inputs_embeds, audio_embeds, audio_codes_mask)

        outputs = self._backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        return outputs.last_hidden_state, outputs.past_key_values

    def generate(
        self,
        output_mode: str = "text",
        **kwargs,
    ):
        """
        Generate tokens with specified output mode.

        Args:
            output_mode: "text" or "audio"
                - "text": use standard HF generate with text_head
                - "audio": custom autoregressive loop with audio_head
        """
        if output_mode == "text":
            return self._base_generate(**kwargs)
        else:
            return self._generate_audio(**kwargs)

    @torch.no_grad()
    def _generate_audio(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.LongTensor] = None,
        audio_codes_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressive audio code generation.

        Returns:
            audio_codes: [B, generated_len] - generated audio code IDs
        """
        if max_new_tokens <= 0:
            return torch.empty((input_ids.shape[0], 0), device=input_ids.device, dtype=torch.long)

        B = input_ids.shape[0]
        device = input_ids.device

        hidden, past_key_values = self.forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        first_logits = self.audio_head(hidden[:, -1, :])
        first_token = sample_next_token(
            logits=first_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        if (first_token == self.n_codes).all():
            return torch.empty((B, 0), device=device, dtype=torch.long)

        generated = first_token.unsqueeze(1)
        if max_new_tokens == 1:
            return generated

        generate_kwargs = dict(kwargs)
        sequences = self._audio_generation_adapter.generate(
            input_ids=generated,
            attention_mask=torch.ones((B, 1), device=device, dtype=attention_mask.dtype),
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            prompt_audio_codes=audio_codes,
            prompt_audio_codes_mask=audio_codes_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            max_new_tokens=max_new_tokens - 1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=generate_kwargs.pop("eos_token_id", self.n_codes),
            pad_token_id=generate_kwargs.pop("pad_token_id", self.n_codes),
            **generate_kwargs,
        )

        trimmed_sequences: List[torch.Tensor] = []
        max_len = 0
        for row in sequences:
            eos_positions = (row == self.n_codes).nonzero(as_tuple=True)[0]
            trim_len = eos_positions[0].item() if len(eos_positions) > 0 else row.shape[0]
            trimmed = row[:trim_len]
            trimmed_sequences.append(trimmed)
            max_len = max(max_len, trimmed.shape[0])

        if max_len == 0:
            return torch.empty((B, 0), device=device, dtype=torch.long)

        if any(seq.shape[0] != max_len for seq in trimmed_sequences):
            logger.warning(
                "Audio generation produced different lengths within the batch; padding shorter rows with 0."
            )
            padded = torch.zeros((B, max_len), device=device, dtype=torch.long)
            for batch_idx, seq in enumerate(trimmed_sequences):
                padded[batch_idx, :seq.shape[0]] = seq
            return padded

        return torch.stack(trimmed_sequences, dim=0)


# ============================================================
# Trainer
# ============================================================

class _SyncStateCallback(TrainerCallback):
    """Force current args to override stale values loaded from trainer_state.json on resume."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = args.save_steps
        state.logging_steps = args.logging_steps
        state.eval_steps = args.eval_steps
        return control


class QwenOmniDualHeadTrainer(Trainer):
    """
    Custom Trainer for dual-head S2TT + Semantic Code TTS training.

    Handles:
    - Separate text/audio loss computation via DualHeadQwenOmni
    - Per-task loss logging
    - LoRA adapter saving/loading
    """

    def __init__(
        self,
        *args,
        eval_collator=None,
        train_sampler=None,
        eval_sampler=None,
        use_lora: bool = False,
        eval_max_steps: int = -1,
        max_seq_len: int = 0,
        loss_weight_text: float = 1.0,
        loss_weight_audio: float = 1.0,
        **kwargs,
    ):
        # This training path does not use DeepSpeed, but recent accelerate imports
        # deepspeed opportunistically during Trainer init if the package is installed.
        # On compute nodes without a local CUDA toolkit, that import can fail while
        # probing op compatibility even though the actual training stack is pure DDP.
        import accelerate.utils.imports as accelerate_imports
        import accelerate.utils.other as accelerate_other

        accelerate_imports.is_deepspeed_available = lambda: False
        accelerate_other.is_deepspeed_available = lambda: False

        super().__init__(*args, **kwargs)
        self.add_callback(_SyncStateCallback())
        self.eval_collator = eval_collator
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.use_lora = use_lora
        self.eval_max_steps = eval_max_steps
        self.model_accepts_loss_kwargs = False
        self.max_seq_len = max_seq_len
        self.loss_weight_text = loss_weight_text
        self.loss_weight_audio = loss_weight_audio

        self._task_loss_sum: Dict[str, float] = {}
        self._task_loss_count: Dict[str, int] = {}
        self._text_loss_sum: float = 0.0
        self._audio_loss_sum: float = 0.0
        self._loss_count: int = 0
        self._skip_count: int = 0

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            base_model = self.model.module if hasattr(self.model, "module") else self.model

            if isinstance(base_model, DualHeadQwenOmni):
                audio_state = {
                    "audio_embed": base_model.audio_embed.state_dict(),
                    "audio_head": base_model.audio_head.state_dict(),
                }
                torch.save(audio_state, os.path.join(output_dir, "audio_components.pt"))
                logger.info(f"Saved audio components to {output_dir}/audio_components.pt")

                if self.use_lora:
                    thinker = base_model.thinker
                    if isinstance(thinker, PeftModel):
                        lora_dir = os.path.join(output_dir, THINKER_LORA_DIR_NAME)
                        os.makedirs(lora_dir, exist_ok=True)
                        thinker.save_pretrained(lora_dir)
                        logger.info(f"Saved thinker LoRA adapter to {lora_dir}")

                self.tokenizer.save_pretrained(output_dir)
            else:
                super().save_model(output_dir, _internal_call=_internal_call)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        base_model = self.model.module if hasattr(self.model, "module") else self.model

        audio_path = os.path.join(resume_from_checkpoint, "audio_components.pt")
        if os.path.exists(audio_path) and isinstance(base_model, DualHeadQwenOmni):
            logger.info(f"Loading audio components from {audio_path}")
            target_device = base_model.audio_embed.weight.device
            audio_state = torch.load(audio_path, map_location=target_device, weights_only=True)
            base_model.audio_embed.load_state_dict(audio_state["audio_embed"])
            base_model.audio_head.load_state_dict(audio_state["audio_head"])

        if not self.use_lora:
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
        logger.info(f"LoRA adapter loaded from checkpoint: {lora_dir}")

    def _save_checkpoint(self, model, trial):
        if self.is_world_process_zero():
            logger.info(f"Begin checkpoint save at step={self.state.global_step}")
        super()._save_checkpoint(model, trial)
        if self.is_world_process_zero():
            logger.info(f"Finished checkpoint save at step={self.state.global_step}")

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        del ignore_keys
        has_labels = "labels" in inputs or "text_labels" in inputs or "audio_labels" in inputs
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
        task_type = inputs.pop("task_type", "unknown")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
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

        base_model = model.module if hasattr(model, "module") else model

        # Determine text_labels and audio_labels
        # New format: separate text_labels and audio_labels
        # Old format (backward compat): single "labels" for text tasks (S2TT/T2T)
        text_labels = inputs.get("text_labels")
        audio_labels = inputs.get("audio_labels")

        # Backward compatibility: if only "labels" is provided, treat as text_labels
        if text_labels is None and audio_labels is None and "labels" in inputs:
            text_labels = inputs["labels"]

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_labels=text_labels,
            audio_labels=audio_labels,
            audio_codes=inputs.get("audio_codes"),
            audio_codes_mask=inputs.get("audio_codes_mask"),
            input_features=inputs.get("audio_features"),
            feature_attention_mask=inputs.get("audio_mask"),
            loss_weight_text=self.loss_weight_text,
            loss_weight_audio=self.loss_weight_audio,
            return_dict=True,
        )
        loss = outputs.loss

        if outputs.text_loss is not None:
            self._text_loss_sum += outputs.text_loss.detach().item()
        if outputs.audio_loss is not None:
            self._audio_loss_sum += outputs.audio_loss.detach().item()
        self._loss_count += 1

        loss_val = loss.detach().item()
        self._task_loss_sum[task_type] = self._task_loss_sum.get(task_type, 0.0) + loss_val
        self._task_loss_count[task_type] = self._task_loss_count.get(task_type, 0) + 1

        return (loss, None) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: float = None):
        if self._task_loss_count:
            for task, count in self._task_loss_count.items():
                if count > 0:
                    logs[f"loss/{task}"] = round(self._task_loss_sum[task] / count, 4)
                    logs[f"count/{task}"] = count
            self._task_loss_sum.clear()
            self._task_loss_count.clear()

        if self._loss_count > 0:
            logs["loss/text"] = round(self._text_loss_sum / self._loss_count, 4)
            logs["loss/audio"] = round(self._audio_loss_sum / self._loss_count, 4)
            self._text_loss_sum = 0.0
            self._audio_loss_sum = 0.0
            self._loss_count = 0

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
        return DataLoader(
            self.eval_dataset,
            batch_sampler=self.eval_sampler,
            collate_fn=self.eval_collator,
            num_workers=0,
            pin_memory=False,
        )


# ============================================================
# Helper Functions
# ============================================================

def load_audio_components(model: DualHeadQwenOmni, checkpoint_dir: str):
    """Load audio_embed and audio_head from checkpoint."""
    audio_path = os.path.join(checkpoint_dir, "audio_components.pt")
    if os.path.exists(audio_path):
        logger.info(f"Loading audio components from {audio_path}")
        target_device = model.audio_embed.weight.device
        audio_state = torch.load(audio_path, map_location=target_device, weights_only=True)
        model.audio_embed.load_state_dict(audio_state["audio_embed"])
        model.audio_head.load_state_dict(audio_state["audio_head"])
        return True
    return False


def log_trainable_parameters(module: nn.Module, module_name: str):
    """Log trainable parameter count for a module."""
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    logger.info(
        f"{module_name} trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_pct:.2f}%)"
    )
