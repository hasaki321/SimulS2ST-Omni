from dataclasses import dataclass
from typing import Optional, Tuple

from src.train.lang_utils import normalize_lang
from src.train.modeling_dual_head import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_SPEECH_END_TOKEN,
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SYSTEM_PROMPT,
)

PROMPT_SPEC_VERSION = "prompt_spec_v1"

TASK_TTS_UNPAIRED = "tts_unpaired"
TASK_TTS_PAIRED_SAME_LANG = "tts_paired_same_lang"
TASK_TTS_PAIRED_CROSS_LANG = "tts_paired_cross_lang"
TASK_S2TT_ASR = "s2tt_asr"
TASK_S2TT_TRANSLATE = "s2tt_translate"
TASK_T2T_TRANSLATE = "t2t_translate"
TASK_T2S_TRANSLATE = "t2s_translate"
TASK_S2S_TRANSLATE = "s2s_translate"
TASK_STREAMING_T2T_TRANSLATE = "streaming_t2t_translate"
TASK_STREAMING_S2TT_ASR = "streaming_s2tt_asr"
TASK_STREAMING_S2TT_TRANSLATE = "streaming_s2tt_translate"
TASK_STREAMING_TTS = "streaming_tts"
TASK_STREAMING_S2S_TRANSLATE = "streaming_s2s_translate"


@dataclass(frozen=True)
class PromptBundle:
    task_type: str
    system_prompt: str
    user_payload: str
    assistant_payload: str


def build_audio_span(n_audio_tokens: int) -> str:
    return (
        DEFAULT_SPEECH_START_TOKEN
        + (DEFAULT_SPEECH_PATCH_TOKEN * n_audio_tokens)
        + DEFAULT_SPEECH_END_TOKEN
    )


def build_text_payload(text: str, lang: str) -> str:
    return text


def resolve_tts_task_type(is_paired: bool, ref_lang: str, tgt_lang: str) -> str:
    if not is_paired:
        return TASK_TTS_UNPAIRED
    if normalize_lang(ref_lang) == normalize_lang(tgt_lang):
        return TASK_TTS_PAIRED_SAME_LANG
    return TASK_TTS_PAIRED_CROSS_LANG


def build_system_prompt(
    task_type: str,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    src_lang: str = "",
    tgt_lang: str = "",
    latency: Optional[int] = None,
) -> str:
    src_lang = normalize_lang(src_lang) if src_lang else ""
    tgt_lang = normalize_lang(tgt_lang) if tgt_lang else ""
    latency_suffix = f" With Latency: {latency}." if latency is not None else ""

    # Text only: Repeaat / Translate
    # Audio input: Transcibe / translate the audio
    # Audio output: The same as Text only

    # Audio input and output: [Audio input] + matches the speaker style
    task_instruction_map = {
        TASK_TTS_UNPAIRED: f"Repeat the user's text exactly as written in {tgt_lang}.",
        TASK_TTS_PAIRED_SAME_LANG: (
            f"Repeat the user's text exactly as written in {tgt_lang}."
            f"Generate speech in {tgt_lang} that matches the speaker style of the user's audio. "
        ),
        TASK_TTS_PAIRED_CROSS_LANG: (
            f"Repeat the user's text exactly as written in {tgt_lang}."
            f"Generate speech in {tgt_lang} that matches the speaker style of the user's audio in {src_lang}. "
        ),

        TASK_T2T_TRANSLATE: f"Translate the user's text from {src_lang} to {tgt_lang}.",
        TASK_T2S_TRANSLATE: f"Translate the user's text from {src_lang} to {tgt_lang}.",

        TASK_S2TT_ASR: f"Transcribe the user's audio in {src_lang}.",
        TASK_S2TT_TRANSLATE: f"Translate the user's audio from {src_lang} to {tgt_lang}.",
        TASK_S2S_TRANSLATE: (
            f"Translate the user's audio from {src_lang} to {tgt_lang}."
            f"Generate speech in {tgt_lang} that matches the speaker style of the user's audio in {src_lang}."
        ),

        TASK_STREAMING_T2T_TRANSLATE: (
            f"Translate the user's increment text from {src_lang} to {tgt_lang}.{latency_suffix}"
        ),
        TASK_STREAMING_S2TT_ASR: f"Transcribe the user's increment audio in {src_lang}.{latency_suffix}",
        TASK_STREAMING_S2TT_TRANSLATE: (
            f"Translate the user's increment audio from {src_lang} to {tgt_lang}.{latency_suffix}"
        ),
        TASK_STREAMING_TTS: (
            f"Repeat the user's increment text exactly as written in {tgt_lang}.{latency_suffix}"
        ),
        TASK_STREAMING_S2S_TRANSLATE: (
            f"Translate the user's increment audio from {src_lang} to {tgt_lang}."
            # f"Generate increment speech in {tgt_lang} that matches the speaker style of the user's audio in {src_lang}."
            f"{latency_suffix}"
        ),
    }
    task_instruction = task_instruction_map[task_type]
    if base_system_prompt:
        return base_system_prompt + "\n" + task_instruction
    return task_instruction


def build_user_payload(
    task_type: str,
    text: str = "",
    src_text: str = "",
    src_lang: str = "",
    tgt_lang: str = "",
    n_audio_tokens: int = 0,
) -> str:
    src_lang = normalize_lang(src_lang) if src_lang else ""
    tgt_lang = normalize_lang(tgt_lang) if tgt_lang else ""
    audio_payload = src_lang + build_audio_span(n_audio_tokens) if n_audio_tokens > 0 else ""

    if task_type == TASK_TTS_UNPAIRED:
        return build_text_payload(text, tgt_lang)
    if task_type in (TASK_TTS_PAIRED_SAME_LANG, TASK_TTS_PAIRED_CROSS_LANG):
        return audio_payload + "\n" + build_text_payload(text, tgt_lang)
    if task_type in (TASK_S2TT_ASR, TASK_S2TT_TRANSLATE, TASK_S2S_TRANSLATE, TASK_STREAMING_S2TT_ASR, TASK_STREAMING_S2TT_TRANSLATE, TASK_STREAMING_S2S_TRANSLATE):
        return audio_payload
    if task_type in (TASK_T2T_TRANSLATE, TASK_T2S_TRANSLATE):
        return build_text_payload(src_text, src_lang)
    if task_type == TASK_STREAMING_TTS:
        return build_text_payload(text, tgt_lang)
    raise ValueError(f"Unsupported task_type: {task_type}")


def build_assistant_payload(
    task_type: str,
    target_text: str,
    tgt_lang: str,
) -> str:
    del task_type
    return build_text_payload(target_text, tgt_lang)


def render_chat_text(
    system_prompt: str,
    user_payload: str,
    assistant_payload: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    rendered = (
        f"{DEFAULT_BOS_TOKEN}system\n{system_prompt}{DEFAULT_EOS_TOKEN}\n"
        f"{DEFAULT_BOS_TOKEN}user\n{user_payload}{DEFAULT_EOS_TOKEN}\n"
    )
    if add_generation_prompt:
        return rendered + f"{DEFAULT_BOS_TOKEN}assistant\n"
    assert assistant_payload is not None, "assistant_payload is required when add_generation_prompt=False"
    return rendered + f"{DEFAULT_BOS_TOKEN}assistant\n{assistant_payload}{DEFAULT_EOS_TOKEN}\n"


def build_prompt_bundle(
    task_type: str,
    base_system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    text: str = "",
    src_text: str = "",
    target_text: str = "",
    src_lang: str = "",
    tgt_lang: str = "",
    n_audio_tokens: int = 0,
    latency: Optional[int] = None,
) -> PromptBundle:
    system_prompt = build_system_prompt(
        task_type=task_type,
        base_system_prompt=base_system_prompt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        latency=latency,
    )
    user_payload = build_user_payload(
        task_type=task_type,
        text=text,
        src_text=src_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        n_audio_tokens=n_audio_tokens,
    )
    assistant_payload = build_assistant_payload(
        task_type=task_type,
        target_text=target_text,
        tgt_lang=tgt_lang,
    )
    return PromptBundle(
        task_type=task_type,
        system_prompt=system_prompt,
        user_payload=user_payload,
        assistant_payload=assistant_payload,
    )


def build_prompt_texts(bundle: PromptBundle) -> Tuple[str, str]:
    prefix_text = render_chat_text(
        system_prompt=bundle.system_prompt,
        user_payload=bundle.user_payload,
        add_generation_prompt=True,
    )
    full_text = render_chat_text(
        system_prompt=bundle.system_prompt,
        user_payload=bundle.user_payload,
        assistant_payload=bundle.assistant_payload,
        add_generation_prompt=False,
    )
    return prefix_text, full_text
