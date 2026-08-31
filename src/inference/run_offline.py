#!/usr/bin/env python3
"""Unified single-example offline inference for ASR, TTS, S2TT, and S2ST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


MODEL_ROOT = Path("models/SimulS2ST-Omni")


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("asr", "tts", "s2tt", "s2st"))
    parser.add_argument("--audio", help="Input speech for ASR/S2TT/S2ST, or reference speech for TTS")
    parser.add_argument("--text", help="Text to synthesize for TTS")
    parser.add_argument("--ref-text", default="", help="Reference transcript used by unpaired TTS prompting")
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Chinese")
    parser.add_argument("--output", required=True, help="Output JSON for text tasks or WAV for speech tasks")
    parser.add_argument("--model-path", default=str(MODEL_ROOT / "offline"))
    parser.add_argument("--checkpoint-path", default=str(MODEL_ROOT / "offline"))
    parser.add_argument("--voicebox-path", default=str(MODEL_ROOT / "voicebox/voicebox.safetensors"))
    parser.add_argument("--vocos-path", default=str(MODEL_ROOT / "voicebox/vocos.safetensors"))
    parser.add_argument("--voicebox-config", default=str(MODEL_ROOT / "voicebox/voicebox_config.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--max-audio-tokens", type=int, default=500)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio:
        raise ValueError(f"--audio is required for {args.task}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.task in {"asr", "s2tt"}:
        from src.inference.omni_talker_s2t_inference import OmniTalkerS2TInference

        model = OmniTalkerS2TInference.from_pretrained(
            checkpoint_path=args.checkpoint_path,
            omni_model_path=args.model_path,
            device=args.device,
        )
        audio = load_audio(args.audio, 16_000).numpy()
        target_lang = args.source_lang if args.task == "asr" else args.target_lang
        text = model.translate_audio(
            audio,
            source_lang=args.source_lang,
            target_lang=target_lang,
            max_new_tokens=args.max_text_tokens,
            do_sample=args.sample,
        )
        output.write_text(
            json.dumps({"task": args.task, "audio": args.audio, "text": text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(text)
        return

    from src.inference.infer_omni_talker_tts import OmniTalkerSemanticTTSInference

    model = OmniTalkerSemanticTTSInference.from_pretrained(
        checkpoint_path=args.checkpoint_path,
        omni_model_path=args.model_path,
        voicebox_path=args.voicebox_path,
        vocos_path=args.vocos_path,
        voicebox_config_path=args.voicebox_config,
        device=args.device,
    )
    audio_16k = load_audio(args.audio, 16_000)
    audio_24k = load_audio(args.audio, 24_000)
    if args.task == "tts":
        if not args.text:
            raise ValueError("--text is required for TTS")
        generated, codes, thinker_text = model.synthesize(
            tgt_text=args.text,
            ref_text=args.ref_text,
            ref_audio_16k=audio_16k,
            ref_audio_24k=audio_24k,
            prompt_format="unpaired" if args.ref_text else "paired",
            ref_lang=args.source_lang,
            tgt_lang=args.target_lang,
            max_text_tokens=args.max_text_tokens,
            max_audio_tokens=args.max_audio_tokens,
            do_sample=args.sample,
        )
    else:
        generated, thinker_text, codes = model.e2e(
            audio_16k=audio_16k,
            audio_24k=audio_24k,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            max_s2tt_new_tokens=args.max_text_tokens,
            max_tts_new_tokens=args.max_audio_tokens,
            do_sample=args.sample,
        )
    sf.write(output, np.asarray(generated), 24_000)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "task": args.task,
                "audio": args.audio,
                "text": args.text,
                "thinker_text": thinker_text,
                "num_codes": int(codes.numel()),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
