#!/usr/bin/env python3
# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


GEMMA_AUDIO_SAMPLE_RATE = 16_000


def build_translation_instruction(lang: str) -> str:
    language_names = {"de": "German", "zh": "Chinese"}
    tgt_lang = language_names[lang]
    return (
        f"Translate the following speech into {tgt_lang}. "
        "Output only the translation, with no explanation or newlines."
    )


def build_audio_messages(lang: str, audio):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_translation_instruction(lang)},
                {"type": "audio", "audio": audio},
            ],
        }
    ]


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, _ = librosa.load(path.as_posix(), sr=target_sr, mono=True)
    return audio


def load_audio_segment(path: Path, target_sr: int, offset: float, duration: float):
    audio, _ = librosa.load(
        path.as_posix(),
        sr=target_sr,
        mono=True,
        offset=offset,
        duration=duration,
    )
    return audio


def build_inputs(model, processor, audio: np.ndarray, lang: str):
    inputs = processor.apply_chat_template(
        build_audio_messages(lang, audio),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return inputs.to(model.device, dtype=model.dtype)


def _decode_generated(processor, inputs, generated_ids) -> str:
    """Decode only the text generated after the Gemma chat prompt."""
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, prompt_len:]
    return processor.decode(
        new_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def translate_file(
    model,
    processor,
    audio_path: Path,
    lang: str,
    max_new_tokens: int = 256,
):
    audio = load_audio(audio_path, GEMMA_AUDIO_SAMPLE_RATE)
    inputs = build_inputs(model, processor, audio, lang)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return _decode_generated(processor, inputs, generated_ids)


def translate_audio_segment(
    model,
    processor,
    audio: np.ndarray,
    lang: str,
    max_new_tokens: int = 256,
):
    inputs = build_inputs(model, processor, audio, lang)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return _decode_generated(processor, inputs, generated_ids)


def load_segments(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("Segments YAML must be a list of objects.")
    for i, segment in enumerate(data):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {i} is not an object.")
        for key in ("wav", "offset", "duration"):
            if key not in segment:
                raise ValueError(f"Segment {i} missing required key: {key}")
    return data


def load_file_order(input_dir: Path):
    file_order_path = input_dir / "FILE_ORDER"
    if not file_order_path.exists():
        raise FileNotFoundError(f"Missing FILE_ORDER file at: {file_order_path}")
    ordered = []
    with file_order_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            ordered.append(name)
    if not ordered:
        raise ValueError(f"FILE_ORDER is empty: {file_order_path}")
    return ordered


def order_audio_files(audio_files, file_order, input_dir: Path):
    rel_map = {}
    name_map = {}
    for p in audio_files:
        rel = p.relative_to(input_dir).as_posix()
        rel_map[rel] = p
        if p.name not in name_map:
            name_map[p.name] = []
        name_map[p.name].append(p)

    ordered = []
    used = set()
    for entry in file_order:
        chosen = rel_map.get(entry)
        candidates = name_map.get(entry, [])
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None:
            print(f"[WARN] FILE_ORDER entry not found among audio files: {entry}")
            continue
        if chosen in used:
            for c in candidates:
                if c not in used:
                    chosen = c
                    break
        if chosen not in used:
            ordered.append(chosen)
            used.add(chosen)
    for p in audio_files:
        if p not in used:
            ordered.append(p)
    return ordered


def one_line(text: str) -> str:
    return " ".join(text.splitlines()).strip()


def order_segments(segments, file_order):
    rank = {name: i for i, name in enumerate(file_order)}
    enumerated = list(enumerate(segments))
    ordered = sorted(
        enumerated,
        key=lambda pair: (
            rank.get(pair[1]["wav"], rank.get(Path(pair[1]["wav"]).name, len(rank))),
            pair[0],
        ),
    )
    return [segment for _, segment in ordered]


def main():
    parser = argparse.ArgumentParser(description="Translate segmented audio with Gemma 4 12B")
    parser.add_argument("--input-dir", required=True, help="Folder containing audio files")
    parser.add_argument("--output-path", required=True, help="Path to store translations")
    parser.add_argument("--segments-yaml", default=None, help="YAML file with segment list (wav, offset, duration)")
    parser.add_argument("--model-id", default="google/gemma-4-12B-it", help="HF model id")
    parser.add_argument("--lang", choices=["de", "zh"], default="de", help="Target language (de=German, zh=Chinese)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quantize", choices=["nf4-2q", "fp-w8a8", "int-w8a8", "int8"], default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)

    segments = None
    if args.segments_yaml:
        segments = load_segments(Path(args.segments_yaml))
        if len(set(s["wav"] for s in segments)) > 1:
            file_order = load_file_order(input_dir)
            segments = order_segments(segments, file_order)
        if not segments:
            print(f"No segments found in {args.segments_yaml}")
            return
    else:
        audio_files = [p for p in input_dir.rglob("*") if p.suffix.lower() == ".wav"]
        file_order = load_file_order(input_dir)
        audio_files = order_audio_files(audio_files, file_order, input_dir)
        if not audio_files:
            print(f"No supported audio files found in {input_dir}")
            return

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    if args.quantize:
        if args.quantize == "nf4-2q":
            # float16 is the most portable compute dtype for bitsandbytes across GPUs
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif args.quantize == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            assert args.quantize == "fp-w8a8" or args.quantize == "int-w8a8"
            if args.quantize == "int-w8a8":
                from torchao.quantization import Int8DynamicActivationInt8WeightConfig
                tao_config = Int8DynamicActivationInt8WeightConfig()
            else:
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                tao_config = Float8DynamicActivationFloat8WeightConfig()
            from transformers import TorchAoConfig
            quant_config = TorchAoConfig(quant_type=tao_config)

        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_id,
            quantization_config=quant_config,
            device_map="auto",
        )

    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_id,
            dtype="auto",
            device_map="auto",
        )
    model.eval()

    with output_path.open("w", encoding="utf-8") as tf:
        if segments is not None:
            total = len(segments)
            for i, segment in enumerate(segments, start=1):
                wav_rel = segment["wav"]
                offset = float(segment["offset"])
                duration = float(segment["duration"])
                audio_path = input_dir / wav_rel
                audio = load_audio_segment(
                    audio_path,
                    target_sr=GEMMA_AUDIO_SAMPLE_RATE,
                    offset=offset,
                    duration=duration,
                )
                translated_text = translate_audio_segment(
                    model=model,
                    processor=processor,
                    audio=audio,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                )

                line = one_line(translated_text)
                tf.write(line + "\n")

                print(
                    f"[{i}/{total}] OK: {audio_path.name} "
                    f"(offset={offset:.3f}s, duration={duration:.3f}s) -> line {i} in {output_path.name}"
                )
        else:
            total = len(audio_files)
            for i, audio_path in enumerate(audio_files, start=1):
                translated_text = translate_file(
                    model=model,
                    processor=processor,
                    audio_path=audio_path,
                    lang=args.lang,
                    max_new_tokens=args.max_new_tokens,
                )
                line = one_line(translated_text)
                tf.write(line + "\n")

                print(
                    f"[{i}/{total}] OK: {audio_path.name} -> line {i} in {output_path.name}"
                )

    print(f"Done. Results in: {output_path}")


if __name__ == "__main__":
    main()
