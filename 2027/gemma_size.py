#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sys
import tempfile

import librosa
import numpy as np
import torch
import yaml
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig


QUANTIZATION_METHODS = ["none", "nf4-2q", "fp-w8a8", "int-w8a8", "int8"]

def serialized_model_size(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        path = f.name
        print(f"Saving temporary model to {path}", file=sys.stderr)
        torch.save(model.state_dict(), path)
        return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(description="Translate segmented audio with Gemma 4 12B")
    parser.add_argument("--model-id", default="google/gemma-4-12B-it", help="HF model id")
    parser.add_argument(
        "--quantize",
        nargs="+",
        choices=QUANTIZATION_METHODS,
        default=QUANTIZATION_METHODS,
        help="Quantization methods to evaluate (defaults to all methods)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    for q_method in args.quantize:
        if q_method == "none":
            model = AutoModelForMultimodalLM.from_pretrained(
                args.model_id,
                dtype="auto",
                device_map="auto",
            )
        else:
            if q_method == "nf4-2q":
                # float16 is the most portable compute dtype for bitsandbytes across GPUs
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif q_method == "int8":
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                assert q_method == "fp-w8a8" or q_method == "int-w8a8"
                if q_method == "int-w8a8":
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

        model.eval()
        model_size_bytes = serialized_model_size(model)
        model_size_gib = model_size_bytes / 1024**3
        print(f"Model size ({q_method}): {model_size_gib:.2f} GiB")


if __name__ == "__main__":
    main()
