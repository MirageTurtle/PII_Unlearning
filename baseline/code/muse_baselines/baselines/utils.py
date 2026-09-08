"""Shared model loading, text reading, and token padding helpers."""

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .task_vector import unlearn as tv_unlearn
from .whos_harry_potter import WHPModelForCausalLM


def read_text(file_path: str) -> str:
    """Read a plain-text dataset."""
    if Path(file_path).suffix != ".txt":
        raise ValueError(f"Expected a .txt file: {file_path}")

    with open(file_path, "r") as f:
        return f.read()


def load_model(
    model_dir: str,
    model_name: str | None = None,
    quantization_config: Any = None,
    reinforced_model_dir: str | None = None,
) -> AutoModelForCausalLM:
    """Load a target model or apply the requested WHP/TV transformation."""

    def extract_alpha(s):
        pattern = r"alpha=([+-]?\d*\.\d+|[+-]?\d+)"
        match = re.search(pattern, s)
        return float(match.group(1)) if match else None

    if model_name is not None:
        alpha = extract_alpha(model_name)
        if "whp" in model_name:
            assert reinforced_model_dir is not None
            model = WHPModelForCausalLM(
                model_dir,
                reinforced_model_dir,
                alpha=alpha if alpha is not None else 1.0,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            return model

        elif "tv" in model_name:
            assert reinforced_model_dir is not None
            model = tv_unlearn(
                model_dir=model_dir,
                some_pt_model_dir=model_dir,
                some_ft_model_dir=reinforced_model_dir,
                alpha=alpha if alpha is not None else 1.0,
            )
            return model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def load_tokenizer(
    tokenizer_dir: str, add_pad_token: bool = True, use_fast: bool = True
) -> AutoTokenizer:
    """Load a tokenizer, optionally using EOS as the padding token."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast)
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir: str,
    model_name: str | None = None,
    tokenizer_dir: str | None = None,
    add_pad_token: bool = True,
    quantization_config: Any = None,
    reinforced_model_dir: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer | None]:
    """Load a model and, when requested, its tokenizer."""
    model = load_model(
        model_dir,
        model_name,
        quantization_config,
        reinforced_model_dir=reinforced_model_dir,
    )
    tokenizer = (
        load_tokenizer(tokenizer_dir, add_pad_token)
        if tokenizer_dir is not None
        else None
    )
    return model, tokenizer


def pad_or_trim_tensor(tensor, target_length, padding_value=0):
    """Right-pad or truncate a one-dimensional token tensor."""
    current_length = tensor.size(0)

    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor

    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor

    else:
        # No change needed
        return tensor
