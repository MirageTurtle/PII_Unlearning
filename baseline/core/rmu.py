"""Standard RMU: random-vector steering with frozen-target retention."""

import json
import math
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import set_seed

from .utils import load_model, load_model_and_tokenizer


def _read_examples(path):
    """Read one example per nonblank TXT line, or JSON strings/text objects."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".txt":
            examples = [line.strip() for line in stream if line.strip()]
        elif path.suffix == ".json":
            examples = json.load(stream)
            if not isinstance(examples, list):
                raise ValueError(f"Expected a JSON list in {path}.")
            examples = [
                item.get("text") if isinstance(item, dict) else item
                for item in examples
            ]
        else:
            raise ValueError(f"Expected a .txt or .json dataset: {path}")
    if not examples or any(
        not isinstance(text, str) or not text.strip() for text in examples
    ):
        raise ValueError(f"Dataset must contain nonempty text examples: {path}")
    return examples


def _decoder_layers(model):
    """Locate common HF decoder blocks without evaluating user-supplied code."""
    for path in (
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    ):
        try:
            layers = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList) and len(layers):
            return layers
    raise ValueError("RMU cannot locate decoder blocks in this model architecture.")


def _select_parameters(model, layer_id, layer_ids):
    layers = _decoder_layers(model)
    if not 0 <= layer_id < len(layers):
        raise ValueError(f"RMU loss layer must be between 0 and {len(layers) - 1}.")
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("RMU update layers must be nonempty and unique.")
    if any(index < 0 or index > layer_id for index in layer_ids):
        raise ValueError("RMU update layers must be between 0 and the loss layer.")
    model.requires_grad_(False)
    for index in layer_ids:
        layers[index].requires_grad_(True)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("RMU update layers contain no parameters.")
    return layers[layer_id], parameters


def _activations(model, inputs, module):
    """Capture a block's output with its graph; always remove the temporary hook."""
    cache = []

    def capture(_module, _inputs, output):
        cache.append(output[0] if isinstance(output, tuple) else output)

    handle = module.register_forward_hook(capture)
    try:
        # The language-model head is unnecessary for an activation loss.
        model.base_model(**inputs, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if len(cache) != 1 or not isinstance(cache[0], torch.Tensor):
        raise ValueError("RMU expected one tensor output from the selected block.")
    return cache[0]


def _masked_mse(activations, target, attention_mask):
    """Mean squared error over real tokens and hidden dimensions, in float32."""
    mask = attention_mask.to(device=activations.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError("RMU received a batch with no unmasked tokens.")
    target = target.to(device=activations.device, dtype=torch.float32)
    difference = activations.float() - target
    return difference[mask].square().mean()


def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str,
    *,
    layer_id: int,
    layer_ids: list[int] | None = None,
    steering_coeff: float = 20.0,
    retain_weight: float = 100.0,
    seed: int = 42,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
):
    """Update selected blocks using a fixed random anchor and retain activations.

    All parameters in the selected blocks are trainable. Layers after the loss
    block cannot receive gradients from this objective and are rejected. Each
    epoch visits every forget example, cycling retain examples as needed.
    """
    if epochs < 1 or per_device_batch_size < 1 or max_len < 2:
        raise ValueError(
            "Epochs/batch size must be positive; max_len must be at least 2."
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not math.isfinite(steering_coeff) or steering_coeff <= 0:
        raise ValueError("RMU steering coefficient must be finite and positive.")
    if not math.isfinite(retain_weight) or retain_weight < 0:
        raise ValueError("RMU retain weight must be finite and non-negative.")
    if not 0 <= seed < 2**32:
        raise ValueError("RMU seed must be between 0 and 2**32 - 1.")
    forget = _read_examples(data_file)
    retain = _read_examples(retain_data_file)
    layer_ids = [layer_id] if layer_ids is None else layer_ids
    set_seed(seed)
    model, tokenizer = load_model_and_tokenizer(
        model_dir, tokenizer_dir=tokenizer_dir or model_dir
    )
    module, parameters = _select_parameters(model, layer_id, layer_ids)
    frozen_model = load_model(model_dir)
    frozen_model.requires_grad_(False)
    frozen_model.eval()
    frozen_module = _decoder_layers(frozen_model)[layer_id]
    model.train()
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    # One U[0, 1) direction per run, fixed across all examples and epochs.
    hidden_size = model.config.hidden_size
    direction = torch.rand(
        1, 1, hidden_size, generator=torch.Generator().manual_seed(seed)
    )
    control_vector = steering_coeff * direction / direction.norm()
    rng = random.Random(seed)

    def tokenize(texts, device):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )
        return {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}

    model_device = model.get_input_embeddings().weight.device
    frozen_device = frozen_model.get_input_embeddings().weight.device
    for epoch in range(epochs):
        rng.shuffle(forget)
        rng.shuffle(retain)
        batches = tqdm(
            range(0, len(forget), per_device_batch_size),
            desc=f"RMU epoch {epoch + 1}/{epochs}",
        )
        for start in batches:
            forget_texts = forget[start : start + per_device_batch_size]
            retain_texts = [
                retain[(start + i) % len(retain)] for i in range(len(forget_texts))
            ]
            forget_inputs = tokenize(forget_texts, model_device)
            retain_inputs = tokenize(retain_texts, model_device)
            optimizer.zero_grad(set_to_none=True)
            forget_activations = _activations(model, forget_inputs, module)
            forget_loss = _masked_mse(
                forget_activations, control_vector, forget_inputs["attention_mask"]
            )
            with torch.no_grad():
                frozen_activations = _activations(
                    frozen_model,
                    {
                        key: value.to(frozen_device)
                        for key, value in retain_inputs.items()
                    },
                    frozen_module,
                )
            retain_activations = _activations(model, retain_inputs, module)
            retain_loss = _masked_mse(
                retain_activations, frozen_activations, retain_inputs["attention_mask"]
            )
            loss = forget_loss + retain_weight * retain_loss
            if not torch.isfinite(loss):
                raise ValueError("RMU loss is not finite; check the training settings.")
            loss.backward()
            if any(parameter.grad is None for parameter in parameters):
                raise ValueError(
                    "Some selected RMU parameters did not receive gradients."
                )
            optimizer.step()
            batches.set_postfix(
                forget=f"{forget_loss.item():.4g}", retain=f"{retain_loss.item():.4g}"
            )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
