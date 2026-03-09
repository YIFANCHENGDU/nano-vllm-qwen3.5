"""Split a HuggingFace Qwen3 safetensors checkpoint into per-layer PyTorch
files suitable for streaming weight loading.

Usage::

    python -m nanovllm.utils.save_layers \\
        --model_path /path/to/Qwen3-0.6B \\
        --output_dir /path/to/streaming_weights

Output layout::

    {output_dir}/
        layer_0000.pt   # weights for decoder layer 0
        layer_0001.pt   # weights for decoder layer 1
        ...
        non_layer.pt    # embed_tokens, final norm, lm_head
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import torch
from safetensors import safe_open


def save_layers(model_path: str, output_dir: str) -> None:
    """Split a safetensors checkpoint into per-decoder-layer ``.pt`` files.

    Each decoder-layer file stores a ``dict[str, Tensor]`` whose keys are the
    weight names *relative* to that layer (i.e. the ``model.layers.{i}.``
    prefix is stripped).  For example, key ``self_attn.q_proj.weight``
    corresponds to ``model.layers.{i}.self_attn.q_proj.weight`` in the
    original checkpoint.

    The ``non_layer.pt`` file stores all remaining weights (token embeddings,
    final RMSNorm, LM head) under their full checkpoint key names.

    Args:
        model_path: Directory containing ``*.safetensors`` checkpoint shards.
        output_dir: Destination directory (created if it does not exist).
    """
    os.makedirs(output_dir, exist_ok=True)

    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_path!r}. "
            "Make sure the model has been downloaded in safetensors format."
        )

    print(f"Found {len(safetensor_files)} safetensors file(s) — scanning weights …")

    layer_weights: dict[int, dict[str, torch.Tensor]] = {}
    non_layer_weights: dict[str, torch.Tensor] = {}

    for file in safetensor_files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                parts = key.split(".")
                # Decoder layer keys look like: model.layers.{idx}.{rest...}
                if (
                    len(parts) >= 4
                    and parts[0] == "model"
                    and parts[1] == "layers"
                ):
                    try:
                        layer_idx = int(parts[2])
                    except ValueError:
                        non_layer_weights[key] = f.get_tensor(key)
                        continue
                    relative_key = ".".join(parts[3:])
                    if layer_idx not in layer_weights:
                        layer_weights[layer_idx] = {}
                    layer_weights[layer_idx][relative_key] = f.get_tensor(key)
                else:
                    non_layer_weights[key] = f.get_tensor(key)

    if not layer_weights:
        raise ValueError(
            "No decoder-layer weights (model.layers.*) found in the checkpoint. "
            "Only Qwen3 dense and MoE models are currently supported."
        )

    num_layers = max(layer_weights.keys()) + 1
    print(f"Saving {num_layers} decoder layers to {output_dir!r} …")

    for layer_idx in range(num_layers):
        out_path = os.path.join(output_dir, f"layer_{layer_idx:04d}.pt")
        torch.save(layer_weights[layer_idx], out_path)
        if (layer_idx + 1) % 10 == 0 or layer_idx == num_layers - 1:
            print(f"  layer {layer_idx + 1}/{num_layers} saved → {out_path}")

    non_layer_path = os.path.join(output_dir, "non_layer.pt")
    torch.save(non_layer_weights, non_layer_path)
    print(f"Non-layer weights saved → {non_layer_path}")
    print(
        f"\nDone.  {num_layers} layer file(s) + 1 non_layer.pt written to {output_dir!r}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a Qwen3 safetensors checkpoint into per-layer .pt files."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the downloaded model directory (contains *.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory for the layer files.",
    )
    args = parser.parse_args()
    save_layers(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
