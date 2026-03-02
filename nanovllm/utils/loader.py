import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# AWQ quantized weight suffixes that are stored as parameters rather than
# the conventional ".weight" suffix.
_AWQ_SUFFIXES = (".qweight", ".qzeros", ".scales")


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_param(model: nn.Module, name: str) -> nn.Parameter:
    """Return a parameter or non-trainable parameter (buffer stored as param)."""
    try:
        return model.get_parameter(name)
    except AttributeError:
        pass
    # Walk the path manually to support both nn.Parameter and plain Tensors
    parts = name.split(".")
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return obj


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Determine whether this is an AWQ quantized weight and strip
                # the suffix so the packed_modules_mapping key lookup works.
                awq_suffix = ""
                base_name = weight_name
                for sfx in _AWQ_SUFFIXES:
                    if weight_name.endswith(sfx):
                        awq_suffix = sfx
                        base_name = weight_name[: -len(sfx)]
                        break

                for k in packed_modules_mapping:
                    if k in base_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = base_name.replace(k, v) + awq_suffix
                        param = _get_param(model, param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = _get_param(model, weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
