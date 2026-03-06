import copy
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoConfig


# Mapping from quantization name to its default configuration dict.
# quant_type: backend identifier (matched in linear.py factory helpers)
# w_bit:      weight bit-width (4 = int4)
# q_group_size: number of input channels per quantization group
_QUANTIZATION_CONFIGS = {
    "awq": {"quant_type": "awq", "w_bit": 4, "q_group_size": 128},
}


def _resolve_model_path(model: str) -> str:
    """Resolve *model* to a local directory path.

    If *model* is already a local directory, it is returned unchanged.
    Otherwise the string is treated as a HuggingFace Hub model ID and the
    model is downloaded to the local cache via ``huggingface_hub.snapshot_download``.

    Args:
        model: Local filesystem path **or** a HuggingFace Hub model ID such as
            ``"Qwen/Qwen3-0.6B-AWQ"``.

    Returns:
        Absolute path to the local model directory.

    Raises:
        ValueError: If *model* is not a local directory and ``huggingface_hub``
            is not installed.
        OSError: If the Hub download fails.
    """
    if os.path.isdir(model):
        return model

    # Not a local directory — try to download from the HuggingFace Hub.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError(
            f"'{model}' is not a local directory. "
            "To use a HuggingFace Hub model ID, install huggingface_hub: "
            "pip install huggingface_hub"
        ) from exc

    return snapshot_download(model)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: str | None = None
    hf_config: "AutoConfig | None" = field(default=None, repr=False)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quant_config: dict | None = None

    def __post_init__(self):
        # Resolve local path or HuggingFace Hub model ID → local directory.
        self.model = _resolve_model_path(self.model)

        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                f"kvcache_block_size must be a multiple of 256, "
                f"got {self.kvcache_block_size}."
            )
        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                f"tensor_parallel_size must be between 1 and 8, "
                f"got {self.tensor_parallel_size}."
            )
        if self.quantization is not None:
            if self.quantization not in _QUANTIZATION_CONFIGS:
                raise ValueError(
                    f"Unsupported quantization '{self.quantization}'. "
                    f"Supported: {list(_QUANTIZATION_CONFIGS)}"
                )
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be "
                f">= max_model_len ({self.max_model_len}). "
                f"Increase max_num_batched_tokens or decrease max_model_len."
            )
        # Detect quantization configuration embedded in the model config.
        raw_qc = getattr(self.hf_config, "quantization_config", None)
        if raw_qc is not None:
            if hasattr(raw_qc, "to_dict"):
                raw_qc = raw_qc.to_dict()
            # Normalize to a plain dict with lowercase keys.
            raw_qc = {k.lower(): v for k, v in raw_qc.items()}
            # Normalize key names used by different AWQ serialisation conventions
            # (e.g. autoawq uses 'quant_method'/'bits'/'group_size' while our
            # internal format uses 'quant_type'/'w_bit'/'q_group_size').
            if "quant_type" not in raw_qc and "quant_method" in raw_qc:
                raw_qc["quant_type"] = raw_qc["quant_method"]
            if "w_bit" not in raw_qc and "bits" in raw_qc:
                raw_qc["w_bit"] = raw_qc["bits"]
            if "q_group_size" not in raw_qc and "group_size" in raw_qc:
                raw_qc["q_group_size"] = raw_qc["group_size"]
            self.quant_config = raw_qc
        elif self.quantization is not None:
            # Use the explicitly requested quantization when the model config
            # does not embed its own quantization_config.
            self.quant_config = copy.deepcopy(_QUANTIZATION_CONFIGS[self.quantization])
