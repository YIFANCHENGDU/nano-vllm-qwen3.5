"""Streaming-weight inference for Qwen3 dense models.

Only *one* decoder layer's weights live in GPU VRAM at any given time.  The
token-embedding table, final RMSNorm, and the LM head are small enough to
keep on the GPU permanently.  Everything else is streamed from the layer
files produced by :mod:`nanovllm.utils.save_layers`.

Typical usage::

    from transformers import AutoConfig, AutoTokenizer
    from nanovllm.streaming import StreamingInference

    hf_config = AutoConfig.from_pretrained("/path/to/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/Qwen3-0.6B")

    model = StreamingInference(
        layer_dir="/path/to/streaming_weights",
        hf_config=hf_config,
        device="cuda",
        prefetch=True,
    )

    token_ids = model.generate(
        input_ids=[tokenizer.encode("Hello, Nano-vLLM!")],
        max_new_tokens=64,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(token_ids[0]))

Only standard PyTorch is required (no flash-attn, no third-party streaming
library).  ``torch.nn.functional.scaled_dot_product_attention`` is used for
the attention kernel (available in PyTorch ≥ 2.0).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from nanovllm.streaming.loader import LayerWeightLoader

# ---------------------------------------------------------------------------
# Functional building blocks  (stateless, no nn.Module)
# ---------------------------------------------------------------------------


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Root-mean-square layer normalisation (Llama / Qwen3 style)."""
    orig_dtype = x.dtype
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(orig_dtype)


def _apply_rotary_emb(
    q: torch.Tensor,  # [batch, seq, heads, head_dim]
    k: torch.Tensor,  # [batch, seq, kv_heads, head_dim]
    positions: torch.Tensor,  # [seq]  — absolute position indices
    rope_base: float,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to *q* and *k*.

    Only the tokens indicated by *positions* get their RoPE frequencies; the
    KV cache entries are already encoded from previous steps.
    """
    device, dtype = q.device, q.dtype

    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )  # [head_dim/2]

    t = positions.float()  # [seq]
    freqs = torch.outer(t, inv_freq)  # [seq, head_dim/2]
    cos = freqs.cos().to(dtype).unsqueeze(1)  # [seq, 1, head_dim/2]
    sin = freqs.sin().to(dtype).unsqueeze(1)  # [seq, 1, head_dim/2]

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, heads, head_dim]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        # Broadcast cos/sin over batch and head dims
        c = cos.unsqueeze(0)  # [1, seq, 1, head_dim/2]
        s = sin.unsqueeze(0)  # [1, seq, 1, head_dim/2]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    return _rotate(q), _rotate(k)


def _layer_forward(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,  # [batch, seq, hidden]
    positions: torch.Tensor,  # [seq]
    kv_entry: list,  # [k_cache | None, v_cache | None]  — mutated in-place
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    rms_norm_eps: float,
) -> torch.Tensor:
    """Single Qwen3 decoder layer forward pass.

    The KV cache tensors inside *kv_entry* are updated in-place to append the
    new K/V tokens produced by this step.  All computations use the weight
    *dtype* to avoid unnecessary casts.

    Args:
        weights:       Per-layer weight dict loaded by :class:`LayerWeightLoader`.
        hidden_states: Input activations, shape ``[batch, seq, hidden_size]``.
        positions:     Absolute position indices for the tokens in this call.
        kv_entry:      Two-element list ``[k_cache, v_cache]`` where each is
                       either ``None`` (first call for this layer) or a tensor
                       of shape ``[batch, past_len, kv_heads, head_dim]``.
        num_heads:     Number of query attention heads.
        num_kv_heads:  Number of key/value attention heads (GQA).
        head_dim:      Per-head feature dimension.
        rope_theta:    RoPE base frequency.
        rms_norm_eps:  Epsilon for RMSNorm layers.

    Returns:
        Updated *hidden_states* of the same shape as the input.
    """
    residual = hidden_states

    # ---- 1. Input layernorm ----------------------------------------
    hidden_states = _rms_norm(
        hidden_states, weights["input_layernorm.weight"], rms_norm_eps
    )

    # ---- 2. Self-attention -----------------------------------------
    batch, seq_len, _ = hidden_states.shape

    q = F.linear(
        hidden_states,
        weights["self_attn.q_proj.weight"],
        weights.get("self_attn.q_proj.bias"),
    )
    k = F.linear(
        hidden_states,
        weights["self_attn.k_proj.weight"],
        weights.get("self_attn.k_proj.bias"),
    )
    v = F.linear(
        hidden_states,
        weights["self_attn.v_proj.weight"],
        weights.get("self_attn.v_proj.bias"),
    )

    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_kv_heads, head_dim)
    v = v.view(batch, seq_len, num_kv_heads, head_dim)

    # Optional per-head QK norms (Qwen3 uses these when qkv_bias=False)
    if "self_attn.q_norm.weight" in weights:
        q = _rms_norm(q, weights["self_attn.q_norm.weight"], rms_norm_eps)
        k = _rms_norm(k, weights["self_attn.k_norm.weight"], rms_norm_eps)

    # Rotary position embeddings (applied only to the new tokens)
    q, k = _apply_rotary_emb(q, k, positions, rope_theta, head_dim)

    # Append new K/V to the cache
    if kv_entry[0] is not None:
        k = torch.cat([kv_entry[0], k], dim=1)
        v = torch.cat([kv_entry[1], v], dim=1)
    kv_entry[0] = k
    kv_entry[1] = v

    # Grouped-query attention: tile KV heads to match Q heads
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k_sdpa = k.repeat_interleave(n_rep, dim=2)
        v_sdpa = v.repeat_interleave(n_rep, dim=2)
    else:
        k_sdpa = k
        v_sdpa = v

    # SDPA expects [batch, heads, seq, head_dim]
    q_t = q.transpose(1, 2)
    k_t = k_sdpa.transpose(1, 2)
    v_t = v_sdpa.transpose(1, 2)

    # During prefill (seq_len > 1) apply a causal mask so token i only
    # attends to tokens 0 … i.  During decode (seq_len == 1) is_causal=False
    # gives the single query unrestricted access to the full KV cache.
    attn_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=(seq_len > 1)
    )

    # [batch, heads, seq_q, head_dim] → [batch, seq_q, hidden]
    attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, num_heads * head_dim)
    hidden_states = F.linear(
        attn_out,
        weights["self_attn.o_proj.weight"],
        weights.get("self_attn.o_proj.bias"),
    )

    # Residual connection
    hidden_states = residual + hidden_states
    residual = hidden_states

    # ---- 3. Post-attention layernorm --------------------------------
    hidden_states = _rms_norm(
        hidden_states, weights["post_attention_layernorm.weight"], rms_norm_eps
    )

    # ---- 4. MLP (SwiGLU / SiluAndMul) ------------------------------
    gate = F.linear(
        hidden_states,
        weights["mlp.gate_proj.weight"],
        weights.get("mlp.gate_proj.bias"),
    )
    up = F.linear(
        hidden_states,
        weights["mlp.up_proj.weight"],
        weights.get("mlp.up_proj.bias"),
    )
    mlp_out = F.silu(gate) * up
    mlp_out = F.linear(
        mlp_out,
        weights["mlp.down_proj.weight"],
        weights.get("mlp.down_proj.bias"),
    )

    # Residual connection
    return residual + mlp_out


# ---------------------------------------------------------------------------
# StreamingInference
# ---------------------------------------------------------------------------


class StreamingInference:
    """Single-GPU, streaming-weight inference engine for Qwen3 dense models.

    Memory profile
    ~~~~~~~~~~~~~~
    * GPU: token embeddings + final norm + LM head (permanent) +
           one decoder-layer weight set (transient, loaded/unloaded per layer) +
           full KV cache (accumulated, but small).
    * CPU: optionally one layer weight dict while being loaded in the
           background (prefetch=True).

    Only PyTorch is required — no flash-attn or third-party streaming library.

    Args:
        layer_dir:  Directory of ``layer_XXXX.pt`` files from
                    :func:`nanovllm.utils.save_layers.save_layers`.
        hf_config:  A HuggingFace ``Qwen3Config`` (or compatible ``AutoConfig``).
        device:     Target compute device (default ``'cuda'``).
        prefetch:   Background-prefetch the next layer while the current one
                    is running (recommended; reduces wall-clock time per token).
        dtype:      Weight/activation dtype.  Defaults to ``hf_config.torch_dtype``
                    if available, otherwise ``torch.float16``.
    """

    def __init__(
        self,
        layer_dir: str,
        hf_config,
        device: str = "cuda",
        prefetch: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = device
        self.hf_config = hf_config
        self.dtype = dtype or getattr(hf_config, "torch_dtype", torch.float16)
        self.num_layers: int = hf_config.num_hidden_layers
        self.num_heads: int = hf_config.num_attention_heads
        self.num_kv_heads: int = hf_config.num_key_value_heads
        self.head_dim: int = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        self.rope_theta: float = float(getattr(hf_config, "rope_theta", 1_000_000.0))
        self.rms_norm_eps: float = hf_config.rms_norm_eps

        # Load permanent (non-layer) weights once and keep on the device
        non_layer_path = os.path.join(layer_dir, "non_layer.pt")
        if not os.path.isfile(non_layer_path):
            raise FileNotFoundError(
                f"non_layer.pt not found in {layer_dir!r}. "
                "Run `python -m nanovllm.utils.save_layers` first."
            )
        non_layer = torch.load(non_layer_path, map_location="cpu", weights_only=True)

        def _p(key: str) -> torch.Tensor:
            return non_layer[key].to(device=device, dtype=self.dtype)

        self.embed_weight = _p("model.embed_tokens.weight")  # [vocab, hidden]
        self.norm_weight = _p("model.norm.weight")  # [hidden]
        if "lm_head.weight" in non_layer:
            self.lm_head_weight = _p("lm_head.weight")
        else:
            # Tied embeddings (e.g. small Qwen3 models)
            self.lm_head_weight = self.embed_weight

        # Per-layer KV cache: list of [k | None, v | None]
        # k/v shape when populated: [batch, seq_len, kv_heads, head_dim]
        self.kv_caches: list[list] = [[None, None] for _ in range(self.num_layers)]

        self.loader = LayerWeightLoader(
            layer_dir=layer_dir,
            num_layers=self.num_layers,
            device=device,
            prefetch=prefetch,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings, output dtype matches :attr:`dtype`."""
        return F.embedding(input_ids, self.embed_weight).to(self.dtype)

    def _final_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final RMSNorm and LM-head projection → vocab logits."""
        h = _rms_norm(hidden_states, self.norm_weight, self.rms_norm_eps)
        return F.linear(h, self.lm_head_weight)

    def _stream_forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        positions: torch.Tensor,  # [seq]
    ) -> torch.Tensor:
        """Run all decoder layers by streaming their weights from disk."""
        for layer_idx in range(self.num_layers):
            # Kick off background loading of the *next* layer
            self.loader.schedule_prefetch(layer_idx + 1)
            # Retrieve current layer (may block briefly if prefetch is still in flight)
            weights = self.loader.get(layer_idx)
            # Cast to the inference dtype
            weights = {k: v.to(self.dtype) for k, v in weights.items()}
            hidden_states = _layer_forward(
                weights=weights,
                hidden_states=hidden_states,
                positions=positions,
                kv_entry=self.kv_caches[layer_idx],
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
                rms_norm_eps=self.rms_norm_eps,
            )
            self.loader.release(weights)
        return hidden_states

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_kv_cache(self) -> None:
        """Clear all KV caches to start a fresh conversation."""
        for entry in self.kv_caches:
            entry[0] = None
            entry[1] = None
        if self.device != "cpu":
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> list[list[int]]:
        """Generate tokens for a single prompt using streaming weight loading.

        Weights for each decoder layer are loaded from disk, used for the
        forward pass, and immediately released before the next layer is
        processed.  Only one layer's weights occupy GPU memory at a time.

        Args:
            input_ids:      List containing one token-id sequence (batch_size=1).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature:    Sampling temperature; ``0.0`` → greedy argmax.
            top_p:          Nucleus-sampling threshold (1.0 = disabled).
            eos_token_id:   Stop token ID; generation halts when produced.

        Returns:
            A list containing the generated token-id sequence (prompt excluded).

        Raises:
            ValueError: If ``len(input_ids) != 1`` (only single-sequence
                        batches are supported in streaming mode).
        """
        if len(input_ids) != 1:
            raise ValueError(
                "StreamingInference supports batch_size=1 only. "
                "For multi-sequence batching use the standard LLM engine."
            )

        self.reset_kv_cache()
        device = self.device
        dtype = self.dtype

        ids = torch.tensor(input_ids, dtype=torch.long, device=device)  # [1, seq]
        prompt_len = ids.size(1)

        # ---- prefill: process the full prompt in one shot ----
        positions = torch.arange(prompt_len, dtype=torch.long, device=device)
        hidden = self._embed(ids).to(dtype)  # [1, prompt_len, hidden]
        hidden = self._stream_forward(hidden, positions)  # KV caches filled

        logits = self._final_logits(hidden[:, -1:, :])  # [1, 1, vocab]
        next_token = self._sample(logits[:, 0, :], temperature, top_p)  # [1]
        generated: list[int] = [next_token.item()]

        if eos_token_id is not None and next_token.item() == eos_token_id:
            return [generated]

        # ---- decode: one token at a time ----
        for step in range(max_new_tokens - 1):
            cur_pos = prompt_len + step
            token_tensor = next_token.unsqueeze(0)  # [1, 1]
            positions = torch.tensor([cur_pos], dtype=torch.long, device=device)
            hidden = self._embed(token_tensor).to(dtype)  # [1, 1, hidden]
            hidden = self._stream_forward(hidden, positions)
            logits = self._final_logits(hidden[:, -1:, :])  # [1, 1, vocab]
            next_token = self._sample(logits[:, 0, :], temperature, top_p)
            generated.append(next_token.item())
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return [generated]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(
        logits: torch.Tensor,  # [batch, vocab]
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Sample the next token from *logits*.

        Supports greedy (temperature=0), temperature-only, and nucleus
        (top-p) sampling.  Returns a 1-D token tensor of length *batch*.
        """
        if temperature == 0.0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_p < 1.0:
            # Sort descending, accumulate softmax probabilities, and mask out
            # tokens whose cumulative probability exceeds top_p.
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(dim=-1)
            cum_probs = probs.cumsum(dim=-1)
            # Keep at least the most-probable token
            remove = cum_probs - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            # Scatter back to original vocabulary order
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
