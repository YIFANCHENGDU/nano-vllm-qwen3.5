import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ReplicatedLinear,
    _make_merged_column_parallel,
    _make_qkv_parallel,
    _make_row_parallel,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = _make_qkv_parallel(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = _make_row_parallel(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = _make_merged_column_parallel(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = _make_row_parallel(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeExpertWeights(nn.Module):
    """Holds the batched 3-D expert weight tensors, matching the HuggingFace
    Qwen3MoeExperts parameter layout so that the weight loader can find them
    under the same paths (``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``)."""

    def __init__(self, num_experts: int, moe_intermediate_size: int, hidden_size: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.gate_up_proj.weight_loader = self._default_loader
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )
        self.down_proj.weight_loader = self._default_loader

    @staticmethod
    def _default_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Mixture-of-Experts block for Qwen3Moe.

    All expert weights are stored in batched 3-D tensors so that the forward
    pass can iterate over only the experts that actually received tokens,
    matching the reference transformers implementation.
    """

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Router is replicated – it is tiny ([num_experts, hidden]) and must
        # see the same logits on every rank to agree on routing decisions.
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)

        # Batched expert weights under the ``experts`` sub-module so that their
        # parameter paths match the HuggingFace checkpoint layout.
        self.experts = Qwen3MoeExpertWeights(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        router_logits = self.gate(hidden_states)  # [tokens, num_experts]
        routing_weights = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # [tokens, top_k]
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # expert_mask: [num_experts, top_k, tokens] (one_hot over expert dim)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)  # [tokens, top_k, experts]
        expert_mask = expert_mask.permute(2, 1, 0)  # [experts, top_k, tokens]

        # Only iterate over experts that received at least one token.
        expert_hit = (expert_mask.sum(dim=(1, 2)) > 0).nonzero(as_tuple=False)
        for idx in expert_hit:
            expert_idx = int(idx[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]  # [n, hidden]

            # gate_up: [2*intermediate, hidden] → output [n, 2*intermediate]
            gate_up_out = F.linear(current_state, self.experts.gate_up_proj[expert_idx])
            gate, up = gate_up_out.chunk(2, dim=-1)
            current_hidden = F.silu(gate) * up  # [n, intermediate]

            # down: [hidden, intermediate] → output [n, hidden]
            current_hidden = F.linear(current_hidden, self.experts.down_proj[expert_idx])

            current_hidden = current_hidden * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
        )
        # Decide dense vs sparse (MoE) MLP for this layer.
        is_moe_layer = (
            config.num_experts > 0
            and layer_idx not in getattr(config, "mlp_only_layers", [])
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    # Packed module mapping: maps individual HF weight names to the combined
    # parameter name and shard identifier used in this model.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
