import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ---------------------------------------------------------------------------
# Global quantization configuration
# ---------------------------------------------------------------------------

_quant_config: dict | None = None


def set_quant_config(config: dict | None):
    global _quant_config
    _quant_config = config


def get_quant_config() -> dict | None:
    return _quant_config


# ---------------------------------------------------------------------------
# Module-level cache for the AWQ GEMM kernel function
# ---------------------------------------------------------------------------

_WQLinearMMFunction = None
_awq_import_attempted = False


def _get_awq_gemm_fn():
    """Return WQLinearMMFunction, or None if autoawq is not available."""
    global _WQLinearMMFunction, _awq_import_attempted
    if not _awq_import_attempted:
        _awq_import_attempted = True
        try:
            from awq.modules.linear.gemm import WQLinearMMFunction as _fn
            _WQLinearMMFunction = _fn
        except ImportError:
            pass
    return _WQLinearMMFunction


# ---------------------------------------------------------------------------
# AWQ quantized linear base
# ---------------------------------------------------------------------------

class AWQLinearBase(nn.Module):
    """Base class for AWQ 4-bit quantized linear layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        w_bit: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.w_bit = w_bit
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        pack = 32 // w_bit  # 8 for 4-bit

        # Quantized weights stored as non-trainable parameters so that
        # model.get_parameter() can find them (weight_loader machinery).
        self.qweight = nn.Parameter(
            torch.zeros(in_features, out_features // pack, dtype=torch.int32),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.zeros(in_features // group_size, out_features // pack, dtype=torch.int32),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.zeros(in_features // group_size, out_features, dtype=torch.float16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float16), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def _awq_forward(self, x: torch.Tensor, bias=None) -> torch.Tensor:
        """Run the AWQ GEMM kernel, falling back to dequantize+matmul."""
        awq_fn = _get_awq_gemm_fn()
        if awq_fn is not None:
            try:
                out_shape = x.shape[:-1] + (self.out_features,)
                inp_dtype = x.dtype
                if inp_dtype != torch.float16:
                    x = x.half()
                with torch.no_grad():
                    out = awq_fn.apply(
                        x, self.qweight, self.qzeros, self.scales,
                        self.w_bit, self.group_size, bias, self.out_features,
                    )
                if inp_dtype != torch.float16:
                    out = out.to(inp_dtype)
                return out.reshape(out_shape)
            except RuntimeError:
                pass
        # Fallback: dequantize then standard matmul (correct but slower)
        weight = self._dequantize()
        return F.linear(x, weight, bias)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize int4 weights to float16, shape [out, in]."""
        pack = 32 // self.w_bit
        shifts = torch.arange(0, 32, self.w_bit, device=self.qweight.device)

        iweight = torch.bitwise_and(
            self.qweight.unsqueeze(-1) >> shifts, 2 ** self.w_bit - 1
        ).reshape(self.in_features, self.out_features).float()

        izeros = torch.bitwise_and(
            self.qzeros.unsqueeze(-1) >> shifts, 2 ** self.w_bit - 1
        ).reshape(self.in_features // self.group_size, self.out_features).float()

        scales = self.scales.float()  # [in//group, out]
        izeros = izeros.repeat_interleave(self.group_size, dim=0)  # [in, out]
        scales = scales.repeat_interleave(self.group_size, dim=0)  # [in, out]

        weight = (iweight - izeros) * scales  # [in, out]
        return weight.to(self.scales.dtype).T  # [out, in]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AWQColumnParallelLinear(AWQLinearBase):
    """AWQ quantized column-parallel linear (output sharded across ranks)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 w_bit: int = 4, group_size: int = 128):
        tp_size = dist.get_world_size()
        per_rank_out = divide(out_features, tp_size)
        super().__init__(in_features, per_rank_out, bias, w_bit, group_size)
        pack = 32 // w_bit
        self.qweight.weight_loader = self._qweight_loader
        self.qzeros.weight_loader = self._qzeros_loader
        self.scales.weight_loader = self._scales_loader

    def _qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(1)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))

    def _qzeros_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(1)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))

    def _scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(1)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._awq_forward(x, self.bias)


class AWQMergedColumnParallelLinear(AWQColumnParallelLinear):
    """AWQ quantized merged column-parallel linear (e.g. gate+up)."""

    def __init__(self, in_features: int, output_sizes: list[int], bias: bool = False,
                 w_bit: int = 4, group_size: int = 128):
        self.output_sizes = output_sizes
        super().__init__(in_features, sum(output_sizes), bias, w_bit, group_size)
        pack = 32 // w_bit
        self.qweight.weight_loader = self._merged_qweight_loader
        self.qzeros.weight_loader = self._merged_qzeros_loader
        self.scales.weight_loader = self._merged_scales_loader

    def _merged_qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                                loaded_shard_id: int):
        pack = 32 // self.w_bit
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size // pack
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size // pack
        src_start = self.tp_rank * shard_size
        param.data.narrow(1, shard_offset, shard_size).copy_(
            loaded_weight.narrow(1, src_start, shard_size)
        )

    def _merged_qzeros_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                               loaded_shard_id: int):
        pack = 32 // self.w_bit
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size // pack
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size // pack
        src_start = self.tp_rank * shard_size
        param.data.narrow(1, shard_offset, shard_size).copy_(
            loaded_weight.narrow(1, src_start, shard_size)
        )

    def _merged_scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                               loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        src_start = self.tp_rank * shard_size
        param.data.narrow(1, shard_offset, shard_size).copy_(
            loaded_weight.narrow(1, src_start, shard_size)
        )


class AWQQKVParallelLinear(AWQColumnParallelLinear):
    """AWQ quantized QKV parallel linear (packs Q, K, V into one tensor)."""

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
                 total_num_kv_heads: int | None = None, bias: bool = False,
                 w_bit: int = 4, group_size: int = 128):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias, w_bit, group_size)
        self.qweight.weight_loader = self._qkv_qweight_loader
        self.qzeros.weight_loader = self._qkv_qzeros_loader
        self.scales.weight_loader = self._qkv_scales_loader

    def _qkv_offsets_sizes(self, shard_id: str):
        pack = 32 // self.w_bit
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            return 0, q_size
        elif shard_id == "k":
            return q_size, kv_size
        else:  # "v"
            return q_size + kv_size, kv_size

    def _qkv_qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                             loaded_shard_id: str):
        pack = 32 // self.w_bit
        offset, size = self._qkv_offsets_sizes(loaded_shard_id)
        param.data.narrow(1, offset // pack, size // pack).copy_(
            loaded_weight.narrow(1, self.tp_rank * size // pack, size // pack)
        )

    def _qkv_qzeros_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                            loaded_shard_id: str):
        pack = 32 // self.w_bit
        offset, size = self._qkv_offsets_sizes(loaded_shard_id)
        param.data.narrow(1, offset // pack, size // pack).copy_(
            loaded_weight.narrow(1, self.tp_rank * size // pack, size // pack)
        )

    def _qkv_scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                            loaded_shard_id: str):
        offset, size = self._qkv_offsets_sizes(loaded_shard_id)
        param.data.narrow(1, offset, size).copy_(
            loaded_weight.narrow(1, self.tp_rank * size, size)
        )


class AWQRowParallelLinear(AWQLinearBase):
    """AWQ quantized row-parallel linear (input sharded across ranks)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 w_bit: int = 4, group_size: int = 128):
        tp_size = dist.get_world_size()
        per_rank_in = divide(in_features, tp_size)
        super().__init__(per_rank_in, out_features, bias, w_bit, group_size)
        self.qweight.weight_loader = self._qweight_loader
        self.qzeros.weight_loader = self._qzeros_loader
        self.scales.weight_loader = self._scales_loader

    def _qweight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def _qzeros_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def _scales_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._awq_forward(x, bias=None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        if self.bias is not None and self.tp_rank == 0:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# Factory helpers: return AWQ or standard class based on quant_config
# ---------------------------------------------------------------------------

def _make_column_parallel(in_features, out_features, bias=False):
    qc = get_quant_config()
    if qc and qc.get("quant_type") == "awq":
        return AWQColumnParallelLinear(
            in_features, out_features, bias,
            w_bit=qc.get("w_bit", 4), group_size=qc.get("q_group_size", 128),
        )
    return ColumnParallelLinear(in_features, out_features, bias)


def _make_merged_column_parallel(in_features, output_sizes, bias=False):
    qc = get_quant_config()
    if qc and qc.get("quant_type") == "awq":
        return AWQMergedColumnParallelLinear(
            in_features, output_sizes, bias,
            w_bit=qc.get("w_bit", 4), group_size=qc.get("q_group_size", 128),
        )
    return MergedColumnParallelLinear(in_features, output_sizes, bias)


def _make_qkv_parallel(hidden_size, head_size, total_num_heads,
                       total_num_kv_heads=None, bias=False):
    qc = get_quant_config()
    if qc and qc.get("quant_type") == "awq":
        return AWQQKVParallelLinear(
            hidden_size, head_size, total_num_heads, total_num_kv_heads, bias,
            w_bit=qc.get("w_bit", 4), group_size=qc.get("q_group_size", 128),
        )
    return QKVParallelLinear(hidden_size, head_size, total_num_heads, total_num_kv_heads, bias)


def _make_row_parallel(in_features, out_features, bias=False):
    qc = get_quant_config()
    if qc and qc.get("quant_type") == "awq":
        return AWQRowParallelLinear(
            in_features, out_features, bias,
            w_bit=qc.get("w_bit", 4), group_size=qc.get("q_group_size", 128),
        )
    return RowParallelLinear(in_features, out_features, bias)


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
