"""Triton-based INT4 W4A16 GEMM kernel for AWQ-style quantization.

Design overview
---------------
AWQ checkpoints store weights in a packed INT4 format:

  qweight:  [K, N // pack]  int32   (pack = 8 for 4-bit; 8 nibbles per int32)
  qzeros:   [K // gs, N // pack]  int32   (packed zero-points, same layout)
  scales:   [K // gs, N]   fp16    (per-group scale factors; gs = group_size)

where K = in_features, N = out_features, gs = group_size (typically 128).

Preprocessing (``preprocess_awq_weights``)
-------------------------------------------
After the checkpoint is loaded we perform a one-time preprocessing step that
converts the packed ``qzeros`` to a float16 *scaled_zeros* tensor:

  scaled_zeros[kg, n] = unpack(qzeros[kg, n // 8], n % 8) * scales[kg, n]

This avoids unpacking zero-points in the inner kernel loop and halves the
zero-point arithmetic during inference.

Triton kernel (``_int4_gemm_kernel``)
--------------------------------------
The fused dequantize + GEMM kernel computes:

  out[M, N] = x[M, K] @ dequant(qweight[K, N])

where the dequantization is performed in tiles without ever materialising the
full FP16 weight matrix.  For each (BLOCK_M × BLOCK_N) output tile the kernel
iterates over the K dimension in BLOCK_K steps:

  1. Load x tile              [BLOCK_M, BLOCK_K] fp32 (cast from fp16)
  2. Load packed qweight      [BLOCK_K, BLOCK_N] int32 (8× repeated reads)
  3. Extract nibble per col   qw_int4 = (packed >> bit_shift) & 0xF
  4. Load scales & sz tiles   [BLOCK_K, BLOCK_N] fp32 (group-indexed along K)
  5. Dequantize               w = qw_int4 * scale - scaled_zero
  6. Accumulate               acc += x @ w  (tl.dot in fp32)

Fallback
--------
If Triton or CUDA is not available the module falls back gracefully to a pure
PyTorch dequantize + F.linear path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional Triton import
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _int4_gemm_kernel(
        # Pointers
        x_ptr, qw_ptr, scales_ptr, sz_ptr, out_ptr,
        # Runtime dimensions
        M, N, K,
        N_PACK,    # = N // 8  (number of int32 columns in qweight)
        n_groups,  # = K // group_size
        # Strides (row strides for row-major contiguous tensors)
        stride_xm,   # x.stride(0)   = K
        stride_om,   # out.stride(0) = N
        # Quantization hyper-param (compile-time constant for unrolling)
        group_size: tl.constexpr,
        # Tile sizes (compile-time constants)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused INT4 dequantize + GEMM.

        Computes  out = x @ dequant(qweight)  in tiles.

        qweight is stored in AWQ packed format [K, N//8] int32 where each
        int32 holds 8 nibbles (int4 values) for 8 consecutive output columns.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # For each output column n:
        #   packed column index  = n // 8
        #   nibble shift (bits)  = (n % 8) * 4  → values 0, 4, 8, …, 28
        pack_col  = offs_n // 8         # [BLOCK_N]
        bit_shift = (offs_n % 8) * 4    # [BLOCK_N]

        # Accumulator in fp32 for numerical stability
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            # Group index along K for scale/zero lookup
            group_idx = offs_k // group_size   # [BLOCK_K]

            # ---------------------------------------------------------------
            # Load x tile [BLOCK_M, BLOCK_K]
            # ---------------------------------------------------------------
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)

            # ---------------------------------------------------------------
            # Load packed qweight → [BLOCK_K, BLOCK_N]
            #
            # For column n, we read qweight[k, n // 8].  Multiple columns map
            # to the same packed int32 (8 consecutive n's share one int32), so
            # some reads are redundant but remain correct and cache-friendly.
            # ---------------------------------------------------------------
            qw = tl.load(
                qw_ptr + offs_k[:, None] * N_PACK + pack_col[None, :],
                mask=(offs_k[:, None] < K) & (pack_col[None, :] < N_PACK),
                other=0,
            )
            # Extract 4-bit nibble for each column
            qw_int4 = (qw >> bit_shift[None, :]) & 0xF   # [BLOCK_K, BLOCK_N]

            # ---------------------------------------------------------------
            # Load scales and pre-computed scaled_zeros [BLOCK_K, BLOCK_N]
            # Both are indexed by (group_idx, n).
            # ---------------------------------------------------------------
            gs_mask = (group_idx[:, None] < n_groups) & (offs_n[None, :] < N)

            scales = tl.load(
                scales_ptr + group_idx[:, None] * N + offs_n[None, :],
                mask=gs_mask,
                other=0.0,
            ).to(tl.float32)

            sz = tl.load(
                sz_ptr + group_idx[:, None] * N + offs_n[None, :],
                mask=gs_mask,
                other=0.0,
            ).to(tl.float32)

            # ---------------------------------------------------------------
            # Dequantize: w = qw_int4 * scale - scaled_zero
            # ---------------------------------------------------------------
            w = qw_int4.to(tl.float32) * scales - sz   # [BLOCK_K, BLOCK_N]

            # ---------------------------------------------------------------
            # Accumulate: acc += x @ w
            # ---------------------------------------------------------------
            acc = tl.dot(x, w, acc, out_dtype=tl.float32)

        # -------------------------------------------------------------------
        # Write output [BLOCK_M, BLOCK_N] cast to fp16
        # -------------------------------------------------------------------
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            acc.to(tl.float16),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def preprocess_awq_weights(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    w_bit: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform AWQ checkpoint layout into kernel-friendly layout.

    Unpacks ``qzeros`` from the packed int32 format used by AWQ checkpoints
    and pre-multiplies them by their corresponding scales to produce
    ``scaled_zeros``.  This preprocessing step is performed once after weight
    loading and avoids redundant zero-point unpacking during every forward
    pass.

    Args:
        qweight: ``[K, N // pack]`` int32 packed weight tensor.
        qzeros:  ``[K // group_size, N // pack]`` int32 packed zero-points.
        scales:  ``[K // group_size, N]`` fp16 scale factors.
        w_bit:   Weight bit-width (default 4).

    Returns:
        A tuple ``(qweight, scales, scaled_zeros)`` where:

        * ``qweight`` – contiguous int32 tensor, same shape as input.
        * ``scales``  – contiguous fp16 tensor, same shape as input.
        * ``scaled_zeros`` – ``[K // group_size, N]`` fp16 tensor with
          ``scaled_zeros[kg, n] = unpacked_zero[kg, n] * scales[kg, n]``.
    """
    pack = 32 // w_bit           # 8 for 4-bit
    n_groups, n_packed = qzeros.shape
    N = n_packed * pack

    # Shifts for extracting each nibble: [pack] = [0, 4, 8, …, 28]
    shifts = torch.arange(0, 32, w_bit, device=qzeros.device, dtype=torch.int32)

    # Unpack zeros: [n_groups, n_packed] → [n_groups, n_packed, pack] → [n_groups, N]
    izeros = torch.bitwise_and(
        qzeros.unsqueeze(-1) >> shifts.view(1, 1, pack),
        (2 ** w_bit - 1),
    )  # [n_groups, n_packed, pack]
    izeros = izeros.reshape(n_groups, N).to(scales.dtype)   # [n_groups, N]

    # Pre-compute the zero-offset term used in dequantization
    scaled_zeros = izeros * scales   # [n_groups, N]

    return qweight.contiguous(), scales.contiguous(), scaled_zeros.contiguous()


# ---------------------------------------------------------------------------
# Public Python interface
# ---------------------------------------------------------------------------

def int4_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    out_features: int | None = None,
) -> torch.Tensor:
    """Fused INT4 dequantize + matrix multiplication (W4A16).

    Computes ``out = x @ dequant(qweight)`` without ever materialising the
    full FP16 weight matrix.  Dequantisation is performed in tiles inside the
    Triton kernel.

    Args:
        x:            Activation tensor ``[..., K]`` in fp16 or bf16.
        qweight:      Packed INT4 weight tensor ``[K, N // 8]`` int32 (AWQ
                      layout: 8 nibbles packed per int32 along the N dim).
        scales:       Scale tensor ``[K // group_size, N]`` fp16.
        scaled_zeros: Pre-computed scaled zero-points ``[K // group_size, N]``
                      fp16 (output of :func:`preprocess_awq_weights`).
        out_features: Expected output width N; inferred from qweight if None.

    Returns:
        Output tensor ``[..., N]`` in fp16.

    Raises:
        RuntimeError: If CUDA is not available.
        ImportError:  If Triton is not installed.
    """
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "Triton is required for int4_gemm. "
            "Install it with: pip install triton"
        )
    if not x.is_cuda:
        raise RuntimeError("int4_gemm requires CUDA tensors.")

    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    M, K = x.shape
    K2, N_PACK = qweight.shape
    N = N_PACK * 8
    if out_features is not None and out_features != N:
        raise ValueError(
            f"out_features={out_features} doesn't match qweight columns {N}"
        )
    if K != K2:
        raise ValueError(
            f"Activation K={K} doesn't match qweight K={K2}"
        )

    n_groups = scales.shape[0]
    group_size = K // n_groups

    # Ensure contiguous fp16 inputs
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    scaled_zeros = scaled_zeros.contiguous()

    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    # Block sizes: BLOCK_K must be a power of 2 and ≥ 16.
    # Using 64 works well for group_size = 128 (two BLOCK_K steps per group).
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = min(64, group_size)
    # Ensure BLOCK_K is at least 16 (required by tl.dot)
    if BLOCK_K < 16:
        BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int4_gemm_kernel[grid](
        x, qweight, scales, scaled_zeros, out,
        M, N, K,
        N_PACK, n_groups,
        x.stride(0),    # stride_xm
        out.stride(0),  # stride_om
        group_size=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.reshape(orig_shape[:-1] + (N,))


def int4_gemm_dequantize_fallback(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch fallback for INT4 W4A16 matmul (for CPU / non-CUDA paths).

    Dequantizes the full weight matrix to fp16 in one shot and then calls
    ``torch.nn.functional.linear``.  Use only when the Triton kernel is not
    available, as this materialises the dense fp16 weight.

    Args:
        x:            ``[..., K]`` fp16 / bf16 activations.
        qweight:      ``[K, N // 8]`` int32 packed weights.
        scales:       ``[K // group_size, N]`` fp16 scales.
        scaled_zeros: ``[K // group_size, N]`` fp16 pre-computed scaled zeros.

    Returns:
        ``[..., N]`` tensor in the same dtype as ``x``.
    """
    K, N_PACK = qweight.shape
    N = N_PACK * 8
    n_groups = scales.shape[0]
    group_size = K // n_groups
    pack = 8  # 32 // 4

    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)

    # Unpack: [K, N//8, 8] → [K, N]
    iweight = torch.bitwise_and(
        qweight.unsqueeze(-1) >> shifts.view(1, 1, pack),
        0xF,
    ).reshape(K, N).to(scales.dtype)   # [K, N]

    # Broadcast scales and scaled_zeros from [K//gs, N] to [K, N]
    iweight = iweight * scales.repeat_interleave(group_size, dim=0)
    iweight = iweight - scaled_zeros.repeat_interleave(group_size, dim=0)

    # F.linear expects weight [N, K]
    return F.linear(x, iweight.T.contiguous())
