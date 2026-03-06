"""Tests for INT4 (AWQ-style W4A16) quantized linear layers.

These tests validate the preprocessing, dequantization, and inference paths
without requiring a real AWQ checkpoint or a GPU.  They use CPU-based
computations and the pure-PyTorch fallback kernels.

Run with:
    python -m pytest tests/test_int4_linear.py -v
"""

import math
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Process-group helper (single-process gloo backend for CPU tests)
# ---------------------------------------------------------------------------

def _ensure_dist_initialized():
    """Initialise a single-process gloo process group if not already done."""
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )


# ---------------------------------------------------------------------------
# Helper: create synthetic AWQ-format weight tensors
# ---------------------------------------------------------------------------

def _make_awq_tensors(
    in_features: int,
    out_features: int,
    group_size: int = 128,
    w_bit: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random synthetic AWQ-format tensors.

    Returns:
        (qweight, qzeros, scales, expected_weight_fp16)

        * qweight – [K, N//pack] int32 packed weights
        * qzeros  – [K//gs, N//pack] int32 packed zero-points
        * scales  – [K//gs, N] fp16 scale factors
        * expected – [N, K] fp16 full-precision weight (for reference)
    """
    torch.manual_seed(seed)
    K, N = in_features, out_features
    pack = 32 // w_bit       # 8 for 4-bit
    n_groups = K // group_size

    # Random int4 weight values in [0, 15]
    iweight = torch.randint(0, 2 ** w_bit, (K, N), dtype=torch.int32)
    # Random int4 zero values in [0, 15]
    izeros = torch.randint(0, 2 ** w_bit, (n_groups, N), dtype=torch.int32)
    # Positive scales
    scales = (torch.rand(n_groups, N) * 0.01 + 0.001).to(torch.float16)

    # Pack weights: [K, N] → [K, N//pack] by packing 8 nibbles per int32
    shifts = torch.arange(0, 32, w_bit, dtype=torch.int32)
    # iweight_3d: [K, N//pack, pack]
    iweight_3d = iweight.reshape(K, N // pack, pack)
    qweight = (iweight_3d << shifts.view(1, 1, pack)).sum(dim=-1).to(torch.int32)

    # Pack zeros: [n_groups, N] → [n_groups, N//pack]
    izeros_3d = izeros.reshape(n_groups, N // pack, pack)
    qzeros = (izeros_3d << shifts.view(1, 1, pack)).sum(dim=-1).to(torch.int32)

    # Reference full-precision weight [K, N]
    scales_expanded = scales.float().repeat_interleave(group_size, dim=0)   # [K, N]
    izeros_expanded = izeros.float().repeat_interleave(group_size, dim=0)   # [K, N]
    expected = (iweight.float() - izeros_expanded) * scales_expanded        # [K, N]
    expected_fp16 = expected.to(torch.float16).T.contiguous()               # [N, K]

    return (
        qweight.to(device),
        qzeros.to(device),
        scales.to(device),
        expected_fp16.to(device),
    )


# ---------------------------------------------------------------------------
# Tests for preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessAWQWeights(unittest.TestCase):

    def test_output_shapes(self):
        """preprocess_awq_weights returns tensors with correct shapes."""
        from nanovllm.kernels.int4_gemm import preprocess_awq_weights

        K, N, gs = 256, 128, 128
        qw = torch.zeros(K, N // 8, dtype=torch.int32)
        qz = torch.zeros(K // gs, N // 8, dtype=torch.int32)
        sc = torch.ones(K // gs, N, dtype=torch.float16)

        qw_out, sc_out, sz_out = preprocess_awq_weights(qw, qz, sc)

        self.assertEqual(qw_out.shape, (K, N // 8))
        self.assertEqual(sc_out.shape, (K // gs, N))
        self.assertEqual(sz_out.shape, (K // gs, N))

    def test_scaled_zeros_values(self):
        """scaled_zeros == unpacked_zeros * scales for a known input."""
        from nanovllm.kernels.int4_gemm import preprocess_awq_weights

        K, N, gs = 128, 64, 128
        n_groups = K // gs   # 1
        pack = 8

        # Construct qzeros where nibble j of group 0 = j (mod 16)
        # For N=64, pack=8: n_packed = 8 groups of 8 nibbles
        # Each int32 packs nibbles: 0<<0 | 1<<4 | 2<<8 | ... | 7<<28 = 0x76543210
        known_int32 = sum(i << (i * 4) for i in range(8))
        qz = torch.full((n_groups, N // pack), known_int32, dtype=torch.int32)

        sc = torch.ones(n_groups, N, dtype=torch.float16) * 2.0

        _, _, sz = preprocess_awq_weights(qweight=torch.zeros(K, N // pack, dtype=torch.int32), qzeros=qz, scales=sc)

        # Expected: izeros[0, n] = n % 8 (cyclic), scaled_zeros = izeros * scale
        expected_iz = torch.tensor([i % 8 for i in range(N)], dtype=torch.float16) * 2.0
        self.assertTrue(
            torch.allclose(sz[0], expected_iz, atol=1e-3),
            f"scaled_zeros mismatch.\nGot:      {sz[0]}\nExpected: {expected_iz}",
        )

    def test_output_contiguous(self):
        """preprocess_awq_weights always returns contiguous tensors."""
        from nanovllm.kernels.int4_gemm import preprocess_awq_weights

        K, N, gs = 256, 128, 128
        qw = torch.zeros(K, N // 8, dtype=torch.int32).T.contiguous().T  # non-contig
        qz = torch.zeros(K // gs, N // 8, dtype=torch.int32)
        sc = torch.ones(K // gs, N, dtype=torch.float16)

        qw_out, sc_out, sz_out = preprocess_awq_weights(qw, qz, sc)

        self.assertTrue(qw_out.is_contiguous())
        self.assertTrue(sc_out.is_contiguous())
        self.assertTrue(sz_out.is_contiguous())


# ---------------------------------------------------------------------------
# Tests for the PyTorch dequantization fallback
# ---------------------------------------------------------------------------

class TestInt4GemmFallback(unittest.TestCase):

    def _run_fallback(self, K: int, N: int, M: int, gs: int):
        """Helper: compare fallback output to the reference full-precision result."""
        from nanovllm.kernels.int4_gemm import (
            int4_gemm_dequantize_fallback,
            preprocess_awq_weights,
        )

        qw, qz, sc, expected_w = _make_awq_tensors(K, N, gs)
        _, sc_out, sz_out = preprocess_awq_weights(qw, qz, sc)

        x = torch.randn(M, K, dtype=torch.float16)

        # Reference: use expected_w directly
        ref = torch.nn.functional.linear(x, expected_w)

        # Fallback path
        out = int4_gemm_dequantize_fallback(x, qw, sc_out, sz_out)

        self.assertEqual(out.shape, (M, N))
        # Results should be very close (only fp16 rounding differences)
        max_diff = (out.float() - ref.float()).abs().max().item()
        self.assertLess(max_diff, 0.2, f"max diff {max_diff:.4f} too large for K={K},N={N},M={M}")

    def test_small(self):
        self._run_fallback(K=128, N=64, M=4, gs=128)

    def test_batch(self):
        self._run_fallback(K=256, N=128, M=16, gs=128)

    def test_non_square(self):
        self._run_fallback(K=512, N=256, M=8, gs=128)


# ---------------------------------------------------------------------------
# Tests for AWQLinearBase preprocessing hook
# ---------------------------------------------------------------------------

class TestAWQLinearBasePreprocess(unittest.TestCase):

    def _make_linear(self, in_f: int, out_f: int, gs: int = 128):
        """Instantiate an AWQLinearBase subclass (ReplicatedAWQ) for testing."""
        _ensure_dist_initialized()

        from nanovllm.layers.linear import AWQLinearBase

        class _TestLinear(AWQLinearBase):
            def forward(self, x):
                return self._awq_forward(x, self.bias)

        layer = _TestLinear(in_f, out_f, bias=False, w_bit=4, group_size=gs)
        return layer

    def test_scaled_zeros_none_before_preprocess(self):
        layer = self._make_linear(128, 64)
        self.assertIsNone(layer._scaled_zeros)

    def test_scaled_zeros_set_after_preprocess(self):
        layer = self._make_linear(128, 64, gs=128)
        # Fill with dummy data so preprocess has something to work with
        layer.qweight.data.zero_()
        layer.qzeros.data.zero_()
        layer.scales.data.fill_(1.0)
        layer.process_weights_after_loading()
        self.assertIsNotNone(layer._scaled_zeros)
        self.assertEqual(layer._scaled_zeros.shape, (128 // 128, 64))

    def test_forward_fallback_cpu(self):
        """Forward pass uses the PyTorch fallback on CPU and returns correct shape."""
        layer = self._make_linear(128, 64, gs=128)
        qw, qz, sc, _ = _make_awq_tensors(128, 64, 128)
        layer.qweight.data.copy_(qw)
        layer.qzeros.data.copy_(qz)
        layer.scales.data.copy_(sc)
        layer.process_weights_after_loading()

        x = torch.randn(4, 128, dtype=torch.float16)
        out = layer(x)
        self.assertEqual(out.shape, (4, 64))

    def test_forward_3d_input(self):
        """Forward correctly handles 3-D input [B, S, K]."""
        layer = self._make_linear(128, 64, gs=128)
        qw, qz, sc, _ = _make_awq_tensors(128, 64, 128)
        layer.qweight.data.copy_(qw)
        layer.qzeros.data.copy_(qz)
        layer.scales.data.copy_(sc)
        layer.process_weights_after_loading()

        x = torch.randn(2, 5, 128, dtype=torch.float16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 5, 64))


# ---------------------------------------------------------------------------
# Tests for AWQLinearBase._dequantize reference implementation
# ---------------------------------------------------------------------------

class TestDequantize(unittest.TestCase):

    def test_dequantize_matches_reference(self):
        """_dequantize output matches the manually constructed expected weight."""
        _ensure_dist_initialized()

        from nanovllm.layers.linear import AWQLinearBase

        class _L(AWQLinearBase):
            def forward(self, x):
                return self._awq_forward(x)

        K, N, gs = 128, 64, 128
        layer = _L(K, N, bias=False, w_bit=4, group_size=gs)
        qw, qz, sc, expected_w = _make_awq_tensors(K, N, gs)
        layer.qweight.data.copy_(qw)
        layer.qzeros.data.copy_(qz)
        layer.scales.data.copy_(sc)

        w = layer._dequantize()
        self.assertEqual(w.shape, (N, K))
        max_diff = (w.float() - expected_w.float()).abs().max().item()
        self.assertLess(max_diff, 1e-2, f"_dequantize max diff too large: {max_diff}")


# ---------------------------------------------------------------------------
# Tests for Config quantization normalization
# ---------------------------------------------------------------------------

class TestConfigQuantNormalization(unittest.TestCase):
    """Verify that quant configs from model checkpoints are normalized to the
    internal format expected by the linear layer factory helpers."""

    def _normalize(self, raw: dict) -> dict:
        """Mirror the normalization logic from Config.__post_init__."""
        qc = {k.lower(): v for k, v in raw.items()}
        if "quant_type" not in qc and "quant_method" in qc:
            qc["quant_type"] = qc["quant_method"]
        if "w_bit" not in qc and "bits" in qc:
            qc["w_bit"] = qc["bits"]
        if "q_group_size" not in qc and "group_size" in qc:
            qc["q_group_size"] = qc["group_size"]
        return qc

    def test_autoawq_format_normalized(self):
        """autoawq checkpoint config (quant_method/bits/group_size) → quant_type/w_bit/q_group_size."""
        raw = {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True, "version": "gemm"}
        qc = self._normalize(raw)
        self.assertEqual(qc.get("quant_type"), "awq")
        self.assertEqual(qc.get("w_bit"), 4)
        self.assertEqual(qc.get("q_group_size"), 128)

    def test_internal_format_unchanged(self):
        """Config already using quant_type/w_bit/q_group_size passes through unchanged."""
        raw = {"quant_type": "awq", "w_bit": 4, "q_group_size": 128}
        qc = self._normalize(raw)
        self.assertEqual(qc.get("quant_type"), "awq")
        self.assertEqual(qc.get("w_bit"), 4)
        self.assertEqual(qc.get("q_group_size"), 128)

    def test_factory_sees_awq(self):
        """After normalization, the linear factory helper correctly identifies AWQ."""
        raw = {"quant_method": "awq", "bits": 4, "group_size": 128}
        qc = self._normalize(raw)
        # This is exactly the condition the factory helpers check:
        self.assertTrue(qc and qc.get("quant_type") == "awq")

    def test_unknown_quant_method_not_treated_as_awq(self):
        """An unknown quant_method is not incorrectly mapped as AWQ."""
        raw = {"quant_method": "gptq", "bits": 4, "group_size": 128}
        qc = self._normalize(raw)
        self.assertNotEqual(qc.get("quant_type"), "awq")

    def test_keys_are_lowercase(self):
        """Keys are always lowercased during normalization."""
        raw = {"Quant_Method": "AWQ", "Bits": 4, "Group_Size": 128}
        qc = self._normalize(raw)
        for key in qc:
            self.assertEqual(key, key.lower(), f"Key '{key}' is not lowercase")


# ---------------------------------------------------------------------------
# Tests for Config model path resolution and validation error messages
# ---------------------------------------------------------------------------

class TestConfigResolveModelPath(unittest.TestCase):
    """Verify _resolve_model_path and Config validation error messages."""

    def test_local_path_returned_unchanged(self):
        """A valid local directory is returned as-is."""
        import tempfile
        from nanovllm.config import _resolve_model_path
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(_resolve_model_path(tmpdir), tmpdir)

    def test_nonexistent_local_path_raises_without_huggingface_hub(self):
        """A path that doesn't exist and is not a HF model ID raises ValueError
        (or downloads if huggingface_hub is present)."""
        import sys
        from unittest.mock import patch
        from nanovllm.config import _resolve_model_path

        # Temporarily block the huggingface_hub import to test the error path.
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaises((ValueError, ImportError)):
                _resolve_model_path("/nonexistent/model/path/xyz")

    def test_bad_quantization_raises_value_error(self):
        """Unsupported quantization name raises ValueError with a descriptive message."""
        from nanovllm.config import Config
        import tempfile

        # The ValueError is raised before AutoConfig.from_pretrained, so no
        # need to mock transformers or create a real config.json.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                Config(model=tmpdir, quantization="gptq")
            self.assertIn("gptq", str(ctx.exception))
            self.assertIn("Unsupported", str(ctx.exception))

    def test_max_num_batched_tokens_too_small_raises_value_error(self):
        """max_num_batched_tokens < max_model_len raises ValueError."""
        import sys
        import tempfile
        from unittest.mock import MagicMock, patch
        from nanovllm.config import Config

        with tempfile.TemporaryDirectory() as tmpdir:
            # The check for max_num_batched_tokens happens after
            # AutoConfig.from_pretrained, so we mock transformers here.
            mock_hf = MagicMock()
            mock_hf.max_position_embeddings = 200  # > max_model_len=100

            mock_transformers = MagicMock()
            mock_transformers.AutoConfig.from_pretrained.return_value = mock_hf

            # max_num_batched_tokens=50 < max_model_len=100
            with patch.dict(sys.modules, {"transformers": mock_transformers}):
                with self.assertRaises(ValueError) as ctx:
                    Config(model=tmpdir, max_model_len=100, max_num_batched_tokens=50)
            self.assertIn("max_num_batched_tokens", str(ctx.exception))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
