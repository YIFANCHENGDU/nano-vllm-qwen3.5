"""pytest configuration for nanovllm unit tests.

The ``nanovllm`` package ``__init__.py`` imports ``LLM`` and ``SamplingParams``
which pull in heavy runtime dependencies (transformers, flash-attn, …) that
are not needed for unit-testing the quantization kernels and linear layers.

This conftest inserts a lightweight stub for ``nanovllm`` into ``sys.modules``
*before* any test module is collected.  Sub-packages (``nanovllm.kernels``,
``nanovllm.layers``, etc.) are still imported from their real source files;
only the top-level ``__init__.py`` is bypassed.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Lightweight stub for the top-level nanovllm package
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent

if "nanovllm" not in sys.modules:
    # Create a minimal package object that resolves to the real filesystem
    # path but does NOT execute nanovllm/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "nanovllm",
        _ROOT / "nanovllm" / "__init__.py",
        submodule_search_locations=[str(_ROOT / "nanovllm")],
    )
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [str(_ROOT / "nanovllm")]
    pkg.__package__ = "nanovllm"
    pkg.__spec__ = spec
    # Register the stub *without* executing __init__.py
    sys.modules["nanovllm"] = pkg
