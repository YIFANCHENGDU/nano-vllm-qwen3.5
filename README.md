<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

### INT4 Quantized Inference (AWQ / W4A16)

Nano-vLLM includes a built-in Triton-based INT4 W4A16 inference path that
requires **no external quantization library**.  Activations are kept in FP16;
only weights are stored in packed INT4 format (8 nibbles per int32, AWQ layout).

#### How it works

```
Checkpoint (qweight / qzeros / scales)
        │
        ▼  load_model()
AWQ linear parameters on GPU
        │
        ▼  process_weights_after_loading()   ← one-time preprocessing
scaled_zeros = unpack(qzeros) * scales       ← [K//gs, N] fp16
        │
        ▼  forward pass
Triton kernel: fused dequantize + GEMM
  for each (BLOCK_M × BLOCK_N) tile:
    w = qw_int4 * scale − scaled_zero   (in tiles, no dense FP16 alloc)
    acc += x @ w
```

#### Kernel-priority fallback chain

| Priority | Condition | Kernel |
|----------|-----------|--------|
| 1 | CUDA available + preprocessing done | Triton INT4 W4A16 |
| 2 | `autoawq` installed | autoawq WQLinearMMFunction |
| 3 | always | pure-PyTorch dequantize + F.linear |

#### Usage

```python
from nanovllm import LLM, SamplingParams

# Option A – model checkpoint already contains quantization_config (AWQ)
llm = LLM("/YOUR/AWQ/MODEL/PATH", enforce_eager=True)

# Option B – pass the quantization type explicitly
llm = LLM("/YOUR/AWQ/MODEL/PATH", quantization="awq", enforce_eager=True)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

The AWQ model can be obtained with the
[autoawq](https://github.com/castor-ai/autoawq) quantisation toolkit:

```bash
# Download a pre-quantised AWQ model from Hugging Face, e.g.:
huggingface-cli download --resume-download Qwen/Qwen3-0.6B-AWQ \
  --local-dir ~/huggingface/Qwen3-0.6B-AWQ/ \
  --local-dir-use-symlinks False
```

If you want to use the optional `autoawq` GEMM kernel (fallback path 2) for
benchmarking or compatibility, install it alongside the package:

```bash
pip install nano-vllm[awq]   # pulls in autoawq>=0.2.0
```

#### Memory savings

| Precision | Qwen3-0.6B VRAM |
|-----------|-----------------|
| BF16 (default) | ~1.2 GB |
| INT4 AWQ       | ~0.4 GB |

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)