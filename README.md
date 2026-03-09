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
* 💾 **Streaming Weight Loading** - Run large models on GPUs with as little as 4 GB VRAM by loading one decoder layer at a time

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

### 4-bit Quantization (AWQ)

To reduce GPU memory usage, load an [AWQ](https://github.com/castor-ai/autoawq)-quantized model checkpoint and pass `quantization="awq"`:

```python
# Install autoawq first: pip install nano-vllm[awq]
llm = LLM("/YOUR/AWQ/MODEL/PATH", quantization="awq", enforce_eager=True)
```

If the model's `config.json` already contains a `quantization_config` block (e.g. it was saved as an AWQ model), the quantization type is detected automatically and the `quantization` argument is not required.

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

---

## Streaming Weight Loading

> **Target use-case:** Running a large Qwen3 model on a single GPU with
> limited VRAM (e.g. 4 GB).  Only one decoder layer's weights occupy GPU
> memory at a time; the rest remain on disk.

Only standard **PyTorch ≥ 2.0** is required — no flash-attn or third-party
streaming library.

### How it works

1. **Layer-wise saving** — The checkpoint is split into individual
   `layer_XXXX.pt` files (one per decoder layer) plus a `non_layer.pt` for
   the token embeddings, final norm and LM head.
2. **Streaming inference** — During a forward pass the engine loads each
   layer's weights, runs the layer, then immediately releases the GPU memory
   before loading the next layer.  The KV cache and the small permanent
   tensors (embeddings, norm, LM head) stay on the GPU throughout.
3. **Async prefetch** — While layer *i* is executing, layer *i+1* is loaded
   from disk into CPU RAM in a background thread, hiding most of the I/O
   latency.

### Step 1 — Save per-layer weights

```bash
python -m nanovllm.utils.save_layers \
    --model_path ~/huggingface/Qwen3-0.6B \
    --output_dir ~/huggingface/Qwen3-0.6B-streaming
```

This produces:

```
~/huggingface/Qwen3-0.6B-streaming/
    layer_0000.pt
    layer_0001.pt
    ...
    non_layer.pt
```

### Step 2 — Run streaming inference

```python
from transformers import AutoConfig, AutoTokenizer
from nanovllm.streaming import StreamingInference

model_path = "~/huggingface/Qwen3-0.6B"
layer_dir  = "~/huggingface/Qwen3-0.6B-streaming"

hf_config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Build the engine (loads only permanent weights to GPU)
model = StreamingInference(
    layer_dir=layer_dir,
    hf_config=hf_config,
    device="cuda",   # or "cpu" for CPU-only
    prefetch=True,   # async prefetch of the next layer (recommended)
)

prompt = "Hello, Nano-vLLM! Tell me about streaming weight loading."
input_ids = tokenizer.encode(prompt)

output_ids = model.generate(
    input_ids=[input_ids],
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output_ids[0]))
```

### API reference

| Parameter | Description |
|-----------|-------------|
| `layer_dir` | Directory of `layer_XXXX.pt` files from `save_layers` |
| `hf_config` | HuggingFace `AutoConfig` for the model |
| `device` | Target device (`'cuda'`, `'cuda:0'`, `'cpu'`, …) |
| `prefetch` | Enable background prefetch of the next layer (default `True`) |
| `dtype` | Override weight dtype (default: `hf_config.torch_dtype`) |

`generate()` supports `temperature`, `top_p`, and `eos_token_id`.
Only `batch_size=1` is supported in streaming mode.

### Memory estimates (Qwen3-0.6B, fp16)

| Component | GPU VRAM |
|-----------|----------|
| Embeddings + norm + LM head | ~50 MB |
| One decoder layer | ~21 MB |
| KV cache (1 K tokens) | ~2 MB |
| **Total peak** | **~75 MB** |

Even with a modest prefetch buffer on CPU, the total VRAM footprint stays well under 1 GB for Qwen3-0.6B.  Larger models scale proportionally — only the per-layer size matters.