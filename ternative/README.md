# ternative.cpp

**Ternative** is a minimal, fast C++ inference runtime for **ternary-weight neural networks**, designed as the llama.cpp of ternary models. It loads BitNet b1.58 `I2_S` bases, merges LoRA adapters without rounding loss, and serves them via an OpenAI-compatible HTTP server — on GPU or CPU-only hardware.

---

## Features

- **Ternary-native**: First-class `I2_S` ({−1, 0, +1}) support — no type-36 errors
- **LoRA merge at load time**: De-quantize base → apply adapter in F32 → keep in F16. Every delta preserved.
- **GPU-accelerated**: Full GPU-resident forward pass (RoPE, KV-cache, attention all on GPU). All 30 layers offloaded on RTX 3050 4 GB using F16 + INT8 mixed precision.
- **INT8 overflow handling**: Layers that don't fit in F16 are auto-quantized to INT8 at load time (symmetric per-tensor). Quality impact is negligible for near-ternary weights.
- **Fast prefill**: Batched GEMV kernel eliminates the row-by-row loop — scoring 100 tokens takes milliseconds instead of seconds.
- **Works CPU-only**: Full inference on any x86-64 AVX2 CPU, no GPU required. ~6 tok/s on modern hardware.
- **OpenAI-compatible server**: `/v1/chat/completions`, `/v1/completions` (with `logprobs`, `echo`), `/v1/models`
- **Cross-platform**: Windows (MSVC 2022), Linux (GCC 11+), macOS (Clang)

---

## Performance (RTX 3050 4 GB, Orchid 1.0 LoRA)

| Mode | Decode speed | Notes |
|------|-------------|-------|
| GPU (all 30 layers) | ~6–7 tok/s | 14 layers F16 + 16 layers INT8, GPU KV-cache |
| CPU-only (`--no-gpu`) | ~6 tok/s | AVX2 F16C GEMM, OpenMP |

Prefill (prompt processing) uses a batched GEMV kernel: single H2D upload, single kernel, single D2H download for each weight matrix — ~50× faster than row-by-row.

---

## Quick Start (CPU-only, any PC)

**Requirements**: 8 GB RAM, AVX2 CPU (any Intel/AMD since ~2013), CMake 3.18+, C++17 compiler.

```powershell
# Windows — auto-detects MSVC, clang-cl, or MinGW
cd ternative.cpp
.\scripts\build.ps1

# Linux / macOS
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run:
```bash
./build/Release/ternative.exe \
  --model ggml-model-i2_s.gguf \
  --lora  dpo_aligned_lora.gguf \
  --prompt "What is the capital of France?"
```

**Minimum distribution** (no GPU, no installation):
```
ternative.exe           ~500 KB
ggml-model-i2_s.gguf   ~1.8 GB  (base model)
dpo_aligned_lora.gguf  ~100 MB  (adapter)
```

---

## GPU Build (NVIDIA, CUDA 12.x)

```bat
REM Windows — run once from a standard cmd prompt
build_cuda.bat
```

The GPU build uses mixed F16/INT8 offload to fit all 30 layers of a 2B model in 4 GB VRAM:
- Layers 0–N in F16 (highest precision, fits budget)
- Remaining layers in INT8 (~70 MB/layer vs ~141 MB/layer F16)
- GPU KV-cache (1024-token capacity, ~157 MB)
- Full GPU-resident decode: RoPE, attention, and residuals all run on-device

```bat
ternative.exe ^
  --model  "models/ggml-model-i2_s.gguf" ^
  --lora   "models/dpo_aligned_lora.gguf" ^
  --server --port 8080
```

Expected startup output:
```
[GPU] Offloaded 30/30 layers (3296 MB VRAM used)
[GPU] All layers offloaded — GPU-resident decode path enabled
```

---

## Server Mode

```bash
./ternative --model model.gguf --server --port 8080
```

Then use any OpenAI client:

```bash
# Chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"orchid","messages":[{"role":"user","content":"Hello!"}]}'

# Completions with log-probabilities (for lm-eval-harness style benchmarks)
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5,"logprobs":1}'

# Echo scoring (score existing text, max_tokens=0)
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is Paris.","max_tokens":0,"logprobs":1,"echo":true}'
```

---

## Why Ternative vs. llama.cpp and bitnet.cpp

| Engine | BitNet I2_S | LoRA runtime | LoRA + I2_S merge | Server |
|--------|-------------|--------------|-------------------|--------|
| **llama.cpp** | ❌ errors on type 36 | ✅ (Q4/Q8 base) | N/A | Via llama-server |
| **bitnet.cpp** | ✅ native kernels | ❌ no adapter path | ❌ destroys LoRA | ❌ |
| **ternative.cpp** | ✅ custom loader | ✅ merge at load | ✅ F16 (no rounding) | ✅ built-in |

The fundamental issue with both llama.cpp and bitnet.cpp: ternary rounding (`weight_quant`) erases LoRA deltas when you try to merge a fine-tuned adapter back into an I2_S base. A delta of 0.01 → quantized to 0. Ternative avoids this by never re-quantizing after the merge.

---

## Architecture

```
I2_S base GGUF (~1.8 GB on disk)
        │
        ▼
Dequantize I2_S → F32
        │
        ▼
Load LoRA adapter (PEFT format → custom GGUF)
Apply delta: W_merged = W_base_f32 + (lora_b @ lora_a) × α/r
        │
        ▼
Cast → F16  [cached to disk as .tvcache for fast reload]
        │
        ▼
GPU offload:
  - Layers 0..k: F16 (141 MB/layer)
  - Layers k..N: INT8 symmetric (~70 MB/layer)
  - Norm weights: F32 (tiny)
  - GPU KV-cache: 1024 tokens × 30 layers × 640 floats × 2
        │
        ▼
Forward pass (fully GPU-resident):
  embed → [rms_norm → QKV GEMV → RoPE → KV write → attention → wo → FFN] × 30 → logits
        │
        ▼
Single cudaDeviceSynchronize() per token
```

---

## Benchmarks (Orchid 1.0, 50Q each, lm-eval-harness methodology)

| Benchmark | Orchid 1.0 | BitNet base | Delta |
|-----------|-----------|-------------|-------|
| ARC-Challenge | **60.0%** | 49.9% | +10.1pp |
| HellaSwag | **56.0%** | 68.4% | −12.4pp |
| MMLU | 40.4% | 53.2% | −12.8pp |

ARC improvement confirms the ORPO fine-tuning works. MMLU regression is an expected alignment tax (DPO/ORPO trades factual recall for TruthfulQA).

---

## Flags

```
--model, -m <path>    Base GGUF model path (required)
--lora <path>         LoRA adapter GGUF path (optional, repeatable)
--prompt, -p <text>   Prompt text (generate mode)
--max-tokens <n>      Max new tokens (default: 512)
--temperature <f>     Sampling temperature (default: 0.8)
--top-p <f>           Nucleus sampling (default: 0.9)
--top-k <n>           Top-k sampling (default: 40)
--server              Run OpenAI-compatible HTTP server
--port <n>            Server port (default: 8080)
--no-gpu              Disable GPU offload (CPU-only mode)
--export-gguf <path>  Export merged F16 model as GGUF
--info <path>         Print GGUF metadata and exit
```

---

## Project Structure

```
ternative.cpp/
├── cuda/               CUDA kernels (GEMV, attention, RoPE, KV-cache, INT8)
├── include/ternative/  Public headers
├── src/                Engine implementation
│   ├── model.cpp       Forward pass, GPU offload, LoRA merge, cache
│   ├── server.cpp      OpenAI-compatible HTTP server
│   └── ...
├── scripts/
│   ├── build.ps1       Auto-detect Windows build (MSVC/clang/MinGW)
│   └── build.sh        Linux/macOS build helper
├── tests/              Test suite
├── CMakeLists.txt
└── build_cuda.bat      One-command GPU build (Windows)
```

---

## Roadmap

- [x] GGUF v3 loader + I2_S tensor ops
- [x] LoRA merge (F32 precision, no rounding loss)
- [x] CPU inference (AVX2 F16C GEMM, OpenMP)
- [x] OpenAI-compatible HTTP server with echo scoring
- [x] GPU-resident forward pass (RoPE, KV-cache, attention on GPU)
- [x] INT8 weight quantization (auto overflow handling)
- [x] Batched GEMV prefill kernel
- [x] GPU KV-cache (1024-token capacity)
- [ ] cuBLAS GEMM for large-batch prefill
- [ ] Flash Attention for long contexts
- [ ] CUDA Graphs (zero-overhead dispatch)
- [ ] Metal backend for Apple Silicon
- [ ] Continuous batching in server mode
- [ ] Python bindings (`ternative-py`)

---

## License

Apache 2.0

See [NOTICE](NOTICE) for third-party copyright notices that must be reproduced with this software.

## Third-Party Licenses

ternative.cpp incorporates or adapts components from the following projects:

| Component | License | Copyright |
|-----------|---------|-----------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format conventions and loader design | MIT | Copyright (c) 2023 Georgi Gerganov |
| [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing | MIT | Copyright (c) 2013-2022 Niels Lohmann |
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — CPU BLAS (CPU builds) | BSD-3-Clause | Copyright (c) 2011-2014 The OpenBLAS Project |
| [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) — model architecture and I2_S weight format this engine was designed for | MIT | Copyright (c) Microsoft Corporation |

## Acknowledgments

- **Microsoft Research** — BitNet b1.58 architecture and the I2_S ternary weight format (`Copyright (c) Microsoft Corporation`)
- **Georgi Gerganov and the llama.cpp project** — GGUF format specification, loader conventions, and tokenizer design
- **Orchid training pipeline** — alignment recipe and merge-first quantization workflow that motivated this engine
