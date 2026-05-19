# 🌸 Orchid 1.0

**First competitive LLM trained and aligned in Colombia** — 2B ternary-weight model built on [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), aligned with ORPO on a single RTX 3050 laptop (4 GB VRAM).

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-MicheRomChis%2Forchid--1.0-yellow)](https://huggingface.co/MicheRomChis/orchid-1.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michelangeloromerochisco/orchid-1.0/blob/main/orchid_colab.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Technical%20Report-red)](https://huggingface.co/MicheRomChis/orchid-1.0/blob/main/orchid-1-0-technical-paper.pdf)

---

## What is Orchid?

Orchid 1.0 is a 2B-parameter language model with **ternary weights** ({−1, 0, +1} at 1.58 bits/weight). It was fine-tuned and aligned entirely on consumer hardware — no cloud compute — through three stages:

1. **SFT-A** — Reasoning and chain-of-thought (50 samples, validation run)
2. **SFT-B** — Identity, knowledge, and multilingual alignment (5,500 samples, ~88 h)
3. **ORPO-3** — Preference alignment without a reference model (2,104 pairs, ~54 h)

The model is multilingual (inherits BitNet's broad language coverage; alignment data focused on English and Spanish), refuses harmful requests, and runs on any PC with 8 GB RAM.

> **Inference note**: Orchid uses the BitNet I2_S ternary format with a separate LoRA adapter. Standard llama.cpp cannot serve this combination. Use **[ternative](https://github.com/michelangeloromerochisco/ternative)** — the purpose-built inference engine.

---

## Quick Start

### Try instantly (no install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/michelangeloromerochisco/orchid-1.0/blob/main/orchid_colab.ipynb)

CPU-only, free, no time limit. First run ~8–10 min (LoRA merge); subsequent runs ~30 s.

### Run locally

**Step 1 — Download models**
```bash
huggingface-cli download MicheRomChis/orchid-1.0 \
  ggml-model-i2_s.gguf dpo_aligned-lora.gguf \
  --local-dir ./orchid-models
```

**Step 2 — Build ternative** (Windows and Linux only — macOS not supported; requires cmake 3.18+, C++17 compiler)
```bash
git clone --depth 1 https://github.com/michelangeloromerochisco/ternative
cd ternative
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
cd ..
```

**Step 3 — Generate**
```bash
# Linux
./ternative/build/ternative \
  --model ./orchid-models/ggml-model-i2_s.gguf \
  --lora  ./orchid-models/dpo_aligned-lora.gguf \
  --prompt "¿Cuál es la capital de Colombia?"

# Windows (PowerShell)
.\ternative\build\Release\ternative.exe `
  --model .\orchid-models\ggml-model-i2_s.gguf `
  --lora  .\orchid-models\dpo_aligned-lora.gguf `
  --prompt "What is photosynthesis? Think step by step."
```

**Step 4 — Run as OpenAI-compatible server**
```bash
./ternative/build/ternative \
  --model ./orchid-models/ggml-model-i2_s.gguf \
  --lora  ./orchid-models/dpo_aligned-lora.gguf \
  --server --port 8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="orchid",
    messages=[{"role": "user", "content": "Explain quantum entanglement simply."}]
)
print(response.choices[0].message.content)
```

---

## Benchmark Results

Scored using the [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) log-probability methodology (50 samples each) via a live ternative server.

### Standard benchmarks

| Benchmark | Orchid 1.0 | BitNet b1.58 base | Delta |
|-----------|----------:|------------------:|------:|
| ARC-Challenge | **56.0%** | 49.9% | **+6.1 pp** |
| HellaSwag (length-norm) | 52.0% | 68.4% | −16.4 pp |
| WinoGrande | **74.0%** | — | — |
| MMLU (57 subjects) | 38.6% | 53.2% | −14.6 pp |

ARC improvement confirms the reasoning fine-tuning transferred. HellaSwag and MMLU regressions are the expected ORPO alignment tax — consistent with published DPO/ORPO literature.

### Internal benchmark (100 questions, 8 categories)

| Rank | Model | Score |
|-----:|-------|------:|
| 1 | Claude 3.5 Sonnet | 89.5% |
| 2 | GPT-4o | 89.2% |
| **3** | **Orchid 1.0** | **87.9%** |
| 4 | BitNet b1.58 base | 84.2% |
| 5 | Kimi k1.5 | 82.2% |
| 6 | Qwen2.5-7B | 78.4% |

> Note: the internal benchmark uses semantic similarity scoring. It is a relative comparison tool, not a substitute for standard NLP benchmarks.

---

## Hardware Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| GPU VRAM | 0 (CPU-only) | 4 GB (RTX 3050 class) |
| RAM | 8 GB | 16 GB |
| Storage | 1.3 GB | 2 GB |
| OS | Windows / Linux | macOS not supported (ARM support planned) |

**GPU mode**: all 30 transformer layers on GPU using F16 + INT8 mixed precision (~3.3 GB VRAM), ~6–7 tok/s.  
**CPU mode**: ~6 tok/s with AVX2 (any Intel/AMD CPU since ~2013).

---

## Model Files

| File | Size | Purpose |
|------|-----:|---------|
| `ggml-model-i2_s.gguf` | ~1.1 GB | BitNet b1.58-2B-4T base (I2_S ternary format) |
| `dpo_aligned-lora.gguf` | ~90 MB | ORPO-3 aligned LoRA adapter (F32, 420 tensors) |

---

## Training Details

All training on a single **NVIDIA RTX 3050 laptop GPU (4 GB VRAM, 16 GB RAM, Windows 11)**.

| Stage | Method | Data | Duration |
|-------|--------|------|----------|
| SFT-A | LoRA r=16 | Reasoning / chain-of-thought (50 samples) | ~1 h |
| SFT-B | LoRA r=16 | 5,500 samples (identity + knowledge) | ~88 h |
| ORPO-2 | LoRA r=8 | 2,038 preference pairs (debiasing) | ~26 h |
| ORPO-3 | LoRA r=8 | 2,104 preference pairs (production) | ~54 h |

**Key memory techniques:**
- Pre-tokenize dataset before loading model (prevents startup OOM)
- `device_map="auto"` — GPU + CPU split via Accelerate
- Gradient checkpointing + `bf16=True`
- ORPO with `ref_model=None` — saves ~1.2 GB vs DPO

---

## Why ternative?

Standard inference stacks cannot serve LoRA-fine-tuned ternary models:

| Engine | I2_S base | Runtime LoRA | I2_S + LoRA |
|--------|:---------:|:------------:|:-----------:|
| llama.cpp | ⚠️ type-36 error | ✓ (Q4/Q8 only) | ✗ |
| bitnet.cpp | ✓ | ✗ no adapter path | ✗ |
| **ternative** | ✓ | ✓ full precision | ✓ |

**The problem**: merging a LoRA adapter into an I2_S base and re-quantizing rounds every delta to zero — the fine-tuning is silently discarded. ternative avoids this by de-quantizing the I2_S base to F32, applying the LoRA delta at full precision, and casting to F16 for inference.

---

## Limitations

- **MMLU at 38.6%** — expected alignment tax from ORPO, documented in the technical paper
- **Spanish coverage** — 80% on internal benchmark; functional but not state-of-the-art
- **Context window** — 4,096 tokens (inherited from BitNet base)
- **ternative required** — llama.cpp produces type-36 errors or silently wrong output
- **No BitsAndBytes** — stacking BNB on top of BitNet's ternary quantization is unsupported
- **Identity requires system prompt** — without a system prompt Orchid may respond generically

---

## Technical Paper

Full methodology, training details, failure modes, and architecture analysis:

**[Orchid 1.0: A Reproducible Recipe for Aligned Ternary-Weight Language Models on Consumer Hardware](https://huggingface.co/MicheRomChis/orchid-1.0/blob/main/orchid-1-0-technical-paper.pdf)**

---

## What's Next — Terse

Orchid 1.0 is the proof of concept. **Terse** is the follow-up: a clean-room ternary sparse transformer family (Mini 1.5B/4.5B, Medium 9B/27B, Pro 27B/81B) with MoE routing, hybrid linear+full attention, recurrent depth, and vision — targeting the same consumer hardware envelope.

---

## Citation

```bibtex
@software{orchid_2026,
  title   = {Orchid 1.0: First Competitive LLM Trained and Aligned in Colombia},
  author  = {Romero Chisco, Michelangelo},
  year    = {2026},
  url     = {https://github.com/michelangeloromerochisco/orchid-1.0},
  license = {Apache-2.0},
  note    = {Fine-tuned from Microsoft BitNet b1.58-2B-4T}
}
```

---

## License

Apache 2.0 — free for research and commercial use.

> **Copyright (c) Microsoft Corporation.** Orchid 1.0 is a fine-tuned derivative of BitNet b1.58-2B-4T (MIT License). The MIT License requires this copyright notice to accompany derivative works.

## Acknowledgments

- **Microsoft Research** — BitNet b1.58-2B-4T base model and architecture
- **Georgi Gerganov and the llama.cpp project** — GGUF format conventions
- **HuggingFace** — Training libraries (PEFT, TRL, Transformers, Accelerate)
