# Forge: Comprehensive 10/10 ML Engineering Roadmap

## Module 1: Tokenization, Embeddings & Data Architecture
### Core Concept
Neural networks process dense vector spaces, not strings. Tokenization is the lossy compression algorithm translating human language into discrete integer indices, which embeddings then project into continuous vector spaces.
### Implementation Steps (`dataset.py`)
1. Ingest raw instruction-tuning data in JSONL format (`{"prompt": "...", "response": "..."}`).
2. Apply standard system chat templates (e.g., Llama-3 instruction template formatting).
3. Instantiate `AutoTokenizer` from Hugging Face; configure explicit padding tokens and sequence truncation boundaries.
4. Tokenize batch text with `return_tensors="pt"`, generating `input_ids`, `attention_mask`, and shifted `labels` for causal language modeling.
### What to Study Outside the Project
- **Byte-Pair Encoding (BPE):** Frequency-based merge rules, subword tokenization, vocabulary collision, and handling out-of-vocabulary tokens.
- **Embedding Spaces:** Cosine similarity, dot products, vector manifold geometry, and positional embeddings (RoPE - Rotary Position Embedding).

---

## Module 2: Parameter-Efficient Fine-Tuning (QLoRA)
### Core Concept
Full fine-tuning a 7-billion parameter model requires ~140GB+ VRAM due to optimizer states and gradients. QLoRA solves this via 4-bit NormalFloat quantization (NF4) and Low-Rank Adaptation (LoRA), freezing base weights and training tiny side matrices ($\Delta W = B \times A$).
### Implementation Steps (`train.py`)
1. Configure `BitsAndBytesConfig` for 4-bit NF4 quantization with double quantization and compute dtype set to `bfloat16`.
2. Load base model via `AutoModelForCausalLM.from_pretrained()`, applying device mapping.
3. Wrap model with `PeftModel` using `LoraConfig`: set rank $r = 16$, scaling alpha $\alpha = 32$, dropout $0.05$, and target projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
4. Initialize `SFTTrainer` from Hugging Face `trl`, passing training arguments (per-device batch size, gradient accumulation steps, learning rate schedule, warmup ratio).
5. Execute training loop, monitor loss convergence, and save serialized LoRA adapter weights (`adapter_model.safetensors`).
### What to Study Outside the Project
- **Matrix Decomposition:** Linear algebra rank, low-rank approximations, SVD (Singular Value Decomposition), and parameter efficiency.
- **Quantization Mathematics:** Absolute maximum quantization, uniform vs non-uniform quantization, NF4 information-theoretic optimality.
- **Optimizer Internals:** AdamW weight decay mechanics, momentum tracking, and why FP16/BF16 precision matters for numerical stability.

---

## Module 3: Evaluation & Validation Harness
### Core Concept
A fine-tuned model can hallucinate pleasant-sounding syntax. Rigorous automated evaluation ensures domain accuracy, prevents catastrophic forgetting, and measures perplexity regressions against the base model.
### Implementation Steps (`eval.py`)
1. Construct a golden evaluation set of 50+ deterministic test prompts with expected outputs.
2. Load base model and merge fine-tuned LoRA adapters into a unified weights state.
3. Generate responses across both models using constrained greedy decoding or nucleus sampling.
4. Implement automated scoring metrics: Exact Match (EM), ROUGE-L, syntax validity checks (e.g., SQL execution tests via SQLite), and LLM-as-a-judge scoring rubrics.
### What to Study Outside the Project
- **Language Model Perplexity:** Information entropy, cross-entropy loss interpretation, and token prediction uncertainty.
- **Evaluation Methodologies:** BLEU, ROUGE, HumanEval, MT-Bench, and prompt sensitivity bias.

---

## Module 4: High-Throughput Serving (vLLM & FastAPI)
### Core Concept
Standard Hugging Face `model.generate()` causes massive memory fragmentation and sequential bottlenecking during inference. vLLM solves this using PagedAttention and continuous batching.
### Implementation Steps (`serve.py`)
1. Initialize vLLM engine (`AsyncLLMEngine`) pointing to the merged model weights directory.
2. Configure GPU memory utilization (`gpu_memory_utilization=0.90`) and max model length limits.
3. Build FastAPI REST application exposing an OpenAI-compatible `/v1/chat/completions` endpoint.
4. Handle asynchronous request streaming, chunking generated tokens back to clients via Server-Sent Events (SSE).
### What to Study Outside the Project
- **KV-Cache Memory Management:** Why autoregressive generation is memory-bound, how Key-Value states grow per token, and memory fragmentation issues.
- **PagedAttention:** Operating system virtual memory pagination applied to GPU KV-caches, zero-copy block allocation, and shared memory sharing across parallel requests.
- **Continuous Batching:** Iteration-level scheduling vs traditional static batching.
