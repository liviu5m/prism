# Forge — Learn & Build Roadmap (phase by phase)

**Read this first.** `ROADMAP.md` explains the *concepts* behind each module, and `roadmap.txt` is the original
short phase list. **This file is the operating manual**: for every phase it gives you
(a) exactly what to build (files, functions, flags, commands), (b) the evidence that proves it works, and
(c) the specific topics to learn so you actually understand what you shipped.

- Task: **Text → SQLite SQL** (English question in, executable SQL out).
- Base model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (1.1 B, 22 layers, hidden 2048, 32 heads / **4 KV heads**, ctx 2048).
- Method: QLoRA (PEFT) + SFT/DPO (TRL). Eval: **SQL execution accuracy**, not vibes. Serve: vLLM + FastAPI on one 4 GB RTX 3050 Laptop.

---

## Contents

- §0 Verified starting point (what is actually in this repo, with evidence)
- §1 The 10/10 scorecard (spec bullet → artefact that proves it) + working rules
- §2 Target repository layout
- **Phase 0** — Repo & environment hygiene
- **Phase 1** — Data: curation, schema conditioning, splits, versioning
- **Phase 2** — Training: config-driven QLoRA/SFT, sweeps, MLflow
- **Phase 3** — Task-specific evaluation harness (execution accuracy)
- **Phase 4** — Preference tuning / DPO (fixing the proven no-op)
- **Phase 5** — Quantization + quality-loss report
- **Phase 6** — Serving: vLLM + FastAPI + queue + metrics + dashboard
- **Phase 7** — A/B serving + autoscaling control loop
- **Phase 8** — Report, CI, packaging, resume numbers
- **Appendix A** — Command cheat sheet · **B** — VRAM/time budget · **C** — Traps to avoid ·
  **D** — Reading list · **E** — Glossary · **F** — 15-day schedule & where to start tomorrow

---

## 0. Verified starting point (measured 2026-09-20, this machine)

| Item | Verified state |
|---|---|
| Repo | 3 commits; `README.md` = 1 line; no tests, no CI, no packaging |
| Training | `src/train.py`: QLoRA, r=8, α=16, **q_proj+v_proj only**, **100 epochs on 50 examples**, `bf16=False`, no eval split, seed 42 → `final_model/` (700 steps) |
| Adapter size | 88 tensors, 1,126,400 trainable params (q: 8×(2048+2048), v: 8×(256+2048), ×22 layers) = 1,132,200×2 B → matches the 2.26 MB file. **0.1 % of the model.** |
| Eval | 5 prompts, and **all 5 are verbatim inside the training set** → any score is memorization |
| DPO | **no-op**: loss pinned at 0.6931 (= log 2), `rewards/margins = 0.0` every step; tensor diff vs the SFT adapter = **0.0 across all 88 tensors** |
| Serving | `src/serve.py` is **0 bytes** |
| Results | `outputs/` is empty → **no number exists anywhere in the project** |
| Live check | base model → garbage; SFT adapter → `SELECT COUNT(*) FROM subscriptions;` while gold is `SELECT COUNT(*) FROM subscriptions WHERE status='active';` → **wrong even on a training prompt** |
| Host | RTX 3050 Laptop, **3.95 GB VRAM** (sm_86, bf16 supported), 14 GB RAM, 354 GB free, driver 610.57.04 / CUDA 13.3 |
| Python | system **3.14.7** only; `uv` available; **Docker has no GPU/CDI** → vLLM must run natively |
| Installed | torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, trl 1.13.0, datasets 5.0.1, bnb 0.50.2, accelerate 1.15.0 |
| vLLM path | vLLM **0.29.0** ships a `cp38-abi3` wheel (runs on 3.14) but pins **torch==2.13.0** → needs its **own venv** |
| KV budget | 2 × 22 layers × 4 KV heads × 64 dim × 2 B = **22.5 KB/token** → ≈50 k cached tokens in ~1.1 GB, so concurrency 8–16 @ 256 tokens fits |

### The five things that are actually broken (fix in this order)
1. **No metrics** → nothing is provable. 2. **Eval leakage** (5/5 prompts in train). 3. **No schema in the prompt** → hard ceiling on accuracy. 4. **DPO trained nothing**. 5. **No serving code at all**.

---

## 1. Definition of "10/10" — the scorecard you keep filling in

A requirement counts only when an **artifact** exists. Keep this table at the top of `README.md` and tick items off.

| Spec bullet | Artifact that proves it |
|---|---|
| Dataset curation, clean, dedupe, versioning | `data/v1/` + `manifest.json` (sha256s) + `DATASET_CARD.md` + a *passing* leakage test |
| LoRA/QLoRA + sweeps tracked in MLflow | `configs/sft/*.yaml`, ≥8 runs in `mlflow.db`, sweep summary table |
| DPO stretch goal | `artifacts/dpo_v1/`, `rewards/margins ≠ 0`, measured exec-acc delta vs SFT |
| Task-specific eval harness | `reports/eval_v1.md`: base vs SFT vs DPO vs quantized, per-slice, with CIs |
| Quantization + quality-loss report | `artifacts/merged_w4a16/` + table of size / VRAM / latency / accuracy delta |
| vLLM + autoscaling + cost/latency dashboard | `forge/serve/*`, load-test numbers, `/dashboard` output, control-loop chart |
| A/B serving | sticky routing, per-variant metrics, a written promotion/rollback decision |
| Reproducible from one config + snapshot | `run_manifest.json` per run: config hash + dataset hash + git SHA + seed + versions |
| Throughput vs untuned base on same HW | `reports/serving_bench.md`: base, SFT, LoRA overhead, HF-generate vs vLLM |
| Graceful queueing, never silently drops | load test with **0 silent drops**: `429` + `Retry-After` + `X-Queue-Depth` counters |

### Non-negotiable working rules
1. **Evidence rule** — no claim in any report without a command that reproduces the number.
2. **One source of truth per run** — a YAML config; scripts never hardcode hyperparameters or paths.
3. **Datasets are immutable snapshots** — `data/v1/…` is written once; a change means `v2`.
4. **No leakage, ever** — the test split is touched *only* by the eval harness, and a test enforces this.
5. **Label what is simulated** — this host has one GPU and Docker cannot see it; any scaled/simulated component is marked as such in the README.

---

## 2. Target repository layout (build towards this)

```
prism/
├── pyproject.toml              # package + deps + console scripts (forge-train, forge-eval, forge-serve)
├── Makefile                    # one target per phase step
├── FORGE_ROADMAP.md            # this file
├── configs/
│   ├── data/v1.yaml            # sources, cleaning rules, split ratios, dedupe thresholds
│   ├── sft/r16_alllinear.yaml  # model, LoRA, SFT hyperparams, seed
│   ├── dpo/v1.yaml             # β, lr, pair-generation settings
│   └── serve.yaml              # engine, queue limits, A/B split, cost model
├── forge/
│   ├── config.py               # load YAML → pydantic model; compute config_hash
│   ├── data/{build,schema,split,version}.py
│   ├── train/{sft,dpo,sweep}.py
│   ├── eval/{harness,metrics,report}.py
│   ├── quant/{quantize,measure}.py
│   └── serve/{engine,router,queue,metrics,app,dashboard,autoscale,loadtest}.py
├── data/
│   ├── raw/                    # untouched sources
│   ├── v1/{train,dev,test}.jsonl + manifest.json + DATASET_CARD.md
│   └── fixtures/               # schema DDL + seeded SQLite DBs (train schema + unseen test schema)
├── artifacts/{sft_v1,merged_fp16,merged_w4a16,dpo_v1}/
├── reports/{eval_v1.md,serving_bench.md,quant_report.md,figures/}
└── tests/                      # pytest, CPU-only, runnable in CI
```

Keep `venv/` (training), add `venv-serve/` (vLLM, torch 2.13) and `venv-quant/` (llm-compressor) — all gitignored.
Also fix the current contradiction: `.gitignore` ignores `data/` and `*.safetensors` while several such files are already tracked, and `__pycache__/*.pyc` is committed. Untrack the pycache, decide one policy per pattern.

---

# Phase 0 — Repo & environment hygiene (½ day)

### Build
1. `pyproject.toml` (PEP 621) with `[project] dependencies` = torch/transformers/peft/trl/datasets/bitsandbytes/accelerate/mlflow,
   `[project.optional-dependencies] serve = ["vllm==0.29.0","fastapi","uvicorn","prometheus-client","httpx"]`, `quant = ["llmcompressor"]`,
   and `[project.scripts] forge-train / forge-eval / forge-serve`.
2. Three venvs, because the pins genuinely conflict (`vllm==0.29.0` pins `torch==2.13.0`, you train on 2.14.0):
   ```bash
   uv venv venv            && uv pip install -e ".[dev]"     # training (keep existing venv if you prefer)
   uv venv venv-serve      && uv pip install -e ".[serve]"   # vLLM + FastAPI (torch 2.13.0)
   uv venv venv-quant      && uv pip install -e ".[quant]"   # llm-compressor (quantization)
   ```
3. Move code into the `forge/` package from section 2 (keeping thin wrapper scripts if you like). Import by module path, never
   `from dataset import ...` (that only works when cwd == `src/` and breaks the moment you add tests or CI).
4. `Makefile`: `make data | make train | make sweep | make eval | make quant | make serve | make bench | make test`.
5. `.gitignore` policy: never ignore `data/v*.jsonl` (they are the product!) but ignore `data/raw/`, `artifacts/`, `reports/figures/`;
   `git rm -r --cached __pycache__ src/__pycache__`; add Git LFS for `*.safetensors` **or** stop committing them and publish to the Hub.
6. `tests/` skeleton with one trivial test, so `make test` works from day one.
7. Replace the 1-line `README.md` with a stub that contains: one-paragraph pitch, architecture diagram (ASCII is fine),
   the scorecard table from section 1, and a **Quickstart** with the exact commands to reproduce everything.

### Prove it
`make test` passes on a clean clone; `git ls-files | grep -c pycache` returns 0; `python -c "import forge"` works from any cwd.

### Learn (≈4 h)
| Topic | Why you need it |
|---|---|
| Python packaging: `pyproject.toml`, src layout, editable installs, entry points | So `forge-eval` is a command, not a file you must be standing next to |
| Dependency resolution & lockfiles (`uv pip compile`, `pip-tools`), why pinning matters | Two tools needing different `torch` versions is a real production problem — you are solving it with venv isolation |
| Reproducibility basics: seeds (`random`, `numpy`, `torch`), cudnn determinism, git SHA capture | Prerequisite for "reproducible from a single config" |
| Git semantics: tracked vs ignored, LFS, `git rm --cached`, commits that are reviewable | Reviewers read your history; binaries + pycache signal sloppiness |
| pytest basics: fixtures, `tmp_path`, parametrize, markers for slow/GPU tests | Your leakage guard and API contract tests live here |

**Self-check:** Why can a venv not satisfy both vLLM's and your training stack's torch pin? What exactly does `-e` change? Why is ignoring `data/` while tracking `data/dataset.jsonl` dangerous for a fresh clone?

---

# Phase 1 — Data: curation, schema conditioning, splitting, versioning (2–3 days)

> This is the highest-ROI phase in the whole project. Model choice and hyperparameters are second-order;
> data quality, prompt informativeness and leakage control decide your headline number.

### 1.1 Build the fixture databases (`forge/data/schema.py`, `data/fixtures/`)
- `shop.sql`: `users(id, name, created_at)`, `orders(id, user_id, total, created_at)`, `order_items(order_id, product_id, quantity, price)`,
  `products(id, name, category_id, price, stock)`, `categories(id, name)`, `subscriptions(id, user_id, status, started_at)`.
- `library.sql`: a **second, never-trained schema** (`authors`, `books`, `loans`, `members`) used only in the test split.
- `seed_db(sql_path, db_path, seed)` inserts deterministic rows: ≥200 orders, ties avoided where gold uses `ORDER BY` (or add a
  unique tiebreaker column), `NULL`s present in some columns, statuses with different casing.
- **Why:** execution accuracy is only meaningful if the DB is deterministic and rich enough that a wrong `WHERE`/`JOIN` changes the result set.

### 1.2 Curation pipeline (`forge/data/build.py`)
Steps, in order, with counts logged at each stage (this table goes in the dataset card):
1. **Load** raw candidate pairs (`data/raw/*.jsonl`) → 2. **Normalise** (strip markdown fences, collapse whitespace, one statement per row) →
3. **Validate**: SQL parses (`sqlglot`) **and** executes on the fixture DB without error; quarantine failures into `data/raw/rejected.jsonl` with the reason →
4. **Dedupe**: exact hash, then near-duplicate by normalised-token Jaccard ≥ 0.85 (or MinHash/LSH if you want the real thing) on both question and SQL →
5. **Balance**: report the distribution over query-pattern buckets (filter, aggregate, join, group-by, subquery, order+limit, date arithmetic) and fill gaps until each bucket has ≥ 15 train examples →
6. **Format**: frozen prompt template (below) → 7. **Split and shard** (below).

**Frozen prompt template** (identical in data build, training, evaluation and serving — put it in *one* function `forge/data/prompt.py::render(schema_ddl, question)` and never inline it elsewhere):
```text
### Schema:
CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, user_id INTEGER, status TEXT, started_at TEXT);
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, created_at TEXT);
### Prompt:
Count active subscriptions
### Response:
SELECT COUNT(*) FROM subscriptions WHERE status = 'active';
```
- Include only the **tables needed** for the question (keep prompts short; measure token-length percentiles with the real tokenizer).
- Sequence budget check: report p50/p95/p99 of prompt+answer tokens; set `max_length` ≥ p99 + 32 so the answer is never truncated
  (truncation silently destroys the training signal and is invisible in the loss curve).

### 1.3 Splits, sizes and leakage controls
| Split | Schema | Size | Purpose |
|---|---|---|---|
| train | shop | 300–500 | learning |
| dev | shop | 60 | model selection, early stopping (never test) |
| test-in | shop | 60 | in-distribution score for the report |
| test-out | library (**unseen**) | 40 | generalisation score — the number that impresses |
Splitting is deterministic: `bucket = int(sha256(normalised_question)[:8], 16) % 100` → 80/10/10, or split by *pattern family*
so that dev/test contain phrasings the model has not seen verbatim.

### 1.4 Version + document (`forge/data/version.py`)
`data/v1/manifest.json`: per-file sha256, row counts, per-stage counts (loaded/validated/deduped/dropped), the config hash,
build-script git SHA, token-length percentiles. `data/v1/DATASET_CARD.md`: motivation, schema, splits, how pairs were produced
and verified (auto-verified by execution + manual review of a 20 % sample), known limits (English only, single statement, one DB dialect), licence.

### 1.5 Leakage guard (`tests/test_no_leakage.py`) — must exist before you train
- normalised questions disjoint across train/dev/test(-in/-out);
- no gold SQL string from dev/test appears in train;
- `test-out` schema DDL never appears in any train/dev row;
- rebuild twice → identical sha256 (byte-for-byte reproducibility).

### Prove it
`make data` twice gives the same `manifest.json` hashes; the leakage test passes; `rejected.jsonl` is non-empty and explained;
p50/p95/p99 token lengths reported; each query-pattern bucket ≥ 15 examples.

### Learn (≈10 h)
| Topic | Why | Study |
|---|---|---|
| Instruction-tuning data formats (Alpaca, ShareGPT, OpenAI messages) and chat templates | Your format is a contract; base TinyLlama has no chat template, so you own it | HF `chat_template` docs; TRL SFT docs |
| Tokenization of code/DDL: BPE merges, subwords of identifiers, digits, quotes | Explains prompt-length cost and why `ORDER BY` snippets get split oddly | your `ROADMAP.md` Module 1; Karpathy’s tokenizer walkthrough |
| Deduplication: exact hashing, Jaccard, MinHash + LSH, shingling | Near-duplicates are your #1 leakage risk with templated SQL | Broder 1997 (MinHash); `datasketch` docs |
| Data quality filtering & synthetic-data verification | More data of low quality is worse than less good data | any 2024–2026 survey on data curation for SFT; LIMA (Zhou et al. 2023) for the quality>quantity argument |
| Splits, stratification, `dev` vs `test` discipline, model selection bias | Using test for early stopping silently inflates your reported number | any ML methodology text (ESL ch. 7) |
| Leakage taxonomy: prompt / answer / schema / near-duplicate / benchmark contamination | The exact bug already in this repo (5/5 prompts) | — |
| Text-to-SQL specifics: schema linking, ambiguity, Spider vs BIRD conventions, **why Exact Match is broken and execution accuracy is used instead** | Determines your metric and your prompt design | Spider (Yu et al. 2018), BIRD (Li et al. 2023), Zhong et al. 2020 on semantic evaluation |
| Data versioning tooling: Git LFS, DVC, lakeFS, HF `datasets` revisions | "One config + dataset snapshot" requirement | tool docs |

**Self-check:** Why does adding the DDL raise the ceiling? Why is `test-out` (unseen schema) a stronger claim than a bigger `test-in`?
Why can 100 epochs on 50 examples look like a beautiful loss curve and still produce a wrong `WHERE` clause? Which of the four leakage types did this repo have?

---

# Phase 2 — Training: config-driven QLoRA/SFT, sweeps, MLflow (2–3 days, sweeps overnight)

> Goal: replace `src/train.py` (hardcoded, 100 epochs on 50 rows, no eval split, chosen by loss) with a
> reproducible, config-driven, swept, MLflow-tracked trainer whose winner is chosen by **dev execution accuracy**.

### 2.1 `forge/config.py` — reproducibility spine
- Load YAML into a pydantic model; refuse unknown keys (typos must fail loudly).
- `config_hash = sha256(canonical_json(cfg))`; write `run_manifest.json` next to every artifact containing:
  config (+hash), dataset manifest sha256, git SHA, seed, package versions (torch/transformers/peft/trl/bnb),
  GPU name + total VRAM, wall-clock, and `mlflow_run_id`.
- Every script takes `--config` and nothing else behavioural. This is what "reproducible from a single versioned config + dataset snapshot" means in practice.

### 2.2 The SFT config that fixes your current run (`configs/sft/r16_alllinear.yaml`)
```yaml
model:  {id: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T, dtype: bfloat16}
quant:  {load_in_4bit: true, quant_type: nf4, double_quant: true, compute_dtype: bfloat16}
lora:   {r: 16, alpha: 32, dropout: 0.05, target_modules: all-linear, bias: none}
sft:
  max_length: 512
  num_train_epochs: 4          # not 100
  learning_rate: 2.0e-4        # LoRA-appropriate, not 2e-5
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8      # effective batch = 16
  bf16: true                   # Ampere supports it; fp16=False/bf16=False was waste
  gradient_checkpointing: true
  optim: adamw_torch_fused
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  early_stopping_patience: 2
  metric_for_best_model: eval_loss   # then re-rank winners by dev exec accuracy
  seed: 42
  report_to: [mlflow]
```
**Why each change** (be able to defend all of them in an interview):
| Change | Reason |
|---|---|
| `all-linear` (q,k,v,o,gate,up,down) instead of q,v | 7 modules instead of 2 → ~12.6 M trainable params vs 1.13 M. Your evidence that q/v-only is too weak: the model drops the `WHERE` clause even on a memorised prompt |
| r 8→16, α 16→32 (α=2r) | More capacity per module; α/r sets the effective update scale |
| lr 2e-5→2e-4 | LoRA tolerates ~10× the full-FT learning rate because only small factors are trained |
| 100→4 epochs with early stopping | 100 epochs on 50 rows = memorisation; selection must happen on **dev**, not on train loss |
| `bf16: true` | sm_86 supports bf16; previous run trained in fp32-relaxtion of a 4-bit model — slow and less stable |
| `double_quant: true`, compute dtype bf16 | Extra memory saving; **keep the autocast dtype and the bnb compute dtype identical** — mismatches here are a classic cause of “training does nothing” |
| eval/save per epoch + `load_best_model_at_end` | Reproducible model selection, no more orphan checkpoints |
| MLflow + config hash | Sweep comparison and provenance |

Add at startup: assert dataset sha256 == the one in the config; print trainable/ total param ratio; log prompt token-length percentiles.

### 2.3 Sweep (`configs/sft/sweep.yaml`, `forge/train/sweep.py`)
- Coarse grid: `r ∈ {8,16,32}` × `lr ∈ {1e-4,2e-4,5e-4}` × `targets ∈ {qv, all-linear}` = 18 cells; run 12 (e.g. r=32 & qv excluded as a-priori dominated) — one overnight, 15–30 min/run on the 3050.
- Then a refinement run around the best cell (dropout 0 vs 0.05, max_length 512 vs 768, epochs 3 vs 6).
- Efficient alternative worth implementing and describing: **successive halving** — 1-epoch proxy for all cells, keep top 1/3, full run only for survivors.
- Rank runs by **dev execution accuracy** (call the Phase-3 harness on each saved adapter), then by dev loss as tiebreak. Log the metric back into MLflow (`mlflow.log_metric`) so the sweep table is one query.

### 2.4 MLflow (`forge/train/sft.py` logging)
```bash
uv pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Log: params (the whole config), metrics per step/epoch (loss, grad_norm, lr, mean_token_accuracy, entropy), dev metrics,
artifacts (config.yaml, run_manifest.json, loss curves) and register the winning adapter as a model version. Tag each run	with dataset version (`v1`), git SHA and sweep cell id.

### 2.5 A second experiment you should run and document (`bf16 LoRA, no quantization`)
TinyLlama fp16/bf16 weights ≈ 2.2 GB. With `gradient_checkpointing` and batch 1–2 at `max_length 512`, plain bf16 LoRA
(often faster + slightly more accurate than QLoRA) may fit in 3.95 GB. Try it, record OOM-or-success in `reports/eval_v1.md`,
and keep the winner. "I measured when QLoRA is actually needed" is a stronger statement than "I used QLoRA".

### Prove it (evidence for the report)
- Re-running one config with the same seed reproduces the dev metric exactly (*or* you document exactly which kernels are non-deterministic — that counts as understanding).
- Sweep table with ≥12 rows and dev exec-accuracy ordering.
- `artifacts/sft_v1/` contains adapter + `run_manifest.json`; `mlflow_run_id` resolves in the UI.
- Train vs dev loss curves show the overfitting gap of your old 100-epoch run against the new one (put both in the README).

### Learn (≈12 h)
| Topic | Why | Study |
|---|---|---|
| LoRA math: `ΔW = BA·(α/r)`, zero-init of `B`, why rank matters, which modules to adapt | You must be able to justify r=16/all-linear | LoRA (Hu et al. 2021); PEFT docs |
| QLoRA: NF4, double quantization, paged optimizers, compute dtype, why 4-bit ≠ 4× slower | Your memory budget is 3.95 GB — this is the enabling trick | QLoRA (Dettmers et al. 2023) |
| SFT loss masking: full-sequence vs **completion-only** loss; how TRL handles it | Are you currently training the model to predict the prompt too? That dilutes the signal | TRL SFT trainer/config docs |
| Collators, padding, sequence packing | Packing changes the effective batch and noise profile | TRL + HF docs |
| Gradient accumulation & effective batch size; gradient checkpointing tradeoffs | Tuning knobs you are already using | — |
| Mixed precision: fp16 vs bf16 vs fp32 master weights; loss scaling; dtype-mismatch failure modes | Your DPO no-op lives in this family of bugs | any AMP write-up; NCCL/AMP notes |
| Optimizers: AdamW variants, fused, 8-bit; weight decay on LoRA params | `adamw_torch` vs `adamw_torch_fused` is a real speed difference | Loshchilov & Hutter 2019 |
| LR schedules: warmup, linear vs cosine, what the loss curve should look like | Diagnose a bad run by shape alone | — |
| Model selection: dev vs test, early stopping, patience, why loss ≠ task metric | Your old run selected nothing at all | ESL ch. 7 |
| HPO strategies: grid, random, Bayesian (TPE), successive halving/ASHA, low-fidelity proxies | 18 cells on one GPU must be pruned intelligently | Bergstra & Bengio 2012; Optuna/ASHA docs |
| MLflow concepts: experiments, runs, params/metrics/artifacts, model registry | Required deliverable and the vocabulary of MLOps interviews | MLflow docs |
| Memorisation vs generalisation diagnostics; catastrophic forgetting | Explains your `WHERE`-clause failure | — |

**Self-check:** number of trainable params at r=8 q/v vs r=16 all-linear (answer: 1.13 M vs ≈12.6 M — verify with `model.print_trainable_parameters()`);
why can LoRA use lr 2e-4 while full fine-tuning would diverge? Why is choosing the winner by train loss a methodological error?
Why must the bnb `compute_dtype` match autocast dtype? What is your effective batch size, and what changes if you double `gradient_accumulation_steps`?

---

# Phase 3 — Task-specific evaluation harness (2 days)

> Goal: replace `src/eval.py` (prints 5 prompts that are *inside the training set*, keeps no numbers) with a harness that
> scores **any** checkpoint — HF+LoRA, merged fp16, quantized, vLLM-served — against the base model, with slices, CIs and a written report.
> This is the single artifact that turns "I fine-tuned a model" into "I improved execution accuracy from X % to Y %".

### 3.1 `forge/eval/harness.py`
- `ModelSpec`: `{kind: hf_peft | hf_merged | vllm, path, adapter, engine_url}` — same interface for every checkpoint.
- Generation: **greedy (`do_sample=False`)** for the headline metric (reproducible), plus an optional `temperature=0.2, n=4` pass for stability.
- Extraction: stop on EOS / next `###` / first `;`; strip markdown fences; keep one statement; normalise (lowercase keywords, collapse whitespace, drop trailing `;`).
- **Template parity**: call `forge/data/prompt.py::render()` — the same function used for training and serving. Never re-inline the template.

### 3.2 `forge/eval/metrics.py` — what to compute
| Metric | Definition | Why |
|---|---|---|
| **Execution accuracy (headline)** | execute pred & gold on the seeded fixture DB, compare result sets | The only metric that matches the real task; ground truth is *executable*, so no LLM judge needed (say this explicitly in the README) |
| Valid-SQL rate | parses with `sqlglot` and prepares/executes without error | Distinguishes "wrong" from "broken" |
| Normalised exact match | whitespace/keyword-cased string equality | Secondary, cheap, comparable to literature |
| Error taxonomy | missed `WHERE` / wrong join / wrong aggregate / hallucinated column / multiple statements / refusal | Drives *data* fixes, not random tuning |
| Forgetting probe | mean NLL / perplexity on 20 held-out generic English paragraphs vs base | Shows the tune did not damage the model globally |
| Degeneration check | distinct-3-gram ratio on generated SQL | Catches repetition/looping |
Result-set comparison rules (write these down, they are the subtle part): column count must match; compare as a **multiset** unless gold has `ORDER BY` (then ordered, with a unique tiebreaker in the fixture data); ignore column *names*, compare values; `1e-6` tolerance for floats/aggregates; NULL == NULL for comparison purposes.
Add **bootstrap 95 % CI** (1000 resamples) and a **McNemar paired test** for every tuned-vs-base comparison — with n=40–60 the point estimate alone is not a claim.

### 3.3 `forge/eval/report.py`
- `reports/eval_v1.json` (machine-readable, one record per model×split×metric) + `reports/eval_v1.md` (human) + `reports/figures/*.png`
  (grouped bar chart with CI whiskers, per-slice heatmap model×slice, before/after example gallery of 5 wins and 5 failures with the error class).
- Report **test-in** and **test-out** separately: in-distribution score vs unseen-schema generalisation. The second is the number you quote.
- Include the base model in every table. A tuned row without a base row is not evidence.

### 3.4 The model matrix you must run
`base → SFT (best sweep cell) → DPO (Phase 4) → merged fp16 → W4A16 quantized → vLLM-served adapter`.
For the last one add a **serving-parity check**: same 20 prompts through the local HF model and through the vLLM endpoint;
assert identical (or documented-difference) outputs. This catches prompt-template drift — the most common silent serving bug.

### 3.5 CLI contract
```bash
forge-eval --models base,artifacts/sft_v1,artifacts/dpo_v1 --split test_in,test_out \
           --out reports/eval_v1 --seed 42        # writes json + md + figures
```

### Prove it
`reports/eval_v1.md` exists (today: nothing), contains base vs tuned with CIs, the leakage test passes, the parity check passes,
and the *old* claim (5 leaked prompts) is explicitly listed in the README as a discarded, invalid measurement.

### Learn (≈12 h)
| Topic | Why | Study |
|---|---|---|
| Metric theory: EM, ROUGE, BLEU, pass@k, perplexity — and how each misleads | You already have `ROUGE` mentioned in `ROADMAP.md`; know why it is wrong for SQL | Lin 2004; Papineni 2002; Chen et al. 2021 (pass@k) |
| Execution accuracy, its pitfalls, test-suite accuracy, semantic equivalence | A wrong-looking query can return the right rows on a small DB (false positive) — say how you mitigate (multiple seeds / test-suite execution) | Spider (Yu et al. 2018); BIRD (Li et al. 2023); Zhong et al. 2020 |
| Decoding determinism: greedy vs beam vs sampling; seed control | Your headline number must be reproducible on demand | HF `generate` docs |
| Statistics: bootstrap CIs, paired tests (McNemar), why n=40 needs intervals, multiple-comparison caution | Prevents over-claiming a 3-point lift | Efron & Tibshirani; McNemar 1947 |
| Error analysis methodology: taxonomy, stratified sampling, fix-the-data-first loop | Interviewers ask "what did the model get wrong and what did you change?" | — |
| LLM-as-a-judge: rubric design, position/verbosity bias, human agreement (Cohen's κ), when *not* to use it | Know the tool and know when ground truth makes it unnecessary | Zheng et al. 2023 (MT-Bench / Judge) |
| Capability-regression & forgetting evaluation; eval suites (HELM, Open LLM Leaderboard practices) | Proves the tune did not damage the model | HELM (Liang et al. 2022) |
| Contract/parity testing between training-time and serving-time pipelines | Template drift is the classic silent failure | — |
| Scientific reporting: tables that lead with the claim, CIs, honest negative results | A 10/10 project reports a *failure* as a finding | — |

**Self-check:** why is ROUGE-L a bad metric here, and what does execution accuracy still get wrong? Why must test-out use an unseen schema?
Why is a 3-point accuracy gain with n=40 not reportable without a CI? Why is "the model got 92 % on prompts it memorised" an invalid sentence?
How would you detect that the served model uses a different prompt template than training?

---

# Phase 4 — Preference tuning / DPO (1–2 days) — your run is currently a proven no-op

> **Evidence in the repo:** `dpo_model/checkpoint-9/trainer_state.json` shows `loss = 0.6931` (= log 2) and
> `rewards/chosen = rewards/rejected = rewards/margins = rewards/accuracies = 0.0` at every step, and the tensor diff between
> `final_model/adapter_model.safetensors`, `dpo_model/adapter_model.safetensors` and `dpo_model/ref/adapter_model.safetensors` is
> **0.0 on all 88 tensors** → the "DPO stretch goal" shipped a bit-identical copy of the SFT adapter. Fix it or drop it honestly.

### 4.1 Step 1 — reproduce the degeneracy in 30 seconds
`forge/train/debug_dpo.py`: load the SFT adapter (4-bit), score one chosen/rejected pair and print
`policy_logp_chosen, policy_logp_rejected, ref_logp_chosen, ref_logp_rejected, margin`.
Expected today: `ref_logp == policy_logp` → margin 0 → loss log 2 → zero learning signal. **Root cause class: the reference path and/or the
adapter dtype make policy and reference numerically identical.**

### 4.2 Step 2 — test the candidate causes, one variable at a time (log every result)
| # | Hypothesis | One-line test | Fix |
|---|---|---|---|
| 1 | Reference computed **with the adapter enabled** (so it equals the policy) | print `ref_logp` vs base-model `logp` on the same pair | `DPOConfig(precompute_ref_log_probs=True)` — also keeps only **one** model in 4 GB at a time |
| 2 | **Dtype mismatch**: adapter saved as bf16/fp32 vs `bnb_4bit_compute_dtype=float16` mixed with `bf16=True` autocast | print `dtype` of every `requires_grad` param and of the forward dtype | one dtype everywhere; LoRA master weights in fp32, autocast bf16, bnb `compute_dtype=bfloat16` |
| 3 | Gradients never reach LoRA params (frozen base + grad checkpointing) | log `grad_norm`; assert it is > 0 in the first 5 steps | `model.enable_input_require_grads()` (you had this) + verify `grad_norm` is logged at all |
| 4 | Prompt tokens not masked → chosen/rejected differ only in the prompt region | compare per-token logprobs of chosen vs rejected regions | pass proper `prompt`/`chosen`/`rejected` fields (TRL masks the prompt automatically) |
| 5 | Optimizer never steps (grad-accum/epochs arithmetic, lr 5e-6 on 20 pairs) | count optimizer steps in the log | raise lr to 1–2e-5, ≥50 pairs, 1–2 epochs |
**Add a permanent guard** so this never recurs silently: `forge/train/check_training_delta.py` compares adapter tensors before/after a run and
fails if `max_abs_diff == 0`; plus a pytest asserting `loss != log 2` after 10 steps on a 20-row fixture. That guard is the difference between a
scipt that "finishes" and an experiment you can trust.

### 4.3 Step 3 — build preference data **on-policy** (this is the real upgrade)
Your current `dpo_data.jsonl` rejected answers are trivially wrong (`SELECT * FROM users ORDER BY total DESC`), so the preference signal is nearly free.
Better loop (`forge/train/make_prefs.py`):
1. sample `n = 4` completions per **train** prompt from the SFT checkpoint (`temperature 0.8`, `top_p 0.95`);
2. score each by the Phase-3 executor (execution accuracy / valid SQL / normalised EM — a **verifiable reward**);
3. keep prompts where `best > worst`; label `chosen = best`, `rejected = worst`; prefer **hard negatives** (executes fine but wrong `WHERE`/aggregate/join);
4. write `data/v1/prefs.jsonl` (400–800 pairs) + counts per difficulty tier + how many prompts were discarded (all-equal or all-wrong).
This turns DPO into *learning from your own model's verified mistakes* — a genuinely strong story, and it is why the preference set is better than hand-written ones.

### 4.4 Step 4 — the DPO config and its gates (`configs/dpo/v1.yaml`)
```yaml
dpo: {beta: 0.1, epochs: 2, learning_rate: 2.0e-5, bf16: true, gradient_checkpointing: true,
      precompute_ref_log_probs: true, loss_type: sigmoid, max_length: 512}
```
Gates before you may claim anything: `rewards/margins ≠ 0` by step 10; `rewards/accuracies` inside 0.4–0.9 (≈1.0 means the task is too easy to learn from;
0.0 means the run is broken as before); weight-delta check passes; and — the only gate that matters — **test-in/test-out execution accuracy improves over SFT with a CI**.
Sweep only `β ∈ {0.05, 0.1}` × `lr ∈ {5e-6, 2e-5}` (4 cells, ~20 min each).

### 4.5 Step 5 — honest fallback (this raises your ceiling if DPO stalls)
Because you own an executor, you have a **verifiable reward**: `RLVR` with **GRPO** (sample k completions, reward = exec accuracy, no reward model, no reference model)
is often a better fit for text-to-SQL than DPO and needs no preference pairs at all. Implement it only after the DPO numbers are in,
and report DPO vs SFT vs GRPO in one table. If DPO loses, **report the loss** — a documented negative result with a diagnosis reads as senior work.

### Learn (≈10 h)
| Topic | Why | Study |
|---|---|---|
| RLHF pipeline: SFT → reward model → PPO; why it is heavy | DPO's whole reason to exist | Christiano et al. 2017; Ouyang et al. 2022 (InstructGPT) |
| DPO derivation: Bradley–Terry preferences, implicit reward `β·log(π/π_ref)`, the loss, the role of β (KL budget) | You must be able to write the objective on a whiteboard | Rafailov et al. 2023 |
| Variants: IPO, KTO (unpaired), ORPO/SimPO, and **GRPO/RLVR** for verifiable tasks | Modern vocabulary; tells you when DPO is the wrong tool | papers + TRL docs |
| Reference models with PEFT: adapter-disabled vs frozen copy, `precompute_ref_log_probs`, VRAM implications | Exactly where your bug lives; 4 GB makes this decisive | TRL DPOTrainer docs |
| Preference data: on-policy sampling, hard negatives, length/format bias, reward hacking, label noise | Your current pairs teach almost nothing | — |
| Diagnostics: margins, reward accuracy, entropy, KL-to-ref, `loss = log 2` meaning, gradient flow through a frozen 4-bit base | Turn a silent failure into a 5-minute diagnosis | — |

**Self-check:** derive why `margin = 0` implies `loss = log 2`. Why does DPO need a reference model at all? Why can `precompute_ref_log_probs` be *both* a
bug fix and a memory win? Why are execution-verified automatic labels better than hand-written rejected SQL? What can go wrong with only 20 preference pairs?

---

# Phase 5 — Quantization + quality-loss report (1 day)

> Goal: produce `artifacts/merged_w4a16/` and a table of **size / VRAM / latency / execution-accuracy** vs the fp16 checkpoint.
> Spec asks for a documented quality loss, not just a smaller file.

### 5.1 Merge first (`forge/quant/merge.py`)
Load base in fp16 + SFT adapter → `merge_and_unload()` → `save_pretrained("artifacts/merged_fp16")` + tokenizer.
On 4 GB, load on CPU (fp16 base ≈ 2.2 GB, 14 GB RAM) if the GPU path OOMs, then save.
**Equivalence check (cheap, catches merge bugs):** greedy-decode 10 prompts with (a) base+adapter and (b) merged checkpoint → outputs must match exactly.

### 5.2 Quantize with vLLM's own toolchain (`forge/quant/quantize.py`, `venv-quant`)
```bash
uv venv venv-quant && uv pip install llmcompressor==0.13.0
# W4A16 = 4-bit weights, 16-bit activations: GPTQ (Hessian-based) or AWQ (activation-aware scaling)
```
- Calibration set: **your own** `data/v1/train.jsonl` rendered prompts (128–256 samples, `max_seq_length 512`, batch 1) — never a random public corpus.
- `group_size 128`, symmetric, W4A16; save with `save_compressed=True` so vLLM can load it directly.
- **Hardware note you must put in the report:** sm_86 (Ampere) has **no FP8** tensor cores (that is sm_89+/Hopper/Blackwell), so FP8 or FP4 aren't options here;
  W4A16 weight-only is the right choice for a 4 GB memory-bound decode workload, W8A8/SmoothQuant is the accuracy-safer alternative if W4A16 degrades.
- Quantize the **base** model too, so the throughput comparison in Phase 6 has a matched (quantized-base vs quantized-tuned) baseline.

### 5.3 Measure (`forge/quant/measure.py`) → `reports/quant_report.md`
| Column | How |
|---|---|
| Disk size (GB) | file sizes of merged_fp16 vs merged_w4a16 |
| Peak VRAM at load | `nvidia-smi --query-gpu=memory.used --format=csv -lms 100` sampler around model load |
| Latency | TTFT + TPOT + p50/p95/p99 from Phase-6 load test at c=1/4/8 |
| Throughput | tokens/s aggregate (and $/1K tokens) |
| Quality | execution accuracy on `test_in` **and** `test_out`, with the bootstrap CI, fp16 vs w4a16 |
| Failure gallery | 5 examples that fp16 got right and w4a16 got wrong, with the error class |
Report the *relative* loss ("−2.1 pts exec accuracy for −68 % VRAM") and state the deployment recommendation. Also record whether
**the smaller model loses more relative quality** than a 7 B model would — that is an expected and reportable finding for a 1.1 B model.

### Prove it
`reports/quant_report.md` has the five columns above with numbers, the merge-equivalence check passes, and the served w4a16 model passes the Phase-3 parity check.

### Learn (≈8 h)
| Topic | Why | Study |
|---|---|---|
| Quantization basics: affine vs symmetric, per-tensor/channel/**group**, scale & zero-point, clipping vs outliers | You will tune `group_size` and explain the tradeoff | any quantization primer |
| GPTQ (layer-wise reconstruction with second-order info) vs AWQ (activation-aware scaling) vs SmoothQuant vs LLM.int8 | Pick methods for the right reason, not by default | Frantar et al. 2022 (GPTQ); Lin et al. 2023 (AWQ); Xiao et al. 2022; Dettmers et al. 2022 |
| NF4 / double quantization (training-time) vs W4A16 (serving-time) | Two different problems: memory for *training* vs memory/latency for *serving* | QLoRA paper |
| Weight-only vs activation quantization; why activations are harder (outliers) | Predicts where your accuracy loss appears | — |
| Calibration sets: size, domain match, seq length; how calibration biases the result | Your calibration data is a documented design decision | llm-compressor docs |
| Deployment formats: compressed-tensors, AWQ/GPTQ checkpoints, GGUF/llama.cpp, FP8/FP4 hardware requirements | vLLM can only serve what it supports; know the matrix | vLLM quantization docs |
| Perplexity vs task accuracy for quantization evaluation | Explains why you re-run the *task* harness | — |

**Self-check:** why is per-group (128) scaling better than per-tensor? Why would 4-bit activations hurt more than 4-bit weights? Why can a 1.1 B model
lose a larger *relative* accuracy fraction than a 7 B model at the same bit width? Why doesn't your Ampere GPU get FP8? Which of size / VRAM / TTFT / throughput
does W4A16 actually improve, and why might throughput barely move for a 1.1 B model?

---

# Phase 6 — Serving: vLLM + FastAPI + queue + metrics + dashboard (2–3 days)

> Goal: delete the empty `src/serve.py` and ship the thing the spec describes — continuous batching, autoscaling signal,
> cost/latency dashboard, and **explicit** handling of overload.

### 6.1 Environment and engine
```bash
uv venv venv-serve && source venv-serve/bin/activate
uv pip install "vllm==0.29.0" fastapi uvicorn prometheus-client httpx   # vLLM pins torch==2.13.0 -> separate venv

vllm serve artifacts/merged_fp16 --served-model-name forge-base \\
  --enable-lora --lora-modules sft=artifacts/sft_v1 dpo=artifacts/dpo_v1 \\
  --max-lora-rank 32 --max-loras 2 \\
  --gpu-memory-utilization 0.85 --max-model-len 1024 --max-num-seqs 8 \\
  --port 8000
```
| Flag | Why it is set that way on a 4 GB card |
|---|---|
| `--enable-lora --lora-modules` | One base model in VRAM, adapters swapped per request → **A/B testing and rollback with zero reload**, and it is exactly how production multi-tenant serving works |
| `--max-lora-rank 32` | Must be ≥ your max `r` (16 in Phase 2) |
| `--max-model-len 1024` | Bounds the KV pool: 22.5 KB/token → a 1024-token sequence needs ~23 MB of KV |
| `--max-num-seqs 8` | Caps concurrent sequences so the scheduler cannot OOM the 4 GB card |
| `--gpu-memory-utilization 0.85` | Leaves headroom for the desktop compositor; peak used is unpredictable otherwise |
| `--enforce-eager` (fallback) | If CUDA-graph capture OOMs, trade throughput for stability — document which you needed |
**Serving constraint you must state in the README:** bitsandbytes 4-bit checkpoints (your QLoRA training format) are *not* a servable
format with LoRA in vLLM — merge to fp16 (or use the W4A16 artifact from Phase 5) and load the adapter dynamically. That is the honest reason
Phase 5 exists.

### 6.2 Gateway (`forge/serve/app.py`)
- `POST /v1/chat/completions` (**OpenAI-compatible**) *and* `POST /v1/completions` + a thin `POST /generate`.
- **Template trap:** `TinyLlama-…-intermediate` is a *base* model with **no chat template**, so vLLM's `messages` path has nothing to apply —
  document that clients use `/v1/completions` (or send pre-rendered prompts), and build every prompt via
  `forge/data/prompt.py::render()`. A wrong template here silently destroys quality; the Phase-3 parity test is what catches it.
- **Streaming:** SSE passthrough with `stream=True`; propagate client disconnects as cancellation so the engine frees the slot
  (a dropped client must not keep generating).
- **Response headers:** `X-Request-Id`, `X-Queue-Time-Ms`, `X-Prefill-Ms`, `X-Prompt-Tokens`, `X-Completion-Tokens`, `X-Variant`, `X-Model`.
- **Accounting:** request usage via `stream_options={"include_usage": true}` (token counts are the input of the cost model).
- Deadline header (`X-Deadline-Ms`) honoured end-to-end; structured JSON logs with the request id.

### 6.3 Bounded queue + admission control (`forge/serve/queue.py`) — the “never silently drops” requirement
- `asyncio.Queue(maxsize=max_queue)` + `max_concurrency` workers; **bounded** means the system has a defined behaviour at overload, not a crash.
- Over capacity → `429 Too Many Requests` + `Retry-After` + body `{error, queue_depth, capacity, retry_after_ms}` and increment
  `rejected_total{reason="queue_full"}`. **Every rejection is explicit, counted and logged** — that is what the spec means.
- Deadline-aware: drop items whose deadline already passed → `504` + `rejected_total{reason="deadline_exceeded"}` (do not run work nobody waits for).
- Two lanes (interactive / batch) with weighted pickup → show that batch requests cannot starve interactive ones.
- Graceful shutdown: stop admitting → `503` + `Retry-After`, drain in-flight work, then exit (`SIGTERM` handler).
- `configs/serve.yaml`: `{max_queue, max_concurrency, request_timeout_s, max_new_tokens, lanes, deadline_ms, cost: {gpu_usd_per_hour}}`.

### 6.4 Metrics + dashboard (`forge/serve/metrics.py`, `forge/serve/dashboard.py`)
- `prometheus-client` on the gateway: counters `requests_total{variant,status}`, `rejected_total{reason}`, `tokens_total{variant,kind}`;
  histograms `queue_wait_seconds`, `ttft_seconds`, `latency_seconds`, `tokens_per_request`; gauges `queue_depth`, `in_flight`,
  `kv_cache_usage_perc`, `gpu_memory_used_bytes`.
- Merge vLLM's own Prometheus output (it exposes `vllm:*` metrics) by scraping `http://localhost:8000/metrics` in a background task and
  re-exposing a single `/metrics` endpoint — one scrape target for the whole system.
- **Cost gauge:** `cost_usd_per_1k_tokens = gpu_usd_per_hour / 3600 / (output_tokens_per_second) * 1000`, documented as a
  *cloud-equivalent normalized* figure (you own the laptop; say so). Also report cost per *successful* request (failures are not free).
- `/stats` JSON + `/dashboard` (single self-contained HTML page, no build step): live p50/p95/p99 latency, tokens/s, queue depth,
  request & rejection rates, $/1K tokens, error rate — screenshot it for the README.
- **Percentiles:** expose Prometheus histograms for aggregates *and* keep an in-process rolling window for exact p50/p95/p99 in the dashboard
  (histogram quantiles are approximations — be able to explain the difference; Grafana/Prometheus in Docker is optional and runs CPU-only here).

### 6.5 Load testing and the serving benchmark (`forge/serve/loadtest.py`) → `reports/serving_bench.md`
- **Closed-loop sweep first:** concurrency c ∈ {1, 2, 4, 8, 16}, 200 requests each over a fixed prompt mix (short/medium), report requests/s,
  aggregate output tokens/s, TTFT p50/p95, end-to-end p50/p95/p99, 429/504 counts, and the queue-depth timeline.
- **Then open-loop (Poisson arrivals at a fixed rate)** to find the saturation knee: raise the offered rate until p99 breaks the SLA (e.g. 3 s).
  Closed-loop tests *cannot* see this — slow clients hide queueing (coordinated omission). Being able to explain that distinction is a strong signal.
- **Baselines on identical hardware (spec requirement):**
  1. HF `transformers.generate()` batch-1 *and* naive static batching, for a handful of prompts, vs vLLM at c=1/4/8 — expect a
     large gap once concurrency > 1 (continuous batching vs static batching);
  2. **untuned base** vs SFT-LoRA vs DPO-LoRA vs W4A16, all on vLLM: the LoRA-overhead number (usually single-digit %) and the
     quantization number from Phase 5 belong in the same table;
  3. cold start (model load) vs warm throughput.
- Overload run: 500 requests at 3× capacity → assert **0 silent drops** (every response is 200, 429 or 504, each counted).

### 6.6 Tests (`tests/`, CPU-only ones must run in CI)
- `test_queue.py`: queue-full → 429 with `Retry-After`; expired deadline → 504; shutdown → 503 then drain.
- `test_api_contract.py`: schema of `/v1/chat/completions` + `/v1/completions` responses (usage fields, headers, `X-Variant`), engine **stubbed**
  so it runs without a GPU.
- `test_metrics.py`: counters/histograms move after a stubbed request; `/stats` percentile maths on a synthetic latency list.
- One GPU-marked integration test: real engine, 3 prompts, asserts the parity check from Phase 3.

### Prove it
`/dashboard` shows live tokens/s, p50/p95/p99, queue depth and $/1K tokens (screenshot in the README); `reports/serving_bench.md` compares
base vs tuned vs quantized and HF vs vLLM; the 500-request overload test reports **zero dropped requests** with an explicit rejection histogram.

### Learn (≈16 h — this is the phase with the most genuine engineering depth)
| Topic | Why | Study |
|---|---|---|
| Prefill vs decode; compute-bound vs **memory-bound** decode; arithmetic intensity / roofline | Explains why batching helps decode and why quantizing *weights* affects speed at all | any GPU-perf primer |
| KV cache: size maths (your 22.5 KB/token), MQA/GQA, growth with batch×seq, why it is the memory bottleneck | You will defend every engine flag with this maths | vLLM docs; GQA (Ainslie et al. 2023) |
| **PagedAttention**: block tables, fragmentation, prefix sharing; prefix caching; chunked prefill | The core technical claim of your serving stack | Kwon et al. 2023 (vLLM, SOSP) |
| **Continuous batching** vs static batching (iteration-level scheduling) | Why HF `generate` collapses at concurrency > 1 | Yu et al. 2022 (Orca) |
| TTFT vs TPOT vs end-to-end latency; throughput vs latency tradeoff; goodput | Correct metric vocabulary for the dashboard | — |
| Queueing theory: Little's law, M/M/1, utilization vs latency knee, why p99 explodes near saturation | Required to design the queue and the autoscaling signal | — |
| Admission control, backpressure vs load shedding, deadline scheduling, graceful degradation | The "never silently drops" requirement | — |
| asyncio/FastAPI: event-loop blocking, async generators, SSE, cancellation propagation, worker model | Your gateway must not block the loop while streaming | FastAPI/Starlette docs |
| Prometheus: counter/gauge/histogram/summary, quantile estimation limits, cardinality, RED/USE | Metrics that are actually right | Prometheus docs |
| Load-test methodology: open vs closed loop, coordinated omission, warmup, steady state | Prevents publishing a meaningless benchmark | Gil Tene, “How NOT to Measure Latency” |
| Cost modelling: GPU-hour amortization, $/1K tokens, cost per successful request, utilization | The dashboard's headline business number | — |
| OpenAI-compatible API surface: chat vs completions, chat templates, usage accounting, streaming usage | Your clients and interviewers both expect this contract | OpenAI API reference; vLLM OpenAI-server docs |
| Optional depth: speculative decoding, CUDA graphs, tensor parallel | Bonus points; know they exist and when they apply | — |

**Self-check:** compute KV bytes/token and the number of concurrent 256-token sequences that fit in 1 GB of KV. Why does static batching waste GPU when
sequence lengths differ? Why does p99 latency spike *before* throughput saturates? Why is returning `429` better than silently dropping, and what must be
counted either way? Why can't the 4-bit training checkpoint be served directly with LoRA? Why does the base model change what the *API contract* looks like?

---

# Phase 7 — A/B serving + autoscaling control loop (1–2 days)

> Goal: route a percentage of traffic to a new checkpoint, compare live metrics, and have a documented promotion/rollback rule.
> Plus a **queue-depth-driven autoscaling control loop** that is real code, honestly labelled as simulated where the host prevents reality.

### 7.1 A/B routing (`forge/serve/router.py`)
```yaml
# configs/serve.yaml
experiment:
  id: ab_sft_vs_dpo
  variants: [{name: sft, weight: 0.90, lora: sft}, {name: dpo, weight: 0.10, lora: dpo}]
```
- Assignment: `bucket = int(sha256(f"{experiment_id}:{user_key}")[:8], 16) % 10000` → variant.
  **Sticky by user** (same caller always sees one variant) so the comparison is not contaminated by within-user mixing; fall back to `request_id` when no user key.
- Dev-only override header `X-Force-Variant: dpo` for smoke tests, disabled in "prod" mode by config.
- Multi-LoRA makes this nearly free: both variants live on the **same engine and the same KV pool**, so the A/B measures the checkpoint difference, not two different machines.
- **Online metrics per variant:** requests, error/429 rate, latency percentiles, tokens/s, and two task-relevant signals you can actually observe without labels:
  1. **valid-SQL rate** computed by sampling ~5 % of responses through the Phase-3 executor;
  2. **human/machine feedback** (`POST /feedback {request_id, rating}` → SQLite) exposed as `feedback_rate{variant}` — the same thumbs-up/down idea as Codewatch's review feedback, which is also where a *real* preference dataset would come from in production.
- **Promotion rule (write it down and honour it):** promote only if (a) ≥200 requests per variant, (b) p95 latency ≤ 1.15 × baseline,
  (c) error rate ≤ baseline + 0.5 pt, (d) valid-SQL rate ≥ baseline, (e) the offline Phase-3 result holds.
  Roll back automatically on any guardrail breach; log the decision with the numbers into `reports/ab_decision.md`.
- **Be honest about power:** at 10 % traffic, detecting a 2-point accuracy difference needs thousands of requests (do the arithmetic and show it).
  So the offline harness is the *primary* evidence and online A/B is the *guardrail* mechanism. Saying that explicitly is worth more than pretending you proved a lift.

### 7.2 Autoscaling control loop (`forge/serve/autoscale.py`)
- **Signal:** scale-out when `queue_wait p95 > 250 ms` **or** `queue_depth > 2 × max_concurrency` sustained for 30 s; scale-in when `queue_depth == 0`
  and `in_flight < 50 %` for 120 s. Hysteresis + cooldown are mandatory (otherwise the loop flaps between 1 and 2 replicas).
- **`ReplicaProvider` interface:** `list() / scale_to(n) / health()`, with three implementations and a config switch:
  | Provider | Reality on this host |
  |---|---|
  | `LocalProvider` | Real: manages worker processes/ports; capped at **1 GPU replica** because there is one 4 GB GPU — it logs the cap instead of pretending |
  | `SimulatedProvider` | Fast-forwards synthetic replica capacity to exercise the control loop, quotas and failure paths; used for the README chart, **labelled simulated** |
  | `DockerProvider` | Implemented but **disabled**: `docker run --gpus all` fails here (`no known GPU vendor found (CDI)`) |
- Expose `replicas_desired`, `replicas_active`, `scale_events_total{reason}`; save the **control-loop chart** (queue depth + desired/active replicas vs time) to `reports/figures/autoscale.png`.
- **GPU-aware graceful degradation** (the part that actually matters on one GPU): when the engine reports KV-cache usage > 95 %, first shrink `max_new_tokens`
  for the batch lane, then shed batch-lane requests with explicit `429` while interactive traffic keeps its SLA. Uniform failure is the anti-pattern; lane-aware degradation is the goal.
- **Production vocabulary to be able to discuss:** Kubernetes HPA with custom metrics, KEDA queue-depth scalers, target-tracking vs step scaling, cooldown periods,
  scale-to-zero vs cold-start (a model replica must materialize weights: 10–30 s here), and why GPU scaling is coarser than CPU scaling.

### Prove it
Two variants served simultaneously with per-variant dashboards; a written `ab_decision.md` with the numbers and the decision; `autoscale.png` from an overload
scenario; and a load test where the system degrades **loudly and correctly** (429s confined to the batch lane, interactive p95 still inside SLA).

### Learn (≈10 h)
| Topic | Why | Study |
|---|---|---|
| Experimentation: randomisation unit, stickiness, guardrail metrics, sample size/power, sequential-testing/peeking pitfalls, novelty effects | You must know when an online A/B *cannot* answer your question | Kohavi et al., *Trustworthy Online Controlled Experiments* |
| Deployment patterns: canary, shadow, blue-green, feature flags, rollback | A/B through multi-LoRA is a canary at the model layer | — |
| SLO / SLI / error budgets; latency budgets per hop | Gives your dashboard meaning | Google SRE book |
| Control theory for autoscalers: target tracking, hysteresis, damping, cooldown, oscillation | Directly shapes `autoscale.py` | HPA/KEDA docs |
| Queue-based vs utilisation-based scaling; why queue depth beats CPU% for inference | The spec asks for queue-depth scaling — know the argument | KEDA + vLLM production guides |
| Cold start / scale-to-zero tradeoff; replica placement; model loading cost | 10–30 s per replica changes the whole scaling policy | — |
| MLOps lifecycle: registry → deploy → monitor → rollback; shadow evaluation | Ties Phases 2–7 into a lifecycle story | — |

**Self-check:** why is per-user stickiness better than per-request randomisation? Roughly how many requests does a 2-point accuracy difference need at 90/10 traffic
(and what is your plan in the meantime)? Why does a queue-depth scaler need cooldown and hysteresis? Why is one GPU replica a coarse scaling unit, and what does that
imply for scale-to-zero? Why is lane-aware degradation better than uniform rejection?

---

# Phase 8 — Make it legible: report, CI, packaging, numbers (1–2 days)

> A 10/10 project is one a stranger can verify in 10 minutes. Everything you built becomes invisible without this phase.

### 8.1 `README.md` (rewrite the 1-line file) — fixed section order
1. **One-paragraph pitch** + architecture diagram (ASCII ok: data → train → eval → quant → serve → A/B).
2. **Results table** (the money shot): base vs SFT vs DPO vs w4a16 × `test_in` / `test_out`, execution accuracy **with 95 % CI**, plus the
   quantized-model quality delta and the latency/VRAM columns.
3. **Serving benchmark table**: HF-`generate` vs vLLM at c=1/4/8 (tokens/s, TTFT, p95), base vs tuned, LoRA overhead %, 429/504 under overload.
4. **Dashboard + control-loop screenshots** (cost/latency and autoscale charts).
5. **Quickstart**: 6 commands that reproduce one number end-to-end (`make data && make train && make eval`).
6. **Honest limitations**: 1.1 B model, English-only single-statement SQL, one schema family, single 4 GB GPU, autoscaling provider simulated and labelled.
7. **What I'd do next** (bigger base model, GRPO with execution rewards, context-length scaling, real traffic, multi-GPU).

### 8.2 CI (`.github/workflows/ci.yml`) — CPU-only, must be green
`ruff` + `pyright` (you already have `pyrightconfig.json`), `pytest -m "not gpu"`, a **data-build determinism job** (build twice → identical manifest hashes),
and the API-contract tests with a stubbed engine. GPU steps cannot run in free CI: document the exact manual commands + expected outputs instead,
and attach the produced reports as artifacts in the release.

### 8.3 Packaging, model cards, release
- Model cards for `sft_v1`, `dpo_v1`, `merged_w4a16`: training data & version hash, hyperparameters, eval table, intended use, limitations, licence.
- `data/v1/DATASET_CARD.md`: motivation, collection/verification process, splits, known biases, licence.
- Push adapter + merged fp16 to the HF Hub (public, verifiable, free) and tag a GitHub release `v1.0` with a CHANGELOG and the reports.
- `make repro` → full pipeline on a fresh clone, with the measured wall-clock documented per step for this hardware.

### 8.4 `reports/REPORT.md` — a mini paper
Problem → data (with the curation funnel counts) → method (LoRA maths, config table) → results (tables with CIs, per-slice) →
ablations (r, targets, lr, QLoRA vs bf16-LoRA, DPO vs SFT) → quantization tradeoff → serving analysis (incl. a queueing explanation of the p99 knee)
→ what failed and why (the original no-op DPO, the leaked 5-prompt eval) → reproduction commands.

### 8.5 The resume bullet, now with real numbers
> *Fine-tuned TinyLlama-1.1B with QLoRA (r=16, all-linear) on a curated 400-pair schema-conditioned text-to-SQL dataset; built a SQL-execution
> evaluation harness showing **+X pts execution accuracy over base (test-out: A % vs B %, 95 % CI)**; cut serving VRAM by **Z %** with W4A16
> (exec-acc −Q pts); served via vLLM with continuous batching at **T tokens/s, p95 = P ms, $C/1K tokens**, and rolled out a new checkpoint
> through a 10 % multi-LoRA A/B canary with queue-depth-driven autoscaling and zero dropped requests under 3× overload.*
Every placeholder must be filled from a report file — never from memory.

### Learn (≈6 h)
| Topic | Why | Study |
|---|---|---|
| Technical writing for engineering artefacts: claim-first tables, method → results → limitations | This phase *is* the deliverable | any strong systems paper you like; Anthropic/OpenAI evals write-ups |
| Chart design: one message per chart, axis labels, units, error bars | Your figures are the first thing a reviewer looks at | — |
| Model Cards (Mitchell et al. 2019) and Datasheets for Datasets (Gebru et al. 2018) | Required for responsible ML, and it looks professional | papers |
| Licensing & provenance for models and data | You are republishing weights and a dataset | HF docs, Apache-2.0/MIT/CC-BY |
| CI/CD basics: workflow files, caching, matrix jobs, artifact upload | Green CI is the credibility signal | GitHub Actions docs |
| Semantic versioning for datasets/models; changelogs | `v1`/`v2` discipline you already committed to | semver.org |

**Self-check:** can a stranger reproduce your headline number with two commands? Does every number in the README trace to a file in `reports/`?
Did you report a negative result honestly somewhere? What is the smallest hardware a reader needs to verify your claims?

---

# Appendix A — Command cheat sheet (the whole project in ~15 commands)
```bash
# Phase 0  (once)
uv venv venv && uv pip install -e ".[dev]" && make test
# Phase 1
make data                                   # build + dedupe + split + manifest + leakage test
# Phase 2
uv pip install mlflow
forge-train --config configs/sft/r16_alllinear.yaml         # one run
forge-train --config configs/sft/sweep.yaml --sweep         # overnight sweep
mlflow ui --backend-store-uri sqlite:///mlflow.db &         # compare runs
# Phase 3
forge-eval --models base,artifacts/sft_v1 --split test_in,test_out --out reports/eval_v1
# Phase 4
forge-train --config configs/dpo/v1.yaml --make-prefs --on-policy
# Phase 5
uv venv venv-quant && uv pip install llmcompressor
forge-quant --config configs/quant/w4a16.yaml --calib data/v1/train.jsonl --out artifacts/merged_w4a16
# Phase 6
uv venv venv-serve && uv pip install "vllm==0.29.0" fastapi uvicorn prometheus-client
vllm serve artifacts/merged_fp16 --enable-lora --lora-modules sft=artifacts/sft_v1 dpo=artifacts/dpo_v1 \\
  --max-lora-rank 32 --max-model-len 1024 --max-num-seqs 8 --gpu-memory-utilization 0.85 --port 8000 &
forge-serve --config configs/serve.yaml        # gateway on :8080, dashboard at /dashboard, metrics at /metrics
forge-bench --concurrency 1,2,4,8,16 --requests 200 --out reports/serving_bench.md
# Phase 7
curl -N localhost:8080/v1/completions -H 'X-User: u1' -d '{"prompt":"...","max_tokens":64}'   # watch X-Variant
forge-autoscale --scenario overload --provider simulated --out reports/figures/autoscale.png
```

# Appendix B — VRAM and time budget on this machine (RTX 3050 Laptop, 3.95 GB)
| Workload | Weights | Extra | Verdict | Time (measured/estimated) |
|---|---|---|---|---|
| QLoRA SFT (4-bit, r=16, all-linear, seq 512, bs 2, grad ckpt) | ~0.75 GB | activations + grad-accum + fp32 adapter | ✅ fits (verify at bs 2, else bs 1) | ~15–30 min / 4 epochs on 400 rows |
| Plain bf16 LoRA (no quantization, seq 512, bs 1) | ~2.2 GB | small optimizer + activations | ⚠️ try it — record success/OOM | ~10–20 min |
| DPO (4-bit policy, `precompute_ref_log_probs`) | ~0.75 GB | one model at a time | ✅ fits | ~10 min / 2 epochs on 400–800 pairs |
| DPO *without* precompute (policy + ref) | ~1.5 GB+ | two models | ⚠️ risky on 4 GB — another reason to use the flag | — |
| Merge fp16 adapter (CPU) | 2.2 GB RAM | — | ✅ 14 GB RAM is enough | ~1 min |
| W4A16 quantization (llm-compressor, CPU/GPU) | ~2.2 GB during calibration | calibration activations | ✅ fits; use 128–256 samples @ 512 tokens | ~10–25 min |
| vLLM fp16 + multi-LoRA + KV | 2.2 GB | ~1.1 GB KV → ~50 k tokens | ✅ `--max-model-len 1024`, `--max-num-seqs 8` | load 10–30 s |
| vLLM W4A16 + KV | ~0.75 GB | ~2.5 GB KV | ✅ more concurrency, better $/1K tokens | load 10–30 s |
| Evaluation run (60 prompts × 128 new tokens, greedy) | as served | — | ✅ few minutes per checkpoint | — |
| Overnight sweep (12 cells) | as training | — | ✅ run with `nohup`, log to MLflow | 3–6 h total |

---

# Appendix C — Traps this project must avoid (first five are already in the repo)
| # | Trap | Symptom | Guard |
|---|---|---|---|
| 1 | **Evaluating on training data** | 5/5 eval prompts are in `dataset.jsonl` | `tests/test_no_leakage.py`, separate `test_out` schema |
| 2 | **Silent no-op training** | DPO loss = log 2, margins 0, tensors unchanged | `check_training_delta.py` + margin/grad-norm gates |
| 3 | **Selecting by loss, not by task metric** | beautiful loss curve, wrong SQL | select on dev **execution accuracy** |
| 4 | **Truncation eating the answer** | `max_length 256` with schema+question+answer | report token percentiles, set `max_length ≥ p99 + 32` |
| 5 | **Empty/unfinished deliverable** | `src/serve.py` is 0 bytes | the scorecard table in section 1 |
| 6 | **Template drift** train ≠ eval ≠ serve | quality collapses only in production | single `render()` function + parity test |
| 7 | **Serving the wrong artefact** | QLoRA 4-bit + LoRA is not a servable combo in vLLM | merge to fp16 / W4A16, document why |
| 8 | **Assuming a chat template exists** | base TinyLlama has none; `messages` payloads misbehave | use `/v1/completions` with your own template, and test it |
| 9 | **Random calibration data** for quantization | unexplained accuracy loss | calibrate on `data/v1/train.jsonl` |
| 10 | **Benchmarking at concurrency 1 only** | vLLM looks useless, continuous batching unproven | always sweep c ∈ {1,2,4,8,16} |
| 11 | **Point estimates on tiny n** | "+3 %" from 40 examples | bootstrap CI + McNemar |
| 12 | **Non-deterministic evaluation** | numbers move between runs, nobody can reproduce | greedy decoding + seeds + manifest |
| 13 | **Unlabelled simulation** | fake autoscaling chart | `provider: simulated` printed on the figure and in the README |
| 14 | **Dataset not versioned/committed** | fresh clone cannot rebuild anything | `data/v1/` tracked + `manifest.json` |
| 15 | **Checkpoint sprawl** | `save_total_limit: 2`, prune stale runs, watch disk | keep only best + last per experiment |
| 16 | **Anecdote-driven tuning** | “this prompt got better” | error taxonomy counts, per-slice tables |

# Appendix D — Consolidated reading list (one priority item per phase)
| Phase | If you read/watch one thing | Then depth |
|---|---|---|
| 0 | Python packaging guide (`pyproject.toml`, src layout) | GitHub Actions docs |
| 1 | Data curation for SFT: LIMA (Zhou et al. 2023) + Spider (Yu et al. 2018) | MinHash/LSH (Broder 1997), BIRD (Li et al. 2023), Zhong et al. 2020 on execution-based eval |
| 2 | LoRA (Hu et al. 2021) + QLoRA (Dettmers et al. 2023) | TRL/PEFT docs, Bergstra & Bengio 2012 (random search), ASHA |
| 3 | Zhong et al. 2020 (execution accuracy) + bootstrap CI chapter | McNemar 1947, MT-Bench (Zheng et al. 2023), HELM (Liang et al. 2022) |
| 4 | DPO (Rafailov et al. 2023) | InstructGPT (Ouyang et al. 2022), IPO/KTO/SimPO, GRPO/RLVR |
| 5 | GPTQ (Frantar et al. 2022) + AWQ (Lin et al. 2023) | SmoothQuant (Xiao et al. 2022), LLM.int8 (Dettmers et al. 2022), llm-compressor docs |
| 6 | vLLM / PagedAttention (Kwon et al. 2023) + Orca (Yu et al. 2022) | queueing theory basics, Prometheus metrics docs, Gil Tene on latency measurement |
| 7 | *Trustworthy Online Controlled Experiments* (Kohavi et al.) | HPA/KEDA docs, Google SRE (SLOs) |
| 8 | Model Cards (Mitchell et al. 2019) + Datasheets (Gebru et al. 2018) | semver, GitHub Actions artifact patterns |

# Appendix E — Glossary (be able to define each in one sentence)
**Training:** SFT · LoRA (r, α, target modules, `ΔW = BA`) · QLoRA (NF4, double quant) · adapter · packing · EOS · completion-only loss ·
epochs vs steps · effective batch size · gradient accumulation · gradient checkpointing · warmup · cosine schedule · early stopping ·
catastrophic forgetting · memorisation vs generalisation · MLflow run/params/metrics · config hash · run manifest.
**Eval:** execution accuracy · exact match · valid-SQL rate · perplexity · pass@k · slice/stratum · error taxonomy · bootstrap CI ·
McNemar test · leakage (prompt/answer/schema/near-duplicate) · contamination · LLM-as-a-judge bias · parity test.
**Quantization:** W4A16 / W8A8 · group size · symmetric vs affine · scale/zero-point · GPTQ · AWQ · SmoothQuant · calibration set ·
compressed-tensors · GGUF · FP8 (sm_89+) · quality-loss report.
**Serving:** prefill vs decode · compute- vs memory-bound · KV cache · GQA/MQA · PagedAttention · block table · prefix caching ·
continuous batching · static batching · TTFT · TPOT · throughput vs latency · goodput · queue depth · admission control · backpressure ·
load shedding · graceful degradation · Little's law · saturation knee · coordinated omission · SLO/SLI · multi-LoRA · canary · rollback ·
$/1K tokens.

# Appendix F — 15-day schedule and where to start tomorrow
| Day | Work | Evidence produced |
|---|---|---|
| 1 | Phase 0 hygiene + fixture DBs + schema-conditioned prompt template | clean `make test`, two SQLite fixtures |
| 2–3 | Phase 1 curation: 400 train / 60 dev / 60 test-in / 40 test-out, dedupe, manifest, dataset card | `data/v1/` + passing leakage test |
| 4 | Phase 3 harness **before** training (build the ruler first) | `reports/eval_base.md` — base model baseline number |
| 5 | Phase 2 rewrite trainer, one sanity run, verify pipeline end-to-end | first honest exec-accuracy number |
| 6 | Phase 2 sweep (12 cells, overnight) | MLflow table, winner chosen by dev exec acc |
| 7 | Phase 3 full matrix on the winner + parity/prep for serving | `reports/eval_v1.md` with CI |
| 8 | Phase 5 merge + W4A16 + quality-loss measurement | `reports/quant_report.md` |
| 9–10 | Phase 6 vLLM engine + gateway + queue + metrics + dashboard | `/dashboard` screenshot, overload test with 0 silent drops |
| 11 | Phase 6 load tests (closed + open loop) vs HF baseline | `reports/serving_bench.md` |
| 12 | Phase 4 DPO fixed with on-policy pairs (or GRPO) + evaluation | `rewards/margins ≠ 0`, delta vs SFT |
| 13 | Phase 7 A/B rollout + autoscale control loop | `ab_decision.md`, `autoscale.png` |
| 14 | Phase 8 README/report/CI/model cards/HF release | public, verifiable project |
| 15 | Buffer: fix what broke, fill scorecard, write the resume bullet from the reports | final `REPORT.md` |

### Start here tomorrow (in this exact order)
1. `uv venv venv-serve` is irrelevant yet — **first fix the ruler**: build the fixture DBs and the schema-conditioned prompt template (Phase 1.1–1.2).
2. Write `forge/eval/harness.py` + `metrics.py` and get a **base-model number on 60 held-out prompts** (Phase 3). You cannot improve what you cannot measure,
   and you currently have no valid measurement at all.
3. Rebuild the dataset to 400/60/60/40 with a leakage test (Phase 1.3–1.5).
4. Re-train with the Phase-2 config (all-linear, r=16, α=32, lr 2e-4, 4 epochs, dev early stopping) and compare on the harness.
5. Only then touch serving, quantization and DPO — in that order.

### The 10/10 promise to yourself
A phase is done when its **evidence file** exists and a stranger can reproduce it with `make <target>`.
If a number is not in `reports/`, it does not exist — and it must not appear in the README or on a resume.
