# PROJECT_STATE.md — session memory bank (keep ≤40 lines, update at the end of each session)

**Goal:** Forge — fine-tune, quantize and serve TinyLlama-1.1B for schema-conditioned Text → SQLite SQL,
with a task-specific execution-accuracy harness, vLLM serving, A/B rollout and a cost/latency dashboard.

**Hardware/constraints:** RTX 3050 Laptop, 3.95 GB VRAM, 14 GB RAM · Python 3.14.7 (`uv`)
· Docker has no GPU/CDI → vLLM native · vLLM 0.29 pins torch 2.13 → `venv-serve/`
· `venv/` = trl 1.13.0, transformers 5.17.0, peft 0.20.0, torch 2.14.0+cu130.

**Verified state (2026-09-20, this session):**
- Fixtures frozen — do **not** re-run `src/db.py` (unseeded Faker changes every gold result): `orders.db` md5
  `abe7caa85cd83651f95166057b7bf45a` (`user, category, shop, item, user_orders`); `cars.db` md5 `5cf91af6fc22c35d6513cbd117816f04`
  (`user, dealership, car, rental, maintenance_log`). Old `oreders.db` (typo) renamed to `orders.db`, content identical (md5).
- `datasets/`: `train.jsonl` 325 · `dev.jsonl` 60 · `test_in.jsonl` 60 (db_id `orders`) · `test_out.jsonl` 73 (db_id `cars`)
  · `sqldata.jsonl` 445 = orders pool (train∪dev∪test_in), **never train on it** · `README.md` = file map + contract + md5s.
  Row fields `id, db_id, question, context, query, pattern, difficulty, tables, source, q_norm_sha256, context_sha256`;
  `context` = exact `sqlite_master` DDL, only the tables that query needs; 7 pattern buckets, ≥7 rows per split.
- Machine-verified by re-executing the finished files: gold SQL runs on its fixture and returns ≥1 row; `context` byte-matches
  the fixture DDL; 0 question / 0 gold-SQL overlap across the four files; 0 cross-schema identifiers; LIMIT always tiebroken.
- Token budget (real TinyLlama tokenizer, prompt+answer): train p50 151 / p95 306 / max 350 · test_out max 399 → `max_length 512`
  (the old 256 truncated ~11 % of rows).
- Old audit still true: `final_model/` r=8 q/v only, 100 epochs on 50 rows; `dpo_model/` no-op (loss log 2, tensor diff 0.0); `src/serve.py` 0 bytes; `outputs/` empty.

**Current phase:** Phase 1 partly done — 1.1 fixture DBs ✔, 1.2 curated data ✔.
**Next action:** write `forge/data/prompt.py::render(schema_ddl, question)` (single frozen template with
`### Schema:` / `### Prompt:` / `### Response:`), then `forge/eval/{harness,metrics}.py` and get the **base model's**
execution accuracy on dev / test_in / test_out. Not before that: training, DPO, quantization, serving.

**Evidence:** `datasets/README.md` exists (verified-properties section). Still missing: `reports/data_v1_build.md`, `datasets/manifest.json`, `tests/test_no_leakage.py`, then `reports/eval_v1.md`, `reports/quant_report.md`, `reports/serving_bench.md`, `reports/ab_decision.md`, `reports/figures/autoscale.png`.

**Decisions locked:** metric = execution accuracy (+ valid-SQL rate) on `test_in`/`test_out`, base model always in the same
table; field names `question`/`context`/`query` (Spider/BIRD style) via one `render()`; train trains, dev selects, test_in/test_out
only report; `max_length 512`; SFT epochs 4 (not 100), lr 2e-4, bf16, completion-only loss; LoRA r=16 α=32 all-linear.

**Open risks:** the data generator / splitter / verifier scripts live in `/tmp` (lost on reboot — the split rule must be
committed as `forge/data/split.py`); `sqldata.jsonl` is a superset and a leakage trap if `train.py` globs `datasets/*.jsonl`;
`user` is a table name in both fixtures (DDL otherwise disjoint); 325 train rows is the accuracy ceiling, not hyperparameters;
autoscaling on this 1-GPU host must be labelled simulated.
