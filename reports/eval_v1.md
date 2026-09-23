# Eval v1 — dev slice (selection split, n=60)

**STATUS: INCOMPLETE — accuracy numbers pending.** The trained adapter exists; no execution-accuracy
number has been produced yet. Phase 2 closes only when the two `?` cells below are filled with real
measurements from the same command shape.

Everything below is either a measured fact (with the command that produced it) or a placeholder.
Nothing here may be filled from memory.

---

## 1. Provenance (frozen snapshot, do not edit)

Dataset hashes — `sha256sum datasets/*.jsonl`:

| file | rows | sha256 |
|---|---|---|
| `datasets/train.jsonl` | 325 | `6475c592ecec1ee72957d28d85906c38c661eec24fc05c3396c7fb3331c7b8dd` |
| `datasets/dev.jsonl` | 60 | `4ca733590aaac27b9959d6bff37984071ad9a8e2be6549f1705e44ad32a019a6` |
| `datasets/test_in.jsonl` | 60 | `778f842df2fba6d3ac7f373615499743adf11de0d4a9b1d0fe6b4b1a7dad1d30` |
| `datasets/test_out.jsonl` | 73 | `17e12db41eb49de51d1b765bd140bc6186e5caf58f866339881d9471c174888d` |
| `datasets/sqldata.jsonl` | 445 | `d4ea797fdfdc5209eb494d713b2c0eb6f6199ca8fb767a1b864567fddcbeb1ed` |

Fixtures — `md5sum orders.db cars.db`: `orders.db` `abe7caa85cd83651f95166057b7bf45a`,
`cars.db` `5cf91af6fc22c35d6513cbd117816f04`. Gold results depend on these bytes; re-running
`src/db.py` (unseeded `Faker()`) invalidates every gold query.

`datasets/sqldata.jsonl` is a superset of train + dev + test_in. It must never be read by the
trainer — only `train.jsonl` and `dev.jsonl`.

## 2. Run under test

| field | value |
|---|---|
| adapter path | `artifacts/runs/2026-09-23_r16_alllinear` (currently still at `artifacts/sft_v1`) |
| base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| LoRA | r=16, alpha=32, dropout=0.05, bias=none, `target_modules` = q,k,v,o,gate,up,down (all-linear) |
| trainable params | 12,615,680 (evidence: `adapter_model.safetensors` = 50,503,544 B ÷ 4 bytes per fp32 param) |
| training | 4 epochs, 84 steps, batch 2 × grad-accum 8, max_length 512, bf16, 4-bit NF4 + double quant |
| seed | 42 |
| loss curve (`checkpoint-84/trainer_state.json`) | train 0.404 → 0.017; eval 0.1039 → 0.0689 → 0.0533 → 0.0533 |
| best metric recorded by trainer | `eval_loss` 0.0533 (epoch 4) |
| dataset | `datasets/train.jsonl` (sha256 above) |
| git SHA | `75b0c1edef46e477f9ae21ced23305276d26f6f5` (working tree dirty) |
| env | torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, trl 1.13.0, bitsandbytes 0.50.2 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |

Reproduce the run: `python3 -m src.train.train` (config inlined in `src/train/train.py`; move it to
`configs/sft/r16_alllinear.yaml` before the next run).

**Loss is not a result.** This repo previously produced a model with a `WHERE`-less answer on a
memorised prompt. Only the table in section 3 counts.

## 3. Execution accuracy — dev (the Phase 2 gate)

Metric: SQL **execution accuracy** — run the predicted statement on `orders.db`, compare the result
set to the result set of the gold `query`. Order-insensitive unless the gold has `ORDER BY`.
Invalid / non-executable predictions count as wrong and are also reported separately as the
valid-SQL rate. Prompt built with `render()` from `src/data/prompt.py` for **all** models — never a
retyped template.

| model | execution accuracy | valid SQL rate | n |
|---|---|---|---|
| base TinyLlama-1.1B | `?` | `?` | 60 |
| `sft r16 all-linear` (checkpoint 84) | `?` | `?` | 60 |

Filled by:

```bash
python3 -m src.eval.exacc --model base --split datasets/dev.jsonl
python3 -m src.eval.exacc --model artifacts/sft_v1 --split datasets/dev.jsonl
```

Gate: `sft_acc > base_acc`. If not, Phase 2 is open; the report then states the null result and the
diagnosis (see section 5).

## 4. Checkpoint selection

Checkpoint choice must be justified by dev execution accuracy, not by `eval_loss`. Training used
`metric_for_best_model="eval_loss"` with `load_best_model_at_end=True`, so the saved adapter is the
epoch-4 checkpoint because eval loss plateaued (0.0533 at epoch 3 and 4). Record here which
checkpoints were scored and what each scored:

| checkpoint | dev execution accuracy |
|---|---|
| `checkpoint-63` | `?` (optional — only if it differs from the final one) |
| `checkpoint-84` | `?` |

## 5. Failure analysis (fill after the run)

For the first few wrong predictions: question, gold SQL, predicted SQL, and the category
(wrong `WHERE`, missing `JOIN`, wrong aggregate, truncated, invalid SQL). This is what makes the
number credible, and it feeds Phase 3's error taxonomy.

| # | question | gold | predicted | category |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

## 6. Definition of done (Phase 2)

- [ ] Both cells in section 3 measured, same 60 dev rows, same prompt template.
- [ ] `sft_acc > base_acc`.
- [ ] `run_manifest.json` written next to the adapter (config, dataset sha256s, seed, git SHA, versions, GPU, wall-clock, dev accuracy).
- [ ] Same-seed re-run reproduces the dev number, or the nondeterministic kernels are named here.
- [ ] `artifacts/runs/2026-09-23_r16_alllinear` exists (run folder renamed, no reuse of `artifacts/sft_v1` for future runs).

## 7. Not in scope here

Full matrix over `test_in` / `test_out`, confidence intervals, per-pattern slices, DPO and
quantized comparisons, and the template-parity test belong to Phase 3 and reuse this same harness
file. MLflow, CI and model cards belong to Phase 8.

