# Eval v1 — base vs SFT on dev, test_in (orders) and test_out (cars, unseen schema) — measured 2026-09-25

**STATUS: MEASURED — all three splits scored 2026-09-25.** dev and test_in (orders schema, seen) and
test_out (cars schema, **unseen**) now have base and SFT execution-accuracy numbers (section 3).
Headline: **0.783 in-distribution, 0.260 on the unseen schema, base 0.000 on all three.**
Still open before this file is quotable: no bootstrap CI yet (section 3.1 carries a labelled
*approximation*), failures not yet exported to `eval_v1.json`, base raw text only partially inspected,
leakage guard deleted in the working tree, `run_manifest.json` untracked, run folder not renamed
(section 6).

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

### 1.1 Leakage check (measured 2026-09-25, ad hoc)

Loading the four files and comparing fields gives: **0 question overlap** (`q_norm_sha256`) and
**0 gold-SQL overlap** between `train` and each of `dev`, `test_in`, `test_out`. The `test_out` table
set is disjoint from train except the documented shared name `user`. `context_sha256` collides 13 times
between train and dev — expected, because the DDL block is a function of the table subset and not of the
question, so the context hash must never be used as a leakage signal.

**Not enforced yet.** The check was written as `tests/test_no_leakage.py` and passed (7 passed,
1 skipped: byte-identical rebuild is impossible while `src/data/build.py` is empty), but that file is
now deleted in the working tree (`git status`: `D tests/test_no_leakage.py`). Until it is committed,
the numbers in section 3 remain provisional: nothing prevents a future dataset edit from leaking.

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

## 3. Execution accuracy — dev, test_in, test_out (the gate and the headline)

Metric: SQL **execution accuracy** — run the predicted statement on the fixture database named by the
row's `db_id` (`orders` → `orders.db`, `cars` → `cars.db`), compare the result set to the result set of
the gold `query`. Order-insensitive unless the gold has `ORDER BY`. Invalid / non-executable predictions
count as wrong and are reported separately as the valid-SQL rate. Prompt built with `render()` from
`src/data/prompt.py` for **all** models — never a retyped template.

| model | split | schema | execution accuracy | valid SQL rate | n |
|---|---|---|---|---|---|
| base TinyLlama-1.1B | dev | orders (seen) | 0/60 = 0.000 | 0/60 = 0.000 | 60 |
| base TinyLlama-1.1B | test_in | orders (seen) | 0/60 = 0.000 | 0/60 = 0.000 | 60 |
| base TinyLlama-1.1B | test_out | cars (**unseen**) | 0/73 = 0.000 | 0/73 = 0.000 | 73 |
| `sft r16 all-linear` c42 | dev | orders | 43/60 = 0.717 | 59/60 = 0.983 | 60 |
| `sft r16 all-linear` c63 | dev | orders | 46/60 = 0.767 | 60/60 = 1.000 | 60 |
| `sft r16 all-linear` c84 | dev | orders | **47/60 = 0.783** | 60/60 = 1.000 | 60 |
| `sft r16 all-linear` (c84) | test_in | orders | **47/60 = 0.783** | 60/60 = 1.000 | 60 |
| `sft r16 all-linear` (c84) | test_out | cars (**unseen**) | **19/73 = 0.260** | 67/73 = 0.918 | 73 |

Measured 2026-09-25, greedy (`do_sample=False`), `max_new_tokens=128`, git `bf10d20` (dirty tree),
runtimes 56-182 s. Logs: `reports/logs/exacc_{dev,test_in,test_out}_*_2026-09-25.log`.

```bash
venv/bin/python -m src.eval.exacc --model base            --file datasets/test_in.jsonl  --db orders
venv/bin/python -m src.eval.exacc --model artifacts/sft_v1 --file datasets/test_in.jsonl  --db orders
venv/bin/python -m src.eval.exacc --model base            --file datasets/test_out.jsonl --db cars
venv/bin/python -m src.eval.exacc --model artifacts/sft_v1 --file datasets/test_out.jsonl --db cars
```

Gate: `sft_acc > base_acc` — **0.783 > 0.000**, passes on every split; no CI is needed to call that.

**The finding that matters: −52 points when the schema changes.** Same weights, same prompt template,
same decoding — only the DDL differs. `test_in` (seen schema, unseen questions) reproduces the dev
number exactly (47/60 both), so the dev split was not lucky and there is no dev overfitting. `test_out`
(73 questions over the cars schema, never in training) drops to 19/73 with 6 statements that do not
execute at all. The model learned the *format* (91.8 % valid SQL on an unfamiliar schema) and failed
*schema linking* (mapping question words to the right columns of a new schema).

Caveats, in the order they affect how this table may be quoted:

1. **No confidence interval yet.** `19/73 = 0.260` with n=73 gives a ±0.10 normal-approximation interval
   (that is an *approximation*, not the bootstrap the roadmap requires) — quote it as "roughly a quarter,
   between ~16 % and ~36 %", never as "26 %". Same for `0.783` at n=60: ±0.10.
2. **±1 question nondeterminism.** The same epoch-4 weights scored 46/60 in an earlier dev run and 47/60
   in the logged run. Two candidate causes, neither excluded: 4-bit bitsandbytes kernels are not
   bit-deterministic, and `artifacts/sft_v1` stores those weights in fp32 while `checkpoint-84` stores
   bf16 (308/308 tensors match, max abs diff `2.2e-4` = one bf16 ulp).
3. **Base 0.000 is now corroborated.** Two of the three logged base failures per split are English prose
   (`The average daily rate of the fleet is 1000.`, `The average length of the item names is 10.`), i.e.
   the base model answers in sentences rather than SQL. One failure per split is an empty prediction
   (`extract()` collapsing on `###`); the raw text behind that single case is still uninspected.
4. **Only the first 3 failures per run are printed** (`if len(fails) < 3`), so the categories below come
   from 3 of 13 dev failures and 3 of 19 test_out failures — not yet a taxonomy.

## 3.1 test_out — why the unseen schema drops (from the logged failures)

| # | id | gold | predicted | category |
|---|---|---|---|---|
| 1 | car_0004 | `SELECT car.make, car.model, dealership.name AS dealership_name FROM car JOIN dealership ON car.dealership_id = dealership.id WHERE dealership.city = 'New Danielborough'` | `SELECT car.vin, car.daily_rate FROM car JOIN dealership ON car.dealership_id = dealership.id WHERE dealership.name LIKE 'New Danielborough%'` | wrong target column for the filter value (`name` vs `city`) + wrong projection |
| 2 | car_0005 | `SELECT make, model FROM car WHERE daily_rate > (SELECT AVG(daily_rate) FROM car)` | `SELECT car.make, car.model, car.year, car.vin, car.daily_rate FROM car WHERE car.daily_rate > (SELECT AVG(x.daily_rate) FROM car AS x WHERE x.dealership_id = car.dealership_id)` | aggregate scope changed (global → per-group) + extra columns |
| 3 | car_0006 | `SELECT make, model, daily_rate FROM car ORDER BY daily_rate DESC, id LIMIT 1 OFFSET 1` | `SELECT dealership_id, MAX(daily_rate) FROM car GROUP BY dealership_id ORDER BY daily_rate DESC, dealership_id LIMIT 1` | wrong query shape (`GROUP BY` instead of `LIMIT/OFFSET`) |

Contrast with the in-distribution failures (section 5), which are dropped/extra columns on the same
tables. On an unseen schema the errors become semantic: wrong column for a filter, wrong aggregate
scope, wrong query shape. Those are *comprehension* failures, and no number of extra epochs fixes them —
the fix is training data with schema diversity and query-shape coverage, not more steps.


## 4. Checkpoint selection

Checkpoint choice is now justified by dev execution accuracy, not by `eval_loss`. Training used
`metric_for_best_model="eval_loss"` with `load_best_model_at_end=True`; the measured scores are:

| checkpoint | dev execution accuracy |
|---|---|
| `checkpoint-42` (epoch 2) | 43/60 = 0.717 |
| `checkpoint-63` (epoch 3) | 46/60 = 0.767 |
| `checkpoint-84` (epoch 4) | **47/60 = 0.783** |

**Kept: `checkpoint-84`.** `artifacts/sft_v1` holds the same epoch-4 weights stored in fp32: 308/308
tensors match `checkpoint-84`, max abs diff `2.2e-4` (one bf16 ulp), which is a storage-dtype rounding,
not a different training result. Accuracy rises monotonically with epochs, so the loss-based and
metric-based choices agree here. The 0.767 vs 0.783 gap is one question — inside the ±1 band noted in
section 3 — so the defensible statement is "epoch 4 ≥ epoch 3, no overfitting observed on dev", not
"epoch 4 is measurably better".


## 5. Failure analysis — dev (3 of the 13 failures at checkpoint-84)

For the first few wrong predictions: question, gold SQL, predicted SQL, and the category
(wrong `WHERE`, missing `JOIN`, wrong aggregate, truncated, invalid SQL). This is what makes the
number credible, and it feeds Phase 3's error taxonomy.

| # | question | gold | predicted | category |
|---|---|---|---|---|
| 1 | ord_0001 | `SELECT name FROM user WHERE created_at >= '2000-01-01'` | `SELECT name, email FROM user WHERE created_at > '2000-01-01'` | wrong `WHERE` boundary (`>` vs `>=`) + extra column |
| 2 | ord_0034 | `SELECT name, email FROM user ORDER BY created_at DESC, id DESC LIMIT 5` | `SELECT name, email, created_at FROM user ORDER BY created_at DESC, id LIMIT 5` | dropped `ORDER BY` tiebreaker + extra column |
| 3 | ord_0098 | `SELECT id, quantity, created_at FROM user_orders ORDER BY created_at DESC, id DESC LIMIT 5` | `SELECT id, created_at FROM user_orders ORDER BY created_at DESC, id LIMIT 5` | dropped tiebreaker + missing `quantity` |

Dominant failure pattern on dev: **column-set drift and dropped `ORDER BY` tiebreakers**. Both change the
result set on the tie cases the fixture was built to punish. At `checkpoint-42` the same questions fail
with different predictions (e.g. ord_0098 `ORDER BY id DESC, created_at`), and epoch 3 → 4 does not fix
them — so these are *data* gaps (the pattern is under-represented in train), not a shortage of epochs.
One invalid statement appeared at `checkpoint-42` (valid 59/60); none at 63 or 84.

Only the first 3 failures are printed per run (`if len(fails) < 3` in `src/eval/exacc.py`), so this table
is a sample of 3 from 13 failures at checkpoint-84, not the full taxonomy — Phase 3 must log every failure.

### 5.1 test_in failures (3 of 13, same model) — same family as dev

| # | id | gold | predicted | category |
|---|---|---|---|---|
| 1 | ord_0027 | `SELECT name, email FROM user ORDER BY created_at DESC, id DESC LIMIT 1` | `SELECT name, email, created_at FROM user ORDER BY created_at DESC, id LIMIT 1` | dropped tiebreaker + extra column |
| 2 | ord_0036 | `SELECT id, quantity FROM user_orders WHERE item_id = 18` | `SELECT id FROM user_orders WHERE item_id = 18` | dropped a requested column |
| 3 | ord_0048 | `SELECT name, price, stock FROM item WHERE stock > 5 ORDER BY price DESC, id LIMIT 5` | `SELECT item.name, item.price FROM item WHERE item.stock > 5 ORDER BY item.price DESC, item.id LIMIT 5` | dropped a requested column (`stock`) |

In-distribution failures are column-set drift and dropped tiebreakers on tables the model has seen —
a data gap, not a capacity gap. The unseen-schema failures (section 3.1) are different in kind: wrong
column semantics, wrong aggregate scope, wrong query shape.


## 6. Definition of done (Phase 2)

- [x] Section 3 cells measured on the same 60 dev rows with the same prompt template (base + c42/c63/c84).
- [x] `sft_acc > base_acc` — 0.783 vs 0.000 (section 3).
- [x] `run_manifest.json` written next to the adapter (`artifacts/sft_v1/run_manifest.json`). Note: `artifacts/` is gitignored, so this copy is local provenance only — add `!artifacts/**/run_manifest.json` to `.gitignore` or keep a tracked copy under `reports/`.
- [x] Reproducibility settled by naming the nondeterminism: one question flips between runs on the same epoch-4 weights — 4-bit bitsandbytes kernels are not bit-deterministic, and the fp32 (`artifacts/sft_v1`) vs bf16 (`checkpoint-84`) storage of those weights is a second candidate cause. Result is `0.783 ± 1.7 pts`, not bit-reproducible.
- [ ] `artifacts/runs/2026-09-23_r16_alllinear` exists (run folder renamed, no reuse of `artifacts/sft_v1` for future runs). **Open.**
- [ ] Base row's empty-prediction caveat resolved by inspecting 3 raw base generations. **Open.**
- [ ] Leakage guard committed to the repository again (`tests/test_no_leakage.py` is deleted in the working tree). **Open.**

### Phase 3 progress (the harness)

- [x] Base and SFT scored on test_in and test_out with the same prompt template and decoding (section 3).
- [x] Logs kept in the repository (`reports/logs/`, 8 runs, no `/tmp`).
- [ ] Every failure exported to `eval_v1.json` with a category (only 3 per run are printed today).
- [ ] Bootstrap 95 % CI (1000 resamples) and McNemar base-vs-SFT per split — section 3 uses a labelled
      normal approximation instead.
- [ ] Per-`pattern` slices (accuracy by the 7 pattern buckets already present in the rows).
- [ ] `sqlglot` parse check for the valid-SQL rate (today validity means "it executed").
- [ ] Serving-parity check (same prompts through HF and through the served endpoint) — belongs to step 7.
- [ ] Regenerate the tables from `eval_v1.json` instead of editing numbers by hand.


## 7. Not in scope here

Full matrix over `test_in` / `test_out`, confidence intervals, per-pattern slices, DPO and
quantized comparisons, and the template-parity test belong to Phase 3 and reuse this same harness
file. MLflow, CI and model cards belong to Phase 8.

Known limitations of the current harness to fix in Phase 3:

- `--split` is accepted but **ignored**: `ROWS` still reads `datasets/dev.jsonl` (`src/eval/exacc.py`).
  Passing `--split datasets/test_out.jsonl` would silently score dev. Also `DB` is hardcoded to
  `orders.db`, so `test_out` (cars schema) cannot be scored at all yet.
- Only the first 3 failures are printed; the failure list is truncated (`if len(fails) < 3`).
- No confidence interval, no per-pattern slice, no `sqlglot` validity check (validity is "it executed").
- Result-set comparison rounds floats to 6 decimals instead of a tolerance check, and sorts rows by
  `repr` — adequate for the gate, not for the final report.

