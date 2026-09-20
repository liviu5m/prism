# datasets/ — text-to-SQL data for Forge

Row contract (identical in every file):

```json
{"id": "ord_0300", "db_id": "orders", "question": "...", "context": "CREATE TABLE ...;",
 "query": "SELECT ...;", "pattern": "order_limit", "difficulty": "hard",
 "tables": ["item"], "source": "handwritten",
 "q_norm_sha256": "f08a7643ef567aa7", "context_sha256": "45d7b7d71cb8274f"}
```

`context` is the exact DDL of **only the tables the query needs**, read from the fixture DB
(`sqlite_master`, whitespace-collapsed). Prompt = `render(context, question)`; label = `query`.

## Files

| File | db_id (fixture) | Rows | Reader | Purpose |
|---|---|---|---|---|
| `train.jsonl` | orders (`orders.db`) | 325 | `train.py` | the **only** file training may read |
| `dev.jsonl` | orders (`orders.db`) | 60 | sweep / checkpoint selection | pick the checkpoint, tune; never a reported number |
| `test_in.jsonl` | orders (`orders.db`) | 60 | eval harness | in-distribution execution accuracy |
| `test_out.jsonl` | cars (`cars.db`) | 73 | eval harness | unseen-schema execution accuracy — the headline number |
| `sqldata.jsonl` | orders | 445 | **nobody** | pool = train ∪ dev ∪ test_in; reused only to re-split |

**Never train on `sqldata.jsonl`** — it is a superset of the train set and contains every
dev/test_in row. Training on it reproduces the original leak (eval prompts inside the training
file). Delete it once the split is frozen.

Ids (`ord_*`, `car_*`) are canonical row ids and are unique **within** each file. `sqldata.jsonl`
shares ids with the split files by design, because the splits are a partition of it.

## Fixtures (frozen — gold results depend on them)

| Fixture | md5 | Schema |
|---|---|---|
| `orders.db` (fixture `orders`) | `abe7caa85cd83651f95166057b7bf45a` | `user`, `category`, `shop`, `item`, `user_orders` |
| `cars.db` | `5cf91af6fc22c35d6513cbd117816f04` | `user`, `dealership`, `car`, `rental`, `maintenance_log` |

`src/db.py` seeds with unseeded `Faker()`/`random`. Re-running it produces different rows, which
invalidates every gold query in this folder. Seed it or keep these two files untouched.

## Verified properties (2026-09-20)

- every `query` executes on its own fixture and returns **≥1 row** (an empty gold makes a broken
  prediction score as correct)
- every `context` statement is byte-identical to a statement in that fixture's `sqlite_master`
- normalised questions and normalised gold SQL: **0 overlap** between train / dev / test_in / test_out
- no cars-schema identifier occurs in any train/dev/test_in `context`, and no orders-schema
  identifier in any test_out `context` (checked with word boundaries)
- every `ORDER BY ... LIMIT` carries a unique tiebreaker (`id`), because prices and quantities tie
- token budget (real TinyLlama tokenizer, prompt+answer): train p50 151 / p95 306 / max 350,
  test_out max 399 → use **`max_length >= 448`** (a 256 budget truncates ~11 % of train rows)
