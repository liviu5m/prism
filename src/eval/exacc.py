import json, re, sqlite3, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.data.prompt import render
import argparse

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="artifacts/sft_v1")
ap.add_argument("--split", default="datasets/dev.jsonl")
ap.add_argument("--db", default="orders")
ap.add_argument("--file", default="datasets/dev.jsonl")
args = ap.parse_args()
MODEL = args.model
ROWS = [json.loads(l) for l in open(args.file) if l.strip()]
DB = args.db+".db"

def extract(text):                   # one statement only
    sql = text.split("###")[0]
    if ";" in sql:
        sql = sql.split(";")[0]
    return " ".join(sql.split())

def run(sql):                        
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()
    finally:
        con.close()

def same(gcols, grows, pcols, prows, ordered):
    if len(gcols) != len(pcols):
        return False
    g = [tuple(round(c, 6) if isinstance(c, float) else c for c in r) for r in grows]
    p = [tuple(round(c, 6) if isinstance(c, float) else c for c in r) for r in prows]
    return g == p if ordered else sorted(g, key=repr) == sorted(p, key=repr)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map={"": 0}, quantization_config=bnb, dtype=torch.bfloat16)
if MODEL != "base":
    model = PeftModel.from_pretrained(model, MODEL)
model.eval()

ok = valid = 0
fails = []
t0 = time.time()
for i, row in enumerate(ROWS, 1):
    prompt = render(row["context"], row["question"])
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    pred = extract(tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True))
    ordered = "order by" in row["query"].lower()
    gcols, grows = run(row["query"])
    try:
        pcols, prows = run(pred)
        valid += 1
        correct = same(gcols, grows, pcols, prows, ordered)
    except Exception as exc:
        correct = False
        if len(fails) < 3:
            fails.append((row["id"], row["query"], pred, f"{type(exc).__name__}: {exc}"))
            continue
    ok += int(correct)
    if not correct and len(fails) < 3:
        fails.append((row["id"], row["query"], pred, "wrong result set"))
    print(f"[{i}/{len(ROWS)}] exec_ok={ok} valid={valid}", flush=True)

print(f"\nMODEL {MODEL}")
print(f"execution accuracy: {ok}/{len(ROWS)} = {ok/len(ROWS):.3f}")
print(f"valid SQL rate:     {valid}/{len(ROWS)} = {valid/len(ROWS):.3f}")
print(f"time: {time.time()-t0:.0f}s")
for rid, gold, pred, why in fails:
    print(f"\n{rid}  ({why})\n  gold: {gold}\n  pred: {pred}")
