import json
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.data.prompt import render
rows = []
with open("datasets/dev.jsonl") as f:
    for l in f:
        rows.append(json.loads(l))

BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL = "artifacts/sft_v1"          

model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
model = PeftModel.from_pretrained(model, MODEL)
model.eval()
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token

for row in rows:
    prompt = render(row["context"], row["question"])
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
    pred = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)
    print(f"\n{row['id']}  ({'wrong result set' if pred != row['query'] else 'correct'})\n  gold: {row['query']}\n  pred: {pred}")

