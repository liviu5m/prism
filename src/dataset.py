import json
from transformers import AutoTokenizer

from data.prompt import render

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def getRecords(file):
    data = [] 
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    records = []

    for d in data:
        r = {
            "prompt": render(d.get("context"), d.get("question")),
            "completion": d.get("query") + tokenizer.eos_token,
        }
        records.append(r)
    return records 

