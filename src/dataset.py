import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def getTokens(file):
    data = [] 
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    resultTokens = []

    for d in data:
        sample = f"### Prompt:\n{d.get("prompt")}\n\n### Response:\n{d.get("response")}<|endoftext|>"
        encoded = tokenizer(sample, truncation=True, max_length=512, padding="max_length")
        resultTokens.append(encoded)
    return resultTokens

def getRecords(file):
    data = [] 
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    records = []

    for d in data:
        sample = f"### Prompt:\n{d.get("prompt")}\n\n### Response:\n{d.get("response")}<|endoftext|>"
        records.append({"text": sample})
    return records 

