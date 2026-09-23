from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, "final_model")
merged_model = model.merge_and_unload()

with open("data/prompts.jsonl", "r") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            prompt = f"### Prompt:\n{data.get("prompt")}\n\n### Response:\n"
            print(data.get("prompt"))
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            outputs = merged_model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(full_text.split("### Response:\n")[-1])
            print(data.get("expected"))
            print("\n");


