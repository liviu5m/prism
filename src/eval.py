from peft import PeftModel
from torch import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, "final_model")
merged_model = model.merge_and_unload()

prompt = "### Prompt:\nGet top 5 users by total order value\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = merged_model.generate(**inputs, max_new_tokens=128, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
