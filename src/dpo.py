import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
import json
from datasets import Dataset
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig 
model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", quantization_config=bnb_config)
model = PeftModel.from_pretrained(base_model, "final_model")
model.enable_input_require_grads()

dataset = Dataset.from_json("data/dpo_data.jsonl")

training_args = DPOConfig(
    output_dir="dpo_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    logging_steps=1,
    num_train_epochs=3,
    beta=0.1,
    gradient_checkpointing=False,
    max_length=256,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
model.save_pretrained("dpo_model")


