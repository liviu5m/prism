from datasets import Dataset
from torch import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from dataset import getRecords

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "final_model"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    dtype=torch.float16,
)

model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], 
    task_type="CAUSAL_LM"
))

data = getRecords("data/dataset.jsonl")
dataset = Dataset.from_list(data)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="final_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=False, bf16=False,
        max_length=256,
        dataset_text_field="text",
        optim="adamw_torch",
        num_train_epochs=10,
    ),
)
trainer.train()
trainer.save_model("final_model")
