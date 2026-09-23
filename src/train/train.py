from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from src.dataset import getRecords, tokenizer
from src.data.prompt import render

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
adapter_path = "final_model"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    dtype=torch.bfloat16,

)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, 
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear"
))

train_ds = Dataset.from_list(getRecords("datasets/train.jsonl"))
eval_ds = Dataset.from_list(getRecords("datasets/dev.jsonl"))

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="artifacts/sft_v1",
        max_length=512,                     
        num_train_epochs=4,               
        learning_rate=2e-4,              
        lr_scheduler_type="cosine",
        per_device_train_batch_size=2,  
        gradient_accumulation_steps=8, 
        warmup_steps=3,
        bf16=True,                    
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=42,
        completion_only_loss=True,
    ),
)
trainer.train()
trainer.save_model("artifacts/sft_v1")
