"""CPU-friendly tiny Marathi model training (125M params)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from pathlib import Path

model_name = "unsloth/gemma-2-9b-bnb-4bit"  # too big for CPU
# Use a tiny model instead
model_name = "EleutherAI/gpt-neo-125M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("json", data_files="marathi_instruct.jsonl", split="train")

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
ds = ds.map(tok, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

model = AutoModelForCausalLM.from_pretrained(model_name)

args = TrainingArguments(
    output_dir="./marathi-tiny",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=200,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=10,
    save_steps=100,
    no_cuda=True,
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model("./marathi-tiny")
tokenizer.save_pretrained("./marathi-tiny")
print("Done")