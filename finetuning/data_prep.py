"""Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning."""
import json
from datasets import load_dataset

SYSTEM = "तुम्ही एक सहाय्यक AI आहात जो मराठीत उत्तम उत्तरे देतो."

def to_chat(example):
    q = example.get("instruction") or example.get("question") or example.get("input", "")
    a = example.get("output") or example.get("answer") or example.get("response", "")
    return {"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"}

# IndicInstruct (Marathi subset)
ds = load_dataset("ai4bharat/IndicInstruct", "mr", split="train")
ds = ds.map(to_chat, remove_columns=ds.column_names)
ds.to_json("data/marathi_instruct.jsonl")

# IndicQA Marathi
qa = load_dataset("ai4bharat/IndicQA", "mr", split="train")
qa = qa.map(lambda x: {"text": f"<start_of_turn>user\n{x['question']}<end_of_turn>\n<start_of_turn>model\n{x['answer']}<end_of_turn>"}, remove_columns=qa.column_names)
qa.to_json("data/marathi_qa.jsonl")
print("Done")