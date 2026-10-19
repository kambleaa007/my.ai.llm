"""Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning."""
import json
from datasets import load_dataset

rows = []

# Load curated Marathi data from JSON file
try:
    with open("marathi_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        rows.append({"text": f"<start_of_turn>user\n{item['q']}<end_of_turn>\n<start_of_turn>model\n{item['a']}<end_of_turn>"})
except Exception as e:
    print("Failed to load marathi_data.json:", e)

# Samanantar Marathi-English parallel corpus
try:
    ds = load_dataset("ai4bharat/samanantar", "mr", split="train[:5000]")
    rows += [{"text": f"<start_of_turn>user\n{r['src']}<end_of_turn>\n<start_of_turn>model\n{r['tgt']}<end_of_turn>"} for r in ds]
except Exception as e:
    print("Samanantar failed:", e)

with open("marathi_instruct.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Wrote", len(rows), "records")