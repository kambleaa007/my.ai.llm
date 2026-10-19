"""Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning."""
import json
from datasets import load_dataset

# Samanantar Marathi-English parallel corpus (instruction-style pairs)
try:
    ds = load_dataset("ai4bharat/samanantar", "mr", split="train[:5000]")
    rows = [{"text": f"<start_of_turn>user\n{r['src']}<end_of_turn>\n<start_of_turn>model\n{r['tgt']}<end_of_turn>"} for r in ds]
except Exception as e:
    print("Samanantar failed:", e)
    rows = [{"text": "<start_of_turn>user\nनमस्कार<end_of_turn>\n<start_of_turn>model\nनमस्कार! कसे आहात?<end_of_turn>"}]

with open("marathi_instruct.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Wrote", len(rows), "records")