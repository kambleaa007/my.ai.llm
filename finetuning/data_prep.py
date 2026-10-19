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
for src_name, cfg in [("ai4bharat/samanantar", "mr"), ("cfilt/iitb-english-marathi-parallel-corpus", None)]:
    try:
        ds = load_dataset(src_name, cfg, split="train[:5000]") if cfg else load_dataset(src_name, split="train[:5000]")
        for r in ds:
            s = r.get("src") or r.get("english") or r.get("en", "")
            t = r.get("tgt") or r.get("marathi") or r.get("mr", "")
            if s and t:
                rows.append({"text": f"<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\n{t}<end_of_turn>"})
        print(f"Loaded {src_name}")
        break
    except Exception as e:
        print(f"{src_name} failed: {e}")

with open("marathi_instruct.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("Wrote", len(rows), "records")