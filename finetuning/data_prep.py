"""Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning."""
import json
import unicodedata
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

# Marathi Alpaca Instruct
try:
    ds = load_dataset("aghatage/marathi-alpaca-llama-finetune", split="train")
    for r in ds:
        q = r.get("instruction", "")
        if r.get("input"):
            q += f"\n\nमाहिती:\n{r['input']}"
        a = r.get("output", "")
        if q and a:
            rows.append({"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"})
    print("Loaded marathi-alpaca")
except Exception as e:
    print("marathi-alpaca failed:", e)

# Marathi Multi-Domain SFT
try:
    ds = load_dataset("grpathak22/marathi-maharashtra-multidomain-SFT-1k", split="train")
    for r in ds:
        q = r.get("instruction") or r.get("question", "")
        a = r.get("output") or r.get("answer", "")
        if q and a:
            rows.append({"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"})
    print("Loaded marathi-multidomain")
except Exception as e:
    print("marathi-multidomain failed:", e)

# AI4Bharat Indic-Instruct (Marathi subset)
try:
    ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", split="train", streaming=True)
    count = 0
    for r in ds:
        if r.get("language") != "mr" and r.get("lang") != "mr":
            continue
        q = r.get("instruction") or r.get("prompt", "")
        a = r.get("output") or r.get("response", "")
        if q and a:
            rows.append({"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"})
            count += 1
            if count >= 5000:
                break
    print(f"Loaded indic-instruct mr: {count}")
except Exception as e:
    print("indic-instruct failed:", e)

# Samanantar Marathi-English parallel corpus
try:
    ds = load_dataset("ai4bharat/samanantar", "mr", split="train[:5000]")
    for r in ds:
        s = r.get("src", "")
        t = r.get("tgt", "")
        if s and t:
            rows.append({"text": f"<start_of_turn>user\n{s}<end_of_turn>\n<start_of_turn>model\n{t}<end_of_turn>"})
    print("Loaded samanantar")
except Exception as e:
    print("samanantar failed:", e)

with open("marathi_instruct.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        text = unicodedata.normalize("NFC", r["text"])
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
print("Wrote", len(rows), "records")