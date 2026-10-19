"""
Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning.
Outputs clean, deduplicated, response-masked compatible JSONL.
"""

import json
import re
import unicodedata
from datasets import load_dataset

rows = []
seen_prompts = set()


def clean_marathi_text(text: str) -> str:
    """Normalize Unicode and clean irregular whitespace/invisible characters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text.strip()


def add_pair(user_text: str, model_text: str, source: str = ""):
    u = clean_marathi_text(user_text)
    m = clean_marathi_text(model_text)

    if not u or not m or len(m) < 4:
        return

    prompt_key = u.lower()
    if prompt_key in seen_prompts:
        return
    seen_prompts.add(prompt_key)

    formatted_text = (
        f"<start_of_turn>user\n{u}<end_of_turn>\n"
        f"<start_of_turn>model\n{m}<end_of_turn>"
    )

    rows.append(
        {
            "messages": [
                {"role": "user", "content": u},
                {"role": "model", "content": m},
            ],
            "text": formatted_text,
            "source": source,
        }
    )


# 1. Load local curated JSON (Instruction/Response or q/a)
try:
    with open("marathi_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "messages" in item:
            u = item["messages"][0]["content"]
            m = item["messages"][1]["content"]
            add_pair(u, m, source="curated_local")
        else:
            q = item.get("q") or item.get("instruction", "")
            a = item.get("a") or item.get("output", "")
            add_pair(q, a, source="curated_local")
    print(f"Loaded curated local data: {len(rows)} samples")
except Exception as e:
    print("Warning: marathi_data.json skipped:", e)

# 2. Smallstep Marathi Alpaca
try:
    ds = load_dataset(
        "smallstepai/marathi-instruction-tuning-alpaca", split="train"
    )
    prev = len(rows)
    for r in ds:
        q = r.get("instruction", "")
        if r.get("input"):
            q += f"\n\nमाहिती:\n{r['input']}"
        a = r.get("output", "")
        add_pair(q, a, source="smallstep")
    print(f"Loaded smallstep: {len(rows) - prev} samples")
except Exception as e:
    print("smallstep failed:", e)

# 3. Marathi Alpaca (aghatage)
try:
    ds = load_dataset(
        "aghatage/marathi-alpaca-llama-finetune", split="train"
    )
    prev = len(rows)
    for r in ds:
        q = r.get("instruction", "")
        if r.get("input"):
            q += f"\n\nमाहिती:\n{r['input']}"
        a = r.get("output", "")
        add_pair(q, a, source="aghatage")
    print(f"Loaded aghatage: {len(rows) - prev} samples")
except Exception as e:
    print("aghatage failed:", e)

# 4. Marathi Multi-Domain SFT
try:
    ds = load_dataset(
        "grpathak22/marathi-maharashtra-multidomain-SFT-1k", split="train"
    )
    prev = len(rows)
    for r in ds:
        q = r.get("instruction") or r.get("question", "")
        a = r.get("output") or r.get("answer", "")
        add_pair(q, a, source="multidomain_sft")
    print(f"Loaded marathi-multidomain: {len(rows) - prev} samples")
except Exception as e:
    print("marathi-multidomain failed:", e)

# 5. AI4Bharat Indic-Instruct (Marathi subset)
try:
    ds = load_dataset(
        "ai4bharat/indic-instruct-data-v0.1", split="train", streaming=True
    )
    prev = len(rows)
    count = 0
    for r in ds:
        lang = r.get("language") or r.get("lang")
        if lang != "mr":
            continue
        q = r.get("instruction") or r.get("prompt", "")
        a = r.get("output") or r.get("response", "")
        if q and a:
            add_pair(q, a, source="indic_instruct")
            count += 1
            if count >= 5000:
                break
    print(f"Loaded indic-instruct: {len(rows) - prev} samples")
except Exception as e:
    print("indic-instruct failed:", e)

# 6. Samanantar Marathi-English Parallel Corpus (With Explicit Instruction Wrapping)
try:
    ds = load_dataset("ai4bharat/samanantar", "mr", split="train[:5000]")
    prev = len(rows)
    for r in ds:
        src = clean_marathi_text(r.get("src", ""))
        tgt = clean_marathi_text(r.get("tgt", ""))

        if src and tgt:
            en_to_mr_prompt = (
                f"खालील इंग्रजी मजकुराचे अचूक मराठीत भाषांतर करा:\n\"{src}\""
            )
            add_pair(en_to_mr_prompt, tgt, source="samanantar_en_mr")

    print(f"Loaded samanantar translations: {len(rows) - prev} samples")
except Exception as e:
    print("samanantar failed:", e)

# Export clean JSONL
output_path = "marathi_instruct.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(
    f"\nSuccessfully generated {len(rows)} high-quality samples in '{output_path}'"
)
"""