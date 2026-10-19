"""Prepare Marathi datasets for Gemma 2 9B QLoRA fine-tuning."""
import json
from datasets import load_dataset

creative = [
 ("एक लहान मराठी कथा लिहा जिथे एक गावातील मुलगा आकाशात उडतो.", "एकदा एका छोट्या गावात रामू नावाचा मुलगा राहत होता. त्याला आकाशात उडायचे होते. एके दिवशी त्याने जादूचे पंख बनवले आणि उडायला लागला. गावकरी आश्चर्यचकित झाले. रामूने ढगांवरून गाव पाहिला आणि हसला. तो एक हिरो बनला."),
 ("मराठीत एक कविता लिहा वसंत ऋतूवर.", "वसंत आला रे वसंत आला,\nफुले उमलली, भ्रमर डुलला.\nकोकिळा गाते गोड गाणे,\nनिसर्ग नटला सुंदर राणे."),
 ("महाराष्ट्राच्या इतिहासातील एक महत्त्वाचा घटना सांगा.", "शिवाजी महाराजांनी १६७४ साली रायगडावर राज्याभिषेक केला. हा मराठा साम्राज्याचा जन्म होता. त्यांनी स्वराज्याची संकल्पना रुजवली."),
 ("एक मराठी शेरा लिहा जीवनावर.", "जीवन हे एक प्रवास आहे,\nप्रत्येक दिवस नवा आहे.\nहसून जा, रडून जा,\nपण थांबू नकोस कधी."),
]

rows = [{"text": f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>"} for q, a in creative]

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