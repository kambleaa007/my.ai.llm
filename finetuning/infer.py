"""Run inference on fine-tuned Marathi Gemma."""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("marathi-gemma-2-9b", max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)

prompt = "<start_of_turn>user\nतुम्ही कोण आहात?<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))