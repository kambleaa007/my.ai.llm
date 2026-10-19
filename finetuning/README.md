# Fine-tuning Guide

1. Open Google Colab (GPU runtime: T4/A100).
2. Install: `!pip install unsloth trl datasets transformers`
3. Run `data_prep.py` to build `marathi_instruct.jsonl`.
4. Run `train_qlora.py`.
5. Download `marathi-gemma-2-9b-gguf` and load in Ollama.

Smoke test first with `max_steps=50` on a 500-sample subset.
"""