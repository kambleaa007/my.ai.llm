# Stage 5: Fine-tuning and Alignment

Turns the pre-trained base model from Stage 4 into a chat assistant: an instruction dataset in Indic languages, supervised fine-tuning, and preference alignment. It is split into 3 parts.

## Directory Structure

indian-llm-architecture/
├── finetuning/
│   ├── sft.py                 # Supervised fine-tuning loop
│   ├── dpo.py                 # Direct Preference Optimization
│   ├── data_builder.py        # Dataset assembly + chat template
│   └── __init__.py
├── configs/
│   ├── sft_1B.yaml            # SFT hyperparameters
│   └── dpo_1B.yaml            # DPO hyperparameters
├── data/
│   ├── instruction/indic_instructions.jsonl
│   └── preferences/indic_preferences.jsonl
└── experiments/{model_size}/
    ├── sft/                   # SFT checkpoint
    └── dpo/                   # Aligned checkpoint

## Part 1: Instruction Dataset

Target: 100K-500K instruction-response pairs, minimum 10K high-quality pairs per major language.

### Existing datasets to leverage

| Dataset | Task | Notes |
|---|---|---|
| IndicQA | Question answering | 11+ Indic languages, natural eval fit |
| IndicNLG | Paraphrase, summarization | Convert to instruction format |
| OASST | Multi-turn chat | Multilingual subset, translate or filter |
| MagPie | Synthetic instructions | High volume, generated |
| NoRobots | Human-written instructions | Small, highest quality — translate manually |
| IndicCorp / IndicGenBench | Benchmarks | Keep held out for Stage 6 |

### Synthetic generation
- Reuse `Quaere.ai/scripts/generate_dataset.py` as the template — it already emits `{instruction, input, output}` JSONL
- Generate English instructions with a strong teacher model, translate responses into each target language
- **Human review is mandatory** for a 10-20% sample per language; unreviewed machine translation produces fluent but culturally wrong data
- Add domain splits: government services, agriculture, education, health, finance, law, agriculture policy, history

### Chat format
Multi-turn conversations serialized with the Stage 1 special tokens:
```
<|system|>You are a helpful assistant that replies in {language}.<|user|>{q1}<|assistant|>{a1}<|user|>{q2}<|assistant|>{a2}<|eos|>
```
- Loss masked on all tokens except assistant responses (`-100` on prompt positions)
- Long conversations truncated from the left so the final assistant turn always survives
- The same tokenizer file from Stage 1 is used — no re-tokenization drift

### Quality controls
- [ ] Minimum 10K high-quality pairs per major language (hi, bn, ta, te, mr, gu, ml)
- [ ] Multi-turn turn boundaries correct; no leaked `<|user|>` inside an assistant turn
- [ ] No PII: run a regex + NER pass over emails, phones, Aadhaar-like numbers, bank accounts
- [ ] No benchmark contamination: n-gram overlap check against IndicQA/IndicNLG test sets
- [ ] Script consistency: response script matches requested language
- [ ] Response length distribution logged per language (a common failure is runaway length in low-resource languages)

## Part 2: Supervised Fine-Tuning (SFT)

**Trainer** (`finetuning/sft.py`)
- Loads the Stage 4 base checkpoint (or SFT/DPO checkpoints for continued runs)
- LoRA/QLoRA for anything under 7B: rank 16-64, alpha 32-128, target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Full fine-tuning is viable for the 125M-350M models
- BF16, gradient checkpointing, packing disabled (SFT needs clean turn boundaries)
- Effective batch 128 sequences via gradient accumulation

**Hyperparameters** (`configs/sft_1B.yaml`)

| Setting | Value |
|---|---|
| Epochs | 3 |
| Peak LR | 2e-5 (full FT) / 2e-4 (LoRA) |
| Schedule | Cosine, 3% warmup |
| Weight decay | 0.01 |
| Grad clip | 1.0 |
| Max seq len | 2048 (extend to 4096 for long chats) |
| Precision | BF16 |
| Seed | 42 |

```bash
python finetuning/sft.py \
  --base_model experiments/1B/export \
  --tokenizer tokenizer/tokenizer.json \
  --dataset data/instruction/indic_instructions.jsonl \
  --output_dir experiments/1B/sft \
  --epochs 3 --lr 2e-5 --batch_size 32 \
  --method lora --lora_rank 16
```

For a single-GPU Colab T4 path, reuse `Quaere.ai/scripts/finetune_colab.py` (Unsloth QLoRA) with the Indic dataset substituted in.

## Part 3: Preference Alignment

DPO is recommended over RLHF as the first approach: no reward model, no PPO, roughly one order of magnitude cheaper, and stable at this scale.

### Preference data generation
- Sample 5K-10K prompts from the SFT evaluation split
- Generate 2-4 candidate responses per prompt from the SFT model (temperature 0.8-1.0)
- Rank pairs with a teacher model (GPT-4 class) or native-speaker annotators
- Store `{"prompt", "chosen", "rejected"}` per line
- Bias the sample toward: safety, refusal quality, factuality, helpfulness, and Indic-specific failure modes (wrong script, wrong-language reply, code-switching errors)
- Target 2-5K usable pairs per major language after filtering

### DPO trainer (`finetuning/dpo.py`)
- Computes `log p(chosen|x)` and `log p(rejected|x)` under the policy and the frozen SFT reference
- Loss: `-log sigmoid(beta * ((logp_chosen - logp_ref_chosen) - (logp_rejected - logp_ref_rejected)))`
- Memory: policy + frozen reference both loaded. With LoRA the reference is free — disable the adapter to get reference logprobs
- Continue from the SFT checkpoint

```bash
python finetuning/dpo.py \
  --model_path experiments/1B/sft \
  --preference_data data/preferences/indic_preferences.jsonl \
  --output_dir experiments/1B/dpo \
  --beta 0.1 --epochs 1 --lr 5e-6
```

**Hyperparameters**
| Setting | Value |
|---|---|
| Beta | 0.1 |
| Epochs | 1 |
| LR | 5e-6 (5e-7 for full FT of larger models) |
| Max pairs | 50K |
| Max prompt len | 1024 |

### ORPO alternative
ORPO folds SFT and preference training into one pass with a single reference-free objective:
`loss = L_SFT + lambda * L_ORPO`, `lambda = 0.1-1.0`. Useful when SFT data and preference data are both small — one run instead of two.

### Export
- Merge LoRA adapters into the base weights before export
- Write the chat template into `tokenizer_config.json` so downstream inference applies it automatically
- Keep the SFT checkpoint on disk; DPO is cheap to redo, SFT is not

## Running

```bash
# 1. Build the dataset
python finetuning/data_builder.py \
  --sources data/instruction/*.jsonl \
  --output data/instruction/indic_instructions.jsonl \
  --min_per_lang 10000 --dedup --pii_filter

# 2. SFT
python finetuning/sft.py --config configs/sft_1B.yaml

# 3. Generate and rank preference pairs
python finetuning/data_builder.py --build_preferences \
  --sft_model experiments/1B/sft --count 10000

# 4. DPO
python finetuning/dpo.py --config configs/dpo_1B.yaml
```

## Verification

- [ ] SFT loss converges (train <2.0, val <2.5)
- [ ] Model responds in the requested language and script
- [ ] Chat format respected: `<|user|>` always followed by `<|assistant|>`
- [ ] DPO win rate >50% against the SFT baseline in head-to-head judging
- [ ] Toxicity reduced (target <0.1% on IndicToxicity, measured in Stage 6)
- [ ] No regression in base-model perplexity beyond 5%
- [ ] No refusal over-triggering: benign prompts still answered
- [ ] Per-language SFT loss reported; no language left far behind the mean