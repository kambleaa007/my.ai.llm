# Marathi LLM — End-to-End Build

Documentation and implementation for a Marathi-first language model for the people of Maharashtra: data pipeline, pre-training, fine-tuning, evaluation, serving, applications, and maintenance. Monolingual Marathi (ISO 639-1: mr) with Marathi-English code-mixing support.

## Quality Standards (Mandatory)
- Diverse, deduplicated instruction data; no repeated templates.
- Inference uses temperature/top-p sampling for varied, non-repetitive answers.
- Human evaluation by native Marathi speakers required before release.
- Continuous iteration; 100% correctness is not achievable—target best-in-class quality.

## Stages

| Stage | Document | Covers |
|---|---|---|
| 1 | `stage-01-data-pipeline.md` | Scraping, cleaning, dedup, filtering, mixing, tokenizer training |
| 2 | `stage-02-model-architecture.md` | Transformer decoder: RoPE, RMSNorm, SwiGLU, GQA, size presets |
| 3 | `stage-03-pretraining-infrastructure.md` | Token packing, training loop, FSDP distribution, Docker |
| 4 | `stage-04-pretraining-run.md` | Smoke run, full run, monitoring, HF export |
| 5 | `stage-05-finetuning.md` | Instruction data, SFT, DPO/ORPO alignment |
| 6 | `stage-06-evaluation.md` | Intrinsic benchmarks, safety/bias, human eval |
| 7 | `stage-07-serving-inference.md` | Quantization, vLLM, API gateway, RAG |
| 8 | `stage-08-applications.md` | Voice (Indic.ai), Socratic (Quaere.ai), code (retroedit), web app |
| 9 | `stage-09-maintenance-mlops.md` | Data refresh, tracking, rollout, cost, release |

Reference documents: `plan.md` (step-by-step build plan), `raw.md` (architecture overview and concepts).

## Structure

- `scraping/` — Web scraping and data collection
- `cleaning/` — Text cleaning and normalization
- `deduplication/` — Near-duplicate removal
- `quality_filtering/` — Perplexity and language ID filtering
- `mixing/` — Data mixing and ratio balancing
- `tokenizer/` — BPE tokenizer training
- `pretraining/` — Model config and architecture
- `finetuning/` — SFT and DPO (stage 5)
- `evaluation/` — Benchmark and safety harnesses (stage 6)
- `serving/` — Quantization, API gateway, RAG (stage 7)
- `mlops/` — Drift, tracking, rollout (stage 9)
- `configs/` — All configuration files
- `scripts/` — Orchestration and utility scripts

## Quick Start

Stage 1 (data pipeline):
```bash
pip install -r requirements.txt
./scripts/run_pipeline.sh all
```

See `scripts/run_pipeline.sh` for the full pipeline.