# India AI Model — Build Plan

Comprehensive roadmap for building an India-first AI language model, from raw data to production serving. The plan is organized into numbered steps grouped into phases. Existing code lives under `indian-llm-architecture/`, `Indic.ai/`, `Quaere.ai/`, and `retroedit/`.

---

## Phase 1: Data Collection & Pipeline

### Step 1 — Define Project Scope and Model Targets

Decide model size, languages, and budget.

| Parameter | Recommended (Start) | Scale-up Target |
|-----------|---------------------|-----------------|
| Model size | 125M–350M params | 1B–7B params |
| Target languages | 12 Indic (hi, bn, ta, te, mr, gu, kn, ml, pa, or, as, sa) + en | All 22 official Indian languages |
| Compute | Single A100 / Colab T4 | Multi-GPU (8x A100) or TPU v5e |
| Corpus size | 10–50 GB cleaned text | 200+ GB |
| Timeline | 2–4 weeks (data) | 3–6 months (full model) |

**Action items:**
- [ ] Freeze the target language list and model size in `configs/pipeline.yaml`
- [ ] Set up cloud storage bucket (S3 / GCS) or local disk with 500 GB free space
- [ ] Install dependencies: `pip install -r indian-llm-architecture/requirements.txt`
- [ ] Verify API keys / access tokens for Wikipedia, news RSS, CommonCrawl

### Step 2 — Scrape Raw Source Data

Collect raw text from Indic web sources.

**Sub-steps:**

2a. **Wikipedia** — `indian-llm-architecture/scraping/wikipedia.py`  
Uses the MediaWiki API to fetch articles for 12 Indic languages. Config-driven via `configs/pipeline.yaml` (`sources.wikipedia`). Output: `data/raw/wikipedia/wikipedia_{lang}.jsonl`.

```bash
python scripts/run_pipeline.py --stage scrape
```

2b. **News** — `indian-llm-architecture/scraping/news.py`  
RSS feeds from major Indian news portals (Jagran, Amarujala, Dinamalar, Eenadu, Anandabazar, Loksatta, Gujarat Mitra, etc.). Output: `data/raw/news/news_{lang}.jsonl`.

2c. **Books** — `configs/pipeline.yaml` → `sources.books`  
Crawl Project Gutenberg (Indic titles) and Internet Archive metadata + full text. Output: `data/raw/books/`.

2d. **CommonCrawl** — `indian-llm-architecture/scraping/commoncrawl.py`  
Query CommonCrawl index for `.in` domains; filter WARC records by Indic Unicode script detection. Output: `data/raw/commoncrawl/commoncrawl_indic.jsonl`. Use `cc-matrix` or `ccdump` for efficiency.

2e. **Code** — `configs/pipeline.yaml` → `sources.code`  
Scrape GitHub for Indian developers' repositories (filter by location/timezone or Indian orgs). Output: `data/raw/code/`.

**Verification:**
- [ ] At least 50K articles per major language (hi, ta, te, bn, mr, gu, ml)
- [ ] 5K+ articles per minority language (as, or, pa, kn, sa)
- [ ] Sample 100 random records from each source; spot-check for correct encoding (UTF-8)

### Step 3 — Clean and Normalize Text

`indian-llm-architecture/cleaning/cleaner.py`

Apply cleaning pipeline to all raw JSONL files:

- Unicode NFKC normalization (handles composed/decomposed Devanagari / Tamil units)
- HTML tag removal (BeautifulSoup)
- URL / email / control character stripping
- Whitespace normalization (collapse, trim)
- Validation: min 20 chars, max 200,000 chars, min 5 words, min 60% alphabetic ratio
- Remove boilerplate (navigation, footers) where possible

Output: `data/cleaned/{source}/{file}.jsonl`

**Verification:**
- [ ] No HTML tags remain in any sample
- [ ] No mojibake / broken Unicode sequences
- [ ] Documents below min length threshold removed (>5% of raw data)

### Step 4 — Deduplicate

`indian-llm-architecture/deduplication/deduplicator.py`

Two-pass deduplication:

4a. **Exact dedup** — MD5 hash of full normalized text; drop identical hashes.  
4b. **Near-dedup** — MinHash (128 hashes) + LSH (14 bands, threshold 0.8) using 5-gram shingles. Removes near-duplicate articles scraped from different sources.

Output: `data/deduped/{source}/{file}.jsonl`

**Verification:**
- [ ] Exact dup rate logged (expected 5–15% for Wikipedia)
- [ ] Near-dedup removed documents logged
- [ ] Sample output checked for remaining duplicates

### Step 5 — Quality Filtering

`indian-llm-architecture/quality_filtering/filter.py`

Filter documents by quality and language:

- Perplexity estimation using character 4-gram entropy (fast heuristic; replace with KenLM later)
- Threshold: 1000.0 (configurable)
- Language validation: `fasttext` `lid-176` model, confidence threshold 0.65
- Document length: 20–50,000 words
- Bad-word / boilerplate filter
- Target languages: hi, bn, ta, te, mr, gu, kn, ml, pa, or, as, sa, en

Output: `data/filtered/{source}/{file}.jsonl`

**Verification:**
- [ ] Per-language token count report generated
- [ ] Top 10 highest perplexity documents inspected
- [ ] Low-confidence language-detected documents < 5% of output

### Step 6 — Data Mixing and Script Balancing

`indian-llm-architecture/mixing/mixer.py`

Combine filtered data into a single curated corpus:

- **70% general text** — Wikipedia, news, web, books
- **20% domain-specific** — code, legal, medical, technical
- **10% high-quality** — curated instruction sets, verified sources
- **Script balancing** — oversample minority languages (sa, as, or, pa, ml, kn, gu) by 3x
- **Code-switching** — target 5% Hinglish/Tanglish mixed-language documents

Output: `data/mixed/mixed.jsonl` (sharded, zstd-compressed)

**Verification:**
- [ ] Per-language token distribution histogram
- [ ] Script coverage report (Devanagari, Tamil, Bengali, Telugu, etc.)
- [ ] Code-switching ratio confirmed via transliteration detection

### Step 7 — Train BPE Tokenizer

`indian-llm-architecture/tokenizer/train.py`

Build a byte-level BPE tokenizer on the mixed corpus:

- Algorithm: BPE (Hugging Face `tokenizers` library)
- Vocab size: 256,000 tokens
- Pre-tokenizer: ByteLevel
- Min frequency: 2
- Special tokens: `<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|sep|>`

Output: `tokenizer/tokenizer.json`, `tokenizer/vocab.json`

**Verification:**
- [ ] Tokenizer encodes all 22 Indic scripts without `<|unk|>` on validation text
- [ ] Token breakdown test: 1000 random sentences, <1% unknown tokens
- [ ] Vocab size confirmed at exactly 256,000

### Step 8 — Data Analysis and Documentation

Produce a data card documenting the corpus:

- Total token count by language
- Source breakdown (Wikipedia %, news %, web %, books %)
- Deduplication statistics
- Quality filter pass/fail rates
- Known biases and limitations

Output: `data/mixed/corpora_card.md`

---

## Phase 2: Pre-training Infrastructure

### Step 9 — Finalize Model Architecture

`indian-llm-architecture/pretraining/config.py` already defines `ModelConfig` with presets (125M, 350M, 750M, 1B, 3B, 7B).

Confirm architecture choices:

- **Architecture:** Transformer decoder (GPT-style)
- **Positional encoding:** RoPE (Rotary Position Embedding)
- **Normalization:** RMSNorm
- **Activation:** SwiGLU (SwiGLU = x * sigmoid(x) * W2)
- **Attention:** GQA (Grouped Query Attention) — 8 KV heads shared across 16 Q heads
- **Token tying:** `tie_word_embeddings: True`

Update `configs/pipeline.yaml` or create `configs/model.yaml` with final hyperparameters.

**Deliverables:**
- [ ] Model config locked for target size
- [ ] Parameter count calculation documented
- [ ] Training FLOPs estimate calculated (≈ 6 × N_params × N_tokens)

### Step 10 — Set Up Training Harness

Create `indian-llm-architecture/pretraining/` module with:

10a. **Model definition** — `model.py`  
Implement the GPT-style transformer using PyTorch with the config from Step 9. Key modules:
- `nn.Embedding` (tied with LM head)
- Transformer blocks with GQA, RoPE, SwiGLU, RMSNorm
- Causal attention mask

10b. **Training loop** — `train.py`  
PyTorch Lightning or plain PyTorch + HuggingFace Trainer. Features:
- Mixed precision (BF16 / FP16)
- Gradient accumulation
- Gradient clipping (1.0)
- Checkpointing (every N steps or GB)
- Logging (TensorBoard / Weights & Biases)

10c. **Distributed config** — `configs/train_config.yaml`
- FSDP (Fully Sharded Data Parallel)
- Tensor parallelism (Mesh-TensorFlow / torch.distributed)
- ZeRO-3 optimizer state sharding

10d. **Dockerfile** for reproducible training environment

**Verification:**
- [ ] Model instantiates with config and produces correct output shape
- [ ] 1 batch forward + backward pass completes without OOM (on smallest model)
- [ ] Distributed training launches with 2 GPUs (NCCL test)

### Step 11 — Data Loading and Tokenization Pipeline

Convert `data/mixed/mixed.jsonl` into training-ready tokenized chunks:

1. Load tokenizer (from Step 7)
2. Encode text → token IDs
3. Pack into fixed-length sequences (4096 or 8192 tokens)
4. Save as binary `.bin` files or use `datasets.load_dataset("json", ...)` with streaming

Output: `data/processed/train_tokens.bin`, `data/processed/val_tokens.bin` (80/20 split)

**Verification:**
- [ ] Tokenization throughput: >50,000 tokens/sec per core
- [ ] No `<|unk|>` tokens in validation split
- [ ] Train/val split confirmed (no data leakage)

### Step 12 — Compute Environment Setup

Provision training infrastructure:

- **Single GPU (dev):** RTX 4090, A100 40GB, or Colab Pro T4
- **Multi-GPU (production):** 8× A100 80GB or H100, NVLink + InfiniBand
- **TPU path (alternative):** TPU v5e (8x) via Google Cloud

Storage: Mount S3/GCS bucket or parallel filesystem (Lustre/GPFS) for dataset + checkpoints.

**Verification:**
- [ ] `torch.cuda.is_available()` returns True (or TPU runtime initializes)
- [ ] NCCL tests pass (`torchrun --standalone`)
- [ ] Disk I/O benchmark: >2 GB/s read for streaming data

---

## Phase 3: Pre-training

### Step 13 — Smoke Train on Subset

Run a quick 1-step training run on 100MB of tokenized data to validate the full pipeline end-to-end.

```bash
python pretraining/train.py --config configs/train_125M_smoke.yaml --max_steps 10
```

**Verification:**
- [ ] Loss starts finite and decreases
- [ ] Gradients are non-zero
- [ ] Checkpoint saves correctly
- [ ] No deadlocks in distributed mode

### Step 14 — Full Pre-training Run

Train the model to target token count:

| Model | Tokens | GPUs | Est. Time | Est. Cost (AWS p4d) |
|-------|--------|------|-----------|---------------------|
| 125M  | 5B     | 1×A100 | 1–2 days | ~$200 |
| 1B    | 20B    | 8×A100 | 1–2 weeks | ~$5,000 |
| 3B    | 50B    | 8×A100 | 3–5 weeks | ~$15,000 |
| 7B    | 150B   | 8×A100 | 6–12 weeks | ~$40,000 |

Hyperparameters (per config):

- Optimizer: AdamW (β1=0.9, β2=0.95, weight_decay=0.01)
- Learning rate: 3e-4 (cosine decay with 2K warmup steps)
- Batch size: 512K tokens (effective, via gradient accumulation)
- Max sequence length: 4096

**Deliverable:** Trained checkpoint at `experiments/{model_size}/checkpoints/`

### Step 15 — Monitor Training

Track metrics via TensorBoard / Weights & Biases:

- Training loss (should trend downward, target <2.0 for 1B model)
- Validation loss / perplexity
- Gradient norms
- Learning rate schedule
- GPU/TPU utilization

Set up early-stopping criteria and checkpoint rotation.

**Verification:**
- [ ] Loss curve is monotonically decreasing (within noise)
- [ ] Val perplexity < 20 for 1B model on Indic text
- [ ] Training is reproducible (seed=42, same config → same result)

---

## Phase 4: Fine-tuning

### Step 16 — Prepare Instruction Dataset

Create high-quality instruction-response pairs in Indic languages:

16a. **Existing datasets to leverage:**
- IndicQA (question answering)
- IndicNLG (paraphrasing, summarization)
- NoRobots (English instruction data, partially translatable)
- OASST (OpenAssistant, multilingual)
- MagPie (generated instruction data)

16b. **Generate synthetic instruction data:**
- Use `Quaere.ai/scripts/generate_dataset.py` as template — adapt for Indic instruction generation
- Translate English instructions to Indic languages (human review required)
- Create multi-turn chat format with `<|system|>`, `<|user|>`, `<|assistant|>` tokens

Output: `data/instruction/indic_instructions.jsonl` (~100K–500K examples)

**Verification:**
- [ ] Minimum 10K high-quality instruction pairs per major language
- [ ] Multi-turn conversations have proper turn boundaries
- [ ] No personally identifiable information (PII)

### Step 17 — Supervised Fine-Tuning (SFT)

Fine-tune the pre-trained model on the instruction dataset:

```bash
python pretraining/sft.py \
  --model_path experiments/1B/checkpoints \
  --dataset data/instruction/indic_instructions.jsonl \
  --output_dir experiments/1B/sft \
  --epochs 3 \
  --lr 2e-5 \
  --batch_size 32
```

**Verification:**
- [ ] SFT loss converges (train < 2.0, val < 2.5)
- [ ] Model generates coherent Indic text in response to prompts
- [ ] Chat format is respected (`<|user|>` → `<|assistant|>`)

### Step 18 — Alignment (DPO / ORPO)

Apply direct preference optimization to align the model with human preferences:

18a. **Generate preference data:**
- Collect pairs of (preferred, non-preferred) responses for 5K–10K prompts
- Use GPT-4 or human annotators to rank responses

18b. **Train DPO:**
```bash
python pretraining/dpo.py \
  --model_path experiments/1B/sft \
  --preference_data data/preferences/indic_preferences.jsonl \
  --output_dir experiments/1B/dpo \
  --beta 0.1 \
  --epochs 1
```

**Verification:**
- [ ] DPO win rate > 50% against baseline SFT model
- [ ] Toxicity score reduced (target <0.1% on IndicToxicity)
- [ ] Helpful and harmless evaluation scores improved

**Note:** RLHF (reward model + PPO) is an alternative but more complex; DPO is recommended as the first approach.

---

## Phase 5: Evaluation

### Step 19 — Intrinsic Evaluation

Evaluate perplexity and downstream task performance:

19a. **Perplexity:** Compute token-level perplexity on held-out Indic text from each language. Target: <15 for 1B model.

19b. **IndicQA:** Question answering benchmark across 11+ Indic languages. Target: comparable to existing IndicBERT baselines.

19c. **IndicNLG:** Paraphrasing and summarization tasks. Compare BLEU, ROUGE, METEOR scores.

19d. **IndicCode:** Code generation in Indian-language contexts (if code was in training mix).

**Verification:**
- [ ] Perplexity table per language
- [ ] QA accuracy table (F1 scores)
- [ ] NLG metric scores vs. baselines (IndicBERT, mBERT, XLM-R)

### Step 20 — Safety and Bias Evaluation

20a. **Toxicity:** Run IndicToxicity dataset and the Aarti benchmarks. Target: toxicity rate <0.15%.

20b. **Bias/fairness:** Evaluate across languages for representation parity. Use IndicBias or custom fairness probes.

20c. **Hallucination:** Apply FActScore or custom claim-checking on generated output.

20d. **Red-teaming:** Have native speaker evaluators test edge cases (medical advice, legal advice, political content).

**Verification:**
- [ ] Toxicity < 5th percentile of public benchmarks
- [ ] No language exhibits disproportionate representation gaps (>10% accuracy delta)
- [ ] Safety mitigations documented

### Step 21 — Human Evaluation

Conduct human evaluation with native speakers from target languages:

- Coherence (1–5 scale)
- Factuality (does the model hallucinate?)
- Cultural appropriateness
- Multilingual code-switching ability

**Target scores:** Coherence ≥ 4.0, factuality ≥ 3.5, cultural appropriateness ≥ 4.0.

---

## Phase 6: Serving & Inference

### Step 22 — Quantization

Quantize the model for efficient inference:

- **Training-time quantization:** Apply GPTQ or SmoothQuant during export
- **Runtime quantization:** INT8, FP8, or 4-bit (NF4) via bitsandbytes
- **GGUF export:** For CPU/edge deployment via `llama.cpp`

```bash
# GGUF export (4-bit)
python retroedit/scripts/export_gguf.py --model_dir experiments/1B/sft --out model-q4_k_m.gguf
```

**Verification:**
- [ ] GGUF model loads in `llama.cpp` CLI
- [ ] Throughput: >20 tokens/sec on CPU (7B Q4), >100 tokens/sec on A100 (1B INT8)
- [ ] Output quality degradation <5% vs. FP16 baseline

### Step 23 — Set Up Inference Server

Deploy the model behind a production-grade API:

23a. **Inference engine:** vLLM (recommended), TensorRT-LLM, or TGI  
23b. **Optimizations:** PagedAttention, continuous batching, speculative decoding  
23c. **API gateway:** FastAPI or Triton Inference Server  
23d. **Features:**
- Rate limiting (token bucket per API key)
- Authentication (JWT / API key)
- Request queuing and load balancing
- Token streaming (SSE / WebSockets)

Output: `/api/` service with `/v1/chat/completions` and `/v1/completions` endpoints

**Verification:**
- [ ] Latency < 500ms p50 for 100-token response (1B model, A100)
- [ ] Throughput > 50 concurrent requests without timeout
- [ ] Streaming works end-to-end

### Step 24 — RAG Pipeline (Optional)

Add retrieval-augmented generation for factual grounding:

24a. **Embeddings index:** Indic-aware embeddings (BAAI/bge-m3 or IndicBERT) indexed with FAISS or Pinecone  
24b. **Document ingestion:** Wikipedia articles, books, legal/medical documents  
24c. **Hybrid search:** BM25 + dense vector similarity + reranker  
24d. **Prompt construction:** System prompt with retrieved context

**Verification:**
- [ ] RAG improves factuality scores by >20% vs. base model
- [ ] Retrieval latency < 200ms for 1000-doc index

---

## Phase 7: Application Layer

### Step 25 — Indic.ai Voice Integration

Integrate the pre-trained LLM with the existing TTS infrastructure in `Indic.ai/`:

- Use the LLM for conversational AI (chat, Q&A)
- Pipe responses through `Indic.ai/tts_server.py` with Piper voices for:
  - Hindi (hi_IN-priyamvada, hi_IN-rohan)
  - Tamil (ta_IN-google-medium — needs download)
  - Telugu (te_IN-maya, te_IN-venkatesh)
  - Bengali (bn_BD-google-medium)
  - Marathi, Malayalam, Gujarati, Urdu voices
- Output: `Indic.ai/voice_agent.py` combining LLM + TTS

**Verification:**
- [ ] LLM + TTS round-trip latency < 3 seconds
- [ ] Audio quality rated ≥4.0/5 by native speakers

### Step 26 — Quaere.ai Socratic Agent

Fine-tune a chat variant on the Socratic method using `Quaere.ai/scripts/generate_dataset.py` as a template:

26a. **Generate Indic instruction data** (Step 16) following the Socratic principle — deep, probing questions  
26b. **Apply fine-tuning** via `Quaere.ai/scripts/finetune_colab.py` (adapt for Indic languages)  
26c. **Deploy** as a web application with FastAPI

Output: `Quaere.ai/app_indic.py` — Indic Socratic chatbot

**Verification:**
- [ ] Model responds with 1–2 probing questions, not direct answers
- [ ] Tone is calm, rigorous, and minimalist across all 12 languages

### Step 27 — RetroEdit Code Agent

Integrate the LLM (or a code-specialized variant) with the `retroedit/` framework for non-monotonic code generation:

27a. **Fine-tune** a code-capable variant on Indic-code datasets (transliteration, mixed-language code comments)  
27b. **Deploy** the SelfHealingEngine (`retroedit/src/engine.py`) with the LLM as the generation backend  
27c. **Test** with the example from `retroedit/README.md`

**Verification:**
- [ ] Generated code passes basic syntax checks (Python, JS, etc.)
- [ ] Self-healing (rollback on error) works end-to-end

### Step 28 — Web Application & API

Build a unified interface:

28a. **Frontend:** React/Vue app with chat UI, voice mode toggle, multi-language selector  
28b. **Backend:** FastAPI gateway routing to:
- `/api/chat` → LLM
- `/api/tts` → Indic.ai voice
- `/api/rag` → RAG pipeline
- `/api/code` → RetroEdit agent
28c. **Dockerization:** Multi-stage Dockerfile, docker-compose for local dev

**Verification:**
- [ ] All endpoints return correct JSON schema
- [ ] Web UI renders correctly in all 12 Indic scripts
- [ ] Docker compose starts all services

---

## Phase 8: Maintenance & Iteration

### Step 29 — Continuous Data Pipeline

Set up a cron job / Airflow DAG to refresh data monthly:

```bash
0 0 1 * * cd indian-llm-architecture && ./scripts/run_pipeline.sh all
```

- Re-scrape Wikipedia and news
- Re-clean, re-dedup, re-filter
- Mix new data with existing corpus
- Re-train tokenizer if >20% new vocabulary detected
- Log data drift metrics

### Step 30 — Model Updates and Versioning

Implement MLOps best practices:

30a. **Checkpoint versioning:** MLflow / DVC tracking  
30b. **A/B testing:** Deploy new checkpoints to 10% of traffic, measure perplexity improvement  
30c. **Rollback:** Auto-rollback if error rate or latency degrades >20%

### Step 31 — Cost Optimization

Monitor and optimize cloud spend:

- Use spot instances for pre-training (save 60–80%)
- Enable checkpointing to S3 for resumability
- Right-size inference instances based on traffic patterns
- Implement request caching for repeated queries

### Step 32 — Community & Documentation

Publish the model with full transparency:

- Model card on HuggingFace (`hfReadme.md` from `retroedit/`)
- Data card (from Step 8)
- Full training logs and hyperparameters
- License: Apache 2.0 or custom Indian open-source license
- GitHub repo with all code, configs, and instructions

---

## Execution Order Recommendation

| Priority | Steps | Description |
|----------|-------|-------------|
| **P1** | 1–7 | Data pipeline: scrape, clean, dedup, filter, mix, tokenize |
| **P2** | 9–15 | Train pre-training infrastructure and run a 125M/350M model |
| **P3** | 16–18 | Fine-tune on Indic instruction data (SFT + DPO) |
| **P4** | 19–21 | Evaluation on Indic benchmarks |
| **P5** | 22–28 | Serving, RAG, voice integration, and applications |
| **P6** | 29–32 | Maintenance, cost optimization, and community release |

## Key Decisions to Resolve Before Starting

Ask when beginning:
1. What is the target model size (125M for proof-of-concept, 1B for production)?
2. Which cloud provider (AWS, GCP, Lambda Labs, RunPod)?
3. Budget ceiling (determines corpus size and model size)?
4. Which languages are priority? (Hindi + 2 others, or all 12?)
5. Are existing Indic datasets (IndicQA, IndicNLG) available for evaluation?

---

*This plan builds on the architecture defined in `indian-llm-architecture/raw.md` and the existing pipeline implementation. Each step is designed to be independently runnable and testable.*
