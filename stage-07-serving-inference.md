# Stage 7: Serving and Inference

Makes the model from Stage 5 fast and cheap to run: quantization, an inference engine, an API gateway, and an optional RAG path for factual grounding. It is split into 4 parts.

## Directory Structure

indian-llm-architecture/
├── serving/
│   ├── quantize.py            # GPTQ / AWQ / SmoothQuant / GGUF export
│   ├── server.py              # FastAPI gateway + OpenAI-compatible API
│   ├── engine.py              # vLLM wrapper, request queue, streaming
│   ├── auth.py                # API key / JWT, rate limiting
│   └── rag.py                 # Retrieval-augmented generation pipeline
├── scripts/
│   ├── export_gguf.py         # llama.cpp conversion
│   └── bench_serving.py       # Latency / throughput benchmark
├── configs/
│   └── serving.yaml           # Engine, quantization, limits
└── models/
    ├── 1B-fp16/               # Baseline weights
    ├── 1B-int8/               # Quantized weights
    └── 1B-q4_k_m.gguf        # CPU/edge weights

## Part 1: Quantization

### Weight quantization

| Method | Bits | VRAM (1B) | Quality loss | Use |
|---|---|---|---|---|
| FP16/BF16 | 16 | ~2 GB | none | Reference |
| SmoothQuant | W8A8 | ~1 GB | <1% | A100 deployment |
| GPTQ | 4 | ~0.7 GB | 2-4% | Best 4-bit quality |
| AWQ | 4 | ~0.7 GB | 1-3% | Faster than GPTQ, better at low bits |
| bitsandbytes NF4 | 4 | ~0.7 GB | 3-5% | Zero-shot, no calibration set |
| GGUF Q4_K_M | 4 | ~0.7 GB | 3-6% | CPU / edge via llama.cpp |

### Activation and cache quantization
- FP8 KV cache (H100 only): halves KV memory at negligible quality cost
- INT8 or NF4 runtime weights via `bitsandbytes` for a no-calibration path

### Calibration
- 128 calibration samples drawn from the Stage 1 filtered corpus, balanced across the 13 languages
- **Calibrate per language cluster, not on Hindi only** — GPTQ on skewed calibration is a common cause of minority-language quality collapse

### GGUF export
```bash
# Merge LoRA, convert to GGUF, quantize
python scripts/export_gguf.py \
  --model_dir experiments/1B/dpo \
  --out models/1B-q4_k_m.gguf \
  --quantize q4_k_m

# Verify it loads and produces sane output
llama-cli -m models/1B-q4_k_m.gguf -p "नमस्ते, आप कैसे हैं?" -n 128
```

## Part 2: Inference Engine

**Recommended: vLLM**

```bash
vllm serve models/1B-int8 \
  --served-model-name indic-llm-1b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --quantization awq \
  --dtype bfloat16 \
  --enable-chunked-prefill
```

**Engine features**
- **PagedAttention**: KV cache in fixed-size pages, near-zero fragmentation, 10-20x throughput over naive implementations
- **Continuous batching**: finished sequences leave the batch immediately; new requests slot in without waiting
- **Chunked prefill**: long prompts are split, so a 4096-token request cannot stall a running decode
- **Speculative decoding**: a 125M draft model proposes 4-8 tokens; the 1B model verifies in one forward pass. 2-3x speedup when the draft agrees often enough. The Stage 1 125M model is a natural draft candidate

**Alternatives**
- TensorRT-LLM: highest throughput, longest build time
- TGI: good batching, simpler config
- llama.cpp `llama-server`: CPU/edge, GGUF only, what `Quaere.ai` points at via `LLAMA_CPP_URL`

## Part 3: API Gateway

**Server** (`serving/server.py`)

OpenAI-compatible endpoints so existing clients work unchanged:
```
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
GET  /health
GET  /metrics
```

**Request extension for Indic**
```json
{
  "model": "indic-llm-1b",
  "messages": [{"role": "user", "content": "..."}],
  "language": "hi",
  "script": "deva",
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 512,
  "stop": ["<|user|>"],
  "seed": 42
}
```
`language` and `script` inject the right system prompt and chat template. Without them, the model guesses the language from the input — reliable for long input, unreliable for short prompts like "hello".

**Streaming**: Server-Sent Events, with the special tokens stripped from the delta so downstream consumers never see `<|assistant|>`.

**Production concerns** (`serving/auth.py`)
- Auth: API key per tenant, or JWT with per-language quota
- Rate limiting: token bucket keyed on `(api_key, language)` — a 12-language model costs more to serve per English prompt
- Request queue with bounded length; return 429 with `Retry-After` on overflow
- Timeouts: request deadline shorter than the load-balancer deadline, so the gateway sheds cleanly
- Load balancing across multiple engine replicas with least-outstanding-requests
- Prompt logging with PII redaction; retention policy documented

**Scaling**: 1B INT8 on an A100 comfortably handles >50 concurrent streams. Above that, add replicas rather than making one engine bigger.

## Part 4: RAG Pipeline (Optional)

Grounding for factual questions, which Stage 6 shows as a real weakness.

**Components** (`serving/rag.py`)
1. **Ingestion**: chunk cleaned Wikipedia, books, and government documents into ~512-token chunks with 64-token overlap
2. **Embeddings**: Indic-aware model. `BAAI/bge-m3` (multilingual, 8192 context) or an IndicBERT fine-tune
3. **Index**: FAISS (self-hosted, `IndexFlatIP` for <1M chunks, `IVF-PQ` beyond) or Qdrant/pgvector for a managed store
4. **Hybrid retrieval**: BM25 (language-specific analyzers for stemming and agglutination) + dense vectors, fused with Reciprocal Rank Fusion
5. **Reranking**: a cross-encoder over the top 50 candidates — the single biggest quality lever in the pipeline
6. **Prompt construction**: retrieved chunks in a `<|context|>` block, with instructions to cite sources and say when the context is insufficient

**Language handling**
- A separate BM25 index per language; agglutinative languages need language-specific analyzers
- Query translation when the user asks in Hinglish but documents are in Devanagari

**Caching**
- Embed the query once per request, cache retrieval results by normalized query hash
- Semantic cache: skip the LLM entirely when a near-identical question was answered recently

## Running

```bash
# 1. Quantize
python serving/quantize.py --model experiments/1B/dpo --method awq --out models/1B-int8

# 2. Serve
python serving/server.py --config configs/serving.yaml --model models/1B-int8

# 3. Smoke test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"indic-llm-1b","language":"ta","messages":[{"role":"user","content":"வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"}]}'

# 4. Benchmark
python scripts/bench_serving.py --concurrency 1,8,32,64 --prompt_len 256 --output_len 256
```

## Verification

- [ ] GGUF loads in `llama.cpp` CLI and generates coherent Indic text
- [ ] Quantized quality degradation <5% vs FP16 on the Stage 6 benchmark suite
- [ ] No per-language quality regression after quantization (re-run benchmarks per language)
- [ ] Latency <500ms p50 for a 100-token response (1B, A100)
- [ ] Throughput >50 concurrent requests with no timeout
- [ ] Streaming works end-to-end with no special-token leakage
- [ ] Rate limiting, auth, and 429 behavior verified
- [ ] RAG improves FActScore by >20% vs. base; retrieval latency <200ms on a 1,000-doc index