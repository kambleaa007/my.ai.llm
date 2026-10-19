# Stage 8: Application Layer

Wires the served model from Stage 7 into the four existing projects: voice (`Indic.ai`), Socratic tutoring (`Quaere.ai`), code editing (`retroedit`), and a unified web application. It is split into 4 parts.

## Directory Structure

project/
├── Indic.ai/                     # Voice agent (exists)
│   ├── tts_server.py             # Piper TTS, FastAPI, existing
│   ├── voice_agent.py            # NEW: LLM + TTS orchestration
│   └── voices/                   # Piper ONNX voices (exists)
├── Quaere.ai/                    # Socratic tutor (exists)
│   ├── server.js                 # Express backend, provider switch (exists)
│   ├── scripts/
│   │   ├── generate_dataset.py   # Socratic dataset generator (exists)
│   │   └── finetune_colab.py     # Unsloth QLoRA (exists)
│   └── public/                   # Dark Academia UI (exists)
├── retroedit/                    # Code editing engine (exists)
│   └── src/engine.py             # SelfHealingEngine (exists)
├── app/                          # NEW: unified web application
│   ├── backend/                  # FastAPI gateway
│   │   ├── main.py               # Routes to /api/chat, /api/tts, /api/rag, /api/code
│   │   └── docker-compose.yml
│   └── frontend/                 # React + Vite chat UI
└── indian-llm-architecture/
    └── serving/                  # Stage 7 gateway (reused as the LLM backend)

## Part 1: Voice Integration (Indic.ai)

`Indic.ai/tts_server.py` already serves offline Piper TTS with no API key and no cloud calls. The new piece is the orchestration layer.

**Voice agent** (`Indic.ai/voice_agent.py`)
- Detects the input language (Stage 1's fasttext LID, or a script-based heuristic), selects the matching voice
- Calls the Stage 7 `/v1/chat/completions` endpoint with `language` set to the detected language
- Streams LLM tokens into TTS at **sentence boundaries**, not per token. Per-token synthesis destroys prosody and roughly triples latency
- Streams audio chunks back over the websocket so playback starts before generation finishes

**Voice coverage**

| Language | Piper voice | Status |
|---|---|---|
| Hindi | `hi_IN-rohan-medium`, `hi_IN-priyamvada-medium` | Present |
| Bengali | `bn_BD-google-medium` | Present |
| Telugu | `te_IN-maya-medium`, `te_IN-venkatesh-medium` | Present |
| Malayalam | `ml_IN-arjun-medium` | Present |
| Marathi | `mr_IN-google-medium` | Present |
| English | `en_US-lessac-medium` | Present |
| Urdu | `ur_PK-fasih-medium` | Present |
| Tamil | `ta_IN-...` | **Missing — download required** |
| Gujarati, Kannada, Odia, Punjabi, Assamese | — | **Missing — availability limited** |

Gap handling: fall back to the closest available voice with a system-prompt note, and answer in text when no reasonable voice exists. Do not silently synthesize a Tamil response with a Hindi voice.

**Operational notes**
- Piper runs at roughly 1-2x real time on CPU; each voice model is ~60MB and loads lazily on first use
- Each voice model carries its own license — verify before commercial deployment
- Windows PowerShell 5.1 must send request bodies as explicit UTF-8 bytes or Devanagari text returns HTTP 422

## Part 2: Socratic Tutor (Quaere.ai)

The existing app forces Socratic behavior through a hardened system prompt. Fine-tuning moves that behavior into the weights, which is what makes it reliable across 12 languages.

**Approach**
1. Generate Socratic instruction data in Indic languages, reusing `Quaere.ai/scripts/generate_dataset.py` as the template — it already emits `{instruction, input, output}` where every output is a question
2. Translate and human-review the outputs per language (Stage 5 Part 1 process)
3. Fine-tune with `Quaere.ai/scripts/finetune_colab.py` (Unsloth QLoRA) for the cheap path, or `finetuning/sft.py` for a full run
4. Export GGUF and point `Quaere.ai` at it via the existing `llamacpp` provider — no server code changes needed

```bash
# Point the existing app at the new model
AI_PROVIDER=llamacpp
LLAMA_CPP_URL=http://localhost:8080
LLAMA_CPP_MODEL=indic-socratic-q4_k_m
```

**Existing deployment options** (all supported today, no code change)
- `openai-compatible` (Groq free tier) — what the live Render deploy uses
- `huggingface` — HF Inference Endpoint, paid
- `ollama` — local
- `llamacpp` — cheapest path to a self-hosted custom model

**Indic extension**: add a language selector to the chat UI, and a system prompt that names the target language explicitly so short prompts like "explain gravity" still get answered in Hindi.

## Part 3: Code Agent (RetroEdit)

`retroedit/` implements non-monotonic decoding over an AST with editor control tokens (ECTs): `BACKSPACE_N`, `LINE_SEEK_K`, `DELETE_RANGE_I_J`, `PATCH_COMMIT`, `EXEC_CHECKPOINT`, `ROLLBACK_ON_FAIL`.

**Integration**
1. **Reserve ECTs in the tokenizer.** Add the control tokens as special tokens in the Stage 1 tokenizer so they occupy fixed IDs, and include them in the Stage 2 vocab size. RetroEdit's `src/control_tokens.py` is the source of truth for the list
2. **Edit-trace pre-training.** Continue training from the Stage 4 base model on (prompt, edit-trace) pairs: teacher-forced generation of code interleaved with ECTs, then the corrected final program. This is the item on the RetroEdit roadmap, and the Stage 3 loop is the training harness
3. **Generation backend.** `SelfHealingEngine` consumes the LLM's token stream and applies ECTs to `RetroEditBuffer`. The KV-cache splice and branch-rollback pieces in `src/kv_cache.py` are the remaining engine work
4. **Serve with ECT passthrough.** Standard servers filter unknown special tokens; the Stage 7 gateway must be configured to emit them

**Value to the Indic model**: Indian-developer code contains non-English identifiers, Indic-language comments, and transliterated APIs. Training on that mixed-script code makes RetroEdit useful on real Indian repositories, not just toy examples.

## Part 4: Unified Web Application

**Backend** (`app/backend/main.py`)
Routes to the four subsystems, all through the Stage 7 gateway:
```
POST /api/chat   -> /v1/chat/completions
POST /api/tts    -> Indic.ai /tts
POST /api/rag    -> Stage 7 RAG pipeline
POST /api/code   -> RetroEdit SelfHealingEngine
```
- Streams responses as SSE for chat and code; binary chunks for TTS
- Holds no model state itself — the Stage 7 gateway owns all inference
- Single source of auth and rate limiting, delegated from `serving/auth.py`

**Frontend** (`app/frontend/`)
- React + Vite, chat UI with streaming output
- Voice mode toggle, streams to `/api/tts`
- Language selector covering all 13 target languages
- Code panel showing the diff RetroEdit produces, with rollback visible as a history step
- Font coverage for all Indic scripts is a hard requirement — a missing conjunct font breaks more of the experience than any model bug. Bundle Noto Sans Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Oriya, Gurmukhi, and Assamese

**Deployment**
- Multi-stage `Dockerfile` for the frontend build, then the backend runtime
- `docker-compose.yml` with `llm-gateway`, `tts`, `rag`, and `frontend` services
- Health checks on every service; the frontend degrades to text-only if TTS is down
- The chat UI must render correctly in all 13 scripts — verify with a rendering test, not a screenshot

## Verification

- [ ] LLM + TTS round-trip latency <3 seconds end to end
- [ ] Audio quality rated >=4.0/5 by native speakers per language
- [ ] Voice selection correct per language; Tamil gap handled explicitly
- [ ] Socratic model responds with 1-2 probing questions, never a direct answer
- [ ] Socratic tone (calm, rigorous, minimal) holds across all 12 languages
- [ ] RetroEdit generated code passes syntax checks (Python, JS)
- [ ] RetroEdit self-healing rollback works end to end
- [ ] All API endpoints return the documented JSON schema
- [ ] Web UI renders correctly in all 13 Indic scripts, including conjuncts
- [ ] `docker compose up` brings up all services healthy