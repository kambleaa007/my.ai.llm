# Stage 9: Maintenance and Release

Keeps the system running after the first release: data refresh, experiment tracking, deployment safety, cost control, and publication. It is split into 5 parts.

## Directory Structure

indian-llm-architecture/
├── mlops/
│   ├── refresh_data.sh         # Monthly pipeline refresh
│   ├── drift.py                # Data + model drift metrics
│   ├── tracking.py             # MLflow / DVC experiment logging
│   └── rollout.py              # A/B traffic split + auto-rollback
├── configs/
│   └── rollout.yaml            # Traffic split, rollback thresholds
├── data/
│   └── corpora_card.md         # Stage 1 deliverable, kept current
├── docs/
│   ├── model_card.md           # HF model card
│   ├── data_card.md
│   └── runbook.md              # Incident response
└── .github/workflows/
    ├── pipeline.yml            # Scheduled data refresh
    └── eval.yml                # Eval suite on every checkpoint

## Part 1: Continuous Data Pipeline

Refresh the corpus monthly so the model tracks language drift.

```cron
0 0 1 * * cd /path/indian-llm-architecture && ./scripts/run_pipeline.sh all
```

**Refresh cycle**
1. Re-scrape Wikipedia (a few thousand articles change monthly) and news (high churn)
2. Re-clean, re-dedup, re-filter the new raw data
3. Mix new data into the existing corpus, rebalancing to the Stage 1 ratios
4. Re-train the tokenizer **only** if new vocabulary exceeds 20% of the current merge count — otherwise existing IDs would shift and invalidate every checkpoint
5. Compute drift metrics and log them

**Idempotency**: the refresh must be safe to re-run. Content-hash every document, skip ones already ingested, and write only into date-partitioned shards so a failed run can be discarded wholesale.

**Drift tracking** (`mlops/drift.py`)
- Per-language token share, week over week
- Per-script character share
- Domain mix shift (general / domain / high-quality)
- Length and vocabulary distributions
- Alert when any language's share moves more than 20% relative

## Part 2: Experiment Tracking and Versioning

**MLflow**
- Every training run logs: full resolved config, git SHA, container digest, seed, loss curves, throughput, final metrics
- Every fine-tuning run logs the base checkpoint ID, dataset version, and hyperparameters
- Model artifacts registered with aliases (`production`, `staging`, `champion`, `challenger`)

**DVC**
- Version the data pipeline outputs and the corpora card
- Each refresh produces a DVC version, so any corpus state is reconstructible

**Checkpoint hygiene**
- Immutable, content-addressed checkpoints in object storage
- A manifest mapping model ID to tokenizer version, base checkpoint, dataset version, and eval results
- Never mutate a released checkpoint; a release is a new artifact

## Part 3: Deployment and Rollback

**A/B testing** (`mlops/rollout.py`)
- Send 10% of traffic to the challenger checkpoint
- Compare on live traffic: perplexity proxy (logprobs of the SFT eval prompts), task completion rate, latency, error rate, and a sampled quality review
- Promote only when the challenger wins on quality without regressing latency or error rate

**Auto-rollback**
Trigger an automatic revert when:
- Error rate rises >20% relative to baseline
- p95 latency degrades >20%
- Validation perplexity regresses >5%
- Any safety metric from Stage 6 regresses past its threshold

**Deployment targets**
- Blue/green behind a load balancer for the gateway
- Rolling update for the TTS and frontend services
- Health checks and a documented runbook at `docs/runbook.md`

## Part 4: Cost Optimization

| Lever | Saving | Notes |
|---|---|---|
| Spot instances for pre-training | 60-80% | Requires checkpoint-and-resume every 30 min |
| Right-sized inference instances | 30-50% | Match instance type to measured concurrency |
| Request caching | 20-40% of tokens | Cache by normalized prompt hash + params |
| Quantization | 2-4x | INT8/FP8 for GPU, Q4_K_M for CPU |
| Speculative decoding | 2-3x | 125M draft against the 1B target |
| Scheduled scale-down | varies | Nightly idle hours for non-production |
| PagedAttention | 10-20x throughput | Free, from vLLM defaults |

**Budget tracking**
- Tag every cloud resource with the project and stage
- Per-stage cost dashboards; alert at 50%, 80%, and 100% of the budget
- Record actual cost against the Stage 4 estimates and use the delta to re-plan the next run

**Ongoing spend shape**: pre-training is a one-off spike; inference is the recurring cost. Optimize inference first — a 3x throughput gain from PagedAttention plus quantization usually matters more than shaving the training bill.

## Part 5: Publication

**Model card** (`docs/model_card.md`)
- Intended use and explicitly out-of-scope use
- Architecture, parameter count, context length, tokenizer
- Training data summary, token counts per language, data cut-off date
- Evaluation results including the per-language tables from Stage 6
- Safety evaluation results and known limitations
- License and any language-specific restrictions
- Intended hardware for local inference

**Data card** (`data/corpora_card.md`, from Stage 1)
- Total tokens by language
- Source breakdown (Wikipedia %, news %, web %, books %, code %)
- Dedup statistics and filter pass rates
- Known biases, coverage gaps, and the languages that remain thin
- Consent, licensing, and robots.txt compliance notes per source

**Repo**
- All code, configs, and step-by-step reproduction instructions
- Pinned dependency versions plus the training container digest
- Full training logs and hyperparameters
- Issue templates for language-quality bug reports, with a script tag (`hi`, `ta`, ...) so problems cluster by language

**Licensing**
- Apache 2.0 for code and model weights
- Verify every scraped source permits redistribution — CommonCrawl carries per-domain terms, and Piper voice models each carry their own license
- Consider a custom license with an Indic-language-access clause if a permissive license is not the goal

**Launch checklist**
- [ ] Model card and data card published
- [ ] Weights on HuggingFace with the tokenizer and chat template included
- [ ] GGUF uploaded for local use
- [ ] Reproduction instructions verified on a clean machine
- [ ] Safety findings from Stage 6 resolved or documented
- [ ] Per-language limitations stated plainly, including which languages the model is weakest in

## Iteration Cadence

| Cadence | Activity |
|---|---|
| Monthly | Data refresh, drift report, cost review |
| Quarterly | Full Stage 6 evaluation, red-team round, budget re-plan |
| On demand | New model size, extended context via RoPE scaling, new language added |
| Per release | Model card update, A/B evaluation, canary rollout |

## Verification

- [ ] Monthly refresh runs idempotently and produces a DVC version
- [ ] Drift metrics computed and logged per language and per script
- [ ] Every training run traceable in MLflow to config, code SHA, and data version
- [ ] A/B rollout promotes only on quality improvement with no latency regression
- [ ] Auto-rollback verified with an injected failure
- [ ] Cost per training run and per 1M served tokens tracked
- [ ] Model card and data card published with per-language results
- [ ] Weights, tokenizer, and GGUF all load from a clean install
- [ ] Every source's redistribution terms verified and recorded