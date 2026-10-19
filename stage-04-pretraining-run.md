# Stage 4: Pre-training Run

Executes the training job produced by Stage 3: a smoke run to validate the pipeline, a full run to the target token count, and the monitoring setup to keep it healthy. It is split into 3 parts.

## Directory Structure

indian-llm-architecture/
├── pretraining/
│   └── train.py               # Entry point (from Stage 3)
├── configs/
│   ├── train_125M_smoke.yaml  # 10-step validation config
│   ├── train_1B.yaml          # Production config
│   └── train_3B.yaml
├── scripts/
│   ├── monitor.sh             # Tail TensorBoard + alert checks
│   └── export_checkpoint.py   # HF-compatible weight export
├── experiments/{model_size}/
│   ├── checkpoints/           # latest.pt, step_*.pt, best.pt
│   ├── logs/                  # tensorboard event files
│   └── config_resolved.yaml   # Exact config used for the run
└── data/processed/
    └── train_tokens.bin

## Part 1: Smoke Run

Validates the entire pipeline end-to-end on 100MB of tokenized data before spending GPU money.

**Smoke config** (`configs/train_125M_smoke.yaml`)
- Model: 125M preset
- Data: first 100MB slice of `train_tokens.bin`
- `max_steps: 10`, `micro_batch_size: 4`, `seq_len: 1024`
- Single GPU, no sharding, `torch.compile` disabled
- Checkpointing enabled so save/load paths are exercised

```bash
python pretraining/train.py --config configs/train_125M_smoke.yaml --max_steps 10
```

**Smoke acceptance criteria**
- [ ] Loss starts finite (expected ~10-11 for a fresh 256k-vocab model) and decreases monotonically over 10 steps
- [ ] Gradients are non-zero on every parameter (no dead ReLU/GQA head)
- [ ] Grad norm stays below the 1.0 clip threshold at least once (not permanently clipping)
- [ ] Checkpoint writes, reloads, and the resumed step matches the uninterrupted run
- [ ] No NCCL hangs when launched with `torchrun --nproc_per_node=2`
- [ ] Throughput measured and recorded as the baseline for MFU

## Part 2: Full Pre-training Run

### Budget and duration

| Model | Tokens | GPUs | Duration (est.) | Cost (AWS p4d @ $32/hr) |
|---|---|---|---|---|
| 125M | 5B | 1x A100 | 1-2 days | ~$200 |
| 350M | 10B | 1-2x A100 | 2-3 days | ~$500 |
| 1B | 20B | 8x A100 | 1-2 weeks | ~$5,000 |
| 3B | 50B | 8x A100 | 3-5 weeks | ~$15,000 |
| 7B | 150B | 8x A100 | 6-12 weeks | ~$40,000 |

### Hyperparameters (production config)

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95) |
| Weight decay | 0.01 (excludes bias/norm/embedding) |
| Peak LR | 3e-4 (1B), 1e-4 (3B), 6e-5 (7B) |
| Schedule | 2,000 warmup steps, cosine to 10% |
| Effective batch | 512K tokens |
| Grad clip | 1.0 (global norm) |
| Max sequence | 4096 |
| Precision | BF16 autocast, FP32 master weights |
| Seed | 42 |

### Launching

```bash
torchrun --nproc_per_node=8 pretraining/train.py \
  --config configs/train_1B.yaml \
  --tokenizer tokenizer/tokenizer.json \
  --data data/processed/train_tokens.bin \
  --val_data data/processed/val_tokens.bin \
  --out_dir experiments/1B \
  --resume latest
```

### Cost-saving options

- Spot instances with automatic checkpoint-and-resume (saves 60-80%)
- `torch.compile` with `mode="max-autotune"` (10-20% throughput)
- Activation checkpointing only on layers where memory is tight
- Reduce `val_tokens.bin` to a fixed 2M-token slice for intermediate evals

### Failure handling

| Failure | Detection | Response |
|---|---|---|
| Loss divergence | Loss becomes NaN/Inf | Roll back 2 checkpoints, lower LR by 2x |
| OOM | CUDA OOM exception | Halve micro-batch, raise accumulation |
| NCCL timeout | Rank stops reporting heartbeat | Kill job, resume from latest checkpoint |
| Throughput collapse | tokens/sec drops >30% | Check for thermal throttling or a stalled dataloader |
| Checkpoint corruption | Reload fails | Keep rotating backups; verify checksum on write |

### Deliverable
Trained checkpoint at `experiments/{model_size}/checkpoints/best.pt`, plus `config_resolved.yaml` recording the exact run configuration.

## Part 3: Monitoring

**Tracked metrics**
- Training loss (EMA over 100 steps)
- Validation loss and perplexity (every 500 steps, on a fixed slice)
- Gradient global norm
- Learning rate (should trace the schedule exactly)
- Tokens/sec and MFU: `6 * N_params * tokens / step_time / (flops_per_sec * world_size)`
- GPU utilization, memory, temperature, power draw
- Data pipeline throughput (blocks/sec consumed)

**Dashboard**
```bash
tensorboard --logdir experiments/1B/logs --port 6006
```

**Alerting** (`scripts/monitor.sh`)
- Loss not decreasing over 500 steps
- Grad norm pinned at the clip value for 50 consecutive steps
- MFU below 35% (indicates dataloader bottleneck, not compute)
- GPU temperature above 85C or throttle reason active
- Checkpoint older than 1 hour during an active run

**Early stopping and rotation**
- Stop if validation loss has not improved for 3 consecutive evaluations
- Rotate: keep `best.pt` plus the last 3 step checkpoints, delete the rest

**Target curves**
- Initial loss ~10-11 (256k vocab, uniform prior: `ln(256000) = 12.45`, lower after Stage 1 init assumptions)
- Final train loss <2.0 for a 1B model
- Validation perplexity <20 on Indic held-out text, ideally <15

**Reproducibility check**
- Re-run 200 steps from `latest.pt` with `seed=42` and confirm the loss curve matches the original run to within float noise

## HF Export

**Exporter** (`scripts/export_checkpoint.py`)
- Reshards FSDP state dict into a single `state_dict`
- Applies weight tying (`lm_head.weight = embed_tokens.weight`)
- Converts BF16 weights back to FP32 for the saved artifact
- Writes `config.json` in HuggingFace `LlamaForCausalLM` format so the model loads with `transformers`
- Copies `tokenizer.json` alongside so the tokenizer and weights stay versioned together

```bash
python scripts/export_checkpoint.py \
  --checkpoint experiments/1B/checkpoints/best.pt \
  --out experiments/1B/export \
  --format hf
```

## Verification

- [ ] Smoke run passes all six acceptance criteria
- [ ] Full run completes the target token count with a monotonically decreasing loss (within noise)
- [ ] Validation perplexity target met for the chosen size
- [ ] Training is reproducible: seed 42 + same config reproduces the curve
- [ ] Exported HF checkpoint loads in `transformers` and produces coherent text
- [ ] Total tokens consumed matches the plan (no silent data starvation)
- [ ] Per-language loss tracked separately to detect language collapse