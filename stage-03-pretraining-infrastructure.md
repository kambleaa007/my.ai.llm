# Stage 3: Pre-training Infrastructure

Everything needed to run a distributed pre-training job: token packing, dataloader, training loop, optimizer/scheduler, sharding config, and a reproducible Docker environment. It is split into 4 parts.

## Directory Structure

indian-llm-architecture/
├── pretraining/
│   ├── config.py              # ModelConfig (from Stage 2)
│   ├── model.py               # TransformerDecoder (from Stage 2)
│   ├── data.py                # Token packing + packed Dataset/DataLoader
│   ├── train.py               # Training loop, AMP, accumulation, checkpointing
│   └── dist.py                # FSDP / DDP / activation-checkpoint helpers
├── configs/
│   ├── train_125M.yaml        # Dev / smoke config
│   ├── train_1B.yaml          # Production config
│   └── train_3B.yaml          # Scale-up config
├── requirements-train.txt     # torch, transformers, deepspeed, wandb
├── Dockerfile                 # Reproducible CUDA training image
└── data/processed/
    ├── train_tokens.bin       # uint16/uint32 packed token stream
    └── val_tokens.bin

## Part 1: Tokenization and Packing

Stage 1 produces `data/mixed/mixed.jsonl` and `tokenizer/tokenizer.json`. This part converts them into flat token arrays.

**Packer** (`pretraining/data.py`)
- Loads the Stage 1 tokenizer, encodes each document, appends `<|eos|>` as a separator
- Concatenates all token streams into one contiguous `uint16` array (256k vocab fits in 16 bits)
- Chunks into fixed-length blocks of `block_size = 4096`
- Drops the final partial block

**Memory-mapped loading**
- `np.memmap` over `train_tokens.bin`, so the dataset costs no RAM
- Each sample is `(block_size,)` int64; the label is the same tensor shifted left by one

**Train/Val split**
- 80/20 split by block index, done before any shuffling
- Split is written to `val_tokens.bin` as a separate file so re-runs are deterministic

**Throughput target**: >50,000 tokens/sec per core. If the packer is slower, shard the corpus across processes.

## Part 2: Training Loop

**Trainer** (`pretraining/train.py`)
- Plain PyTorch loop (no Lightning dependency) for full control over sharding
- Optimizer: AdamW with `betas=(0.9, 0.95)`, `weight_decay=0.01`, `eps=1e-8`
- No weight decay on bias, norm weights, or embeddings
- LR schedule: linear warmup over 2,000 steps, then cosine decay to 10% of peak
- Peak LR: `3e-4` for small models (<1B), `1e-4` for 3B+, `6e-5` for 7B
- Mixed precision: BF16 autocast with FP32 master weights (`torch.amp.GradScaler` only for FP16)
- Gradient clipping: global norm `1.0`
- Gradient accumulation: micro-batch x accumulation = 512K effective tokens
- Activation checkpointing on every transformer block (optional, trades ~30% speed for large memory savings)
- `torch.compile` enabled for the transformer block stack

**Loss**
- Cross-entropy over the 256k vocab, computed in FP32
- Ignore index set to `-100` so padding and separator positions do not contribute
- Optional 5% document-masking trick for long-context robustness

**Checkpointing**
- Save every N steps (default 2,000) and when wall-clock exceeds 30 minutes
- Contains: `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, `step`, `config`, `tokens_seen`
- Keep last 3 checkpoints plus the best-validation checkpoint
- Write to local disk, then sync to S3/GCS asynchronously

**Logging**
- TensorBoard or Weights & Biases
- Every 10 steps: loss, tokens/sec, grad norm, LR
- Every 500 steps: validation loss and perplexity on a fixed 2M-token slice

## Part 3: Distributed Configuration

**Parallelism strategy** by model size:

| Model | TP | PP | DP (= world_size / TP / PP) | Sharding |
|---|---|---|---|---|
| 125M | 1 | 1 | 1 | DDP (or single GPU) |
| 1B | 1 | 1 | 8 | FSDP full shard |
| 3B | 1 | 1 | 8 | FSDP full shard + activation ckpt |
| 7B | 2 | 1 | 4 | FSDP full shard + TP=2 |

- **FSDP** shards parameters, gradients, and optimizer states across the DP group — memory scales as `1/N`
- **Tensor parallelism** splits attention/MLP weight matrices inside a layer; requires NVLink
- **Pipeline parallelism** splits layers across stages; micro-batches flow through. Lower bandwidth need, more bubble overhead
- **ZeRO-3** in DeepSpeed is equivalent to FSDP full shard

**Launcher** (`pretraining/dist.py`)
```bash
torchrun --nproc_per_node=8 pretraining/train.py --config configs/train_1B.yaml
```

Multi-node:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         pretraining/train.py --config configs/train_1B.yaml
```

**Sequence parallelism** for TP>1 splits the RMSNorm and dropout activations along the sequence dimension, cutting activation memory by TP size.

**Key environment variables**
```
NCCL_DEBUG=INFO
NCCL_P2P_DISABLE=0
TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_DEVICE_MAX_CONNECTIONS=1
```

## Part 4: Compute and Reproducibility

**Hardware targets**

| Tier | Hardware | Use |
|---|---|---|
| Dev | RTX 4090 24GB / Colab T4 16GB | 125M, short runs |
| Production | 8x A100 80GB or H100 | 1B-7B |
| Alternative | TPU v5e (8x) via GCP | 1B-7B (use `torch_xla`) |

Storage: parallel filesystem (Lustre/GPFS) or S3/GCS for dataset and checkpoints. Never read checkpoints from a spinning-disk share during training.

**Dockerfile** (`Dockerfile`)
- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- Installs PyTorch with the matching CUDA wheel, then `requirements-train.txt`
- Copies the repo, sets `PYTHONPATH`, exposes an entrypoint wrapping `torchrun`
- Pin exact versions; a container digest is the only reproducible training environment

**Reproducibility**
- Seed everything (`seed=42`): Python `random`, NumPy, `torch.manual_seed`, and per-rank offsets for the data sampler
- `torch.use_deterministic_algorithms(True)` where it does not kill throughput
- Log the full resolved config, git SHA, and container digest into every checkpoint

**Data loader workers**
- `num_workers=8` per rank, `pin_memory=True`, `persistent_workers=True`
- Pre-fetch 2 batches per worker to overlap H2D copy with compute

## Running

```bash
pip install -r requirements-train.txt

# 1. Tokenize and pack (from Stage 1 output)
python pretraining/data.py \
  --tokenizer tokenizer/tokenizer.json \
  --input data/mixed/mixed.jsonl \
  --out_dir data/processed \
  --block_size 4096 --val_split 0.2

# 2. Smoke run, single GPU
python pretraining/train.py --config configs/train_125M.yaml --max_steps 10

# 3. Full run, 8 GPUs
torchrun --nproc_per_node=8 pretraining/train.py --config configs/train_1B.yaml

# 4. Resume from checkpoint
torchrun --nproc_per_node=8 pretraining/train.py \
  --config configs/train_1B.yaml --resume experiments/1B/checkpoints/latest.pt
```

## Verification

- [ ] Packed `.bin` contains no `<|unk|>` and has no train/val overlap
- [ ] Tokenization throughput >50K tokens/sec per core
- [ ] Single-GPU forward+backward completes without OOM at the largest 125M micro-batch
- [ ] `torchrun --nproc_per_node=2` completes 20 steps (NCCL sanity)
- [ ] Loss is finite and decreasing on the first 100 steps
- [ ] Checkpoint saves, reloads, and reproduces the next-step loss exactly
- [ ] Same seed + same config reproduces the same loss curve