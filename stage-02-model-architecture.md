# Stage 2: Model Architecture

Defines the Transformer decoder that consumes the tokenizer and corpus produced by Stage 1. The architecture is GPT-style (decoder-only) with modern efficiency choices: RoPE positions, RMSNorm, SwiGLU feed-forward, Grouped Query Attention, and tied embeddings.

## Directory Structure

indian-llm-architecture/
├── pretraining/
│   ├── config.py              # ModelConfig + size presets (125M ... 7B)
│   ├── model.py               # TransformerDecoder: blocks, attention, MLP, head
│   ├── layers.py              # RMSNorm, RoPE, SwiGLU, GQA primitives
│   └── __init__.py
├── configs/
│   ├── model.yaml             # Final locked hyperparameters per size
│   └── pipeline.yaml          # Vocab size must match Stage 1 tokenizer
└── scripts/
    └── count_params.py        # Parameter count + FLOPs estimate

## Architecture Decisions

| Choice | Value | Why |
|---|---|---|
| Architecture | Transformer decoder (GPT-style) | Simplest scalable LM, matches Llama/Qwen |
| Positional encoding | RoPE (theta=10000) | No learned position table, extrapolates better |
| Normalization | RMSNorm (eps=1e-5) | Fewer ops than LayerNorm, no mean subtraction |
| Activation | SwiGLU | Consistently beats ReLU/GELU at equal params |
| Attention | GQA (16 Q heads / 8 KV heads) | Cuts KV cache ~2x, enables fast decode |
| Token tying | `tie_word_embeddings: True` | Saves 256k x hidden params |
| Dropout | 0.0 (all positions) | Data is large; dropout hurts at this scale |

## Model Presets

Defined in `pretraining/config.py` as `MODEL_SIZES`:

| Size | Layers | Hidden | Q Heads | KV Heads | FFN | Params (approx) |
|------|--------|--------|---------|---------|-----|-----------------|
| 125M | 12 | 768 | 12 | 4 | 2048 | ~125M |
| 350M | 24 | 1024 | 16 | 4 | 2816 | ~350M |
| 750M | 24 | 1536 | 16 | 4 | 4096 | ~750M |
| 1B | 24 | 2048 | 16 | 8 | 5632 | ~1.1B |
| 3B | 28 | 3072 | 24 | 8 | 8192 | ~3.2B |
| 7B | 32 | 4096 | 32 | 8 | 11008 | ~7B |

Shared across all presets: `vocab_size=256000`, `max_position_embeddings=4096`, tied embeddings, RoPE theta 10000.

Retrieve a config with:

```python
from pretraining.config import get_model_config
cfg = get_model_config("1B")
```

## Model Components

### `ModelConfig` (`pretraining/config.py`)
Dataclass carrying every hyperparameter. Notable fields:
- `vocab_size=256000`, `intermediate_size` sized ~8/3 x hidden for SwiGLU
- `num_key_value_heads < num_attention_heads` enables GQA
- `bos_token_id=2`, `eos_token_id=3`, `pad_token_id=0` — must match Stage 1 special token order
- `rope_scaling: Optional[str]` — set to `"linear"` or `"dynamic"` when extending context beyond 4096
- `use_cache: bool = True` — enables KV cache for autoregressive generation

### Transformer Block (`pretraining/model.py`)
Per layer:
1. RMSNorm -> GQA self-attention -> residual add
2. RMSNorm -> SwiGLU MLP -> residual add

Attention uses a causal mask so position `t` only attends to `<= t`. KV cache stores one entry per layer per generated token.

### Primitives (`pretraining/layers.py`)
- **RMSNorm**: `x * rsqrt(mean(x^2) + eps) * weight` — no mean subtraction, no bias
- **RoPE**: rotates Q/K by position-dependent angles; applied per attention head dimension pair
- **SwiGLU**: `down_proj(silu(gate_proj(x)) * up_proj(x))`, `intermediate_size` split 2:1 for gate/up
- **GQA attention**: K/V heads are repeated (`repeat_interleave`) to match Q head count before SDPA

### Embedding and Head
- Input: `nn.Embedding(vocab_size, hidden_size)`
- Output: `nn.Linear(hidden_size, vocab_size)` with `weight = embedding.weight` when tied
- Logits computed in FP32 for stable cross-entropy even under BF16 autocast

## Parameter Counting and FLOPs

Per-layer parameter breakdown (hidden `h`, ffn `f`, kv heads `k`, heads `a`):
- Attention: `4 * h^2 + 2 * h * (k/a) * h`
- MLP: `3 * h * f`
- Norms: `2 * h`

Embedding (tied) contributes `vocab_size * h` counted once.

Training FLOPs estimate: `6 * N_params * N_tokens`. At 1B params and 20B tokens that is ~1.2e20 FLOPs, which is roughly 1-2 weeks on 8x A100.

## Indic-Specific Considerations

- **Vocab size 256k**: Agglutinative languages (Tamil, Telugu, Kannada, Malayalam) explode word-form counts. Larger vocab buys rare-token coverage and lowers sequence length (more tokens per sequence), which is the dominant cost driver.
- **Conjunct tokenization**: Sanskrit/Telugu stack characters (e.g. `क्ष`). ByteLevel BPE merges them into single tokens; verify coverage after Stage 1.
- **Code-switching**: Shared romanized/Indic tokens mean Hinglish/Tanglish segments tokenize more efficiently than separate models would allow.
- **Context length**: 4096 is a floor. Long documents (Wikipedia articles, books) are packed with EOS separators rather than truncated.

## Locking the Config

1. Pick a size preset in `pretraining/config.py`
2. Mirror it into `configs/model.yaml`
3. Confirm `vocab_size` matches `tokenizer/tokenizer.json` from Stage 1
4. Verify token IDs: `pad=0`, `bos=2`, `eos=3`

## Verification

- [ ] Model instantiates and returns logits of shape `(batch, seq_len, 256000)`
- [ ] Parameter count matches the preset table within 5%
- [ ] Causal mask verified: changing a later token does not alter earlier logits
- [ ] GQA KV cache size is half the MHA equivalent at 1B preset
- [ ] `torch.compile` succeeds and does not change outputs
- [ ] Forward + backward pass fits in GPU memory at the target micro-batch size