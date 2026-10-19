"""Model configuration for Indian LLM pre-training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer decoder (GPT-style) configuration."""

    vocab_size: int = 64000
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    intermediate_size: int = 8192
    hidden_act: str = "swiglu"
    norm_eps: float = 1e-5
    norm_type: str = "rmsnorm"
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    use_cache: bool = True
    tie_word_embeddings: bool = True
    bos_token_id: int = 2
    eos_token_id: int = 3
    pad_token_id: int = 0


MODEL_SIZES = {
    "125M": ModelConfig(vocab_size=64000, hidden_size=768, num_layers=12,
                        num_attention_heads=12, num_key_value_heads=4,
                        intermediate_size=2048, max_position_embeddings=4096),
    "350M": ModelConfig(vocab_size=64000, hidden_size=1024, num_layers=24,
                        num_attention_heads=16, num_key_value_heads=4,
                        intermediate_size=2816, max_position_embeddings=4096),
    "750M": ModelConfig(vocab_size=64000, hidden_size=1536, num_layers=24,
                        num_attention_heads=16, num_key_value_heads=4,
                        intermediate_size=4096, max_position_embeddings=4096),
    "1B": ModelConfig(vocab_size=64000, hidden_size=2048, num_layers=24,
                      num_attention_heads=16, num_key_value_heads=8,
                      intermediate_size=5632, max_position_embeddings=4096),
    "3B": ModelConfig(vocab_size=64000, hidden_size=3072, num_layers=28,
                      num_attention_heads=24, num_key_value_heads=8,
                      intermediate_size=8192, max_position_embeddings=4096),
    "7B": ModelConfig(vocab_size=64000, hidden_size=4096, num_layers=32,
                      num_attention_heads=32, num_key_value_heads=8,
                      intermediate_size=11008, max_position_embeddings=4096),
}


def get_model_config(size: str = "1B") -> ModelConfig:
    if size not in MODEL_SIZES:
        raise KeyError(f"Unknown model size: {size}. Available: {list(MODEL_SIZES.keys())}")
    return MODEL_SIZES[size]
