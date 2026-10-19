"""
Train a BPE tokenizer on the mixed Indic corpus.
Uses Hugging Face Tokenizers library for speed.
"""

import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Special tokens for the model
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|sep|>",
]


class TokenizerTrainer:
    """Train a BPE tokenizer on Indic text."""

    def __init__(
        self,
        vocab_size: int = 64000,
        min_frequency: int = 2,
        special_tokens: list = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self.tokenizer = None

    def train(self, data_files: list, output_dir: str) -> str:
        """Train the tokenizer and save it."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize BPE tokenizer with byte-level pre-tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=byte_level_alphabet(),
        )

        logger.info(f"Training BPE tokenizer with vocab_size={self.vocab_size}")
        logger.info(f"Training files: {data_files}")

        tokenizer.train(files=data_files, trainer=trainer)

        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Also save vocab for inspection
        vocab = tokenizer.get_vocab()
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to {vocab_path} ({len(vocab)} tokens)")

        self.tokenizer = tokenizer
        return str(tokenizer_path)

    def encode_file(self, input_path: str, output_path: str) -> int:
        """Tokenize a JSONL file and save token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained yet")

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text", "")
                if not text:
                    continue
                tokens = self.tokenizer.encode(text).ids
                fout.write(json.dumps({"ids": tokens, "source": record.get("source", "")}) + "\n")
                count += 1

        logger.info(f"Tokenized {count} records -> {output_path}")
        return count


def byte_level_alphabet():
    """Return the byte-level alphabet for initial BPE training."""
    return bytes(range(256)).decode("latin-1")


if __name__ == "__main__":
    import sys
    import yaml

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pipeline.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "tokenizer"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tok_config = config["tokenizer"]
    data_files = sys.argv[3:] if len(sys.argv) > 3 else ["data/mixed/mixed.jsonl"]

    trainer = TokenizerTrainer(
        vocab_size=tok_config["vocab_size"],
        min_frequency=tok_config["min_frequency"],
        special_tokens=tok_config["special_tokens"],
    )
    trainer.train(data_files, output_dir)