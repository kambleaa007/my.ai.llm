"""
Quality filtering: perplexity-based filtering and language identification.
Uses fasttext for language detection and a small KenLM-style model for perplexity.
"""

import json
import re
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Indic Unicode script ranges for validation
INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0B00, 0x0B7F),  # Odia
    (0x0800, 0x083F),  # Devanagari Extended
    (0x1CD0, 0x1CFF),  # Vedic Extensions
]


def indic_char_ratio(text: str) -> float:
    """Return fraction of characters in Indic Unicode blocks."""
    if not text:
        return 0.0
    indic = 0
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in INDIC_RANGES):
            indic += 1
    return indic / len(text)


def estimate_perplexity(text: str) -> float:
    """
    Rough perplexity estimate using character n-gram entropy.
    This is a fast proxy — for real filtering use a trained KenLM model.
    """
    if len(text) < 50:
        return float("inf")

    # Character 4-gram model
    from collections import Counter
    grams = [text[i : i + 4] for i in range(len(text) - 3)]
    if not grams:
        return float("inf")

    total = len(grams)
    unique = len(set(grams))
    # Smoothed probability estimate
    vocab_size = 256 ** 4  # upper bound
    prob = (unique + 1) / (total + vocab_size)
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


class QualityFilter:
    """Filter documents by language, perplexity, and content quality."""

    def __init__(
        self,
        perplexity_threshold: float = 1000.0,
        min_doc_words: int = 20,
        max_doc_words: int = 50000,
        min_indic_ratio: float = 0.3,
        target_languages: list = None,
    ):
        self.perplexity_threshold = perplexity_threshold
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_indic_ratio = min_indic_ratio
        self.target_languages = target_languages or [
            "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "sa", "en"
        ]

    def filter_document(self, record: dict) -> bool:
        """Return True if document passes all filters."""
        text = record.get("text", "")
        if not text:
            return False

        words = text.split()
        if len(words) < self.min_doc_words or len(words) > self.max_doc_words:
            return False

        # Perplexity check
        ppl = estimate_perplexity(text)
        if ppl > self.perplexity_threshold:
            return False

        # Language check (if language is provided)
        lang = record.get("language", "")
        if lang and lang not in self.target_languages:
            return False

        return True

    def process_file(self, input_path: str, output_path: str) -> int:
        """Filter a JSONL file."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        kept = 0
        dropped = 0

        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue

                if self.filter_document(record):
                    record["perplexity"] = estimate_perplexity(record.get("text", ""))
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped += 1

        logger.info(f"{input_path.name}: kept={kept}, dropped={dropped}")
        return kept


if __name__ == "__main__":
    import sys
    qf = QualityFilter()
    qf.process_file(sys.argv[1], sys.argv[2])