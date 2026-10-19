"""
Text cleaning and normalization for Indic text.
Handles Unicode normalization, HTML stripping, whitespace cleanup.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Common patterns to remove
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class TextCleaner:
    """Clean and normalize raw text for training."""

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 200000,
        min_words: int = 5,
        normalize_unicode: bool = True,
        unicode_form: str = "NFKC",
        strip_whitespace: bool = True,
        remove_emails: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        min_alpha_ratio: float = 0.6,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.normalize_unicode = normalize_unicode
        self.unicode_form = unicode_form
        self.strip_whitespace = strip_whitespace
        self.remove_emails = remove_emails
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.min_alpha_ratio = min_alpha_ratio

    def clean(self, text: str) -> Optional[str]:
        """Apply all cleaning steps. Returns None if text fails validation."""
        if not text or not isinstance(text, str):
            return None

        # Remove HTML tags
        if self.remove_html:
            text = HTML_PATTERN.sub(" ", text)

        # Remove URLs
        if self.remove_urls:
            text = URL_PATTERN.sub(" ", text)

        # Remove emails
        if self.remove_emails:
            text = EMAIL_PATTERN.sub(" ", text)

        # Remove control characters
        text = CONTROL_CHAR_PATTERN.sub(" ", text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize(self.unicode_form, text)

        # Whitespace cleanup
        if self.strip_whitespace:
            text = WHITESPACE_PATTERN.sub(" ", text)
            text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
            text = text.strip()

        # Length checks
        if len(text) < self.min_length or len(text) > self.max_length:
            return None

        # Word count check
        words = text.split()
        if len(words) < self.min_words:
            return None

        # Alpha ratio check (allow Indic scripts + Latin + spaces)
        alpha_chars = sum(1 for c in text if c.isalpha() or c.isspace())
        if alpha_chars / len(text) < self.min_alpha_ratio:
            return None

        return text

    def process_file(self, input_path: str, output_path: str) -> int:
        """Process a JSONL file, writing cleaned output."""
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

                text = record.get("text", "")
                cleaned = self.clean(text)
                if cleaned is None:
                    dropped += 1
                    continue

                record["text"] = cleaned
                record["char_count"] = len(cleaned)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

        logger.info(f"{input_path.name}: kept={kept}, dropped={dropped}")
        return kept


if __name__ == "__main__":
    import sys
    cleaner = TextCleaner()
    cleaner.process_file(sys.argv[1], sys.argv[2])