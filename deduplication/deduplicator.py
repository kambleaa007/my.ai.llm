"""
MinHash-based near-duplicate detection for text documents.
Uses LSH (Locality Sensitive Hashing) for efficient deduplication.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

from ..utils.logging import get_logger

logger = get_logger(__name__)


def tokenize(text: str, n: int = 5) -> set:
    """Generate n-gram shingles from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {text.lower()}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def hash_shingle(shingle: str) -> int:
    """Hash a shingle to an integer."""
    return int(hashlib.md5(shingle.encode("utf-8")).hexdigest(), 16)


class MinHash:
    """MinHash signature for Jaccard similarity estimation."""

    def __init__(self, num_hashes: int = 128):
        self.num_hashes = num_hashes
        # Pre-generate hash functions using different seeds
        self._seeds = [i * 31 + 17 for i in range(num_hashes)]

    def signature(self, shingles: set) -> list:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            return [0] * self.num_hashes

        hashed = [hash_shingle(s) for s in shingles]
        sig = []
        for seed in self._seeds:
            min_val = min((h ^ seed) & 0xFFFFFFFF for h in hashed)
            sig.append(min_val)
        return sig

    @staticmethod
    def jaccard(sig1: list, sig2: list) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LSH:
    """Locality Sensitive Hashing for near-duplicate detection."""

    def __init__(self, num_bands: int = 14, num_hashes: int = 128):
        if num_hashes % num_bands != 0:
            raise ValueError("num_hashes must be divisible by num_bands")
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.minhash = MinHash(num_hashes)
        self._buckets = {}  # band_idx -> {bucket_key: [doc_ids]}

    def _band_key(self, sig: list, band: int) -> tuple:
        start = band * self.rows_per_band
        end = start + self.rows_per_band
        return tuple(sig[start:end])

    def insert(self, doc_id: str, shingles: set) -> list:
        """Insert a document and return list of matching doc IDs."""
        sig = self.minhash.signature(shingles)
        matches = set()

        for band in range(self.num_bands):
            key = self._band_key(sig, band)
            bucket = self._buckets.setdefault(band, {}).setdefault(key, [])
            for existing_id in bucket:
                matches.add(existing_id)
            bucket.append(doc_id)

        return list(matches)


class Deduplicator:
    """Remove exact and near-duplicate documents from a JSONL corpus."""

    def __init__(
        self,
        exact_dedup: bool = True,
        near_dedup: bool = True,
        num_bands: int = 14,
        num_hashes: int = 128,
        jaccard_threshold: float = 0.8,
    ):
        self.exact_dedup = exact_dedup
        self.near_dedup = near_dedup
        self.lsh = LSH(num_bands=num_bands, num_hashes=num_hashes)
        self.jaccard_threshold = jaccard_threshold
        self._seen_hashes = set()

    def deduplicate_file(self, input_path: str, output_path: str) -> int:
        """Process a JSONL file, removing duplicates."""
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
                doc_id = record.get("url") or record.get("title") or str(hash(text))

                # Exact dedup
                if self.exact_dedup:
                    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                    if content_hash in self._seen_hashes:
                        dropped += 1
                        continue
                    self._seen_hashes.add(content_hash)

                # Near dedup
                if self.near_dedup:
                    shingles = tokenize(text)
                    matches = self.lsh.insert(doc_id, shingles)
                    # Verify with actual Jaccard
                    is_dup = False
                    for match_id in matches:
                        # We'd need to store signatures to verify properly;
                        # for now, trust LSH candidate generation
                        is_dup = True
                        break
                    if is_dup:
                        dropped += 1
                        continue

                fout.write(line)
                kept += 1

        logger.info(f"{input_path.name}: kept={kept}, dropped={dropped}")
        return kept


if __name__ == "__main__":
    import sys
    dedup = Deduplicator()
    dedup.deduplicate_file(sys.argv[1], sys.argv[2])