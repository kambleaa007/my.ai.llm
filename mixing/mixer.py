"""
Data mixing: balance ratios across sources, languages, and scripts.
"""

import json
import random
from pathlib import Path
from typing import List

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataMixer:
    """Mix cleaned data according to configured ratios."""

    def __init__(
        self,
        general_ratio: float = 0.70,
        domain_ratio: float = 0.20,
        high_quality_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.general_ratio = general_ratio
        self.domain_ratio = domain_ratio
        self.high_quality_ratio = high_quality_ratio
        self.rng = random.Random(seed)

    def _load_jsonl(self, path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _sample(self, pool: List[dict], ratio: float, total: int) -> List[dict]:
        n = int(total * ratio)
        n = min(n, len(pool))
        return self.rng.sample(pool, n)

    def mix(
        self,
        general_files: List[str],
        domain_files: List[str],
        high_quality_files: List[str],
        output_path: str,
        target_total: int = 1000000,
    ) -> None:
        """Mix data from three tiers and write output."""
        general_pool = []
        for f in general_files:
            general_pool.extend(self._load_jsonl(f))

        domain_pool = []
        for f in domain_files:
            domain_pool.extend(self._load_jsonl(f))

        hq_pool = []
        for f in high_quality_files:
            hq_pool.extend(self._load_jsonl(f))

        logger.info(
            f"Pools: general={len(general_pool)}, domain={len(domain_pool)}, hq={len(hq_pool)}"
        )

        general_sample = self._sample(general_pool, self.general_ratio, target_total)
        domain_sample = self._sample(domain_pool, self.domain_ratio, target_total)
        hq_sample = self._sample(hq_pool, self.high_quality_ratio, target_total)

        mixed = general_sample + domain_sample + hq_sample
        self.rng.shuffle(mixed)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in mixed:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Mixed {len(mixed)} records -> {output_path}")


if __name__ == "__main__":
    import sys
    mixer = DataMixer()
    mixer.mix(
        general_files=sys.argv[1].split(","),
        domain_files=sys.argv[2].split(","),
        high_quality_files=sys.argv[3].split(","),
        output_path=sys.argv[4],
    )