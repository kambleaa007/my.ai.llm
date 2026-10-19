"""
CommonCrawl subset scraper for .in domain and Indic-script text.
Uses the CommonCrawl index API to find relevant URLs.
"""

import gzip
import io
import json
import re
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger(__name__)

CC_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-2024-30-index?url=.in/*&output=json"
INDIC_SCRIPT_RANGES = [
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
]


def has_indic_script(text: str) -> bool:
    """Check if text contains characters from Indic Unicode blocks."""
    for char in text:
        code = ord(char)
        for lo, hi in INDIC_SCRIPT_RANGES:
            if lo <= code <= hi:
                return True
    return False


class CommonCrawlScraper:
    """Fetch and extract text from CommonCrawl .in domain pages."""

    def __init__(self, output_dir: str, max_pages: int = 50000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "IndicLLM-DataPipeline/1.0"})

    def fetch_index(self) -> Iterator[dict]:
        """Yield index entries for .in URLs."""
        resp = self.session.get(CC_INDEX_URL, timeout=60)
        for line in resp.iter_lines():
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def fetch_cdx(self, url: str, offset: int, length: int) -> bytes:
        """Fetch a CDX-serialized WARC record."""
        cdx_url = f"https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-30/segments/{url}"
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        resp = self.session.get(cdx_url, headers=headers, timeout=30)
        return resp.content

    def extract_text(self, warc_bytes: bytes) -> str:
        """Extract HTML text from a WARC record."""
        try:
            # WARC records are gzip-compressed
            decompressed = gzip.decompress(warc_bytes)
            # Find the HTTP response body
            parts = decompressed.split(b"\r\n\r\n", 2)
            if len(parts) < 3:
                return ""
            html = parts[2]
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            import re as _re
            text = _re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            logger.warning(f"WARC extraction failed: {e}")
            return ""

    def run(self) -> None:
        """Scrape CommonCrawl for Indic-script text."""
        output_file = self.output_dir / "commoncrawl_indic.jsonl"
        count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for entry in self.fetch_index():
                if count >= self.max_pages:
                    break
                url = entry.get("url", "")
                offset = int(entry.get("offset", 0))
                length = int(entry.get("length", 0))

                warc_bytes = self.fetch_cdx(entry.get("filename", ""), offset, length)
                text = self.extract_text(warc_bytes)

                if text and has_indic_script(text) and len(text) > 200:
                    f.write(
                        json.dumps(
                            {
                                "source": "commoncrawl",
                                "url": url,
                                "text": text,
                                "char_count": len(text),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1

                time.sleep(0.01)

        logger.info(f"CommonCrawl: extracted {count} pages -> {output_file}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/commoncrawl"
    CommonCrawlScraper(output_dir=out).run()