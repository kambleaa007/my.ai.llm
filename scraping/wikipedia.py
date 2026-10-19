"""
Wikipedia scraper for Indic languages.
Downloads Wikipedia dumps or uses the MediaWiki API for Indic language Wikipedia.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

import requests
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ISO 639-1 codes for Indic languages
INDIC_LANGS = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
}


class WikipediaScraper:
    """Scrape Indic Wikipedia articles via the MediaWiki API."""

    def __init__(self, output_dir: str, max_pages_per_lang: int = 50000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pages = max_pages_per_lang
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "IndicLLM-DataPipeline/1.0"})

    def _api_url(self, lang_code: str) -> str:
        return f"https://{lang_code}.wikipedia.org/w/api.php"

    def _fetch_pages(
        self, lang_code: str, continue_from: Optional[str] = None
    ) -> dict:
        """Fetch a batch of page titles from the API."""
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "aplimit": "500",
            "apfilterredir": "nonredirects",
            "apnamespace": "0",
        }
        if continue_from:
            params["apcontinue"] = continue_from

        resp = self.session.get(self._api_url(lang_code), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fetch_page_content(self, lang_code: str, titles: list[str]) -> dict:
        """Fetch full page content by title."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "exintro": "0",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": "|".join(titles),
            "redirects": "1",
        }

        resp = self.session.get(self._api_url(lang_code), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _extract_text(self, html_text: str) -> str:
        """Strip remaining HTML and normalize whitespace."""
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "sup", "span"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def scrape_language(self, lang_code: str) -> Iterator[dict]:
        """Yield cleaned article dicts for one language."""
        logger.info(f"Scraping Wikipedia for {INDIC_LANGS.get(lang_code, lang_code)}")
        output_file = self.output_dir / f"wikipedia_{lang_code}.jsonl"
        count = 0
        continue_from = None

        while count < self.max_pages:
            data = self._fetch_pages(lang_code, continue_from)
            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            titles = [p["title"] for p in pages if "title" in p]
            content_data = self._fetch_page_content(lang_code, titles)
            page_map = content_data.get("query", {}).get("pages", {})

            for page_id, page in page_map.items():
                if page_id == "-1":
                    continue
                raw_text = page.get("extract", "")
                clean_text = self._extract_text(raw_text)
                if len(clean_text) < 50:
                    continue

                yield {
                    "source": "wikipedia",
                    "language": lang_code,
                    "language_name": INDIC_LANGS.get(lang_code, lang_code),
                    "title": page.get("title", ""),
                    "url": page.get("fullurl", ""),
                    "text": clean_text,
                    "char_count": len(clean_text),
                }
                count += 1

            continue_from = data.get("continue", {}).get("apcontinue")
            if not continue_from:
                break
            time.sleep(0.1)  # rate limiting

        logger.info(f"Finished {lang_code}: {count} articles")

    def run(self) -> None:
        """Scrape all configured Indic languages."""
        for lang_code in INDIC_LANGS:
            output_file = self.output_dir / f"wikipedia_{lang_code}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for article in self.scrape_language(lang_code):
                    f.write(json.dumps(article, ensure_ascii=False) + "\n")
            logger.info(f"Saved {lang_code} to {output_file}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/wikipedia"
    scraper = WikipediaScraper(output_dir=out)
    scraper.run()