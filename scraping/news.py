"""
News scraper for Indian news websites.
Supports Hindi, Tamil, Telugu, Bengali news portals via RSS feeds.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger(__name__)

# RSS feed URLs for major Indian news portals
NEWS_FEEDS = {
    "hi": [
        ("Hindi", "https://www.jagran.com/feed"),
        ("Hindi", "https://www.amarujala.com/feed"),
        ("Hindi", "https://www.hindustantimes.com/rssfeeds/indiannews"),
    ],
    "ta": [
        ("Tamil", "https://www.dinamalar.com/feed"),
        ("Tamil", "https://www.thanthi.tv/feed"),
        ("Tamil", "https://www.dinamal.com/feed"),
    ],
    "te": [
        ("Telugu", "https://www.eenadu.net/feed"),
        ("Telugu", "https://www.abcandh.com/feed"),
        ("Telugu", "https://www Sakshi.com/feed"),
    ],
    "bn": [
        ("Bengali", "https://www.anandabazar.com/feed"),
        ("Bengali", "https://www.eisamay.com/feed"),
        ("Bengali", "https://www.bengalnews24.com/feed"),
    ],
    "mr": [
        ("Marathi", "https://www.loksatta.com/feed"),
        ("Marathi", "https://www.punyanagari.com/feed"),
    ],
    "gu": [
        ("Gujarati", "https://www.gujaratmitra.com/feed"),
        ("Gujarati", "https://www.divyabhaskar.com/feed"),
    ],
}


class NewsScraper:
    """Scrape news articles from RSS feeds of Indian news portals."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "IndicLLM-DataPipeline/1.0", "Accept": "text/html,application/xhtml+xml"}
        )

    def _fetch_feed(self, url: str) -> Optional[feedparser.FeedParserDict]:
        """Parse an RSS feed."""
        try:
            return feedparser.parse(url)
        except Exception as e:
            logger.warning(f"Failed to fetch feed {url}: {e}")
            return None

    def _extract_article_text(self, url: str) -> Optional[str]:
        """Fetch and extract article body text."""
        try:
            resp = self.session.get(url, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Try to find main content
            article = soup.find("article") or soup.find(
                "div", class_=re.compile(r"(content|article|story|body)", re.I)
            )
            if article:
                text = article.get_text(separator=" ")
            else:
                text = soup.get_text(separator=" ")

            text = re.sub(r"\s+", " ", text).strip()
            return text if len(text) > 200 else None
        except Exception as e:
            logger.warning(f"Failed to extract {url}: {e}")
            return None

    def scrape_feed(self, lang_code: str, portal_name: str, feed_url: str) -> Iterator[dict]:
        """Yield articles from a single RSS feed."""
        feed = self._fetch_feed(feed_url)
        if not feed or not feed.entries:
            return

        for entry in feed.entries[:100]:  # last 100 articles per feed
            url = entry.get("link", "")
            title = entry.get("title", "")
            summary = entry.get("summary", "")

            # Try to get full article text
            full_text = self._extract_article_text(url)
            text = full_text or summary

            if not text or len(text) < 200:
                continue

            yield {
                "source": "news",
                "portal": portal_name,
                "language": lang_code,
                "title": title,
                "url": url,
                "published": entry.get("published", ""),
                "text": text,
                "char_count": len(text),
            }
            time.sleep(0.05)

    def run(self) -> None:
        """Scrape all configured news feeds."""
        for lang_code, feeds in NEWS_FEEDS.items():
            output_file = self.output_dir / f"news_{lang_code}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for portal_name, feed_url in feeds:
                    logger.info(f"Scraping {portal_name} ({lang_code})")
                    for article in self.scrape_feed(lang_code, portal_name, feed_url):
                        f.write(json.dumps(article, ensure_ascii=False) + "\n")
            logger.info(f"Finished {lang_code} news -> {output_file}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/news"
    NewsScraper(output_dir=out).run()