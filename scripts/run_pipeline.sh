#!/usr/bin/env bash
# Indian LLM Data Pipeline Runner
# Usage: ./scripts/run_pipeline.sh [scrape|clean|dedup|filter|mix|tokenize|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STAGE="${1:-all}"

echo "============================================"
echo " Indian LLM Data Pipeline - Stage: $STAGE"
echo "============================================"

case "$STAGE" in
  scrape)
    echo ">>> Stage 1: Scraping raw data..."
    python -m scraping.wikipedia data/raw/wikipedia
    python -m scraping.news data/raw/news
    # python -m scraping.commoncrawl data/raw/commoncrawl
    ;;
  clean)
    echo ">>> Stage 2: Cleaning text..."
    python -m scripts.run_pipeline --stage clean
    ;;
  dedup)
    echo ">>> Stage 3: Deduplication..."
    python -m scripts.run_pipeline --stage dedup
    ;;
  filter)
    echo ">>> Stage 4: Quality filtering..."
    python -m scripts.run_pipeline --stage filter
    ;;
  mix)
    echo ">>> Stage 5: Data mixing..."
    python -m scripts.run_pipeline --stage mix
    ;;
  tokenize)
    echo ">>> Stage 6: Tokenizer training..."
    python -m scripts.run_pipeline --stage tokenize
    ;;
  all)
    echo ">>> Running full pipeline..."
    python -m scripts.run_pipeline --stage all
    ;;
  *)
    echo "Usage: $0 [scrape|clean|dedup|filter|mix|tokenize|all]"
    exit 1
    ;;
esac

echo ">>> Done."