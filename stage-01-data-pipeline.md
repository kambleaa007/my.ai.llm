# Stage 1: Data Pipeline

The data pipeline collects, cleans, deduplicates, filters, and tokenizes raw text for training an Indian LLM. It is split into 6 stages.

## Directory Structure

indian-llm-architecture/
├── configs/
│   └── pipeline.yaml          # All configuration: sources, ratios, vocab size, etc.
├── scraping/
│   ├── wikipedia.py           # MediaWiki API -> 12 Indic languages
│   ├── news.py                # RSS feeds -> Hindi/Tamil/Telugu/Bengali/Marathi/Gujarati
│   └── commoncrawl.py         # CommonCrawl .in domain + Indic script detection
├── cleaning/
│   └── cleaner.py             # Unicode NFKC, HTML/URL strip, length/alpha validation
├── deduplication/
│   └── deduplicator.py        # MinHash + LSH near-duplicate removal
├── quality_filtering/
│   └── filter.py              # Perplexity + language ID filtering
├── mixing/
│   └── mixer.py               # 70/20/10 ratio balancing + script oversampling
├── tokenizer/
│   └── train.py               # BPE tokenizer (256k vocab, byte-level pre-tokenizer)
├── utils/
│   └── logging.py
└── scripts/
    ├── run_pipeline.py        # Python orchestrator (all 6 stages)
    └── run_pipeline.sh        # Shell wrapper

## Stage Details

### Stage 1: Scraping
Collect raw text from multiple sources.

**Wikipedia** (`scraping/wikipedia.py`)
- Uses MediaWiki API for 12 Indic languages: hi, bn, ta, te, mr, gu, kn, ml, pa, or, as, sa
- Fetches up to 50,000 pages per language
- Extracts plain text, strips HTML, normalizes whitespace
- Output: `data/raw/wikipedia/wikipedia_{lang}.jsonl`

**News** (`scraping/news.py`)
- RSS feeds from major Indian news portals
- Covers: Hindi (Jagran, Amarujala), Tamil (Dinamalar, Thanthi), Telugu (Eenadu), Bengali (Anandabazar), Marathi (Loksatta), Gujarati (Gujarat Mitra, Divya Bhaskar)
- Fetches full article text from each link
- Output: `data/raw/news/news_{lang}.jsonl`

**CommonCrawl** (`scraping/commoncrawl.py`)
- Queries CommonCrawl index for .in domain pages
- Filters for Indic script characters using Unicode block detection
- Extracts text from WARC records
- Output: `data/raw/commoncrawl/commoncrawl_indic.jsonl`

### Stage 2: Cleaning
Normalize and validate raw text.

**TextCleaner** (`cleaning/cleaner.py`)
- Removes HTML tags, URLs, emails, control characters
- Unicode NFKC normalization (handles composed/decomposed forms)
- Whitespace normalization (collapses multiple spaces, trims)
- Validation: min 20 chars, max 200,000 chars, min 5 words, min 60% alpha ratio
- Output: `data/cleaned/{source}/{file}.jsonl`

### Stage 3: Deduplication
Remove exact and near-duplicate documents.

**Deduplicator** (`deduplication/deduplicator.py`)
- Exact dedup: MD5 hash of full text
- Near dedup: MinHash (128 hashes) + LSH (14 bands)
- Jaccard similarity threshold: 0.8
- 5-gram shingle tokenization
- Output: `data/deduped/{source}/{file}.jsonl`

### Stage 4: Quality Filtering
Filter by language and content quality.

**QualityFilter** (`quality_filtering/filter.py`)
- Perplexity estimation using character 4-gram entropy
- Threshold: 1000.0 (configurable)
- Language validation against target languages
- Document length: 20-50,000 words
- Output: `data/filtered/{source}/{file}.jsonl`

### Stage 5: Data Mixing
Balance ratios across sources and languages.

**DataMixer** (`mixing/mixer.py`)
- General text: 70% (Wikipedia, news, web)
- Domain-specific: 20% (code, legal, medical)
- High-quality: 10% (curated, verified)
- Script balancing: oversample minority languages (Sanskrit, Assamese, Odia, Punjabi, Malayalam, Kannada, Gujarati) by 3x
- Code-switching: 5% target for Hinglish/Tanglish
- Output: `data/mixed/mixed.jsonl`

### Stage 6: Tokenizer Training
Train BPE tokenizer on the mixed corpus.

**TokenizerTrainer** (`tokenizer/train.py`)
- Algorithm: Byte Pair Encoding (BPE)
- Vocab size: 256,000 tokens
- Pre-tokenizer: ByteLevel (handles any Unicode byte)
- Min frequency: 2
- Special tokens: <|pad|>, <|unk|>, <|bos|>, <|eos|>, ,
- Output: `tokenizer/tokenizer.json`, `tokenizer/vocab.json`

## Configuration

All settings in `configs/pipeline.yaml`:
- Source URLs and language codes
- Cleaning thresholds
- Dedup parameters
- Quality filter thresholds
- Mixing ratios
- Tokenizer hyperparameters

## Running the Pipeline

cd indian-llm-architecture
pip install -r requirements.txt

# Run all stages
./scripts/run_pipeline.sh all

# Or run individually
./scripts/run_pipeline.sh scrape      # Stage 1: collect raw data
./scripts/run_pipeline.sh clean       # Stage 2: normalize text
./scripts/run_pipeline.sh dedup       # Stage 3: remove duplicates
./scripts/run_pipeline.sh filter      # Stage 4: quality filtering
./scripts/run_pipeline.sh mix         # Stage 5: balance ratios
./scripts/run_pipeline.sh tokenize    # Stage 6: train BPE tokenizer