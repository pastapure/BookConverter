# YouTube Transcript Pipeline

Fetch YouTube video transcripts, clean them up into proper Markdown documents (without losing any content), and upload them to the [askdomainexpert.com](https://askdomainexpert.com) knowledge base.

## Quick Start

```bash
# Set up
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env         # then edit .env with your credentials

# Run
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Usage

```bash
python main.py <youtube-url> [options]
```

**Options:**
| Flag | Description |
|------|-------------|
| `-o, --output FILE` | Output .md path (default: auto-named in `./output/`) |
| `--no-upload` | Skip KB upload, just save the .md file |
| `--lang LANG` | Transcript language code (default: `en`) |
| `--proxy URL` | HTTP/HTTPS proxy (bypass YouTube IP blocks) |
| `--cookies FILE` | Netscape-format cookies.txt for YouTube auth |

**Proxy/cookies** can also be set in `.env` as `YT_PROXY` and `YT_COOKIES`.

## Pipeline Steps

1. **Fetch** — Full transcript via `youtube-transcript-api` (all segments, no missing parts)
2. **Process** — Correct ASR errors (repeated word stutter, missing punctuation) while preserving every word, then format as clean Markdown
3. **Upload** — Authenticate (OAuth2 password flow) and upload to askdomainexpert.com docs API

## Project Structure

```
youtube-transcript-pipeline/
├── main.py                        # CLI entry point
├── transcript_pipeline/
│   ├── fetcher.py                 # YouTube transcript fetching (proxy/cookies support)
│   ├── processor.py               # ASR error correction + Markdown formatting
│   └── uploader.py                # KB API upload (OAuth2 + multipart POST)
├── tests/
│   ├── test_fetcher.py            # 13 tests (extract + fetch + list)
│   └── test_processor.py          # 43 tests (cleaning + markdown + E2E)
├── .env.example                   # Template for credentials
├── requirements.txt
└── README.md
```

## Testing

```bash
pytest tests/ -v
```

> **Note:** Live YouTube tests will fail from cloud provider IPs (YouTube blocks them). They pass from a residential/home IP. Run `pytest tests/ -v -k "not fetch_transcript"` to skip live network tests.
