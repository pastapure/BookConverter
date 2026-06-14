---
name: convert-pdf
description: Convert PDF files to Markdown using Docling with GPU acceleration, formula enrichment, and picture descriptions.
arguments:
  - name: file
    description: Path to the PDF file (.pdf) or ZIP archive (.zip) to convert
    required: true
  - name: output_dir
    description: Directory to save the output Markdown file(s). Defaults to config.json output_base_dir.
    required: false
---

# PDF / ZIP → Markdown Converter

Convert PDF documents to Markdown using IBM Docling on GPU, with LaTeX formula extraction, LLaVA-powered picture descriptions, and embedded images.

## Quick start

```bash
# Single PDF
python main.py --mode pdf --input "<file>" --output-dir "<output_dir>"

# ZIP of PDFs (extracts + converts all)
python main.py --mode zip --input "<file>.zip" --output-dir "<output_dir>"
```

## Execution

```bash
cd D:/Projects/BookConverter && D:/Projects/BookConverter/.venv/Scripts/python.exe main.py --mode pdf --input "<file>" --output-dir "<output_dir>"
```

For ZIP archives, replace `--mode pdf` with `--mode zip`.

## Features

- **GPU-accelerated** via Docling (CUDA)
- **Formula enrichment** — LaTeX math extracted from PDF equations
- **Picture descriptions** — LLaVA via Ollama API captions images
- **Embedded images** — base64-embedded in output Markdown
- **Incremental** — ZIP mode skips already-converted files

## Config

`config.json` provides default I/O paths:
```json
{"input_dir": "F:/NCERT_Books/geo", "output_base_dir": "F:/NCERT_Books/downloads/processed"}
```

## Requirements

```bash
pip install -r requirements/base.txt
```

Windows: `HF_HUB_DISABLE_SYMLINKS=1` and `HF_HUB_ENABLE_HF_TRANSFER=0` are set automatically for HuggingFace compatibility.
