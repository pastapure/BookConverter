---
name: generate-image
description: Generate AI images using Ideogram 4.0 running locally on GPU. Takes a text prompt and produces a PNG image.
arguments:
  - name: prompt
    description: Text description of the image to generate (required). E.g. "a ginger cat wearing a wizard hat in a library"
    required: true
  - name: height
    description: Image height in pixels — multiple of 16, 256–2048 (default 1024)
    required: false
  - name: width
    description: Image width in pixels — multiple of 16, 256–2048 (default 1024)
    required: false
  - name: steps
    description: Inference steps — higher = better quality but slower (default 48, range 1–100)
    required: false
  - name: seed
    description: Random seed for reproducibility (omit for random)
    required: false
  - name: guidance_scale
    description: CFG scale — higher = more prompt adherence (default uses recommended schedule)
    required: false
---

# Image Generation (Ideogram 4.0)

Generate AI images from text prompts using Ideogram 4.0 running **fully locally** on GPU.

## Requirements

- **GPU**: RTX 5080 16GB (or better)
- **RAM**: 32GB+ recommended (model loads ~28GB total)
- **Dependencies**: `pip install -r requirements/ideogram.txt`
- **Model**: Auto-downloaded from HuggingFace on first run (`HF_TOKEN` in `.env`)
- **Cache**: Model stored at `C:/huggingface_models/ideogram4`

## Quick start

```bash
python main.py --mode ideogram --prompt "<prompt>" --height 1024 --width 1024

# With seed for reproducibility
python main.py --mode ideogram --prompt "<prompt>" --seed 42

# High quality (48 steps, full resolution)
python main.py --mode ideogram --prompt "<prompt>" --height 2048 --width 2048 --steps 48 --seed 42

# Fast preview (20 steps, 512px)
python main.py --mode ideogram --prompt "<prompt>" --height 512 --width 512 --steps 20
```

## Execution

When invoked, run:

```bash
cd D:/Projects/BookConverter && D:/Projects/BookConverter/.venv/Scripts/python.exe main.py --mode ideogram --prompt "<prompt>" --height <height> --width <width> --steps <steps> [--seed <seed>] [--guidance-scale <scale>]
```

The model loads lazily on first call (~3 min warmup), then generates in ~30–60s depending on resolution and steps.

## Output

Images saved to `data/ideogram/assets/ideogram4_<width>x<height>_<seed>_<timestamp>.png`.

## Tips

| Goal | Settings |
|------|----------|
| Fast preview | `--height 512 --width 512 --steps 15` |
| Best quality | `--height 2048 --width 2048 --steps 48` |
| Reproducible | `--seed <number>` |
| Creative/artistic | Lower guidance scale or omit for recommended schedule |
| Exact prompt match | `--guidance-scale 7.0` |

## Config (.env)

| Var | Default | Description |
|-----|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token for gated model download |
| `IDEOGRAM_CACHE_DIR` | `C:/huggingface_models/ideogram4` | Where model weights are stored |
| `IDEOGRAM_SKIP_ENHANCER` | `1` | Skip prompt enhancer head (saves ~3GB RAM) |
