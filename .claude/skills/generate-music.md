---
name: generate-music
description: Generate AI music from style tags and lyrics using HeartMuLa-oss-3B. Also transcribe lyrics from audio.
arguments:
  - name: tags
    description: Comma-separated style tags — e.g. 'bhangra, energetic, punjabi' or 'acoustic, romantic, bollywood'
    required: true
  - name: lyrics
    description: Lyrics text for the song, or path to a .txt file containing lyrics
    required: true
  - name: duration
    description: Target duration in seconds (default 30)
    required: false
  - name: output_name
    description: Output MP3 filename (default auto-generated from tags)
    required: false
  - name: action
    description: 'generate' for music generation, 'transcribe' to extract lyrics from audio
    required: false
---

# HeartMuLa Music Generation

Generate AI music from style tags + lyrics using HeartMuLa-oss-3B (3B parameter music generation model) on GPU.

## Quick start

```bash
# Generate music
python main.py --mode heartmula --generate \
  --tags "bhangra, energetic, punjabi" \
  --lyrics "My lyrics text here" \
  --duration 30

# Generate with lyrics file
python main.py --mode heartmula --generate \
  --tags "acoustic, romantic" \
  --lyrics path/to/lyrics.txt \
  --output-name my_song.mp3

# Transcribe lyrics from audio
python main.py --mode heartmula --transcribe-lyrics --input song.mp3
```

## Execution

```bash
cd D:/Projects/BookConverter && D:/Projects/BookConverter/.venv/Scripts/python.exe main.py --mode heartmula --generate --tags "<tags>" --lyrics "<lyrics>" --duration <duration> [--output-name <name>.mp3]
```

## Style tag examples

From the project's tag files:

| Style | Tags file |
|-------|-----------|
| Bhangra | `data/heartmula/tags/tags_3_bhangra.txt` |
| EDM | `data/heartmula/tags/tags_2_edm.txt` |
| Bollywood | `data/heartmula/tags/tags_1_bollywood.txt` |
| Acoustic | `data/heartmula/tags/tags_4_acoustic.txt` |
| Latin | `data/heartmula/tags/tags_4_latin.txt` |
| Reggaeton | `data/heartmula/tags/tags_4_reggaeton.txt` |

## API (when server is running)

```bash
# Generate
curl -X POST http://localhost:8000/heartmula/generate \
  -H "Content-Type: application/json" \
  -d '{"tags":"bhangra, energetic","lyrics":"Your lyrics here","duration":30}'

# Transcribe lyrics
curl -X POST http://localhost:8000/heartmula/transcribe \
  -F "file=@song.mp3"
```

## Checkpoints

Model weights at `heartmula/ckpt/HeartMuLa-oss-3B/` (~6 GB). Loaded lazily as a singleton on first use.

## Output

Generated music saved to `heartmula/assets/` (or the specified `--output-name` path).
