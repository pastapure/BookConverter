---
name: tts-inference
description: Generate Hindi speech from text using fine-tuned XTTS-v2 voice model. Model loaded once, then fast generation. Also supports multi-speaker podcast from Markdown.
arguments:
  - name: text
    description: Hindi text to synthesize (inline). Use for short prompts.
    required: false
  - name: file
    description: Path to .txt file with Hindi text to read. Best for long texts.
    required: false
  - name: markdown
    description: Path to .md file with **SPEAKER_00:** and **SPEAKER_01:** labels for multi-speaker podcast.
    required: false
  - name: voice
    description: Voice to use — 'female' (hindi_tts_female) or 'male' (voice3). Default: female.
    required: false
  - name: output
    description: Output audio file path. Default: output/<name>.wav
    required: false
---

# TTS Inference (XTTS-v2)

Generate Hindi speech using fine-tuned XTTS-v2 models. Model loads once (~70s), then each generation is ~7s.

## Voice models

| Voice | Path | Gender |
|-------|------|--------|
| female | `models/hindi_tts_female/` | Female |
| male | `models/voice3/hindi_voice3_xtts-*/` | Male |

## Usage

```bash
# Short text
python infer.py --text "नमस्ते, यह मेरी आवाज़ है।"

# Long text file
python infer.py --file data/samples/test_paragraph.txt --output output/speech.wav

# Multi-speaker podcast from Markdown
python infer.py --markdown data/samples/transcript.md --output output/podcast.mp3
```

## How it works

1. Imports TTS library (~37s one-time)
2. Loads 5.3 GB model from disk (~29s one-time)
3. Model stays in GPU memory — each generation is ~7s
4. Auto-splits long text into chunks to respect token limits

## Output

WAV or MP3 file at 24000 Hz. For Markdown input, segments are concatenated with 800ms pauses between speaker turns.
