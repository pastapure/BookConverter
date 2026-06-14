# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-modal AI pipeline monorepo** with five subsystems under a unified CLI (`main.py`):

| Subsystem | What it does | Source |
|---|---|---|
| **BookConverter** | PDF → Markdown via Docling (GPU-accelerated, formula + picture enrichment) | `src/book_converter/` |
| **Transcript Pipeline** | Audio/video → text via faster-whisper, with diarization, speaker ID, TTS podcast | `src/transcript_pipeline/` |
| **HeartMuLa Service** | AI music generation from style tags + lyrics (HeartMuLa-oss-3B) | `src/heartmula_service/` |
| **XTTS Finetuning** | Coqui XTTS-v2 voice cloning (full + LoRA) | `src/finetune/` |
| **ElevenLabs → Coqui** | ElevenLabs API → Coqui XTTS-v2 training datasets | `src/transcript_pipeline/elevenlabs_tts.py` |

The `README_pipeline.md` is **stale** — it describes a deleted YouTube-transcript architecture. The project transcribes local media files directly via Whisper and converts PDFs via Docling.

## Unified Entry Point

All functionality is accessed through `main.py` with a `--mode` flag (defaults to server):

```bash
# API server (all services)
python main.py --mode server                    # http://0.0.0.0:8000
python main.py --mode server --port 9000

# PDF conversion (BookConverter)
python main.py --mode pdf --input book.pdf --output-dir output/pdf
python main.py --mode zip --input archives.zip --output-dir output/zip

# Transcription (backward-compatible: --cli works without --mode)
python main.py --mode transcript --cli audio.m4a --diarize --no-upload
python main.py --cli audio.m4a --diarize --tts --no-upload

# HeartMuLa music generation
python main.py --mode heartmula --generate --tags bhangra --lyrics song.txt

# XTTS finetuning
python main.py --mode finetune --dataset voice3

# Standalone TTS from existing .md
python main.py --tts-from-md output/transcript.md
```

## Build / Install / Test

```bash
# Base: PDF conversion + transcription + API + upload
pip install -r requirements/base.txt

# Optional: diarization + speaker identification (NO HF token needed)
pip install -r requirements/diarization.txt

# Optional: edge-tts podcast generation (free, NO API key)
pip install -r requirements/tts.txt

# Optional: Coqui XTTS-v2 local voice cloning (NO API key)
pip install -r requirements/coqui.txt

# Optional: ElevenLabs TTS → Coqui dataset generation (needs API key)
pip install -r requirements/elevenlabs.txt

# Run tests (src/ is added to path via conftest.py)
pytest tests/ -v
pytest tests/test_transcript/test_processor.py::TestFixRepeatedWords -v
```

`ffmpeg` and `ffprobe` must be on PATH.

## Environment Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Purpose | Default |
|---|---|---|
| `WHISPER_MODEL` | Model size (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `WHISPER_DEVICE` | Torch device for API mode (`cuda`/`cpu`) | `cuda` |
| `ENABLE_DIARIZATION` | Enable diarization globally (`1`/`true`/`yes`) | off |
| `SPEAKER_ID_THRESHOLD` | Cosine similarity threshold for speaker ID (0.0–1.0) | `0.6` |
| `KB_USERNAME` / `KB_PASSWORD` | Credentials for KB upload | — |
| `KB_UPLOAD_URL` | KB document upload endpoint | `https://askdomainexpert.com/documents/upload` |
| `KB_TOKEN_URL` | KB OAuth2 login endpoint | `https://askdomainexpert.com/login` |
| `TTS_VOICE_MAP_JSON` | Voice mapping for edge-tts podcast (JSON string) | `{"SPEAKER_00":"en-US-GuyNeural","SPEAKER_01":"en-US-JennyNeural"}` |
| `TTS_PAUSE_MS` | Pause between speaker turns in ms | `800` |
| `TTS_DEFAULT_VOICE_00` / `_01` | Per-speaker fallback voices | `en-US-GuyNeural` / `en-US-JennyNeural` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (for dataset generation) | — |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID | `lxYfHSkYm1EzQzGhdbfc` |
| `TTS_BACKEND` | TTS backend: `edge` or `coqui` | `edge` |
| `COQUI_REFERENCE_WAV` | Reference WAV for Coqui XTTS-v2 voice cloning | — |

**No HuggingFace token is required** — diarization and speaker identification use the public SpeechBrain ECAPA-TDNN model.

**No API key is required for edge-tts** — podcast generation uses Microsoft Edge TTS, free with 500+ voices.

## Architecture

### Directory layout

```
BookConverter/
├── main.py                          # Unified CLI + server entry point
├── conftest.py                      # Adds src/ to sys.path for tests
├── config.json                      # BookConverter: PDF/ZIP I/O paths
├── .env.example
├── requirements/                    # Split requirement files
│   ├── base.txt, tts.txt, diarization.txt, coqui.txt, elevenlabs.txt
├── src/
│   ├── book_converter/              # PDF/ZIP → Markdown (Docling)
│   │   ├── convert_pdf.py, process_zip.py
│   ├── transcript_pipeline/         # Audio → text → podcast
│   │   ├── audio_transcriber.py     # Core Whisper engine
│   │   ├── diarizer.py              # SpeechBrain speaker diarization
│   │   ├── speaker_identifier.py    # Voiceprint matching
│   │   ├── processor.py             # ASR correction + Markdown
│   │   ├── tts_generator.py         # edge-tts podcast generation
│   │   ├── coqui_tts.py             # Coqui XTTS-v2 voice cloning + TTS
│   │   ├── elevenlabs_tts.py        # ElevenLabs → Coqui dataset generator
│   │   ├── api.py                   # FastAPI router
│   │   └── uploader.py              # OAuth2 KB upload
│   ├── heartmula_service/           # AI music generation
│   │   ├── generator.py, api.py
│   └── finetune/                    # XTTS-v2 finetuning (config.py, train.py, …)
├── heartmula/                       # Vendored HeartMuLa library
│   └── heartlib/src/heartlib/       # Music generation + lyrics transcription
├── data/
│   ├── tts/                         # TTS datasets (voice3/, bella/)
│   ├── heartmula/                   # tags/, lyrics/, assets/
│   └── books/                       # BookConverter data
├── models/                          # Model checkpoints (xtts_v2/, voice3/)
└── tests/
    └── test_transcript/             # Processor + TTS generator tests
```

### Pipeline flow (Transcript)

```
media file → [ffmpeg extract audio] → [faster-whisper transcribe] → [diarize] → [speaker ID] → [clean + Markdown] → [upload to KB]
                                                                            ↑              ↑                         ↓
                                                                     SpeechBrain    ECAPA-TDNN            [TTS podcast MP3]
                                                                     ECAPA-TDNN     voiceprints          (edge-tts or Coqui)
```

### Subsystem: BookConverter (PDF/ZIP → Markdown)

Uses IBM Docling for GPU-accelerated PDF conversion with:
- **Formula enrichment** — LaTeX math extraction
- **Picture description** — LLaVA via Ollama API for image captions
- **Embedded images** — base64-embedded in output Markdown
- **ZIP mode** — extracts ZIP archives, converts all contained PDFs, skips already-converted files

Configured via `config.json`:
```json
{"input_dir": "F:/NCERT_Books/geo", "output_base_dir": "F:/NCERT_Books/downloads/processed"}
```

Windows HF workaround: sets `HF_HUB_DISABLE_SYMLINKS=1` and `HF_HUB_ENABLE_HF_TRANSFER=0`.

### Subsystem: HeartMuLa Music Generation

Vendored library at `heartmula/heartlib/`. Two pipelines:
- **MusicGenerationPipeline** — generates music from style tags + lyrics text using HeartMuLa-oss-3B
- **LyricsTranscriptionPipeline** — transcribes lyrics from audio

The service wrapper (`src/heartmula_service/`) provides lazy-loaded pipeline singletons and a FastAPI router with endpoints:
- `GET /heartmula/health` — model + GPU status
- `POST /heartmula/generate` — tags + lyrics → MP3
- `POST /heartmula/transcribe` — audio → lyrics text

Checkpoints live at `heartmula/ckpt/HeartMuLa-oss-3B/` (~6 GB).

### Transcript Pipeline modules

- **`audio_transcriber.py`** — Core engine. `prepare_audio()` decides ffmpeg extraction vs passthrough. `transcribe_file()` orchestrates: prepare → transcribe → optional diarization → optional speaker ID. Two Whisper backends: faster-whisper (CTranslate2, preferred) → openai-whisper (fallback). Diarization/speaker-ID failures are logged as warnings and never lose the transcription.

- **`diarizer.py`** — Speaker diarization via SpeechBrain ECAPA-TDNN embeddings + spectral clustering. **No HF token needed.** Pipeline: VAD (energy-based) → extract embeddings per speech segment → spectral clustering (auto-detects speaker count via silhouette score) → assign SPEAKER_00, SPEAKER_01, etc. `assign_speakers()` maps diarization labels onto Whisper segments by maximum temporal overlap. `HAS_PYANNOTE` is always `True` (SpeechBrain is always available).

- **`speaker_identifier.py`** — Maps diarized SPEAKER_00 to actual names using reference voice samples. Builds ECAPA-TDNN voice profiles from reference WAVs, crops audio by speaker turn, extracts embeddings, matches via cosine similarity. Segments below threshold keep their original label. `HAS_EMBEDDING` is always `True`.

- **`processor.py`** — ASR correction pipeline: (1) `_fix_repeated_words()` removes stutter-duplicates of function words only (preserves content-word repeats like "go go go"); (2) `_fix_punctuation()` adds periods and capitalises; (3) `_join_and_normalize()` joins segment lines into flowing paragraphs. `rewrite_as_markdown()` produces the final Markdown with frontmatter, speaker-attributed sections, or standard paragraph flow. Never summarizes — preserves all content.

- **`api.py`** — FastAPI router. Endpoints: `GET /health`, `POST /transcribe` (sync), `POST /transcribe/async` + `GET /transcribe/{task_id}` (background, in-memory task store). All endpoints accept optional `reference_samples` JSON form field for speaker ID. `create_app()` builds standalone app with CORS.

- **`tts_generator.py`** — Podcast audio via Microsoft Edge TTS (`edge-tts`). `parse_speaker_segments_from_md()` extracts `**SPEAKER_XX:**` blocks. `generate_podcast()` (async) and `generate_podcast_from_md()` (async) produce multi-voice MP3s. `generate_podcast_sync()` wraps for CLI. Voice resolution cascades: explicit param → `TTS_VOICE_MAP_JSON` env → `TTS_DEFAULT_VOICE_00`/`_01` env → `DEFAULT_VOICE_MAP` → `DEFAULT_FALLBACK_VOICE`. Flag: `HAS_TTS` (True when both `edge-tts` and `pydub` installed).

- **`coqui_tts.py`** — Local XTTS-v2 voice cloning (no API key). `CoquiTTSEngine` supports zero-shot (base model + reference WAV) and fine-tuned (custom checkpoint) modes. `MultiSpeakerCoquiEngine` maps speaker labels to different voice references for multi-voice podcast generation. XTTS-v2 model (~1.9 GB) auto-downloads on first use. Monkey-patches `torch.load` for PyTorch ≥2.6 compat. Flag: `HAS_COQUI_DEPS` (True when `TTS` package installed).

- **`elevenlabs_tts.py`** — Generates Coqui XTTS-v2 training datasets from ElevenLabs API. `extract_sentences_from_md()` parses transcripts into sentences. `ElevenLabsTTS.generate_coqui_dataset()` uses ThreadPoolExecutor for parallel synthesis, outputs LJSpeech-format `wavs/` + `metadata.csv`. Supports `--save-sentences` dry-run mode to preview before spending API credits. Flag: `HAS_ELEVENLABS_DEPS`.

- **`uploader.py`** — OAuth2 password-flow → Bearer token → multipart upload of `.md` to KB.

- **`__init__.py`** — Version `0.3.0`. Re-exports all public API surface.

### TTS backend comparison

| Feature | edge-tts | Coqui XTTS-v2 |
|---|---|---|
| API key | None | None |
| Model download | None | ~1.9 GB auto-download |
| Voices | 500+ Microsoft neural | Voice cloning from reference WAV |
| GPU | No | Yes (CUDA) |
| Speed | Fast (cloud) | Slow (local inference) |
| Multi-speaker | Via voice map | Via MultiSpeakerCoquiEngine |
| Use case | Quick podcast | Custom cloned voice |

### Key design choices

- **HF-free** — Diarization and speaker ID use public SpeechBrain models downloaded on first use. No token, no registration.
- **Diarization is best-effort** — failures are logged to stderr and never lose the transcription.
- **Device fallback** — faster-whisper tries `cuda` then `cpu` when device is `auto`.
- **Audio preparation is lazy** — WAV/FLAC at 16 kHz passthrough without ffmpeg.
- **Async tasks are in-memory** — lost on restart. Swap to Redis/DB for multi-process.
- **No summarization** — processor preserves every word, corrects only artifacts.
- **`src/` is the Python path root** — `conftest.py` and `main.py` both add `src/` to `sys.path`. All imports use the package name (e.g. `from transcript_pipeline.processor import ...`). Do NOT use relative imports across subsystems.

## Coqui XTTS-v2 Finetuning (Voice Cloning)

Finetune Coqui XTTS-v2 on custom voices. Scripts in `src/finetune/`, data in `data/tts/`, models in `models/`.

**VENV:** Use `D:\Projects\BookConverter\.venv` (Python 3.11.9, PyTorch 2.8.0+cu128, Coqui TTS installed).

**GPU required** — RTX 5080 (16 GB) is the primary training device.

### Available datasets

| Dataset | Path | Clips | Language | Voice |
|---|---|---|---|---|
| voice3 | `data/tts/voice3/` | 250 | Hindi | Female |
| bella | `data/tts/bella/` | 167 | English | Female |

All datasets use **LJSpeech format**: `metadata.csv` with `filename|text` lines and a `wavs/` subdirectory.

### Configuring a finetuning run

**All settings must be set in `.env`** — there are no hardcoded defaults in the code.
Copy the XTTS section from `.env.example` to your `.env` and adjust as needed.

Missing required variables will raise a clear error at startup listing exactly what needs to be set.

```bash
# .env — XTTS finetuning section (ALL variables are REQUIRED)
XTTS_PROJECT_ROOT=D:/Projects/BookConverter
XTTS_DATASET_PATH=D:/Projects/BookConverter/data/tts/voice3
XTTS_DATASET_NAME=voice3
XTTS_LANGUAGE=hi
XTTS_EVAL_SPLIT_RATIO=0.1
XTTS_PRETRAINED_MODEL_ROOT=D:/Projects/BookConverter/models/xtts_v2
XTTS_SPEAKER_REFERENCE_ROOT=D:/Projects/BookConverter/models/voice3
XTTS_SPEAKER_REFERENCE_WAV=D:/Projects/BookConverter/models/voice3/reference.wav
XTTS_OUTPUT_PATH=D:/Projects/BookConverter/models/voice3
XTTS_OUTPUT_DIR_NAME=training_output
XTTS_EPOCHS=50
XTTS_BATCH_SIZE=2
XTTS_EVAL_BATCH_SIZE=1
XTTS_GRAD_ACCUM_STEPS=32        # Effective batch = BATCH_SIZE × this (64)
XTTS_LEARNING_RATE=5e-06
XTTS_SAVE_N_CHECKPOINTS=5
XTTS_RUN_NAME=hindi_voice3_finetune
XTTS_PROJECT_NAME=xtts_finetuning_project
XTTS_SAVE_STEP=50
XTTS_PLOT_STEP=25
XTTS_GPU=0
XTTS_TEST_SENTENCES_JSON=D:/Projects/BookConverter/data/tts/voice3/test_sentences.json
XTTS_INFERENCE_TEMPERATURE=0.75
XTTS_INFERENCE_REPETITION_PENALTY=5.0
```

For a full list with descriptions, see `.env.example`.
Runtime patching via `main.py --dataset bella` still works — it overrides the env-derived values at runtime.

### Running finetuning

```bash
python main.py --mode finetune --dataset voice3
# Or directly:
cd src/finetune
D:/Projects/BookConverter/.venv/Scripts/python.exe train.py
```

### Running inference with fine-tuned model

```bash
cd src/finetune

# --text is required; auto-detects best checkpoint from XTTS_OUTPUT_PATH
D:/Projects/BookConverter/.venv/Scripts/python.exe inference.py \
  --text "नमस्ते, यह मेरी आवाज़ है।" --lang hi

# Explicit checkpoint + output path
D:/Projects/BookConverter/.venv/Scripts/python.exe inference.py \
  --text "Hello world" --lang en \
  --checkpoint D:/Projects/BookConverter/models/voice3/best_model_1017.pth \
  --output my_clip.wav
```

Inference reads all paths from `.env` / config.  Set `XTTS_CHECKPOINT_PATH` to skip auto-detection.
Temperature and repetition penalty come from `XTTS_INFERENCE_TEMPERATURE` / `XTTS_INFERENCE_REPETITION_PENALTY` in `.env` (or override via `--temperature` / `--repetition-penalty`).

## HeartMuLa Music Generation (Usage)

```bash
# CLI — generate music
python main.py --mode heartmula --generate \
  --tags "bhangra, energetic, punjabi" \
  --lyrics "My lyrics text here" \
  --duration 30 --output-name my_song.mp3

# CLI — transcribe lyrics from audio
python main.py --mode heartmula --transcribe-lyrics --input song.mp3

# API (when server is running)
curl -X POST http://localhost:8000/heartmula/generate \
  -H "Content-Type: application/json" \
  -d '{"tags":"bhangra, energetic","lyrics":"Your lyrics here","duration":30}'
```

## Speaker Identification (Transcript Pipeline)

```bash
# CLI with inline JSON
python main.py --cli podcast.mp3 --diarize \
  --speakers '{"Alice":"refs/alice.wav","Bob":"refs/bob.wav"}'

# CLI with JSON file
python main.py --cli podcast.mp3 --diarize --speakers @speakers.json

# API
curl -X POST http://localhost:8000/transcript/transcribe \
  -F "file=@podcast.mp3" -F "diarize=true" \
  -F 'reference_samples={"Alice":"/data/alice.wav","Bob":"/data/bob.wav"}'
```

## ElevenLabs → Coqui Dataset Generation

```bash
# Preview sentences (no API call, no cost)
python -m transcript_pipeline.elevenlabs_tts input.md --save-sentences preview.txt

# Generate dataset
python -m transcript_pipeline.elevenlabs_tts input.md \
  --api-key sk_xxxx --voice-id lxYfHSkYm1EzQzGhdbfc \
  --output-dir ./my_dataset

# From Python
from transcript_pipeline.elevenlabs_tts import generate_coqui_dataset_from_md
generate_coqui_dataset_from_md("transcript.md", "./my_dataset", api_key="sk_xxxx")
```

Output format (LJSpeech):
```
my_dataset/
├── wavs/
│   ├── 0001.wav
│   └── 0002.wav
└── metadata.csv    # 0001.wav|First sentence text.
```
