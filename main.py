#!/usr/bin/env python3
"""
BookConverter & AI Pipeline — Unified CLI
==========================================
Single entry point for PDF conversion, audio transcription, TTS podcast
generation, HeartMuLa music generation, and XTTS voice finetuning.

Modes:
    python main.py --mode pdf --input book.pdf
    python main.py --mode zip --input archives.zip
    python main.py --mode transcript --cli audio.m4a --diarize
    python main.py --mode heartmula --generate --tags bhangra --lyrics song.txt
    python main.py --mode finetune --dataset voice3
    python main.py --mode server
    python main.py --tts-from-md output/transcript.md
"""
import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

# Fix Windows console encoding for Unicode filenames
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Source path ──────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Add heartlib to path for heartmula mode
HEARTLIB_DIR = Path(__file__).resolve().parent / "heartmula" / "heartlib" / "src"
if HEARTLIB_DIR.exists() and str(HEARTLIB_DIR) not in sys.path:
    sys.path.insert(0, str(HEARTLIB_DIR))

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════
# CLI Argument Setup
# ═══════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="BookConverter & AI Pipeline — unified entry point",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# ── Mode ──
parser.add_argument("--mode", choices=["pdf", "zip", "transcript", "heartmula", "finetune", "ideogram", "server"],
                    help="Operation mode")

# ── PDF/ZIP mode ──
parser.add_argument("--input", help="Input file or directory")
parser.add_argument("--output-dir", help="Output directory")

# ── Transcript mode ──
parser.add_argument("--cli", help="Audio/video file to transcribe (transcript mode)")
parser.add_argument("--no-upload", action="store_true", help="Skip KB upload")
parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
parser.add_argument("--speakers", help="Speaker reference JSON or @file")
parser.add_argument("--tts", action="store_true", help="Generate podcast MP3 after transcription")
parser.add_argument("--tts-backend", choices=["edge", "coqui"], default="edge")
parser.add_argument("--tts-voice-map", help="JSON voice mapping for TTS")
parser.add_argument("--tts-output", help="Custom TTS output path")
parser.add_argument("--tts-pause", type=int, help="Pause between speaker turns (ms)")
parser.add_argument("--tts-from-md", help="Generate podcast from existing Markdown file")

# ── HeartMuLa mode ──
parser.add_argument("--generate", action="store_true", help="Generate music (heartmula mode)")
parser.add_argument("--transcribe-lyrics", action="store_true", help="Transcribe lyrics from audio")
parser.add_argument("--tags", help="Music style tags or path to tags file")
parser.add_argument("--lyrics", help="Lyrics text or path to lyrics file")
parser.add_argument("--duration", type=float, default=30.0, help="Music duration in seconds")
parser.add_argument("--output-name", default="generated.mp3")

# ── Ideogram mode ──
parser.add_argument("--prompt", help="Text prompt describing the image to generate")
parser.add_argument("--height", type=int, default=1024, help="Image height (256–2048, multiple of 16)")
parser.add_argument("--width", type=int, default=1024, help="Image width (256–2048, multiple of 16)")
parser.add_argument("--steps", type=int, default=48, help="Inference steps (higher=better, slower)")
parser.add_argument("--guidance-scale", type=float, help="CFG scale (None=recommended schedule)")
parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
parser.add_argument("--no-prompt-upsampling", action="store_true",
                    help="Disable local prompt upsampling")

# ── Finetune mode ──
parser.add_argument("--dataset", choices=["voice3", "bella"], help="Dataset for finetuning")

# ── Server mode ──
parser.add_argument("--port", type=int, default=8000, help="Server port")
parser.add_argument("--host", default="0.0.0.0", help="Server host")


# ═══════════════════════════════════════════════════════════════════
# Mode: PDF Conversion
# ═══════════════════════════════════════════════════════════════════

def run_pdf():
    from book_converter.convert_pdf import convert_pdf_to_markdown
    input_path = args.input or os.environ.get("PDF_INPUT")
    output_dir = args.output_dir or os.environ.get("PDF_OUTPUT_DIR", "output/pdf")
    if not input_path:
        print("ERROR: --input required for PDF mode", file=sys.stderr)
        sys.exit(1)
    convert_pdf_to_markdown(input_path, output_dir)


def run_zip():
    input_path = args.input
    output_dir = args.output_dir or "output/zip"
    if not input_path:
        print("ERROR: --input required for ZIP mode", file=sys.stderr)
        sys.exit(1)
    print(f"Processing ZIP: {input_path} -> {output_dir}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "process_zip", SRC_DIR / "book_converter" / "process_zip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# ═══════════════════════════════════════════════════════════════════
# Mode: Transcript Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_tts_from_md():
    md_path = args.tts_from_md
    if not os.path.isfile(md_path):
        print(f"ERROR: File not found: {md_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from transcript_pipeline.tts_generator import generate_podcast_sync
    except ImportError:
        print("ERROR: TTS dependencies not installed.", file=sys.stderr)
        print("  Install with: pip install -r requirements/tts.txt", file=sys.stderr)
        sys.exit(1)

    voice_map = None
    if args.tts_voice_map:
        voice_map = json.loads(args.tts_voice_map)

    output = args.tts_output or md_path.replace(".md", ".mp3")
    pause = args.tts_pause

    generate_podcast_sync(md_path, output, voice_map=voice_map, pause_ms=pause)
    print(f"Podcast saved to: {output}")


def run_cli():
    input_file = args.cli
    if not input_file:
        print("ERROR: --cli <file> required", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(input_file):
        print(f"ERROR: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribing: {input_file}")

    # ── Reference samples (speaker identification) ──
    reference_samples = None
    if args.speakers:
        if args.speakers.startswith("@"):
            with open(args.speakers[1:], encoding="utf-8") as f:
                reference_samples = json.load(f)
        else:
            reference_samples = json.loads(args.speakers)

    # ── Transcribe ──
    from transcript_pipeline.audio_transcriber import transcribe_file
    transcription_result = transcribe_file(
        input_file,
        model_size=os.getenv("WHISPER_MODEL", "base"),
        device=os.getenv("WHISPER_DEVICE", "cuda"),
        diarize=args.diarize,
        reference_samples=reference_samples,
    )

    if "error" in transcription_result:
        print(f"ERROR: {transcription_result['error']}", file=sys.stderr)
        sys.exit(1)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_file).stem
    safe_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()

    # ── Rewrite as Markdown ──
    from transcript_pipeline.processor import rewrite_as_markdown
    result = rewrite_as_markdown(
        transcription_result["text"],
        title=stem,
        video_id=stem,
        segments=transcription_result.get("segments"),
    )
    safe_stem = safe_stem.replace(" ", "_")
    md_path = os.path.join(output_dir, f"{safe_stem}.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Transcript saved: {md_path}")

    # ── Optional upload ──
    if not args.no_upload:
        try:
            from transcript_pipeline.uploader import upload_document
            upload_document(md_path)
        except Exception as e:
            print(f"Upload skipped: {e}", file=sys.stderr)

    # ── Optional TTS ──
    if args.tts:
        voice_map = None
        if args.tts_voice_map:
            voice_map = json.loads(args.tts_voice_map)
        tts_output = args.tts_output or md_path.replace(".md", ".mp3")

        if args.tts_backend == "coqui":
            from transcript_pipeline.coqui_tts import CoquiTTSEngine, HAS_COQUI_DEPS
            if not HAS_COQUI_DEPS:
                print("Coqui deps not installed. Falling back to edge-tts.", file=sys.stderr)
            else:
                engine = CoquiTTSEngine()
                engine.generate_podcast(md_path, tts_output, voice_map=voice_map)
                print(f"Podcast (Coqui): {tts_output}")
                return

        from transcript_pipeline.tts_generator import generate_podcast_sync
        generate_podcast_sync(md_path, tts_output, voice_map=voice_map, pause_ms=args.tts_pause)
        print(f"Podcast: {tts_output}")


# ═══════════════════════════════════════════════════════════════════
# Mode: HeartMuLa
# ═══════════════════════════════════════════════════════════════════

def run_heartmula():
    from heartmula_service.generator import generate_music, transcribe_lyrics

    if args.generate:
        if not args.tags or not args.lyrics:
            print("ERROR: --tags and --lyrics required for music generation", file=sys.stderr)
            sys.exit(1)
        output = generate_music(
            tags=args.tags, lyrics=args.lyrics,
            output_path=f"data/heartmula/assets/{args.output_name}",
            duration=args.duration,
        )
        print(f"Music saved: {output}")

    elif args.transcribe_lyrics:
        if not args.input:
            print("ERROR: --input required for lyrics transcription", file=sys.stderr)
            sys.exit(1)
        lyrics = transcribe_lyrics(args.input)
        print(f"Lyrics:\n{lyrics}")


# ═══════════════════════════════════════════════════════════════
# Mode: Ideogram
# ═══════════════════════════════════════════════════════════════

def run_ideogram():
    from ideogram_service.generator import generate_image

    if not args.prompt:
        print("ERROR: --prompt required for ideogram mode", file=sys.stderr)
        print("  Example: python main.py --mode ideogram --prompt \"a cat wearing a wizard hat\"", file=sys.stderr)
        sys.exit(1)

    result = generate_image(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        prompt_upsampling=not args.no_prompt_upsampling,
    )
    print(f"\nImage saved: {result['images'][0]}")
    print(f"Size: {result['width']}x{result['height']}")
    print(f"Steps: {result['num_inference_steps']}")
    if result.get("seed") is not None:
        print(f"Seed: {result['seed']}")
    print(f"Time: {result['elapsed_seconds']}s")


# ═══════════════════════════════════════════════════════════════════
# Mode: Finetune
# ═══════════════════════════════════════════════════════════════════

def run_finetune():
    dataset = args.dataset or "voice3"
    finetune_dir = SRC_DIR / "finetune"
    sys.path.insert(0, str(finetune_dir))
    os.chdir(finetune_dir)

    # Override config for selected dataset (patches env-driven values at runtime)
    import config as ft_config
    if dataset == "bella":
        ft_config.DATASET_PATH = str(SRC_DIR.parent / "data" / "tts" / "bella")
        ft_config.DATASET_NAME = "bella"
        ft_config.LANGUAGE = "en"
        ft_config.RUN_NAME = "bella_finetune"
        ft_config.SPEAKER_REFERENCE_ROOT = str(SRC_DIR.parent / "models" / "bella")
        ft_config.SPEAKER_REFERENCE_WAV = str(SRC_DIR.parent / "models" / "bella" / "reference.wav")
        ft_config.OUTPUT_PATH = str(SRC_DIR.parent / "models" / "bella")
        # Switch test sentences to bella's JSON (if it exists)
        _bella_sentences = SRC_DIR.parent / "data" / "tts" / "bella" / "test_sentences.json"
        if _bella_sentences.exists():
            import json as _json
            ft_config.TEST_SENTENCES = _json.loads(_bella_sentences.read_text(encoding="utf-8"))

    import train as ft_train
    ft_train.main()


# ═══════════════════════════════════════════════════════════════════
# Mode: Server (all services)
# ═══════════════════════════════════════════════════════════════════

def run_server():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="BookConverter & AI Pipeline")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Transcript pipeline
    from transcript_pipeline.api import router as transcript_router
    app.include_router(transcript_router, prefix="/transcript")

    # HeartMuLa service
    try:
        from heartmula_service.api import router as heartmula_router
        app.include_router(heartmula_router)
        print("HeartMuLa service mounted at /heartmula")
    except ImportError as e:
        print(f"HeartMuLa not available: {e}")

    # Ideogram service
    try:
        from ideogram_service.api import router as ideogram_router
        app.include_router(ideogram_router)
        print("Ideogram service mounted at /ideogram")
    except ImportError as e:
        print(f"Ideogram not available: {e}")

    @app.get("/")
    async def root():
        return {"service": "BookConverter & AI Pipeline", "docs": "/docs"}

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parser.parse_args()

    # ── Standalone TTS mode (no --mode needed) ──
    if args.tts_from_md:
        run_tts_from_md()
        sys.exit(0)

    # ── CLI transcript mode (no --mode needed, backward compat) ──
    if args.cli:
        run_cli()
        sys.exit(0)

    # ── Mode dispatch ──
    if not args.mode:
        # Default: start server
        print("No mode specified. Starting server on http://0.0.0.0:8000")
        run_server()
        sys.exit(0)

    mode_handlers = {
        "pdf": run_pdf,
        "zip": run_zip,
        "transcript": run_cli,
        "heartmula": run_heartmula,
        "finetune": run_finetune,
        "ideogram": run_ideogram,
        "server": run_server,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler()
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
