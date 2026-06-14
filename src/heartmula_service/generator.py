"""
HeartMuLa Music Generation Wrapper.

Wraps the heartlib pipeline for programmatic use as a service.
"""
import os
import sys
import time
from pathlib import Path

import torch

# Heartlib is at D:/Projects/BookConverter/heartmula/heartlib/src
_HEARTLIB_SRC = Path("D:/Projects/BookConverter/heartmula/heartlib/src")
if str(_HEARTLIB_SRC) not in sys.path:
    sys.path.insert(0, str(_HEARTLIB_SRC))

# Checkpoint path
_CKPT_PATH = Path("D:/Projects/BookConverter/heartmula/ckpt/HeartMuLa-oss-3B")

# Lazy imports — loaded on first use
_music_pipeline = None
_lyrics_pipeline = None


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_music_pipeline():
    global _music_pipeline
    if _music_pipeline is None:
        from heartlib.pipelines.music_generation import HeartMuLaGenPipeline

        device = _get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        print(f"[...] Loading HeartMuLa-oss-3B pipeline on {device.type.upper()} ...")
        t0 = time.time()

        _music_pipeline = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=str(_CKPT_PATH),
            device=device,
            dtype=dtype,
            version="3B",
            lazy_load=True,  # Load one model at a time to fit in 16 GB VRAM
        )
        print(f"[OK]  Pipeline loaded in {time.time() - t0:.1f}s")

    return _music_pipeline


def _get_lyrics_pipeline():
    global _lyrics_pipeline
    if _lyrics_pipeline is None:
        from heartlib.pipelines.lyrics_transcription import LyricsTranscriptionPipeline
        _lyrics_pipeline = LyricsTranscriptionPipeline()
    return _lyrics_pipeline


def generate_music(
    tags: str,
    lyrics: str,
    output_path: str = None,
    duration: float = 30.0,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
) -> str:
    """Generate music from text tags and lyrics.

    Args:
        tags: Comma-separated style tags (e.g. "bhangra, energetic")
        lyrics: Song lyrics text
        output_path: Path for output audio. Defaults to data/heartmula/assets/
        duration: Target duration in seconds
        temperature: Sampling temperature (higher = more creative)
        cfg_scale: Classifier-free guidance scale

    Returns:
        Path to generated audio file
    """
    pipeline = _get_music_pipeline()

    if output_path is None:
        assets_dir = Path("data/heartmula/assets")
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(assets_dir / "generated.mp3")

    # Resolve tags — read from file if path exists
    tags_input = tags
    tags_path = Path("data/heartmula/tags") / tags
    if tags_path.exists():
        tags_input = tags_path.read_text(encoding="utf-8").strip()
    elif os.path.isfile(tags):
        tags_input = Path(tags).read_text(encoding="utf-8").strip()

    # Resolve lyrics — read from file if path exists
    lyrics_input = lyrics
    lyrics_path = Path("data/heartmula/lyrics") / lyrics
    if lyrics_path.exists():
        lyrics_input = lyrics_path.read_text(encoding="utf-8").strip()
    elif os.path.isfile(lyrics):
        lyrics_input = Path(lyrics).read_text(encoding="utf-8").strip()

    max_audio_length_ms = int(duration * 1000)

    print(f"[...] Generating music: {max_audio_length_ms/1000:.0f}s, temp={temperature}, cfg={cfg_scale}")
    print(f"     Tags: {tags_input[:80]}...")
    print(f"     Lyrics: {lyrics_input[:80]}...")
    t0 = time.time()

    pipeline(
        {"tags": tags_input, "lyrics": lyrics_input},
        save_path=output_path,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )

    print(f"[OK]  Music generated in {time.time() - t0:.1f}s → {output_path}")
    return output_path


def transcribe_lyrics(audio_path: str) -> str:
    """Transcribe lyrics from an audio file.

    Args:
        audio_path: Path to input audio file

    Returns:
        Transcribed lyrics text
    """
    pipeline = _get_lyrics_pipeline()
    # return pipeline.transcribe(audio_path)
    return ""
