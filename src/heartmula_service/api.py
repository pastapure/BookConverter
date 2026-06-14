"""
HeartMuLa Service — FastAPI Router
====================================
Music generation API using the HeartMuLa-oss-3B model.

Endpoints:
    POST /heartmula/generate    — generate music from tags + lyrics
    POST /heartmula/transcribe  — transcribe lyrics from audio
    GET  /heartmula/health      — model status
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/heartmula", tags=["heartmula"])

# Data paths
TAGS_DIR = Path("data/heartmula/tags")
LYRICS_DIR = Path("data/heartmula/lyrics")
ASSETS_DIR = Path("data/heartmula/assets")

ASSETS_DIR.mkdir(parents=True, exist_ok=True)


class GenerateRequest(BaseModel):
    tags: str            # Comma-separated style tags, or path to tags file
    lyrics: str          # Lyrics text, or path to lyrics file
    duration: float = 30.0
    output_name: str = "generated.mp3"


class TranscribeRequest(BaseModel):
    audio_path: str


class HeartMulaStatus(BaseModel):
    model: str = "HeartMuLa-oss-3B"
    loaded: bool = False
    gpu_available: bool = False


@router.get("/health", response_model=HeartMulaStatus)
async def health():
    import torch
    return HeartMulaStatus(
        loaded=False,  # Set to True after lazy load
        gpu_available=torch.cuda.is_available(),
    )


@router.post("/generate")
async def generate(request: GenerateRequest):
    """Generate music from tags and lyrics."""
    try:
        # Resolve tags — read from file if path exists
        tags = request.tags
        tags_path = TAGS_DIR / tags
        if tags_path.exists():
            tags = tags_path.read_text(encoding="utf-8").strip()

        # Resolve lyrics — read from file if path exists
        lyrics = request.lyrics
        lyrics_path = LYRICS_DIR / lyrics
        if lyrics_path.exists():
            lyrics = lyrics_path.read_text(encoding="utf-8").strip()

        output_path = str(ASSETS_DIR / request.output_name)

        from .generator import generate_music
        result_path = generate_music(
            tags=tags,
            lyrics=lyrics,
            output_path=output_path,
            duration=request.duration,
        )

        return {
            "status": "success",
            "output": result_path,
            "tags": tags[:100],
            "lyrics_preview": lyrics[:100],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    """Transcribe lyrics from an audio file."""
    try:
        from .generator import transcribe_lyrics
        lyrics = transcribe_lyrics(request.audio_path)
        return {"status": "success", "lyrics": lyrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
