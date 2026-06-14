"""
Ideogram Service — FastAPI Router
==================================
Local image generation using Ideogram 4.0 via HuggingFace Diffusers.

Endpoints:
    POST /ideogram/generate  — text-to-image generation
    GET  /ideogram/health    — pipeline status
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ideogram", tags=["ideogram"])


# ── Request / Response models ─────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    height: int = Field(default=1024, ge=256, le=2048,
                        description="Image height (multiple of 16)")
    width: int = Field(default=1024, ge=256, le=2048,
                       description="Image width (multiple of 16)")
    num_inference_steps: int = Field(default=48, ge=1, le=100,
                                     description="Flow-matching steps (higher=better quality)")
    guidance_scale: float | None = Field(default=None, ge=1.0, le=20.0,
                                         description="CFG scale (None=recommended schedule)")
    seed: int | None = Field(default=None,
                             description="Random seed for reproducibility")
    prompt_upsampling: bool = Field(default=True,
                                    description="Expand prompt to JSON via local enhancer")
    prompt_upsampling_temperature: float = Field(default=1.0, ge=0.1, le=2.0)


class IdeogramStatus(BaseModel):
    service: str = "Ideogram 4.0 (local)"
    gpu_available: bool = False
    pipeline_loaded: bool = False
    prompt_enhancer_available: bool = False


# ── Pipeline state tracking ─────────────────────────────────────────────


def _is_pipeline_loaded() -> bool:
    """Check if the pipeline has been lazy-loaded."""
    try:
        from .generator import _pipeline
        return _pipeline is not None
    except Exception:
        return False


def _is_enhancer_available() -> bool:
    """Check if the prompt enhancer head is available."""
    try:
        from .generator import _prompt_enhancer_head
        return _prompt_enhancer_head is not None
    except Exception:
        return False


# ── Health ──────────────────────────────────────────────────────────────


@router.get("/health", response_model=IdeogramStatus)
async def health():
    """Check pipeline and GPU status."""
    import torch
    return IdeogramStatus(
        gpu_available=torch.cuda.is_available(),
        pipeline_loaded=_is_pipeline_loaded(),
        prompt_enhancer_available=_is_enhancer_available(),
    )


# ── Generate (text-to-image) ────────────────────────────────────────────


@router.post("/generate")
async def generate(request: GenerateRequest):
    """Generate an image from a text prompt using local Ideogram 4.0."""
    try:
        from .generator import generate_image

        result = generate_image(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            prompt_upsampling=request.prompt_upsampling,
            prompt_upsampling_temperature=request.prompt_upsampling_temperature,
        )
        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
