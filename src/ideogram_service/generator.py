"""
Ideogram 4.0 Image Generation Wrapper.
======================================
Runs Ideogram 4.0 **fully locally** via the HuggingFace Diffusers pipeline.
No cloud API key required — the model downloads once then runs on-device.

Supports:
    - Text-to-image generation (``generate_image``)
    - Optional local prompt upsampling (``Ideogram4PromptEnhancerHead``)

Requirements:
    pip install -r requirements/ideogram.txt

GPU required (16GB+ VRAM recommended — NF4 quantization used by default).

Set ``HF_TOKEN`` in ``.env`` for gated model download (one-time).
Get a token at https://huggingface.co/settings/tokens
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

# ── Module-level lazy pipeline (matches HeartMuLa pattern) ──────────────
_pipeline = None
_prompt_enhancer_head = None

# ── Model config ─────────────────────────────────────────────────────────
MODEL_ID = "ideogram-ai/ideogram-4-nf4"
PROMPT_ENHANCER_ID = "diffusers/qwen3-vl-8b-instruct-lm-head"

# ── Cache directory (where model weights are stored) ─────────────────────
# Override with IDEOGRAM_CACHE_DIR in .env
DEFAULT_CACHE_DIR = os.getenv("IDEOGRAM_CACHE_DIR", "C:/huggingface_models/ideogram4")
CACHE_DIR = Path(DEFAULT_CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Output directory (where generated images are saved) ───────────────────
OUTPUT_DIR = Path("data/ideogram/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_STEPS = 48


# ── Device helper ─────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Work around NF4 dispatch bug ────────────────────────────────────────
# diffusers v0.39 auto-dispatches quantized models → model.to(device)
# crashes on NF4 meta tensors.  Patch the import in modeling_utils.
import diffusers.models.modeling_utils as _dmu
_dmu.dispatch_model = lambda model, *a, **kw: model

# ── Availability flags ────────────────────────────────────────────────────

try:
    from diffusers import Ideogram4Pipeline, Ideogram4PromptEnhancerHead
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

try:
    import outlines  # noqa: F401
    HAS_OUTLINES = True
except ImportError:
    HAS_OUTLINES = False


# ── Lazy pipeline loading (matches HeartMuLa _get_music_pipeline) ────────


def _get_pipeline():
    """Lazily load the Ideogram4Pipeline on first call."""
    global _pipeline, _prompt_enhancer_head

    if _pipeline is not None:
        return _pipeline

    if not HAS_DIFFUSERS:
        raise RuntimeError(
            "diffusers not installed. "
            "Install with: pip install -r requirements/ideogram.txt"
        )

    hf_token = os.getenv("HF_TOKEN", "") or None
    device = _get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[...] Loading Ideogram 4.0 pipeline on {device.type.upper()} ...")
    print(f"     Model: {MODEL_ID}")
    print(f"     Cache: {CACHE_DIR}")
    t0 = time.time()

    # ── Load prompt enhancer head (fully local prompt upsampling) ────
    # SKIP if system RAM is tight — the main pipeline alone needs ~30 GB.
    _skip_enhancer = os.getenv("IDEOGRAM_SKIP_ENHANCER", "1") == "1"
    if not _skip_enhancer and HAS_OUTLINES:
        try:
            print(f"     Loading prompt enhancer head: {PROMPT_ENHANCER_ID}")
            _prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(
                PROMPT_ENHANCER_ID,
                torch_dtype=dtype,
                token=hf_token,
                cache_dir=str(CACHE_DIR),
            )
            print(f"     [OK]  Prompt enhancer loaded (local upsampling enabled)")
        except Exception as exc:
            print(f"     [WARNING] Prompt enhancer not available: {exc}")
            _prompt_enhancer_head = None
    elif _skip_enhancer:
        print(f"     [INFO]  Prompt enhancer skipped (IDEOGRAM_SKIP_ENHANCER=1)")
        _prompt_enhancer_head = None
    else:
        print(f"     [INFO]  'outlines' not installed — prompt upsampling disabled.")
        _prompt_enhancer_head = None

    # ── Load the main pipeline ────────────────────────────────────────
    # No RAM budget — let Windows pagefile handle spillover.
    # NF4 DiTs → GPU (~12 GB), text encoder → CPU (whatever fits).
    os.environ.setdefault("SAFETENSORS_FAST_GPU", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _pipeline = Ideogram4Pipeline.from_pretrained(
        MODEL_ID,
        prompt_enhancer_head=_prompt_enhancer_head,
        torch_dtype=dtype,
        token=hf_token,
        cache_dir=str(CACHE_DIR),
        # NF4 DiTs → GPU, text encoder → CPU (avoids RAM exhaustion)
        device_map="balanced",
        max_memory={0: "14GB", "cpu": "16GB"},
    )

    # Sequential offload after balanced placement.
    if device.type == "cuda":
        _pipeline.enable_sequential_cpu_offload("cuda")
        print(f"     [OK]  Sequential offload enabled")

    print(f"[OK]  Pipeline loaded in {time.time() - t0:.1f}s "
          f"(device={device.type}, dtype={dtype})")

    print(f"[OK]  Pipeline loaded in {time.time() - t0:.1f}s "
          f"(device={device.type}, dtype={dtype})")

    return _pipeline


# ── Public API ────────────────────────────────────────────────────────────


def generate_image(
    prompt: str,
    output_path: str = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = None,
    seed: int = None,
    prompt_upsampling: bool = True,
    prompt_upsampling_temperature: float = 1.0,
) -> dict:
    """Generate an image from a text prompt using local Ideogram 4.0.

    Args:
        prompt: Natural-language description of the image to generate.
        output_path: Optional path for the output image.  If ``None``,
                    auto-generates a filename under
                    ``data/ideogram/assets/``.
        height: Output image height in pixels (multiple of 16, 256–2048).
               Default 1024.
        width: Output image width in pixels (multiple of 16, 256–2048).
              Default 1024.
        num_inference_steps: Number of flow-matching steps.  Higher = better
                            quality but slower.  Default 48 (recommended).
        guidance_scale: Constant classifier-free guidance scale.  Higher =
                       more prompt adherence, less diversity.  If ``None``,
                       uses the recommended per-step schedule (7.0, dropping
                       to 3.0 for final 3 polish steps).
        seed: Random seed for reproducibility.  ``None`` = non-deterministic.
        prompt_upsampling: If ``True``, expand the prompt into Ideogram 4's
                          native structured JSON caption using the local
                          prompt enhancer head.  Requires ``outlines``.
        prompt_upsampling_temperature: Sampling temperature for prompt
                                      upsampling (1.0 = default).

    Returns:
        A dict with keys:
        - ``status``: ``"success"``
        - ``images``: list of local file paths (length 1)
        - ``prompt``: the prompt used
        - ``height``: output image height
        - ``width``: output image width
        - ``seed``: random seed used (None if not set)
        - ``num_inference_steps``: steps used
        - ``elapsed_seconds``: generation time

    Raises:
        RuntimeError: If dependencies are missing, HF_TOKEN is not set, or
                     generation fails.
    """
    pipe = _get_pipeline()
    device = _get_device()

    print(f"[...] Generating image with Ideogram 4.0 (local) ...")
    print(f"     Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"     Size: {width}x{height}, Steps: {num_inference_steps}")
    if guidance_scale is not None:
        print(f"     CFG scale: {guidance_scale}")
    if prompt_upsampling and _prompt_enhancer_head is not None:
        print(f"     Prompt upsampling: local (on-device)")

    t0 = time.time()

    # ── Set up generator for reproducibility ──────────────────────────
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"     Seed: {seed}")

    # ── Run the pipeline ──────────────────────────────────────────────
    try:
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_upsampling=prompt_upsampling and _prompt_enhancer_head is not None,
            prompt_upsampling_temperature=prompt_upsampling_temperature,
            generator=generator,
            output_type="pil",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ideogram 4 pipeline failed: {exc}"
        ) from exc

    images = result.images
    if not images:
        raise RuntimeError("Pipeline returned no images.")

    # ── Save image ────────────────────────────────────────────────────
    if output_path is None:
        seed_str = f"_{seed}" if seed is not None else ""
        timestamp = int(time.time())
        output_path = str(OUTPUT_DIR / f"ideogram4_{width}x{height}{seed_str}_{timestamp}.png")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    images[0].save(output_path, format="PNG")

    elapsed = time.time() - t0

    print(f"[OK]  Image generated in {elapsed:.1f}s → {output_path}")
    print(f"     Size: {images[0].size[0]}x{images[0].size[1]}")

    return {
        "status": "success",
        "images": [os.path.abspath(output_path)],
        "prompt": prompt,
        "height": height,
        "width": width,
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "elapsed_seconds": round(elapsed, 1),
    }
