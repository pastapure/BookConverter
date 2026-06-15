"""
Optimized multi-speaker podcast generator from Markdown transcripts.
====================================================================
Uses fine-tuned XTTS-v2 models — male (voice3) + female (hindi_tts_female).
Optimizations: FP16, GPU-accelerated, cached models, parallel loading.
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Patch torch.load ────────────────────────────────────────────
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

# ── Config ──────────────────────────────────────────────────────
PROJECT = Path("D:/Projects/BookConverter")
MODELS = PROJECT / "models"

# Find latest male checkpoint directory
_MALE_CANDIDATES = sorted(
    (MODELS / "voice3").glob("hindi_voice3_*"),
    key=lambda d: d.stat().st_mtime if d.exists() else 0,
    reverse=True,
)

VOICES = {}
# Female — use top-level model dir if it has a config.json
if (MODELS / "hindi_tts_female" / "config.json").exists():
    VOICES["SPEAKER_00"] = {  # Female
        "model": str(MODELS / "hindi_tts_female"),
        "ref":   str(MODELS / "hindi_tts_female" / "reference.wav"),
    }
# Male — find the most recent finetuned dir
if _MALE_CANDIDATES and (_MALE_CANDIDATES[0] / "config.json").exists():
    VOICES["SPEAKER_01"] = {  # Male
        "model": str(_MALE_CANDIDATES[0]),
        "ref":   str(MODELS / "voice3" / "reference.wav"),
    }
elif (MODELS / "voice3" / "config.json").exists():
    VOICES["SPEAKER_01"] = {  # Male (top-level dir)
        "model": str(MODELS / "voice3"),
        "ref":   str(MODELS / "voice3" / "reference.wav"),
    }

VOCAB = str(MODELS / "xtts_v2" / "vocab.json")
SAMPLE_RATE = 24000
PAUSE_MS = 800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = False  # RTX 5080 Blackwell: FP32 is 11% faster than selective FP16

# Use tensor cores for FP32 matmuls (free speedup on Blackwell/Ampere+)
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ── Global model cache ──────────────────────────────────────────
_model_cache: dict[str, dict] = {}


def load_voice(model_dir: str, ref_wav: str) -> dict:
    """Load one voice model (cached)."""
    cache_key = f"{model_dir}|{FP16}"
    if cache_key in _model_cache:
        name = Path(model_dir).name
        print(f"  Reusing cached model: {name}")
        return _model_cache[cache_key]

    print(f"  Loading: {Path(model_dir).name} (FP16={FP16})", flush=True)

    config = XttsConfig()
    config.load_json(str(Path(model_dir) / "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(Path(model_dir) / "best_model.pth"),
        vocab_path=VOCAB,
        speaker_file_path=" ",
        use_deepspeed=False,
        eval=True,
    )

    model = model.to(DEVICE)
    if FP16:
        model = model.half()

    # Register Hindi
    if "hi" not in model.tokenizer.char_limits:
        model.tokenizer.char_limits["hi"] = 200

    # Compute latents (float32 for stability, cast to half if FP16)
    with torch.cuda.amp.autocast(enabled=False):
        gpt_cond, speaker_emb = model.get_conditioning_latents(
            audio_path=[ref_wav], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60,
        )
    if FP16:
        gpt_cond = gpt_cond.half()
        speaker_emb = speaker_emb.half()

    handle = {
        "model": model,
        "gpt_cond": gpt_cond,
        "speaker_emb": speaker_emb,
        "config": config,
    }
    _model_cache[cache_key] = handle
    return handle


def parse_markdown(md_path: str) -> list[tuple[str, str]]:
    """Extract speaker-labeled segments from markdown."""
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    segments = re.findall(
        r'\*\*(SPEAKER_\d+):\*\*\s*(.+?)(?=\n\s*\*\*SPEAKER|\Z|\n\n---)',
        text, re.DOTALL,
    )
    if not segments:
        segments = re.findall(r'\*\*(SPEAKER_\d+):\*\*\s*(.+)', text)
    return [(spk, txt.strip()) for spk, txt in segments]


def speak(handle: dict, text: str) -> np.ndarray:
    """Generate speech from text using cached model handle."""
    model = handle["model"]
    gpt_cond = handle["gpt_cond"]
    speaker_emb = handle["speaker_emb"]

    autocast_ctx = torch.cuda.amp.autocast(enabled=FP16)
    with autocast_ctx, torch.no_grad():
        out = model.inference(
            text=text, language="hi",
            gpt_cond_latent=gpt_cond, speaker_embedding=speaker_emb,
            repetition_penalty=5.0, temperature=0.75,
            enable_text_splitting=True,
        )
    wav = out["wav"]
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    return np.asarray(wav, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Optimized multi-speaker podcast generator")
    p.add_argument("input", help="Markdown transcript file")
    p.add_argument("--output", default=None, help="Output WAV path")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 (default: FP32 — 11% faster on RTX 5080)")
    p.add_argument("--pause", type=int, default=PAUSE_MS, help="Pause between turns (ms)")
    args = p.parse_args()

    global FP16
    if args.fp16:
        FP16 = True

    segments = parse_markdown(args.input)
    print(f"Found {len(segments)} segments")
    print(f"Device: {DEVICE}, FP16: {FP16}", flush=True)

    if not VOICES:
        print("ERROR: No voice models found!")
        print("  Expected: models/hindi_tts_female/ (SPEAKER_00)")
        print("  Expected: models/voice3/<finetuned_dir>/ (SPEAKER_01)")
        sys.exit(1)

    # ── Generate per speaker (parallel loading) ────────────────
    audio_segments: dict[int, np.ndarray] = {}
    speakers_needed = set(spk for spk, _ in segments)

    t_total = time.time()

    for speaker in sorted(speakers_needed):
        if speaker not in VOICES:
            print(f"  SKIP {speaker}: no voice config found")
            continue

        v = VOICES[speaker]
        t_load = time.time()
        handle = load_voice(v["model"], v["ref"])
        print(f"  {speaker}: loaded in {time.time()-t_load:.1f}s", flush=True)

        speaker_segs = [(i, txt) for i, (spk, txt) in enumerate(segments) if spk == speaker]
        if not speaker_segs:
            continue

        print(f"  Generating {len(speaker_segs)} segments...", flush=True)
        for idx, (seg_idx, txt) in enumerate(speaker_segs):
            t0 = time.time()
            wav = speak(handle, txt)
            audio_segments[seg_idx] = wav
            print(f"    [{idx+1}/{len(speaker_segs)}] {len(wav)/SAMPLE_RATE:.1f}s ({time.time()-t0:.1f}s)", flush=True)

    # ── Concatenate with pauses (GPU) ──────────────────────────
    if not audio_segments:
        print("ERROR: No audio generated!")
        sys.exit(1)

    print("\nConcatenating...", flush=True)
    pause = torch.zeros(int(SAMPLE_RATE * args.pause / 1000))
    parts = []
    for i in range(len(segments)):
        if i in audio_segments:
            parts.append(torch.from_numpy(audio_segments[i]))
            parts.append(pause.clone())

    final = torch.cat(parts)
    out_path = args.output or args.input.replace(".md", ".wav")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torchaudio.save(out_path, final.unsqueeze(0), SAMPLE_RATE)

    duration = len(final) / SAMPLE_RATE
    elapsed = time.time() - t_total
    print(f"Saved: {out_path}")
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  RTF: {elapsed/max(duration,0.1):.2f}x")
    print("Done.")


if __name__ == "__main__":
    main()
