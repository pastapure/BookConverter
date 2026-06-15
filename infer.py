"""
Fast XTTS-v2 voice cloning — FP16, GPU-cached, warm model.
=========================================================
Model stays in GPU memory between prompts (near-instant generation).

Usage:
  python infer.py                          # interactive mode
  python infer.py --text "नमस्ते"          # single prompt
  python infer.py --server                 # HTTP API on port 8765
  python infer.py --markdown transcript.md  # full podcast from markdown
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── PyTorch 2.6+ compat ──────────────────────────────────────────
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

import numpy as np
import torch
import torchaudio

# ── Config ────────────────────────────────────────────────────────
PROJECT = Path("D:/Projects/BookConverter")
MODELS = PROJECT / "models"
XTTS_V2 = MODELS / "xtts_v2"
VOICE3 = MODELS / "voice3"

# Auto-detect latest male finetuned checkpoint
_FT_DIRS = sorted(
    VOICE3.glob("hindi_voice3_*finetune*"),
    key=lambda d: d.stat().st_mtime if d.exists() else 0,
    reverse=True,
)
CKPT_DIR = _FT_DIRS[0] if _FT_DIRS else VOICE3 / "hindi_voice3_xtts-June-14-2026_04+39PM-0000000"

CHECKPOINT = str(CKPT_DIR / "best_model.pth")
CONFIG = str(CKPT_DIR / "config.json")
# Fallback: check if config.json is in the parent dir
if not os.path.isfile(CONFIG):
    CONFIG = str(VOICE3 / "config.json")

REF_WAV = str(VOICE3 / "reference.wav")
VOCAB = str(XTTS_V2 / "vocab.json")
LANG = "hi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = False  # RTX 5080 Blackwell: FP32 is 11% faster than selective FP16

# Use tensor cores for FP32 matmuls (free speedup on Blackwell/Ampere+)
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_model(fp16: bool = FP16) -> tuple:
    """Load XTTS model, optionally FP16, return (model, gpt_cond, speaker_emb)."""
    t_total = time.time()

    t0 = time.time()
    from TTS.tts.models.xtts import Xtts
    from TTS.tts.configs.xtts_config import XttsConfig
    print(f"[TIMING] imports: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    config = XttsConfig()
    config.load_json(CONFIG)
    print(f"[TIMING] config: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model = Xtts.init_from_config(config)
    print(f"[TIMING] init: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model.load_checkpoint(
        config, checkpoint_path=CHECKPOINT,
        vocab_path=VOCAB, speaker_file_path=" ",
        use_deepspeed=False, eval=True,
    )
    print(f"[TIMING] load_checkpoint: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    model = model.to(DEVICE)
    torch.cuda.synchronize()
    print(f"[TIMING] model.to(device): {time.time()-t0:.1f}s", flush=True)

    # Register Hindi
    if LANG not in model.tokenizer.char_limits:
        model.tokenizer.char_limits[LANG] = 200

    t0 = time.time()
    print("Computing speaker latents...", flush=True)
    # Compute latents in fp32 FIRST, before any FP16 casting
    with torch.cuda.amp.autocast(enabled=False):
        gpt_cond, speaker_emb = model.get_conditioning_latents(
            audio_path=[REF_WAV], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60,
        )
    print(f"[TIMING] latents: {time.time()-t0:.1f}s", flush=True)

    # NOW cast GPT to FP16 for faster inference (after latents are cached)
    if fp16:
        # Selective FP16: only GPT (transformer) — keep hifigan/speaker_encoder in FP32
        if hasattr(model, "gpt") and model.gpt is not None:
            model.gpt = model.gpt.half()
        if hasattr(model, "hifigan_decoder"):
            try:
                model.hifigan_decoder = model.hifigan_decoder.float()
            except Exception:
                pass
        gpt_cond = gpt_cond.half()
        speaker_emb = speaker_emb.half()
    torch.cuda.synchronize()
    print(f"[TIMING] FP16={fp16}: {time.time()-t_total:.1f}s", flush=True)
    VRAM = f"{torch.cuda.memory_allocated(0)/1024**3:.1f} GB"
    print(f"Ready! VRAM: {VRAM}\n", flush=True)
    return model, gpt_cond, speaker_emb, fp16


def speak(model, gpt_cond, speaker_emb, text, language=LANG, fp16=False):
    """Generate speech from text using cached model."""
    print(f"[TIMING] speak: {len(text)} chars", flush=True)

    # FP16 model needs autocast for inference
    autocast_ctx = torch.cuda.amp.autocast(enabled=fp16)
    with autocast_ctx, torch.no_grad():
        t0 = time.time()
        out = model.inference(
            text=text, language=language,
            gpt_cond_latent=gpt_cond, speaker_embedding=speaker_emb,
            repetition_penalty=5.0, temperature=0.75,
            enable_text_splitting=True,
        )
    duration = time.time() - t0
    print(f"[TIMING] inference: {duration:.1f}s", flush=True)
    wav = out["wav"]
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    return np.asarray(wav, dtype=np.float32), duration


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, help="Single prompt")
    p.add_argument("--file", type=str, help="Text file to read prompt from")
    p.add_argument("--markdown", type=str, help="Markdown file with SPEAKER_XX labels")
    p.add_argument("--server", action="store_true", help="HTTP server mode")
    p.add_argument("--output", type=str, default=None, help="Output WAV path")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 (default: FP32 — 11% faster on RTX 5080)")
    args = p.parse_args()

    fp16 = FP16 or args.fp16
    model, gpt_cond, speaker_emb, fp16 = load_model(fp16=fp16)

    if args.markdown:
        import re
        md = open(args.markdown, encoding="utf-8").read()
        segs = re.findall(r'\*\*(SPEAKER_\d+):\*\*\s*(.+?)(?=\n\*\*SPEAKER|\n\n---|\Z)', md, re.DOTALL)
        print(f"Segments: {len(segs)}")
        pause = torch.zeros(int(SAMPLE_RATE := 24000 * 0.8))
        parts = []
        for i, (spk, txt) in enumerate(segs):
            wav, dur = speak(model, gpt_cond, speaker_emb, txt.strip(), fp16=fp16)
            parts.append(torch.from_numpy(wav))
            parts.append(pause.clone())
            print(f"  [{i+1}/{len(segs)}] {spk}: {len(wav)/SAMPLE_RATE:.1f}s ({dur:.1f}s)")
        final = torch.cat(parts)
        out_path = args.output or args.markdown.replace(".md", ".mp3")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, final.unsqueeze(0), SAMPLE_RATE)
        print(f"Saved: {out_path} ({len(final)/SAMPLE_RATE:.0f}s, {os.path.getsize(out_path)} bytes)")

    elif args.server:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json as _json
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                data = _json.loads(body)
                text = data.get("text", "")
                lang = data.get("language", LANG)
                wav, dur = speak(model, gpt_cond, speaker_emb, text, lang, fp16)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                import tempfile
                tmp = Path(tempfile.gettempdir()) / "tts_out.wav"
                torchaudio.save(str(tmp), torch.from_numpy(wav).unsqueeze(0), 24000)
                self.wfile.write(tmp.read_bytes())
                print(f"  Served: {text[:50]}... ({dur:.1f}s)")
            def log_message(self, *a): pass
        print("Server: http://localhost:8765\n  POST {\"text\": \"...\"}  ->  WAV")
        HTTPServer(("0.0.0.0", 8765), H).serve_forever()

    elif args.file:
        t0 = time.time()
        text = open(args.file, encoding="utf-8").read().strip()
        print(f"[TIMING] file read: {time.time()-t0:.1f}s", flush=True)
        wav, dur = speak(model, gpt_cond, speaker_emb, text, fp16=fp16)
        t0 = time.time()
        out_path = args.output or args.file.replace(".txt", ".wav")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        print(f"[TIMING] save: {time.time()-t0:.1f}s", flush=True)
        print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes, {len(wav)/24000:.1f}s, gen: {dur:.1f}s)")

    elif args.text:
        wav, dur = speak(model, gpt_cond, speaker_emb, args.text, fp16=fp16)
        out_path = args.output or "output/tts_out.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torchaudio.save(out_path, torch.from_numpy(wav).unsqueeze(0), 24000)
        print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes, {len(wav)/24000:.1f}s, gen: {dur:.1f}s)")

    else:
        print("Interactive mode. Type Hindi text. Ctrl+C to exit.\n")
        i = 0
        while True:
            try:
                text = input("> ").strip()
                if not text: continue
                wav, dur = speak(model, gpt_cond, speaker_emb, text, fp16=fp16)
                out_path = f"output/tts_{i}.wav"
                os.makedirs("output", exist_ok=True)
                torchaudio.save(out_path, torch.from_numpy(wav).unsqueeze(0), 24000)
                print(f"  -> {out_path} ({len(wav)/24000:.1f}s, gen: {dur:.1f}s)")
                i += 1
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")
                break
