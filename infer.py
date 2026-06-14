"""
Fast XTTS-v2 voice cloning — model loaded ONCE, then prompt repeatedly.
Model stays in GPU memory between prompts (near-instant generation).

Usage:
  python infer.py                          # interactive mode
  python infer.py --text "नमस्ते"          # single prompt
  python infer.py --server                 # HTTP API on port 8765
"""

import os, sys, time, argparse
from pathlib import Path

# PyTorch 2.6+ compat
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

import numpy as np
import torch, torchaudio

# ── Config ──────────────────────────────────────────────────────────
MODEL_DIR = Path("D:/Projects/BookConverter/models/voice3")
CKPT_DIR = MODEL_DIR / "hindi_voice3_xtts-June-14-2026_04+39PM-0000000"
CHECKPOINT = str(CKPT_DIR / "best_model.pth")
CONFIG = str(CKPT_DIR / "config.json")
REF_WAV = str(MODEL_DIR / "reference.wav")
VOCAB = "D:/Projects/BookConverter/models/xtts_v2/vocab.json"
LANG = "hi"

def load_model():
    t_total = time.time()

    t0 = time.time()
    from TTS.tts.models.xtts import Xtts
    print(f"[TIMING] import Xtts: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    from TTS.tts.configs.xtts_config import XttsConfig
    print(f"[TIMING] import XttsConfig: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    config = XttsConfig()
    config.load_json(CONFIG)
    print(f"[TIMING] load config.json: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    model = Xtts.init_from_config(config)
    print(f"[TIMING] Xtts.init_from_config: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    model.load_checkpoint(config, checkpoint_path=CHECKPOINT,
                          vocab_path=VOCAB, speaker_file_path=" ",
                          use_deepspeed=False, eval=True)
    print(f"[TIMING] load_checkpoint: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    model = model.cuda()
    print(f"[TIMING] model.cuda(): {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    print("Computing speaker latents...", flush=True)
    gpt_cond, speaker_emb = model.get_conditioning_latents(
        audio_path=[REF_WAV], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
    )
    print(f"[TIMING] get_conditioning_latents: {time.time() - t0:.1f}s", flush=True)

    print(f"[TIMING] TOTAL load_model: {time.time() - t_total:.1f}s", flush=True)
    print("Ready!\n", flush=True)
    return model, gpt_cond, speaker_emb


def speak(model, gpt_cond, speaker_emb, text, language=LANG):
    """Generate speech from text (single chunk, no splitting)."""
    print(f"[TIMING] speak: {len(text)} chars (single chunk)", flush=True)
    # Monkey-patch: add Hindi to tokenizer char_limits (model was fine-tuned on Hindi)
    if language not in model.tokenizer.char_limits:
        model.tokenizer.char_limits[language] = 200  # matches training max_text_length

    t0 = time.time()
    out = model.inference(
        text=text, language=language,
        gpt_cond_latent=gpt_cond, speaker_embedding=speaker_emb,
        repetition_penalty=5.0, temperature=0.75,
        enable_text_splitting=True,
    )
    duration = time.time() - t0
    print(f"[TIMING] speak inference: {duration:.1f}s", flush=True)
    return out["wav"], duration


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, help="Single prompt")
    p.add_argument("--file", type=str, help="Text file to read prompt from")
    p.add_argument("--markdown", type=str, help="Markdown file with SPEAKER_XX labels")
    p.add_argument("--server", action="store_true", help="HTTP server mode")
    p.add_argument("--output", type=str, default=None, help="Output WAV path")
    args = p.parse_args()

    model, gpt_cond, speaker_emb = load_model()

    if args.markdown:
        import re
        md = open(args.markdown, encoding="utf-8").read()
        segs = re.findall(r'\*\*(SPEAKER_\d+):\*\*\s*(.+?)(?=\n\*\*SPEAKER|\n\n---|\Z)', md, re.DOTALL)
        print(f"Segments: {len(segs)}")
        SR = 24000
        pause = torch.zeros(int(SR * 0.8))
        parts = []
        for i, (spk, txt) in enumerate(segs):
            wav, dur = speak(model, gpt_cond, speaker_emb, txt.strip())
            parts.append(torch.tensor(wav))
            parts.append(pause.clone())
            print(f"  [{i+1}/{len(segs)}] {spk}: {len(wav)/SR:.1f}s ({dur:.1f}s)")
        final = torch.cat(parts)
        out_path = args.output or args.markdown.replace(".md", ".mp3")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, final.unsqueeze(0), SR)
        print(f"Saved: {out_path} ({len(final)/SR:.0f}s, {os.path.getsize(out_path)} bytes)")
    elif args.server:
        # Quick HTTP server for external calls
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json as _json
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                data = _json.loads(body)
                text = data.get("text", "")
                lang = data.get("language", LANG)
                wav, dur = speak(model, gpt_cond, speaker_emb, text, lang)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                buf = torchaudio.save("", torch.tensor(wav).unsqueeze(0), 24000, format="wav")
                # torchaudio doesn't support in-memory easily; save temp
                import tempfile
                tmp = Path(tempfile.gettempdir()) / "tts_out.wav"
                torchaudio.save(str(tmp), torch.tensor(wav).unsqueeze(0), 24000)
                self.wfile.write(tmp.read_bytes())
                print(f"  Served: {text[:50]}... ({dur:.1f}s)")
            def log_message(self, *a): pass  # quiet
        print("Server: http://localhost:8765\n  POST {\"text\": \"...\"}  ->  WAV")
        HTTPServer(("0.0.0.0", 8765), H).serve_forever()

    elif args.file:
        t0 = time.time()
        text = open(args.file, encoding="utf-8").read().strip()
        print(f"[TIMING] file read: {time.time() - t0:.1f}s", flush=True)
        wav, dur = speak(model, gpt_cond, speaker_emb, text)
        t0 = time.time()
        out_path = args.output or args.file.replace(".txt", ".wav")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, torch.tensor(wav).unsqueeze(0), 24000)
        print(f"[TIMING] torchaudio.save: {time.time() - t0:.1f}s", flush=True)
        print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes, {len(wav)/24000:.1f}s, gen: {dur:.1f}s)")
    elif args.text:
        wav, dur = speak(model, gpt_cond, speaker_emb, args.text)
        out_path = args.output or "output/tts_out.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torchaudio.save(out_path, torch.tensor(wav).unsqueeze(0), 24000)
        print(f"Saved: {out_path} ({os.path.getsize(out_path)} bytes, {len(wav)/24000:.1f}s, gen: {dur:.1f}s)")

    else:
        print("Interactive mode. Type Hindi text. Ctrl+C to exit.\n")
        i = 0
        while True:
            try:
                text = input("> ").strip()
                if not text: continue
                wav, dur = speak(model, gpt_cond, speaker_emb, text)
                out_path = f"output/tts_{i}.wav"
                os.makedirs("output", exist_ok=True)
                torchaudio.save(out_path, torch.tensor(wav).unsqueeze(0), 24000)
                print(f"  → {out_path} ({len(wav)/24000:.1f}s, gen: {dur:.1f}s)")
                i += 1
            except (KeyboardInterrupt, EOFError):
                print("\nDone.")
                break
