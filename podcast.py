"""
Multi-speaker podcast generator from Markdown transcripts.
Uses fine-tuned XTTS-v2 models — male (voice3) + female (hindi_tts_female).
"""
import os, sys, re, time, argparse
from pathlib import Path
import torch, torchaudio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

# ── Voice config ─────────────────────────────────────────────────
VOICES = {
    "SPEAKER_00": {  # Female
        "model": "D:/Projects/BookConverter/models/hindi_tts_female",
        "ref":   "D:/Projects/BookConverter/models/hindi_tts_female/reference.wav",
    },
    "SPEAKER_01": {  # Male
        "model": "D:/Projects/BookConverter/models/voice3/hindi_voice3_xtts-June-14-2026_04+39PM-0000000",
        "ref":   "D:/Projects/BookConverter/models/voice3/reference.wav",
    },
}
VOCAB = "D:/Projects/BookConverter/models/xtts_v2/vocab.json"
SAMPLE_RATE = 24000
PAUSE_MS = 800  # pause between speaker turns


def load_voice(model_dir: str):
    """Load one voice model."""
    d = Path(model_dir)
    config = XttsConfig(); config.load_json(str(d / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(d / "best_model.pth"),
                          vocab_path=VOCAB, speaker_file_path=" ",
                          use_deepspeed=False, eval=True)
    model.cuda()
    return model


def parse_markdown(md_path: str):
    """Extract speaker-labeled segments from markdown."""
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    segments = re.findall(r'\*\*(SPEAKER_\d+):\*\*\s*(.+?)(?=\n\*\*SPEAKER|\n\n---|\Z)', text, re.DOTALL)
    if not segments:
        segments = re.findall(r'\*\*(SPEAKER_\d+):\*\*\s*(.+)', text)
    return [(spk, txt.strip()) for spk, txt in segments]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Markdown transcript file")
    p.add_argument("--output", default=None, help="Output WAV path")
    args = p.parse_args()

    segments = parse_markdown(args.input)
    print(f"Found {len(segments)} segments")

    # ── Generate per speaker (load one model at a time to fit GPU) ─
    audio_segments = {}
    speakers_needed = set(spk for spk, _ in segments)

    for speaker in sorted(speakers_needed):
        v = VOICES[speaker]
        print(f"\nLoading {speaker} voice ({v['model'].split('/')[-1]})...")
        model = load_voice(v["model"])

        gpt_cond, speaker_emb = model.get_conditioning_latents(
            audio_path=[v["ref"]], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60
        )

        speaker_segs = [(i, txt) for i, (spk, txt) in enumerate(segments) if spk == speaker]
        print(f"  Generating {len(speaker_segs)} segments...")

        for idx, (seg_idx, txt) in enumerate(speaker_segs):
            t0 = time.time()
            out = model.inference(
                text=txt, language="hi",
                gpt_cond_latent=gpt_cond, speaker_embedding=speaker_emb,
                repetition_penalty=5.0, temperature=0.75,
            )
            audio_segments[seg_idx] = out["wav"]
            print(f"  [{idx+1}/{len(speaker_segs)}] {len(out['wav'])/SAMPLE_RATE:.1f}s ({time.time()-t0:.1f}s)")

        del model; torch.cuda.empty_cache()

    # ── Concatenate with pauses ─────────────────────────────────
    print("\nConcatenating...")
    pause = torch.zeros(int(SAMPLE_RATE * PAUSE_MS / 1000))
    parts = []
    for i in range(len(segments)):
        if i in audio_segments:
            parts.append(torch.tensor(audio_segments[i]))
            parts.append(pause.clone())

    final = torch.cat(parts)
    out_path = args.output or args.input.replace(".md", ".wav")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torchaudio.save(out_path, final.unsqueeze(0), SAMPLE_RATE)
    duration = len(final) / SAMPLE_RATE
    print(f"Saved: {out_path}\n  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
