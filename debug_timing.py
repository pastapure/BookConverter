"""Debug timing — find where time is spent."""
import time, torch, os, sys

_p = print
def log(msg, t0):
    _p(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

t0 = time.time()
log("start", t0)

# Fix torch.load
o = torch.load
def patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return o(*a, **k)
torch.load = patched_load
log("torch.load patched", t0)

log("importing TTS...", t0)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
log("TTS imported", t0)

log("loading config...", t0)
cfg = XttsConfig()
cfg.load_json("models/xtts_v2/config.json")
log("config loaded", t0)

log("init model...", t0)
model = Xtts.init_from_config(cfg)
log("model inited", t0)

log("loading checkpoint (1.9 GB base model)...", t0)
model.load_checkpoint(
    cfg,
    checkpoint_path="models/xtts_v2/model.pth",
    vocab_path="models/xtts_v2/vocab.json",
    speaker_file_path=" ",
    use_deepspeed=False,
    eval=True,
)
log("checkpoint loaded", t0)

log("moving to GPU...", t0)
model.cuda()
log("GPU ready", t0)

log("computing speaker latents...", t0)
gc, se = model.get_conditioning_latents(
    audio_path=["models/voice3/reference.wav"],
    gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60,
)
log("latents computed", t0)

log("reading text...", t0)
text = open("data/samples/test_paragraph.txt", encoding="utf-8").read().strip()
log(f"text: {len(text)} chars", t0)

log("generating speech...", t0)
out = model.inference(
    text=text, language="hi",
    gpt_cond_latent=gc, speaker_embedding=se,
    repetition_penalty=5.0, temperature=0.75,
)
log(f"speech generated: {len(out['wav'])/24000:.1f}s audio", t0)

os.makedirs("output", exist_ok=True)
out_path = "output/debug_test.wav"
import torchaudio
torchaudio.save(out_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
log(f"saved: {out_path} ({os.path.getsize(out_path)} bytes)", t0)

_p(f"\n=== Total: {time.time()-t0:.1f}s ===", flush=True)
