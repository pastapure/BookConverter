"""
Coqui TTS — Local Voice Cloning
================================
Uses Coqui XTTS-v2 for in-context voice cloning.  No API key, no per-char
cost, no fine-tuning required.  Provide a reference WAV and XTTS generates
new speech in that voice.

Requirements::

    pip install TTS

The XTTS-v2 model (~1.9 GB) is downloaded automatically on first use and
cached to ``~/.local/share/tts/``.

Usage (from pipeline)::

    from transcript_pipeline.coqui_tts import CoquiTTSEngine

    engine = CoquiTTSEngine(reference_wav="voice_samples/bella.wav")
    engine.synthesize_to_file("Hello world.", "output.wav")

Usage (standalone)::

    python -m transcript_pipeline.coqui_tts \\
        --reference tests/input/coqui_training_dataset/wavs/0001.wav \\
        --text "Hello, this is my cloned voice." \\
        --output cloned_test.wav
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Use tensor cores for FP32 matmuls (free speedup on Blackwell/Ampere+)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ── PyTorch 2.6+ compat: allow TTS checkpoint classes ──────────────────────────
# PyTorch >=2.6 defaults to weights_only=True for torch.load, but Coqui TTS
# checkpoints contain custom classes that need weights_only=False.

import torch
try:
    torch.serialization.add_safe_globals([
        type("XttsConfig", (), {}),
        type("XttsArgs", (), {}),
        type("XttsAudioConfig", (), {}),
    ])
except Exception:
    pass

# Monkey-patch torch.load to default weights_only=False (restores pre-2.6 behaviour)
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# ── Availability flags ─────────────────────────────────────────────────────────

HAS_TORCH = True

try:
    from TTS.api import TTS as _TTS_API
    HAS_TTS = True
except ImportError:
    _TTS_API = None
    HAS_TTS = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

HAS_COQUI_DEPS = HAS_TORCH and HAS_TTS

# ── Defaults ──────────────────────────────────────────────────────────────────

# XTTS-v2 model name for the TTS API
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS-v2 expects 22050 Hz mono
XTTS_SAMPLE_RATE = 22050

# Default reference audio length to use for cloning (seconds)
DEFAULT_REF_LENGTH = 12.0


# ── Engine ────────────────────────────────────────────────────────────────────


class CoquiTTSEngine:
    """
    Local TTS engine using Coqui XTTS-v2 voice cloning.

    Two modes:
    - **Zero-shot** (default): Uses the base XTTS-v2 model with a reference WAV.
    - **Fine-tuned**: Uses a custom fine-tuned checkpoint.

    Args:
        reference_wav: Path to a reference WAV file for voice cloning.
        finetuned_dir: Path to a fine-tuned checkpoint directory
                       (containing config.json + best_model.pth).
        device: Torch device (``cuda`` or ``cpu``, auto-detected if ``None``).
    """

    # Global cache for finetuned models (keyed by model_dir|fp16)
    _ft_cache: dict[str, dict] = {}

    def __init__(
        self,
        reference_wav: str,
        finetuned_dir: Optional[str] = None,
        device: Optional[str] = None,
        fp16: Optional[bool] = None,
    ):
        if not HAS_TTS:
            raise ImportError(
                "Coqui TTS is required. Install with: pip install TTS"
            )
        if not os.path.isfile(reference_wav):
            raise FileNotFoundError(f"Reference WAV not found: {reference_wav}")

        self.reference_wav = os.path.abspath(reference_wav)
        self.finetuned_dir = finetuned_dir

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # FP16: disabled by default — RTX 5080 Blackwell benchmark proves FP32 is 11% faster
        if fp16 is None:
            self.fp16 = False
        else:
            self.fp16 = fp16

        # Lazy-loaded
        self._tts: Optional[_TTS_API] = None
        self._ft_model = None  # fine-tuned XTTS model
        self._ft_gpt_latent = None
        self._ft_spk_emb = None

    @property
    def is_finetuned(self) -> bool:
        """Whether this engine is using a fine-tuned checkpoint."""
        return bool(self.finetuned_dir)

    @property
    def tts(self) -> _TTS_API:
        """Lazy-initialise the TTS model (downloads ~1.9 GB on first use).

        Only used in zero-shot mode — fine-tuned mode uses :meth:`_load_ft_model`.
        """
        if self._tts is None:
            print(f"[Coqui] Loading XTTS-v2 on {self.device}...")
            self._tts = _TTS_API(
                model_name=XTTS_MODEL_NAME,
                progress_bar=True,
                gpu=(self.device == "cuda"),
            )
            print("[Coqui] Model loaded.")
        return self._tts

    def _load_ft_model(self):
        """Load the fine-tuned XTTS model from ``finetuned_dir`` (FP16 + cached)."""
        if self._ft_model is not None:
            return

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        ft_dir = os.path.abspath(self.finetuned_dir)
        config_path = os.path.join(ft_dir, "config.json")
        ckpt_path = os.path.join(ft_dir, "best_model.pth")

        # Find vocab.json — try sibling xtts_v2 first, then parent pretrained_model
        vocab_path = os.path.join(
            os.path.dirname(ft_dir), "..", "pretrained_model", "vocab.json"
        )
        xtts_v2_vocab = os.path.join(
            os.path.dirname(os.path.dirname(ft_dir)), "xtts_v2", "vocab.json"
        )
        if os.path.isfile(xtts_v2_vocab):
            vocab_path = xtts_v2_vocab

        # Check model cache
        cache_key = f"{ft_dir}|{self.fp16}"
        if cache_key in CoquiTTSEngine._ft_cache:
            cached = CoquiTTSEngine._ft_cache[cache_key]
            self._ft_model = cached["model"]
            self._ft_gpt_latent = cached["gpt_cond"]
            self._ft_spk_emb = cached["speaker_emb"]
            name = os.path.basename(ft_dir)
            print(f"[Coqui] Using cached fine-tuned model: {name}")
            return

        print(f"[Coqui] Loading fine-tuned model from: {ft_dir}")
        config = XttsConfig()
        config.load_json(config_path)

        self._ft_model = Xtts.init_from_config(config)
        self._ft_model.load_checkpoint(
            config,
            checkpoint_path=ckpt_path,
            vocab_path=vocab_path,
            speaker_file_path=" ",
            use_deepspeed=False,
            eval=True,
        )

        self._ft_model = self._ft_model.to(self.device)

        # Register Hindi if applicable
        if "hi" not in self._ft_model.tokenizer.char_limits:
            self._ft_model.tokenizer.char_limits["hi"] = 200

        # Cache the conditioning latents FIRST (in FP32) before FP16 casting
        with torch.cuda.amp.autocast(enabled=False):
            self._ft_gpt_latent, self._ft_spk_emb = (
                self._ft_model.get_conditioning_latents(
                    audio_path=[self.reference_wav],
                    gpt_cond_len=30,
                    gpt_cond_chunk_len=4,
                    max_ref_length=60,
                )
            )

        # NOW cast to FP16 (after latents are cached)
        if self.fp16:
            if hasattr(self._ft_model, "gpt") and self._ft_model.gpt is not None:
                self._ft_model.gpt = self._ft_model.gpt.half()
            if hasattr(self._ft_model, "hifigan_decoder"):
                try:
                    self._ft_model.hifigan_decoder = self._ft_model.hifigan_decoder.float()
                except Exception:
                    pass
            self._ft_gpt_latent = self._ft_gpt_latent.half()
            self._ft_spk_emb = self._ft_spk_emb.half()

        # Save to cache
        CoquiTTSEngine._ft_cache[cache_key] = {
            "model": self._ft_model,
            "gpt_cond": self._ft_gpt_latent,
            "speaker_emb": self._ft_spk_emb,
        }
        print("[Coqui] Fine-tuned model ready (FP16={}).".format(self.fp16))

    # ── Public API ────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        language: str = "en",
    ) -> bytes:
        """
        Synthesize text using the cloned voice.

        Args:
            text: Text to speak.
            language: ISO language code.

        Returns:
            Raw WAV audio bytes (22050 Hz, mono, 16-bit PCM).
        """

        if self.is_finetuned:
            self._load_ft_model()
            autocast_ctx = torch.cuda.amp.autocast(enabled=self.fp16)
            with autocast_ctx, torch.no_grad():
                out = self._ft_model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self._ft_gpt_latent,
                    speaker_embedding=self._ft_spk_emb,
                    repetition_penalty=5.0,
                    temperature=0.75,
                )
            import numpy as np
            wav = out["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            samples = np.clip(wav, -1.0, 1.0)
            return (samples * 32767).astype(np.int16).tobytes()

        wav_numpy = self.tts.tts(
            text=text,
            speaker_wav=self.reference_wav,
            language=language,
        )

        # tts() returns a list of float values. Convert to 16-bit PCM bytes.
        import numpy as np

        samples = np.array(wav_numpy, dtype=np.float32)
        # Clamp to [-1, 1]
        samples = np.clip(samples, -1.0, 1.0)
        # Convert to int16
        int_samples = (samples * 32767).astype(np.int16)
        return int_samples.tobytes()

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
    ) -> str:
        """
        Synthesize text and save as a WAV file.

        Args:
            text: Text to speak.
            output_path: Where to save the WAV file.
            language: ISO language code.

        Returns:
            Absolute path to the saved WAV file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        import numpy as np

        if self.is_finetuned:
            self._load_ft_model()
            autocast_ctx = torch.cuda.amp.autocast(enabled=self.fp16)
            with autocast_ctx, torch.no_grad():
                out = self._ft_model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=self._ft_gpt_latent,
                    speaker_embedding=self._ft_spk_emb,
                    repetition_penalty=5.0,
                    temperature=0.75,
                )
            samples = np.array(out["wav"], dtype=np.float32)
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            samples = np.clip(samples, -1.0, 1.0)
            sr = 24000  # XTTS native output rate
        else:
            wav_numpy = self.tts.tts(
                text=text,
                speaker_wav=self.reference_wav,
                language=language,
            )
            samples = np.array(wav_numpy, dtype=np.float32)
            samples = np.clip(samples, -1.0, 1.0)
            sr = XTTS_SAMPLE_RATE
        samples = np.clip(samples, -1.0, 1.0)

        # Use Python's built-in wave module (avoids torchaudio/soundfile DLL issues)
        import wave
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        int_samples = (samples * 32767).astype(np.int16)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(int_samples.tobytes())

        print(f"[Coqui] Synthesized {len(text)} chars -> {output_path}")
        return os.path.abspath(output_path)

    # ── Podcast generator (replaces edge-tts pipeline) ────────────────────

    def generate_podcast(
        self,
        segments: list[dict],
        output_path: str,
        pause_ms: int = 800,
    ) -> str:
        """
        Generate a multi-speaker podcast from segments.

        Each segment should have a ``speaker`` and ``text`` key.  If multiple
        speakers are present, each should have a corresponding reference WAV
        configured via ``voice_map``.

        Args:
            segments: List of ``{"speaker": "...", "text": "..."}`` dicts.
            output_path: Output MP3 path.
            pause_ms: Pause between turns in milliseconds.

        Returns:
            Absolute path to the output MP3.

        Note:
            For single-speaker segments, the same reference_wav is used for all.
            For multi-speaker, use :class:`MultiSpeakerCoquiEngine` instead.
        """
        if not HAS_PYDUB:
            raise ImportError("pydub is required for podcast generation.")

        total = len(segments)
        segment_files: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, seg in enumerate(segments):
                wav_path = os.path.join(tmpdir, f"seg_{i:04d}.wav")
                self.synthesize_to_file(
                    text=seg.get("text", ""),
                    output_path=wav_path,
                )
                segment_files.append(wav_path)

            # Concatenate with pauses
            combined = AudioSegment.empty()
            for i, seg_file in enumerate(segment_files):
                combined += AudioSegment.from_wav(seg_file)
                if i < len(segment_files) - 1:
                    combined += AudioSegment.silent(duration=pause_ms)

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            combined.export(output_path, format="mp3", bitrate="192k")

        print(f"[Coqui] Podcast: {total} segments -> {output_path}")
        return os.path.abspath(output_path)


class MultiSpeakerCoquiEngine:
    """
    Multi-speaker Coqui TTS that maps speaker labels to different voice references.

    Args:
        voice_map: Dict mapping speaker labels to reference WAV paths.
        device: Torch device.
        fp16: Enable FP16 half-precision (default: auto-detect).
    """

    def __init__(
        self,
        voice_map: dict[str, str],
        device: Optional[str] = None,
        fp16: Optional[bool] = None,
    ):
        self.voice_map = voice_map
        self.device = device
        self.fp16 = fp16

        # One engine per speaker (they share the same TTS model in memory
        # after the first init, but keep separate ref_wav handles)
        self._engines: dict[str, CoquiTTSEngine] = {}

    def _get_engine(self, speaker: str) -> CoquiTTSEngine:
        if speaker not in self._engines:
            ref_wav = self.voice_map.get(speaker)
            if ref_wav is None:
                raise KeyError(
                    f"No reference WAV for speaker '{speaker}'. "
                    f"Available: {list(self.voice_map.keys())}"
                )
            self._engines[speaker] = CoquiTTSEngine(
                reference_wav=ref_wav,
                device=self.device,
                fp16=self.fp16,
            )
        return self._engines[speaker]

    def generate_podcast(
        self,
        segments: list[dict],
        output_path: str,
        pause_ms: int = 800,
    ) -> str:
        """
        Generate a multi-speaker podcast.

        Args:
            segments: List of ``{"speaker": "SpeakerName", "text": "..."}`` dicts.
            output_path: Output MP3 path.
            pause_ms: Pause between turns.

        Returns:
            Absolute path to the output MP3.
        """
        if not HAS_PYDUB:
            raise ImportError("pydub is required for podcast generation.")

        total = len(segments)
        segment_files: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, seg in enumerate(segments):
                speaker = seg.get("speaker", "SPEAKER_00")
                engine = self._get_engine(speaker)
                wav_path = os.path.join(tmpdir, f"seg_{i:04d}.wav")
                engine.synthesize_to_file(
                    text=seg.get("text", ""),
                    output_path=wav_path,
                )
                segment_files.append(wav_path)

            combined = AudioSegment.empty()
            for i, seg_file in enumerate(segment_files):
                combined += AudioSegment.from_wav(seg_file)
                if i < len(segment_files) - 1:
                    combined += AudioSegment.silent(duration=pause_ms)

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            combined.export(output_path, format="mp3", bitrate="192k")

        print(f"[Coqui] Multi-speaker podcast: {total} segments -> {output_path}")
        return os.path.abspath(output_path)


# ── Convenience ───────────────────────────────────────────────────────────────


def list_xtts_languages() -> list[str]:
    """List languages supported by XTTS-v2."""
    return [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
    ]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Coqui XTTS-v2 voice cloning — generate speech from a reference voice sample."
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to reference WAV file for voice cloning."
    )
    parser.add_argument(
        "--text", required=True,
        help="Text to synthesize."
    )
    parser.add_argument(
        "--output", default="coqui_output.wav",
        help="Output WAV file path."
    )
    parser.add_argument(
        "--language", default="en",
        help="Language code (default: en)."
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (cuda or cpu, auto-detected if omitted)."
    )

    args = parser.parse_args()

    engine = CoquiTTSEngine(
        reference_wav=args.reference,
        device=args.device,
    )
    engine.synthesize_to_file(
        text=args.text,
        output_path=args.output,
        language=args.language,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    _cli()
