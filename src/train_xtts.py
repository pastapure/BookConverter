"""
XTTS-v2 Fine-tuning — based on official Coqui TTS example
https://github.com/idiap/coqui-ai-TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py

Usage:
  python src/train_xtts.py
"""

import os
from pathlib import Path

# Load .env file so os.getenv() picks up settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# PyTorch 2.6+ compat
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig

# ── Config (override via env or edit here) ────────────────────────────
RUN_NAME = os.getenv("XTTS_RUN_NAME", "hindi_voice3_xtts")
PROJECT_NAME = os.getenv("XTTS_PROJECT_NAME", "XTTS_trainer")
LANGUAGE = os.getenv("XTTS_LANGUAGE", "hi")
DATASET_NAME = os.getenv("XTTS_DATASET_NAME", "voice3")
EPOCHS = int(os.getenv("XTTS_EPOCHS", "300"))
BATCH_SIZE = int(os.getenv("XTTS_BATCH_SIZE", "3"))
GRAD_ACCUM = int(os.getenv("XTTS_GRAD_ACCUM_STEPS", "84"))
LR = float(os.getenv("XTTS_LEARNING_RATE", "5e-06"))
GPU = os.getenv("XTTS_GPU", "0")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

# ── Paths (from .env, with fallbacks) ──────────────────────────────────
PROJECT_ROOT = Path(os.getenv("XTTS_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATASET_PATH = Path(os.getenv("XTTS_DATASET_PATH", PROJECT_ROOT / "data" / "tts" / "voice3"))
MODEL_PATH = Path(os.getenv("XTTS_PRETRAINED_MODEL_ROOT", PROJECT_ROOT / "models" / "xtts_v2"))
OUTPUT_PATH = Path(os.getenv("XTTS_OUTPUT_PATH", PROJECT_ROOT / "models" / "voice3"))
SPEAKER_REF_WAV = os.getenv("XTTS_SPEAKER_REFERENCE_WAV", str(PROJECT_ROOT / "models" / "voice3" / "reference.wav"))
SPEAKER_REF = [SPEAKER_REF_WAV]

# Read test sentences from JSON if available
TEST_SENTENCES_JSON = os.getenv("XTTS_TEST_SENTENCES_JSON", "")
TEST_SENTENCES = []
if TEST_SENTENCES_JSON and Path(TEST_SENTENCES_JSON).exists():
    import json as _json
    TEST_SENTENCES = _json.loads(Path(TEST_SENTENCES_JSON).read_text(encoding="utf-8"))
else:
    TEST_SENTENCES = [
        {"text": "जटिल mathematical equation solve कर सकता है जिससे हल करने में इंसानों को सालो लग जाएंगे।", "speaker_wav": SPEAKER_REF, "language": LANGUAGE},
    ]

# ── Verify local model files ──────────────────────────────────────────
for fname in ["dvae.pth", "mel_stats.pth", "vocab.json", "model.pth"]:
    path = MODEL_PATH / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

# ── Dataset ───────────────────────────────────────────────────────────
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name=DATASET_NAME,
    path=str(DATASET_PATH),
    meta_file_train="metadata.csv",
    language=LANGUAGE,
)

# ── Test sentences ────────────────────────────────────────────────────
# (loaded from XTTS_TEST_SENTENCES_JSON in paths section above)


def main():
    # ── Model args ────────────────────────────────────────────────
    model_args = GPTArgs(
        max_conditioning_length=66150,   # halved to avoid Windows TDR
        min_conditioning_length=33075,
        max_wav_length=132300,           # halved to ~6s audio
        max_text_length=200,
        mel_norm_file=str(MODEL_PATH / "mel_stats.pth"),
        dvae_checkpoint=str(MODEL_PATH / "dvae.pth"),
        xtts_checkpoint=str(MODEL_PATH / "model.pth"),
        tokenizer_file=str(MODEL_PATH / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050, output_sample_rate=24000
    )
    audio_config.dvae_sample_rate = 22050  # compat with TTS 0.22.0 internals

    # ── Trainer config ────────────────────────────────────────────
    trainer_config = GPTTrainerConfig(
        output_path=str(OUTPUT_PATH),
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="XTTS-v2 fine-tune on voice3 Hindi",
        datasets=[dataset_config],
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=0,  # 0 = no grouping, avoids empty dataloader
        eval_batch_size=1,   # smaller eval batch to fit GPU
        num_loader_workers=0,  # 0 = safe on Windows
        eval_split_max_size=64,
        eval_split_size=0.2,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=100,
        save_n_checkpoints=1,
        save_checkpoints=False,
        print_eval=False,
        epochs=EPOCHS,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
        },
        test_sentences=TEST_SENTENCES,
    )

    # ── Load data ─────────────────────────────────────────────────
    train_samples, eval_samples = load_tts_samples(
        [dataset_config], eval_split=True,
        eval_split_max_size=64, eval_split_size=0.2,
    )
    print(f"Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    # ── Early stopping callback ────────────────────────────────────
    PATIENCE = int(os.getenv("XTTS_EARLY_STOP_PATIENCE", "10"))
    MIN_DELTA = float(os.getenv("XTTS_EARLY_STOP_MIN_DELTA", "0.001"))

    best_loss = float("inf")
    no_improve_count = 0

    def early_stop_cb(trainer_obj):
        nonlocal best_loss, no_improve_count
        # Read eval loss from the trainer's running averages
        avg_values = trainer_obj.keep_avg_eval.avg_values if trainer_obj.keep_avg_eval else {}
        loss = avg_values.get("avg_loss", None)
        if loss is None:
            return
        if best_loss - loss > MIN_DELTA:
            best_loss = loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"\n  Early stop: eval loss not improved for {PATIENCE} epochs "
                      f"(best={best_loss:.4f}, current={loss:.4f})")
                trainer_obj.stop_training = True

    # ── Train ─────────────────────────────────────────────────────
    model = GPTTrainer.init_from_config(trainer_config)

    trainer = Trainer(
        args=TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=GRAD_ACCUM,
        ),
        config=trainer_config,
        output_path=str(OUTPUT_PATH),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks={"on_epoch_end": early_stop_cb},
    )
    trainer.fit()
    print("Training finished.")


if __name__ == "__main__":
    main()
