# TTS Speed Optimization Benchmark Report

**Date:** 2026-06-15  
**Task:** t_f1e07974 — Test optimized TTS conversion and measure speedup  
**GPU:** NVIDIA GeForce RTX 5080 (17.1 GB VRAM, Compute 12.0)  
**Model:** XTTS-v2 finetuned for Hindi (hindi_voice3_finetune_v3)  
**Torch:** 2.8.0+cu128  
**Python:** 3.11.9

## Test Samples (Hindi text, varying length)

| Sample | Characters | Description |
|--------|-----------|-------------|
| short | 20 | Simple greeting |
| medium | 97 | Weather description |
| long | 326 | Cultural overview of India |
| xlong | 632 | Technology and AI discussion |
| paragraph | 660 | Comprehensive TTS evaluation text |

## Results Summary

| Metric | FP16 (Optimized) | FP32 (Baseline) | Δ |
|--------|:----------------:|:----------------:|:-:|
| **Total Inference** | 208.0s | 187.5s | **FP32 11% faster** |
| **Model Load Time** | 177.5s | 99.0s | **FP32 44% faster** |
| **Average RTF** | 1.97x | 1.75x | **FP32 13% lower** |
| **VRAM Usage** | 1.0 GB | 1.8 GB | **FP16 44% less** |
| **Audio Quality** | ✓ All valid | ✓ All valid | **Equal** |

## Per-Sample Speed Comparison

| Sample | FP16 (s) | FP32 (s) | Speedup | Faster |
|--------|:--------:|:--------:|:-------:|:------:|
| short | 4.41 | 3.61 | 0.82x | FP32 22% |
| medium | 12.14 | 9.61 | 0.79x | FP32 26% |
| long | 38.46 | 35.55 | 0.92x | FP32 8% |
| xlong | 77.49 | 67.12 | 0.87x | FP32 15% |
| paragraph | 75.54 | 71.64 | 0.95x | FP32 5% |
| **TOTAL** | **208.00** | **187.50** | **0.90x** | **FP32 11%** |

## Acceptance Criteria Evaluation

1. **Documented speedup (aim for ≥2× improvement under GPU)** ❌ FAIL
   - Net result: FP16 is **10% slower** than FP32 baseline
   - No sample achieved even 1× speedup; FP32 is faster on all samples
   - Model loading is also 44% slower with FP16 (177.5s vs 99.0s)

2. **Audio quality maintained (no artifacts/truncation)** ✓ PASS
   - All 10 WAV files (5 FP16 + 5 FP32) verified valid by torchaudio
   - No corruption, truncation, or duration mismatches detected
   - Audio durations are consistent between modes (<0.5s difference per sample)

## Analysis

**Why FP16 didn't help on RTX 5080:**

1. **Blackwell architecture tensor cores** — RTX 5080 has exceptional FP32 throughput via 5th-gen tensor cores. The FP32→FP16 performance gap is much narrower than on older GPUs.

2. **Selective FP16 casting** — The optimization casts only the GPT (transformer) to half-precision, leaving hifigan_decoder and speaker encoder in FP32. This creates format conversion overhead at every GPT→decoder boundary.

3. **Model loading overhead** — FP16 requires computing latents first, then casting the model. This adds 78s overhead vs native FP32 loading.

4. **autocast overhead** — The `torch.cuda.amp.autocast()` context manager adds dispatch overhead, especially for a model where only parts are FP16.

**What does work:**
- `torch.no_grad()` — eliminates gradient tracking overhead (already applied)
- Model caching — keeps model warm in GPU memory between calls
- VRAM savings — FP16 uses 44% less VRAM (1.0 vs 1.8 GB), useful for multi-model setups

**Recommendation:**
- Drop selective FP16 casting for RTX 5080 (and likely all Blackwell+ GPUs)
- Keep `torch.no_grad()` and model caching
- Consider `torch.compile()` for further speedups if supported
- The real bottleneck is the autoregressive GPT inference, which is inherently sequential

## Raw Data

Saved to `benchmark_results/benchmark_report.json`.  
Audio files: `benchmark_results/fp16/*.wav` and `benchmark_results/fp32/*.wav`
