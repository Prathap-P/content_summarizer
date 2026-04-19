"""Voxtral TTS backend — mlx-audio based, Apple Silicon only.

Lazy-loaded model singleton. All heavy imports are inside function bodies
so the main Flask process never loads model weights. Runs in a spawn
subprocess via process_runner.run_in_subprocess().
"""

import gc
import os
import re
import random
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOICES = ["neutral_male", "neutral_female"]
SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 350                            # short chunks prevent autoregressive quality drift
TARGET_LUFS = -23.0                              # ITU-R BS.1770 integrated loudness target (LUFS)
INTER_CHUNK_SILENCE_MS = 400                     # silence gap between stitched chunks (ms)

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_model = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _get_model():
    """Load the Voxtral model once; return the cached instance on subsequent calls."""
    global _model
    if _model is None:
        from mlx_audio.tts.utils import load  # heavy import — inside function body
        model_id = os.getenv("VOXTRAL_MODEL_ID", "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit")
        print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Loading model: {model_id}")
        _model = load(model_id)
        print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Model loaded")
    return _model


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at natural speech boundaries.

    Priority order: paragraph breaks → sentence ends → clause boundaries.
    Chunks never exceed max_chars characters.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        sentences = re.split(r'(?<=[.!?…])\s+', para)
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(sent) > max_chars:
                    # Long sentence — split at clause boundaries
                    clauses = re.split(r'(?<=[,;])\s+', sent)
                    sub = ""
                    for clause in clauses:
                        if len(sub) + len(clause) + 1 <= max_chars:
                            sub = f"{sub} {clause}".strip()
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = clause
                    current = sub if sub else ""
                else:
                    current = sent
        if current:
            chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _create_silence(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a float32 silence array of the given duration."""
    return np.zeros(int(sample_rate * duration_ms / 1000), dtype=np.float32)


def _crossfade(seg1: np.ndarray, seg2: np.ndarray, overlap_ms: int = 50) -> np.ndarray:
    """Blend two audio segments with a crossfade to eliminate click artifacts."""
    overlap = int(SAMPLE_RATE * overlap_ms / 1000)
    if len(seg1) < overlap or len(seg2) < overlap:
        return np.concatenate([seg1, seg2])
    fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    blended = seg1[-overlap:] * fade_out + seg2[:overlap] * fade_in
    return np.concatenate([seg1[:-overlap], blended, seg2[overlap:]])


def _normalize_loudness(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to TARGET_LUFS using ITU-R BS.1770 integrated loudness.

    Falls back to RMS normalization (-23 dBFS ≈ 0.07 RMS) when the chunk is
    too short for pyloudnorm's 400ms minimum measurement window.
    """
    import pyloudnorm as pyln  # heavy import — inside function body; installed as mlx-audio dep

    _FALLBACK_RMS = 0.07  # ≈ -23 dBFS, matches TARGET_LUFS
    _MIN_LUFS_SAMPLES = int(0.4 * SAMPLE_RATE)  # pyloudnorm requires at least 400ms

    if len(audio) < _MIN_LUFS_SAMPLES:
        # Too short for integrated loudness — use RMS fallback
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
        return audio * (_FALLBACK_RMS / rms)

    try:
        meter = pyln.Meter(SAMPLE_RATE)
        measured = meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(measured) or measured < -70.0:
            # Measurement unreliable (silence or near-silence) — RMS fallback
            rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
            return audio * (_FALLBACK_RMS / rms)
        normalized = pyln.normalize.loudness(audio.astype(np.float64), measured, TARGET_LUFS)
        # Clip to [-1, 1] to prevent downstream clipping in WAV writer
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    except Exception:
        # pyloudnorm failed for any reason — safe RMS fallback
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
        return (audio * (_FALLBACK_RMS / rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_audio(text: str, voice: str | None = None) -> np.ndarray:
    """Generate TTS audio from text using a randomly selected Voxtral voice.

    Splits long text into ≤350-char chunks, generates each chunk separately,
    stitches with 400ms silence gaps and 50ms crossfades, and returns a single
    24 kHz float32 numpy array — same contract as kokoro_tts.generate_audio().

    Args:
        text: Input text (any length).
        voice: Voxtral voice name. If None, a random voice from VOICES is chosen.

    Returns:
        float32 numpy array at 24 kHz sample rate.
    """
    model = _get_model()
    if not voice:
        voice = random.choice(VOICES)
    print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Selected voice: {voice}")

    chunks = _split_into_chunks(text)
    print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Generating audio for {len(text)} chars in {len(chunks)} chunks")

    silence = _create_silence(INTER_CHUNK_SILENCE_MS)
    segments: list[np.ndarray] = []

    for i, chunk in enumerate(chunks):
        print(f"[DEBUG]   [VOXTRAL_TTS] Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
        parts: list[np.ndarray] = []
        for result in model.generate(text=chunk, voice=voice):
            parts.append(np.array(result.audio, dtype=np.float32))

        if not parts:
            print(f"[WARNING] [VOXTRAL_TTS] model.generate() yielded nothing for chunk {i + 1} — skipping")
            continue

        chunk_audio = np.concatenate(parts) if len(parts) > 1 else parts[0]

        # Normalize per-chunk loudness to ITU-R BS.1770 target (-23 LUFS)
        chunk_audio = _normalize_loudness(chunk_audio)

        segments.append(chunk_audio)

        if i < len(chunks) - 1:
            segments.append(silence)

        if (i + 1) % 10 == 0:
            gc.collect()

    if not segments:
        print(f"[ERROR]   [{_ts()}] [VOXTRAL_TTS] No audio segments generated — returning empty array")
        return np.zeros(1, dtype=np.float32)

    # Stitch with crossfade
    final = segments[0]
    for seg in segments[1:]:
        final = _crossfade(final, seg)

    print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Audio generated: {len(final)} samples, {len(final) / SAMPLE_RATE:.2f}s")
    return final
