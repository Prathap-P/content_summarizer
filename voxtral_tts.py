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

VOICES = ["neutral_male"]
SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 500                            # short chunks prevent autoregressive quality drift
INTER_CHUNK_SILENCE_MS = 250                     # silence gap between stitched chunks (ms)

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
        model_id = os.getenv("VOXTRAL_MODEL_ID", "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")
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


def _normalize_peak(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Normalize audio to the given peak amplitude."""
    max_val = float(np.max(np.abs(audio)))
    if max_val < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_val * peak).astype(np.float32)


def _reduce_noise(audio: np.ndarray) -> np.ndarray:
    """Apply spectral gating noise reduction to the assembled audio."""
    import noisereduce as nr  # heavy import — inside function body
    try:
        return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.6).astype(np.float32)
    except Exception:
        return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_audio(text: str, voice: str | None = None) -> np.ndarray:
    """Generate TTS audio from text using a randomly selected Voxtral voice.

    Splits long text into ≤500-char chunks, generates each chunk separately,
    stitches with 250ms silence gaps and 50ms crossfades, and returns a single
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

        # Normalize per-chunk loudness to 0.9 peak
        chunk_audio = _normalize_peak(chunk_audio)

        segments.append(chunk_audio)

        if i < len(chunks) - 1:
            segments.append(silence.copy())

        if (i + 1) % 10 == 0:
            gc.collect()

    if not segments:
        print(f"[ERROR]   [{_ts()}] [VOXTRAL_TTS] No audio segments generated — returning empty array")
        return np.zeros(1, dtype=np.float32)

    # Stitch with crossfade
    final = segments[0]
    for seg in segments[1:]:
        final = _crossfade(final, seg)

    final = _reduce_noise(final)
    final = _normalize_peak(final)

    print(f"[INFO]    [{_ts()}] [VOXTRAL_TTS] Audio generated: {len(final)} samples, {len(final) / SAMPLE_RATE:.2f}s")
    return final
