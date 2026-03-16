"""Audio download and Whisper transcription pipeline.

Flow:
    get_transcript_via_whisper(url, video_id)
        ├─ get_video_duration()   — metadata-only fetch, warn if > 90 min
        ├─ download_audio()       — yt-dlp → yt_audio/<id>.mp3
        │       └─ REUSES cached file if it already exists and is non-empty
        └─ transcribe_audio()     — mlx-whisper large-v3
                └─ GPU memory released in finally block (Apple Silicon only)
"""

import gc
import platform
from datetime import datetime
from pathlib import Path

import mlx_whisper
import yt_dlp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YT_AUDIO_DIR = Path("yt_audio")
YT_AUDIO_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DURATION_WARNING_THRESHOLD_SECONDS = 90 * 60

# Platform guard: Metal memory cache only exists on Apple Silicon.
_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# ---------------------------------------------------------------------------
# yt-dlp client strategy (2026-03+)
#
# Do NOT specify extractor_args / player_client overrides.
#
# History of why explicit client pinning was removed:
#   ios         — Now requires a GVS PO Token; without it every audio/video
#                 format is skipped, leaving only thumbnail storyboards.
#   mweb        — Same; also requires GVS PO Token since early 2026.
#   tv_embedded — Marked unsupported in yt-dlp ≥ 2026.03.
#   android     — Returns only one combined format (360p) without PO token.
#
# yt-dlp's built-in automatic client selection (android_vr fallback as of
# 2026.03.13) resolves all DASH audio formats (139/140/249/251) without any
# cookie or PO token.  Let it do its job.
# ---------------------------------------------------------------------------

# Stable YouTube DASH audio format IDs in preference order:
#   140 = AAC  128k m4a  — present on virtually every public video since 2013
#   251 = Opus 160k webm — present on most modern videos
#   139 = AAC   48k m4a  — fallback for age-restricted or older videos
# Generic quality-based selectors follow as last resort.
_AUDIO_FORMAT = "140/251/139/bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _release_gpu_memory() -> None:
    """Release MLX/Metal GPU memory after Whisper use.

    No-op on non-Apple-Silicon platforms so the code runs unmodified on Linux/CUDA.
    """
    gc.collect()
    if _IS_APPLE_SILICON:
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception as e:
            print(f"[WARNING] Could not clear MPS cache: {e}")
    print(f"[INFO]    [{_ts()}] Whisper GPU memory released")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_video_duration(url: str) -> int | None:
    """Return video duration in seconds without downloading any media.

    Uses yt-dlp's automatic client selection — no extractor_args override.
    Returns None on any error.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("duration")
    except Exception as e:
        print(f"[WARNING] Could not fetch video duration: {e}")
        return None


def download_audio(url: str, video_id: str) -> Path | None:
    """Download the audio track from a YouTube URL using yt-dlp.

    Saves to ``yt_audio/<video_id>.mp3``.

    Idempotent: if the file already exists and has a non-zero size it is reused
    without re-downloading. This means a crash mid-transcription does not force
    a full re-download on retry.

    Returns the resolved Path on success, None on failure.
    """
    output_path = YT_AUDIO_DIR / f"{video_id}.mp3"

    # Reuse cached file instead of re-downloading
    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(
            f"[INFO]    [{_ts()}] Reusing cached audio: "
            f"{output_path.name} ({size_mb:.1f} MB)"
        )
        return output_path

    output_template = str(YT_AUDIO_DIR / video_id)
    ydl_opts = {
        "format": _AUDIO_FORMAT,
        "outtmpl": output_template + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"[ERROR]   yt-dlp download failed: {e}")
        return None

    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"[ERROR]   Audio file missing or empty after download: {output_path}")
        return None

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(
        f"[INFO]    [{_ts()}] Audio downloaded: {output_path.name} ({size_mb:.1f} MB)"
    )
    return output_path


def transcribe_audio(audio_path: Path) -> str | None:
    """Transcribe an audio file with mlx-whisper (large-v3 by default).

    GPU memory is always released in the ``finally`` block so the ~3 GB of MLX
    weights are not held for the lifetime of the Flask process.

    Returns plain transcript text, or None on failure or empty result.
    """
    print(
        f"[INFO]    [{_ts()}] Starting Whisper transcription: "
        f"{audio_path.name}  (model: {WHISPER_MODEL})"
    )
    try:
        result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=WHISPER_MODEL)
        text = result.get("text", "").strip()
        if not text:
            print("[WARNING] Whisper returned an empty transcript.")
            return None
        print(f"[INFO]    [{_ts()}] Whisper transcription complete: {len(text):,} chars")
        return text
    except Exception as e:
        print(f"[ERROR]   Whisper transcription failed: {e}")
        return None
    finally:
        _release_gpu_memory()


def get_transcript_via_whisper(url: str, video_id: str) -> str:
    """Full Whisper pipeline: duration check → audio download → transcription.

    Returns:
        Transcript text string on success.
        A string prefixed with ``"Error:"`` on any failure.
    """
    duration = get_video_duration(url)
    if duration is not None:
        minutes = duration // 60
        print(f"[INFO]    [{_ts()}] Video duration: {minutes} min ({duration}s)")
        if duration > DURATION_WARNING_THRESHOLD_SECONDS:
            print(
                f"[WARNING] Video is {minutes} min — "
                f"Whisper transcription may take 5–15 min on large-v3"
            )

    print(f"[INFO]    [{_ts()}] Downloading audio via yt-dlp...")
    audio_path = download_audio(url, video_id)
    if not audio_path:
        return "Error: Failed to download audio from YouTube."

    transcript = transcribe_audio(audio_path)
    if not transcript:
        return "Error: Whisper transcription failed or produced empty output."

    return transcript
