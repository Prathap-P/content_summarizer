"""
video_producer.py — Local video production pipeline.

Handles SRT generation, thumbnail creation, and FFmpeg video assembly.
No model calls — all data comes from already-generated audio + script.
"""
from __future__ import annotations

import os
import re
import subprocess
import wave
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _wrap_text(text: str, max_chars: int = 30) -> str:
    """Wrap *text* into lines of at most *max_chars* by splitting on words.

    Returns a string with ``\\n`` separating lines (Python newline, NOT
    literal backslash-n).
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return "\n".join(lines) if lines else text

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_audio_duration(wav_path: Path) -> float:
    """Return duration of a WAV file in seconds using the built-in wave module.

    Falls back to ``0.0`` and logs a warning if the file cannot be opened.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception as exc:
        print(f"[WARNING] Could not read WAV duration for {wav_path}: {exc}")
        return 0.0

def generate_srt(script: str, duration_seconds: float, output_path: Path) -> Path:
    """Generate an SRT subtitle file from *script*.

    Splits the script into sentences on ``. ``, ``? ``, ``! ``, groups them
    into subtitle blocks of ~8 words, and distributes timestamps
    proportionally by cumulative character count over *duration_seconds*.

    Parameters
    ----------
    script:
        The spoken-word script text.
    duration_seconds:
        Total audio duration in seconds.
    output_path:
        Destination path (must already carry ``.srt`` extension).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    # --- tokenise into sentences, keeping punctuation ---
    raw_sentences = re.split(r"(?<=[.?!])\s+", script.strip())
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    # --- group into ~8-word subtitle blocks ---
    blocks: list[str] = []
    current_words: list[str] = []
    for sentence in raw_sentences:
        for word in sentence.split():
            current_words.append(word)
            if len(current_words) >= 8:
                blocks.append(" ".join(current_words))
                current_words = []
    if current_words:
        blocks.append(" ".join(current_words))
    if not blocks:
        blocks = [script.strip()]

    total_chars = sum(len(b) for b in blocks)

    # --- build SRT entries ---
    srt_lines: list[str] = []
    elapsed_chars = 0
    for idx, block in enumerate(blocks, start=1):
        start_time = (elapsed_chars / total_chars) * duration_seconds if total_chars else 0.0
        elapsed_chars += len(block)
        end_time = (elapsed_chars / total_chars) * duration_seconds if total_chars else duration_seconds

        srt_lines.append(str(idx))
        srt_lines.append(f"{_fmt_timestamp(start_time)} --> {_fmt_timestamp(end_time)}")
        srt_lines.append(block)
        srt_lines.append("")

    output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    return output_path

def generate_thumbnail(title: str, output_path: Path) -> Path:
    """Generate a 1280×720 JPEG thumbnail with gradient background and title.

    Tries the ``gradients`` lavfi source first; falls back to a solid dark
    colour if that filter is unavailable on the installed FFmpeg build.

    Single quotes in *title* are escaped for FFmpeg's filter parser; newlines
    inserted by word-wrapping are converted to the literal ``\\n`` sequence
    that FFmpeg drawtext understands.

    Parameters
    ----------
    title:
        Human-readable title to render on the thumbnail.
    output_path:
        Destination path (must carry ``.jpg`` extension).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    if not Path(font_path).exists():
        font_path = "Arial"

    # Wrap to ~30 chars per line then escape for FFmpeg drawtext:
    #   - Replace actual newlines with the literal \n FFmpeg drawtext expects
    #   - Escape single quotes with \' for FFmpeg's filter parser (NOT shell-style)
    wrapped = _wrap_text(title, max_chars=30)
    escaped = (
        wrapped
        .replace("'", "\\'")     # FFmpeg filter parser quote escape
        .replace("\n", "\\n")    # Python newline → literal \n for drawtext
    )

    drawtext_vf = (
        f"drawtext=fontfile='{font_path}'"
        f":text='{escaped}'"
        f":fontcolor=white"
        f":fontsize=52"
        f":shadowx=2:shadowy=2"
        f":x=(W-text_w)/2:y=(H-text_h)/2"
    )

    def _run(use_gradient: bool) -> tuple[bool, str]:
        bg = (
            "gradients=size=1280x720:c0=#0d1b2a:c1=#1b2a4a:duration=1"
            if use_gradient
            else "color=c=#0d1b2a:size=1280x720:duration=1"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", bg,
            "-vf", drawtext_vf,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return r.returncode == 0, r.stderr

    ok, _ = _run(True)
    if not ok:
        print("[WARNING] gradients filter unavailable — falling back to solid dark background")
        ok, stderr = _run(False)
        if not ok:
            raise RuntimeError(f"FFmpeg thumbnail generation failed:\n{stderr[-500:]}")

    return output_path

def assemble_video(
    wav_path: Path,
    srt_path: Path,
    output_path: Path,
    duration_seconds: float,
) -> Path:
    """Assemble a 1920×1080 MP4 with waveform visualisation and burnt-in subtitles.

    FFmpeg inputs
    -------------
    0 — lavfi ``color`` background (1920×1080, 30 fps, *duration_seconds* long)
    1 — WAV audio file

    The waveform is rendered from the audio stream via ``showwaves`` and
    overlaid at y=780 (bottom quarter). Subtitles are chained inside
    ``filter_complex`` to avoid the FFmpeg restriction on mixing simple
    and complex filtergraphs.

    Parameters
    ----------
    wav_path:
        Path to the source WAV file.
    srt_path:
        Path to the SRT subtitle file.
    output_path:
        Destination ``.mp4`` path.
    duration_seconds:
        Audio duration used to size the background colour source.

    Returns
    -------
    Path
        *output_path* after successful encoding.

    Raises
    ------
    RuntimeError
        If FFmpeg exits with a non-zero code.
    """
    # SRT path: forward slashes, single-quotes escaped for FFmpeg filter parser
    srt_str = str(srt_path).replace("\\", "/").replace("'", "\\'")

    # NOTE: showwaves requires an audio stream — use [1:a] not [0:v].
    # Subtitles filter is chained inside filter_complex to avoid the
    # FFmpeg error: "simple and complex filtergraphs cannot be used together".
    filter_complex = (
        f"[1:a]showwaves=s=1920x300:mode=line:colors=0x4fc3f7[waves];"
        f"[0:v][waves]overlay=0:780[vcomp];"
        f"[vcomp]subtitles='{srt_str}':force_style='FontSize=24,PrimaryColour=&H00FFFFFF&'[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=#0d1b2a:size=1920x1080:rate=30:duration={duration_seconds}",
        "-i", str(wav_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ]

    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Running FFmpeg assembly ({duration_seconds:.1f}s)…")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[ERROR]   FFmpeg assembly failed (exit {result.returncode})")
        raise RuntimeError(result.stderr[-500:])
    return output_path

def produce_video(audio_file_basename: str, script: str, title: str) -> dict:
    """Orchestrate the full local video production pipeline.

    Parameters
    ----------
    audio_file_basename:
        Filename only, e.g. ``kokoro_500word_20260422_143022.wav``.
        The file must exist under ``kokoro_outputs/``.
    script:
        The condensed spoken-word script used when generating the audio.
    title:
        Human-readable title for the thumbnail.

    Returns
    -------
    dict
        ``{"mp4_path": str, "srt_path": str, "thumb_path": str,
           "duration_seconds": float}``
    """
    os.makedirs("youtube_outputs", exist_ok=True)

    wav_path = Path("kokoro_outputs") / audio_file_basename
    stem = Path(audio_file_basename).stem

    srt_path = Path("youtube_outputs") / f"{stem}.srt"
    thumb_path = Path("youtube_outputs") / f"{stem}_thumb.jpg"
    mp4_path = Path("youtube_outputs") / f"{stem}.mp4"

    # Step 1: audio duration
    t0 = datetime.now()
    duration = get_audio_duration(wav_path)
    print(f"[INFO]    [{t0.strftime('%H:%M:%S')}] Audio duration: {duration:.1f}s")
    if duration <= 0.0:
        raise RuntimeError(f"Cannot produce video: invalid audio duration ({duration}s) for {wav_path}")

    # Step 2: SRT
    t1 = datetime.now()
    generate_srt(script, duration, srt_path)
    elapsed = (datetime.now() - t1).total_seconds()
    print(f"[INFO]    [{t1.strftime('%H:%M:%S')}] SRT generated → {srt_path} ({elapsed:.2f}s)")

    # Step 3: thumbnail
    t2 = datetime.now()
    generate_thumbnail(title, thumb_path)
    elapsed = (datetime.now() - t2).total_seconds()
    print(f"[INFO]    [{t2.strftime('%H:%M:%S')}] Thumbnail generated → {thumb_path} ({elapsed:.2f}s)")

    # Step 4: video assembly
    t3 = datetime.now()
    assemble_video(wav_path, srt_path, mp4_path, duration)
    elapsed = (datetime.now() - t3).total_seconds()
    print(f"[INFO]    [{t3.strftime('%H:%M:%S')}] Video assembled → {mp4_path} ({elapsed:.2f}s)")

    return {
        "mp4_path": str(mp4_path),
        "srt_path": str(srt_path),
        "thumb_path": str(thumb_path),
        "duration_seconds": duration,
    }
