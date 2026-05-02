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

def _download_youtube_thumbnail(video_id: str, output_path: Path) -> bool:
    """Download the original YouTube video thumbnail.

    Tries maxresdefault (1280×720) first, falls back to hqdefault (480×360).
    YouTube returns a small grey placeholder for maxresdefault when it doesn't
    exist — detected by file size < 5 KB and retried with hqdefault.

    Returns True if a thumbnail was saved to *output_path*, False otherwise.
    """
    import ssl
    import urllib.request
    # Use a permissive SSL context — img.youtube.com certificate chain may not
    # be in the system trust store on some macOS Python installs.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for quality in ("maxresdefault", "hqdefault", "sddefault"):
        thumb_url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        try:
            with urllib.request.urlopen(thumb_url, context=ctx) as resp:
                data = resp.read()
            if len(data) > 5000:  # real thumb always >5 KB
                output_path.write_bytes(data)
                print(
                    f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
                    f"YouTube thumbnail downloaded ({quality}) → {output_path.name}"
                )
                return True
            # placeholder grey image — try next quality
        except Exception as exc:
            print(f"[WARNING] YouTube thumbnail download failed ({quality}): {exc}")
    return False


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

    word_counts = [len(b.split()) for b in blocks]
    total_words = sum(word_counts)

    # --- build SRT entries ---
    srt_lines: list[str] = []
    elapsed_words = 0
    for idx, block in enumerate(blocks, start=1):
        start_time = (elapsed_words / total_words) * duration_seconds if total_words else (idx - 1) / len(blocks) * duration_seconds
        elapsed_words += word_counts[idx - 1]
        end_time = (elapsed_words / total_words) * duration_seconds if total_words else idx / len(blocks) * duration_seconds

        srt_lines.append(str(idx))
        srt_lines.append(f"{_fmt_timestamp(start_time)} --> {_fmt_timestamp(end_time)}")
        srt_lines.append(block)
        srt_lines.append("")

    output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    return output_path

def generate_thumbnail(title: str, output_path: Path) -> Path:
    """Generate a 1280×720 JPEG thumbnail using Pillow.

    Dark gradient background with centred white title text. No FFmpeg
    dependency — works regardless of which filters the installed FFmpeg
    was compiled with.
    """
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1280, 720

    # --- background: vertical dark-blue gradient ---
    img = Image.new("RGB", (W, H))
    top_colour = (13, 27, 42)     # #0d1b2a
    bot_colour = (27, 42, 74)     # #1b2a4a
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        r = int(top_colour[0] + t * (bot_colour[0] - top_colour[0]))
        g = int(top_colour[1] + t * (bot_colour[1] - top_colour[1]))
        b = int(top_colour[2] + t * (bot_colour[2] - top_colour[2]))
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # --- font: try system Helvetica, fall back to default ---
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.load_default()
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, size=52)
                break
            except Exception:
                continue

    # --- wrap and centre text ---
    wrapped = _wrap_text(title, max_chars=30)
    lines = wrapped.split("\n")

    line_height = 60
    total_text_h = len(lines) * line_height
    y_start = (H - total_text_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (W - text_w) // 2
        y = y_start + i * line_height
        # shadow
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))

    img.save(str(output_path), "JPEG", quality=90)
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

    # Escape characters special in FFmpeg's filter_complex option parser.
    # Do NOT wrap the path in single quotes — FFmpeg treats 'value' as a quoted
    # block that consumes colons, merging force_style into the filename and
    # leaving [vout] dangling (exit 234 / "Invalid argument").
    srt_str = (
        str(srt_path)
        .replace("\\", "\\\\")   # must be first; covers Windows paths
        .replace("'",  "\\'")
        .replace(":",  "\\:")
        .replace("[",  "\\[")
        .replace("]",  "\\]")
        .replace(";",  "\\;")
    )

    fc_with_subs = (
        f"[1:a]showwaves=s=1920x300:mode=line:colors=0x4fc3f7[waves];"
        f"[0:v][waves]overlay=0:780[vcomp];"
        f"[vcomp]subtitles={srt_str}:force_style=FontSize=24[vout]"
    )
    fc_without_subs = (
        f"[1:a]showwaves=s=1920x300:mode=line:colors=0x4fc3f7[waves];"
        f"[0:v][waves]overlay=0:780[vout]"
    )

    def _build_cmd(filter_complex_str: str) -> list:
        return [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=#0d1b2a:size=1920x1080:rate=30:duration={duration_seconds}",
            "-i", str(wav_path),
            "-filter_complex", filter_complex_str,
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
    result = subprocess.run(_build_cmd(fc_with_subs), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # Fall back to no-subtitles for any failure — subtitles filter requires
        # libass which may not be compiled into the local FFmpeg build.
        print(f"[WARNING] Subtitles filter failed (exit {result.returncode}) — retrying without subtitles")
        result = subprocess.run(_build_cmd(fc_without_subs), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"[ERROR]   FFmpeg assembly failed (exit {result.returncode})")
            raise RuntimeError(result.stderr[-500:])
    return output_path

def produce_video(audio_file_basename: str, script: str, title: str, video_id: str = "") -> dict:
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
    video_id:
        YouTube video ID used to download the original thumbnail.
        If empty, thumbnail step is skipped.

    Returns
    -------
    dict
        ``{"mp4_path": str, "srt_path": str, "thumb_path": str,
           "duration_seconds": float}``
    """

    os.makedirs(Path(__file__).parent / "youtube_outputs", exist_ok=True)

    wav_path = Path(__file__).parent / "kokoro_outputs" / audio_file_basename
    stem = Path(audio_file_basename).stem

    if not wav_path.exists():
        raise RuntimeError(f"Audio file not found: {wav_path}")

    srt_path = Path(__file__).parent / "youtube_outputs" / f"{stem}.srt"
    thumb_path = Path(__file__).parent / "youtube_outputs" / f"{stem}_thumb.jpg"
    mp4_path = Path(__file__).parent / "youtube_outputs" / f"{stem}.mp4"

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

    # Step 3: thumbnail — use original YouTube thumbnail when video_id is available.
    # Custom thumbnail generation (generate_thumbnail) is disabled for now;
    # re-enable it below once the design is finalized.
    t2 = datetime.now()
    if video_id:
        ok = _download_youtube_thumbnail(video_id, thumb_path)
        if not ok:
            print(f"[WARNING] Could not download YouTube thumbnail — video will upload without thumbnail")
    else:
        # generate_thumbnail(title, thumb_path)  # TODO: re-enable custom thumbnail
        print(f"[INFO]    [{t2.strftime('%H:%M:%S')}] No video_id provided — skipping thumbnail")
    elapsed = (datetime.now() - t2).total_seconds()
    print(f"[INFO]    [{t2.strftime('%H:%M:%S')}] Thumbnail step done ({elapsed:.2f}s)")

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
