"""Standalone utility: list all available subtitles/transcripts for a YouTube video.

Usage:
    python list_subtitles.py <youtube_url>
    python list_subtitles.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

This script is NOT imported by the main application.
It reuses extract_video_id() from youtube_transcript_fetcher.py.
"""

import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_fetcher import extract_video_id


def list_available_transcripts(youtube_url: str) -> None:
    """Print all available transcript tracks for the given YouTube URL.

    Shows for each track:
      - Language name and code
      - Whether it is auto-generated or manually created
      - Whether it is translatable to other languages
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print(f"[ERROR] Could not extract a video ID from: {youtube_url}")
        return

    print(f"\nVideo ID : {video_id}")
    print(f"URL      : https://www.youtube.com/watch?v={video_id}\n")

    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
    except Exception as exc:
        print(f"[ERROR] Failed to retrieve transcript list: {exc}")
        return

    manual = []
    generated = []

    for t in transcript_list:
        entry = {
            "language": t.language,
            "code": t.language_code,
            "translatable": t.is_translatable,
        }
        if t.is_generated:
            generated.append(entry)
        else:
            manual.append(entry)

    if manual:
        print(f"--- Manually created ({len(manual)}) ---")
        for t in manual:
            translatable = "translatable" if t["translatable"] else "not translatable"
            print(f"  [{t['code']:10s}]  {t['language']}  ({translatable})")
    else:
        print("--- Manually created: none ---")

    print()

    if generated:
        print(f"--- Auto-generated ({len(generated)}) ---")
        for t in generated:
            translatable = "translatable" if t["translatable"] else "not translatable"
            print(f"  [{t['code']:10s}]  {t['language']}  ({translatable})")
    else:
        print("--- Auto-generated: none ---")

    total = len(manual) + len(generated)
    print(f"\nTotal tracks found: {total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_subtitles.py <youtube_url>")
        sys.exit(1)

    list_available_transcripts(sys.argv[1])
