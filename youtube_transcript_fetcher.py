"""YouTube transcript fetching.

Two distinct paths — callers choose which one they need:

  get_youtube_transcript(url)
      Transcript-queue path.  Calls YouTubeTranscriptApi().fetch() directly—
      no list_transcripts(), no metadata API calls, no Whisper.
      Returns an "Error:" string when the API fails so the caller can surface
      a message telling the user to use the Audio Queue instead.

  get_transcript_via_whisper(url, video_id)   [from whisper_transcriber]
      Audio-queue path.  Always downloads audio via yt-dlp + mlx-whisper.
"""

import re
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi

# get_transcript_via_whisper is NOT used here — it belongs to the audio-queue
# path only. Import it from whisper_transcriber directly when needed.


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def extract_video_id(video_link: str) -> str | None:
    """Extract the 11-character video ID from any recognised YouTube URL format.

    Handles: youtube.com/watch, youtu.be, /shorts/, /embed/, /live/,
    and youtube.com/?v= query-param variants.
    Returns None if no valid ID can be found.
    """
    pattern = (
        r"(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?|live)/|\S*?[?&]v=)"
        r"|youtu\.be/|youtube\.com/shorts/)"
        r"([a-zA-Z0-9_-]{11})"
    )
    match = re.search(pattern, video_link)
    return match.group(1) if match else None


def get_youtube_transcript(video_link: str) -> str:
    """Fetch a transcript for a YouTube video using the Transcript API only.

    This is the **transcript-queue path**.  It never downloads audio and never
    calls Whisper.

    Uses ``YouTubeTranscriptApi().fetch()`` directly — no ``list_transcripts()``
    call, no extra round-trips, no dependency on transcript metadata APIs that
    are unreliable.  If the fetch fails for any reason (no transcript, private
    video, API error), returns an ``"Error:"`` string so the caller can surface
    it to the user (who should then use the Audio Queue instead).

    Returns:
        Plain transcript text on success.
        A string prefixed with ``"Invalid YouTube"`` or ``"Error:"`` on failure.
    """
    video_id = extract_video_id(video_link)
    if not video_id:
        return "Invalid YouTube URL format."

    print(f"[INFO]    [{_ts()}] Fetching transcript via API for video: {video_id}")
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        if not fetched:
            return (
                f"Error: No transcript available for video {video_id}. "
                f"Try the Audio Queue to transcribe via Whisper instead."
            )
        # youtube-transcript-api >= 0.7 returns FetchedTranscriptSnippet objects
        # (attribute access); older versions returned plain dicts (key access).
        # Support both to avoid breaking on any installed version.
        def _get_text(entry) -> str:
            try:
                return entry.text
            except AttributeError:
                return entry["text"]

        text = " ".join(_get_text(e) for e in fetched)
        print(f"[INFO]    [{_ts()}] Transcript fetched: {len(text):,} chars")
        return text
    except Exception as e:
        print(f"[WARNING] Transcript API fetch failed for {video_id}: {e}")
        return (
            f"Error: Could not fetch transcript for video {video_id} ({e}). "
            f"Try the Audio Queue to transcribe via Whisper instead."
        )