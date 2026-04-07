"""YouTube transcript fetching.

Two distinct paths — callers choose which one they need:

  get_youtube_transcript(url)
      Transcript-queue path.  Three-step fallback chain, no audio download:
        1. fetch(video_id, languages=['en']) — fast English path.
        2. list(video_id) + translate('en') — library translate path,
           cookie-free, no yt-dlp.  Default foreign-language fallback.
        3. yt-dlp extract_info automatic_captions['en'] — kept as last-resort
           alternative; needs YTDLP_COOKIES_BROWSER set to avoid 429.
      Returns an "Error:" string when all three steps fail.

  get_transcript_via_whisper(url, video_id)   [from whisper_transcriber]
      Audio-queue path.  Always downloads audio via yt-dlp + mlx-whisper.
"""

import json
import os
import re
from datetime import datetime

import yt_dlp
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# Browser yt-dlp uses to read cookies for YouTube auto-translated captions.
# YouTube's timedtext endpoint returns 429 for programmatic requests to
# auto-translated tracks unless the request carries fresh browser session
# cookies.  Set YTDLP_COOKIES_BROWSER=none to disable (e.g. on a server with
# no browser); the fetch will then be attempted without cookies and may 429.
# Valid values: safari (default on macOS), chrome, firefox, chromium, edge.
_YTDLP_COOKIES_BROWSER: str = os.getenv("YTDLP_COOKIES_BROWSER", "safari")

# get_transcript_via_whisper is NOT used here — it belongs to the audio-queue
# path only. Import it from whisper_transcriber directly when needed.


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fetch_translated_via_api(video_id: str) -> str:
    """Fetch a translated-to-English transcript using youtube-transcript-api.

    Calls ``list(video_id)`` to find any auto-generated, translatable
    non-English track, then calls ``.translate('en').fetch()`` which appends
    ``&tlang=en`` to YouTube's own timedtext URL — the same translation service
    the web player uses.  No cookies, no yt-dlp, no browser reads.

    Raises ``ValueError`` if no translatable non-English track is found, or
    re-raises any exception from the API (e.g. ``NotTranslatable``,
    ``RequestBlocked``).
    """
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)
    # Find any auto-generated track that YouTube agrees can be translated.
    # Prefer 'hi' but accept any non-English generated track with is_translatable.
    candidate = None
    for t in transcript_list:
        if t.language_code != 'en' and t.is_generated and t.is_translatable:
            if t.language_code == 'hi':
                candidate = t
                break
            if candidate is None:
                candidate = t  # take first match; override if 'hi' found later
    if candidate is None:
        raise ValueError(
            f"No translatable auto-generated non-English transcript for {video_id}"
        )
    print(
        f"[INFO]    [{_ts()}] Translating '{candidate.language}' "
        f"({candidate.language_code}) → English via API for {video_id}"
    )
    fetched = candidate.translate('en').fetch()
    text = " ".join(e.text for e in fetched)
    return text


def _fetch_ytdlp_auto_translated_english(video_id: str) -> str:
    """Fetch YouTube's auto-translated English captions via yt-dlp.

    Last-resort fallback.  yt-dlp's extract_info() exposes the translated
    track via automatic_captions['en'] using its own client negotiation,
    bypassing the ``isTranslatable: false`` flag that the InnerTube mobile API
    can return.  Reads browser cookies (YTDLP_COOKIES_BROWSER) to avoid 429.

    Returns the plain English text, or raises on any failure.
    """
    ydl_opts: dict = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    if _YTDLP_COOKIES_BROWSER.lower() != 'none':
        ydl_opts['cookiesfrombrowser'] = (_YTDLP_COOKIES_BROWSER,)
        print(f"[DEBUG]   yt-dlp using cookies from browser: {_YTDLP_COOKIES_BROWSER}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}", download=False
        )
        auto_captions = info.get('automatic_captions', {})
        en_formats = auto_captions.get('en', [])
        if not en_formats:
            raise ValueError(f"No auto-translated English captions for {video_id}")

        json3_url = next(
            (f['url'] for f in en_formats if f['ext'] == 'json3'), None
        )
        if not json3_url:
            raise ValueError("json3 format not available in auto-translated captions")

        resp = ydl.urlopen(json3_url)
        data = json.loads(resp.read().decode('utf-8'))

        lines = []
        for event in data.get('events', []):
            text = ''.join(s.get('utf8', '') for s in event.get('segs', [])).strip()
            if text:
                lines.append(text)

        return ' '.join(lines)


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

    Three-step fallback chain:
      1. fetch(video_id, languages=['en']) — fast English path, no extra
         round-trip.  Handles English and English auto-generated tracks.
      2. list(video_id) + translate('en') — cookie-free library translate path.
         Finds a non-English auto-generated track with is_translatable=True and
         calls .translate('en').fetch().  This is the default foreign-language
         fallback; no yt-dlp, no browser reads, no cookies.
      3. yt-dlp automatic_captions['en'] — last-resort fallback.  Bypasses the
         isTranslatable: false restriction that the InnerTube API can return.
         Reads browser cookies via YTDLP_COOKIES_BROWSER to avoid 429.
    If all three steps fail, returns an ``"Error:"`` string — no audio fallback.

    Returns:
        Plain transcript text on success.
        A string prefixed with ``"Invalid YouTube"`` or ``"Error:"`` on failure.
    """
    video_id = extract_video_id(video_link)
    if not video_id:
        return "Invalid YouTube URL format."

    print(f"[INFO]    [{_ts()}] Fetching transcript via API for video: {video_id}")

    # Step 1 — try English directly (fast path, no extra round-trip)
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        text = " ".join(e.text for e in fetched)
        print(f"[INFO]    [{_ts()}] English transcript fetched: {len(text):,} chars")
        return text
    except Exception:
        print(f"[INFO]    [{_ts()}] No English transcript for {video_id}, trying API translate path...")

    # Step 2 — cookie-free translate via youtube-transcript-api.
    # list() finds all available tracks; we pick a non-English auto-generated
    # one with is_translatable=True and call .translate('en').fetch().
    try:
        text = _fetch_translated_via_api(video_id)
        if not text:
            raise ValueError("Empty transcript returned")
        print(f"[INFO]    [{_ts()}] API translate path succeeded: {len(text):,} chars")
        return text
    except Exception as e:
        print(f"[INFO]    [{_ts()}] API translate path failed for {video_id}: {e}. Falling back to yt-dlp...")

    # Step 3 — yt-dlp last-resort: reads browser cookies to avoid 429.
    # youtube-transcript-api can return isTranslatable: false on some Hindi
    # auto-generated tracks even when YouTube's web player serves auto-translate.
    # yt-dlp's own client negotiation bypasses that restriction.
    try:
        print(f"[INFO]    [{_ts()}] Fetching auto-translated English captions via yt-dlp for {video_id}")
        text = _fetch_ytdlp_auto_translated_english(video_id)
        if not text:
            raise ValueError("Empty transcript returned")
        print(f"[INFO]    [{_ts()}] yt-dlp translate path succeeded: {len(text):,} chars")
        return text
    except Exception as e:
        print(f"[WARNING] All transcript paths failed for {video_id}: {e}")
        return f"Error: No accessible English or auto-translated transcript for video {video_id} ({e})."