"""YouTube Data API v3 upload, quota tracking, and auto-publish helpers."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_TOKEN_FILE = Path(__file__).parent / "token.json"
_QUOTA_FILE = Path(__file__).parent / "youtube_quota.json"
_PT_TIMEZONE = "America/Los_Angeles"

_CATEGORY_IDS: dict[str, str] = {
    "tech": "28",
    "science": "28",
    "social": "22",
    "news": "25",
}

_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

def _get_youtube_service() -> Any:
    """Load credentials from token.json and return a YouTube API v3 service object."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    if not _TOKEN_FILE.exists():
        raise RuntimeError(
            f"token.json not found at {_TOKEN_FILE}. "
            "Run `python youtube_auth.py` to authorise the app first."
        )

    creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), _SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _TOKEN_FILE.write_text(creds.to_json())
        except RefreshError as exc:
            raise RuntimeError(
                f"Token refresh failed — re-run `python youtube_auth.py`. ({exc})"
            ) from exc

    return build("youtube", "v3", credentials=creds)

def _today_pt() -> str:
    """Return today's date string in Pacific Time (YYYY-MM-DD)."""
    return datetime.now(timezone.utc).astimezone(ZoneInfo(_PT_TIMEZONE)).strftime("%Y-%m-%d")

def _read_quota() -> dict:
    """Read quota file, creating it with defaults if missing."""
    if not _QUOTA_FILE.exists():
        data: dict = {"date": "", "used": 0}
        tmp = _QUOTA_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, _QUOTA_FILE)
        return data
    return json.loads(_QUOTA_FILE.read_text())

def _check_quota(units_needed: int) -> None:
    """Raise RuntimeError if adding units_needed would exceed the daily 10,000-unit limit."""
    data = _read_quota()
    today = _today_pt()
    used = data["used"] if data.get("date") == today else 0

    if used + units_needed > 10000:
        raise RuntimeError(
            f"Daily YouTube quota exceeded (used: {used}/10000). "
            "Resets at midnight Pacific Time."
        )

def _consume_quota(units: int) -> None:
    """Increment the quota counter by units, resetting if the PT date has changed."""
    data = _read_quota()
    today = _today_pt()

    if data.get("date") != today:
        data = {"date": today, "used": 0}

    data["used"] = data["used"] + units
    tmp = _QUOTA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, _QUOTA_FILE)

def upload_video(
    mp4_path: "str | Path",
    title: str,
    description: str,
    thumb_path: "str | Path",
    srt_path: "str | Path",
    category: str,
) -> str:
    """Upload a video to YouTube (private), set thumbnail and captions, return video ID.

    Consumes ~1700 quota units. Raises RuntimeError if quota would be exceeded.
    """
    from googleapiclient.http import MediaFileUpload

    _check_quota(1700)

    service = _get_youtube_service()
    mp4_path = Path(mp4_path)
    thumb_path = Path(thumb_path)
    srt_path = Path(srt_path)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": _CATEGORY_IDS.get(category.lower(), "22"),
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,
        },
    }

    media = MediaFileUpload(str(mp4_path), mimetype="video/mp4", resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Starting YouTube upload: {mp4_path.name}")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Upload progress: {pct}%")

    yt_video_id: str = response["id"]
    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Upload complete — video ID: {yt_video_id}")

    # Consume quota immediately after upload succeeds (before optional steps that may fail)
    _consume_quota(1700)

    # Set thumbnail
    try:
        thumb_media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg", resumable=False)
        service.thumbnails().set(videoId=yt_video_id, media_body=thumb_media).execute()
        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Thumbnail set for {yt_video_id}")
    except Exception as exc:
        print(f"[WARNING] Thumbnail upload failed for {yt_video_id}: {exc} — continuing")

    # Upload captions
    try:
        caption_body = {
            "snippet": {
                "videoId": yt_video_id,
                "language": "en",
                "name": title[:150],
                "isDraft": False,
            }
        }
        caption_media = MediaFileUpload(str(srt_path), mimetype="application/octet-stream", resumable=False)
        service.captions().insert(
            part="snippet", body=caption_body, media_body=caption_media
        ).execute()
        print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Captions uploaded for {yt_video_id}")
    except Exception as exc:
        print(f"[WARNING] Caption upload failed for {yt_video_id}: {exc} — continuing")

    return yt_video_id

def check_and_publish(yt_video_id: str) -> dict:
    """Check processing status and publish the video if processing has succeeded.

    Returns a dict with keys: status, published (bool), and optionally error.
    """
    try:
        service = _get_youtube_service()
        list_response = service.videos().list(
            part="status,processingDetails", id=yt_video_id
        ).execute()

        items = list_response.get("items", [])
        if not items:
            return {"status": "not_found", "published": False}

        item = items[0]
        processing = item.get("processingDetails", {})
        processing_status = processing.get("processingStatus", "")

        if processing_status == "terminated":
            print(f"[WARNING] Video {yt_video_id} processing terminated (YouTube failed to process it).")
            return {"status": "failed", "published": False}

        if processing_status != "succeeded":
            print(
                f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
                f"Video {yt_video_id} still processing: {processing_status}"
            )
            return {"status": "processing", "published": False}

        update_body = {
            "id": yt_video_id,
            "status": {"privacyStatus": "public"},
        }
        service.videos().update(part="status", body=update_body).execute()
        print(
            f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
            f"Video {yt_video_id} published successfully."
        )
        return {"status": "published", "published": True}

    except Exception as e:
        print(f"[ERROR]   check_and_publish failed for {yt_video_id}: {e}")
        return {"status": "error", "published": False, "error": str(e)}
