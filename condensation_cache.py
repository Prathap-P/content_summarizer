"""
Pipeline-wide resume / checkpoint cache for the condensation pipeline.

Each processing run is identified by a 16-char hex key derived from the
canonical URL identifier and the active model key.  Progress is saved to
  condensation_cache/<key>.json
using atomic writes (write to .tmp then os.replace) so a crash never
leaves a corrupt checkpoint file.

Checkpoint stages (in order):
  0  raw_content       — transcript / article text after fetch / Whisper
  1  map_chunks        — content split stored so resume uses identical chunks
  2  map_results[i]    — MAP output per chunk (string keys)
  3  reduce_results[i] — REDUCE output per batch (string keys)
  4  consolidation_result — optional final consolidation output
  5  final_output      — fully condensed text, condensation pipeline done
  6  audio_file_path   — Kokoro TTS output path
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qsl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path("condensation_cache")
CHECKPOINT_TTL_HOURS = 24
MAX_RETRIES_PER_STEP = 3

# Query-string keys that are tracking noise and must be stripped before hashing
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "ref", "source", "campaign")


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------

def _strip_tracking_params(query: str) -> str:
    """Remove known tracking query parameters and sort survivors."""
    pairs = parse_qsl(query, keep_blank_values=True)
    cleaned = sorted(
        (k, v) for k, v in pairs
        if not any(k.lower().startswith(prefix) for prefix in _TRACKING_PREFIXES)
    )
    return urlencode(cleaned)


def _canonicalize_url(url: str, mode: str) -> str:
    """Return a stable canonical identifier for the URL.

    YouTube: extract video_id (all URL variants collapse to the same ID).
    News   : strip tracking params, sort survivors, lowercase scheme+host+path.
    """
    if mode == "youtube":
        # Import here to avoid circular imports; only needed for YouTube mode.
        from youtube_transcript_fetcher import extract_video_id
        video_id = extract_video_id(url.strip())
        if video_id:
            return f"yt:{video_id}"
        # Fallback: treat as opaque URL
        return f"yt_raw:{url.strip().lower()}"

    # News path
    parsed = urlparse(url.strip())
    canonical_query = _strip_tracking_params(parsed.query)
    canonical = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path,
        query=canonical_query,
        fragment="",  # fragments are client-side, not part of the resource
    )
    return canonical.geturl()


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def compute_cache_key(
    url: str,
    mode: str,
    model_key: str,
    fetch_mode: str = "transcript",
) -> str:
    """Derive a stable 16-char hex key from (canonical_url, model_key, fetch_mode).

    Identical video URLs in all variants map to the same key.
    Different models never share a key.
    fetch_mode ('transcript' | 'audio') is included so a Whisper-forced audio run
    never reuses a cached transcript-API run for the same video.
    """
    canonical_id = _canonicalize_url(url, mode)
    raw = f"{canonical_id}|{model_key}|{fetch_mode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=CHECKPOINT_TTL_HOURS)).isoformat()


def _is_expired(data: dict) -> bool:
    try:
        expires_at = datetime.fromisoformat(data["expires_at"])
        return datetime.now(timezone.utc) >= expires_at
    except (KeyError, ValueError):
        return True  # malformed → treat as expired


def _fresh_checkpoint(url: str, mode: str, model_key: str, fetch_mode: str = "transcript") -> dict:
    """Return a blank checkpoint dict with all fields initialized."""
    return {
        "url": url,
        "mode": mode,
        "model_key": model_key,
        "fetch_mode": fetch_mode,
        "created_at": _now_iso(),
        "expires_at": _expires_iso(),
        # Stage 0 — raw content
        "raw_content": None,
        "source": None,          # "whisper" | "transcript_api" | "news_loader"
        # Stage 1 — split chunks (list of strings)
        "map_chunks": None,
        # Stage 2 — MAP results (str-keyed dict: {"0": "...", "1": "..."})
        "map_results": {},
        "map_retry_counts": {},
        # Stage 3 — REDUCE results (str-keyed dict)
        "reduce_batches_total": None,
        "reduce_results": {},
        "reduce_retry_counts": {},
        # Stage 4 — consolidation
        "consolidation_result": None,
        "consolidation_retries": 0,
        # Stage 5 — final condensation output
        "final_output": None,
        # Stage 6 — TTS
        "audio_file_path": None,
    }


# ---------------------------------------------------------------------------
# Atomic I/O
# ---------------------------------------------------------------------------

def _checkpoint_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _tmp_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.tmp"


def save_checkpoint(key: str, data: dict) -> None:
    """Atomically persist checkpoint data to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(key)
    final = _checkpoint_path(key)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, final)
    except Exception as e:
        print(f"[ERROR]   save_checkpoint: failed to write {final}: {e}")
        # Best-effort cleanup of tmp file
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def load_checkpoint(key: str) -> Optional[dict]:
    """Load a checkpoint.  Returns None if missing or expired."""
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if _is_expired(data):
            print(f"[INFO]    Checkpoint {key} has expired — ignoring")
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARNING] Could not load checkpoint {key}: {e} — ignoring")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_checkpoint(
    url: str,
    mode: str,
    model_key: str,
    fetch_mode: str = "transcript",
) -> tuple[str, dict]:
    """Create a fresh in-memory checkpoint; does NOT write to disk.

    Writing is deferred to the first successful stage so we never leave
    empty files for aborted runs.

    Args:
        fetch_mode: ``'transcript'`` or ``'audio'``.  Must match the value
                    used when the cache key was computed so resume works
                    correctly across retries.

    Returns:
        (key, data)
    """
    key = compute_cache_key(url, mode, model_key, fetch_mode)
    data = _fresh_checkpoint(url, mode, model_key, fetch_mode)
    print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] New checkpoint created: key={key}")
    return key, data


def get_progress_summary(data: dict) -> dict:
    """Return a human-readable progress dict suitable for error responses."""
    map_total = len(data.get("map_chunks") or [])
    map_done = len(data.get("map_results", {}))
    reduce_total = data.get("reduce_batches_total") or 0
    reduce_done = len(data.get("reduce_results", {}))
    return {
        "raw_content_cached": data.get("raw_content") is not None,
        "map_chunks_total": map_total,
        "map_chunks_done": map_done,
        "reduce_batches_total": reduce_total,
        "reduce_batches_done": reduce_done,
        "consolidation_done": data.get("consolidation_result") is not None,
        "final_output_cached": data.get("final_output") is not None,
        "audio_cached": data.get("audio_file_path") is not None,
    }


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def purge_expired_checkpoints() -> int:
    """Remove expired .json checkpoints and orphaned .tmp files.

    Returns the number of files removed.
    """
    if not CACHE_DIR.exists():
        return 0

    removed = 0
    now = datetime.now(timezone.utc)

    for path in CACHE_DIR.iterdir():
        try:
            if path.suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if _is_expired(data):
                        path.unlink()
                        print(f"[PURGE]   Removed expired checkpoint: {path.name}")
                        removed += 1
                except (json.JSONDecodeError, OSError):
                    # Corrupt file — remove it
                    path.unlink(missing_ok=True)
                    print(f"[PURGE]   Removed corrupt checkpoint: {path.name}")
                    removed += 1

            elif path.suffix == ".tmp":
                # Orphaned tmp file older than 1 hour
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if (now - mtime).total_seconds() > 3600:
                    path.unlink(missing_ok=True)
                    print(f"[PURGE]   Removed orphaned tmp file: {path.name}")
                    removed += 1
        except Exception as e:
            print(f"[WARNING] purge_expired_checkpoints: error processing {path.name}: {e}")

    if removed:
        print(f"[PURGE]   Total files removed: {removed}")
    return removed
