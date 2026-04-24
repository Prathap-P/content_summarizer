"""One-time CLI OAuth setup for the YouTube Data API v3.

Run this script once to authorise the app and save credentials to token.json.
Subsequent runs (and all upload calls) reuse or silently refresh the token.

Usage:
    python youtube_auth.py
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

SCOPES: list[str] = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

_PROJECT_ROOT = Path(__file__).parent
_CLIENT_SECRETS = _PROJECT_ROOT / "client_secrets.json"
_TOKEN_FILE = _PROJECT_ROOT / "token.json"

def _load_existing_credentials() -> Optional[Any]:
    """Return valid credentials from token.json if they exist, else None."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not _TOKEN_FILE.exists():
        return None

    creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
            return creds
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Failed to refresh token: {exc} — re-authorising.")
            return None

    return None

def _save_credentials(creds: Any) -> None:
    """Write credentials to token.json."""
    _TOKEN_FILE.write_text(creds.to_json())

def main() -> None:
    """Run the OAuth flow or confirm an existing valid token."""
    # ------------------------------------------------------------------ #
    # Bail early if credentials are already valid                          #
    # ------------------------------------------------------------------ #
    creds = _load_existing_credentials()
    if creds is not None:
        print(
            f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
            f"token.json is valid — no re-authorisation needed."
        )
        return

    # ------------------------------------------------------------------ #
    # Ensure client_secrets.json is present                               #
    # ------------------------------------------------------------------ #
    if not _CLIENT_SECRETS.exists():
        print(
            "[ERROR]   client_secrets.json not found.\n"
            "          To fix this:\n"
            "            1. Go to https://console.cloud.google.com/apis/credentials\n"
            "            2. Create or select an OAuth 2.0 Client ID (Desktop app).\n"
            "            3. Click 'Download JSON'.\n"
            "            4. Rename the downloaded file to 'client_secrets.json'.\n"
            f"            5. Place it in: {_PROJECT_ROOT}"
        )
        return

    # ------------------------------------------------------------------ #
    # Run the local-server OAuth flow                                      #
    # ------------------------------------------------------------------ #
    from google_auth_oauthlib.flow import InstalledAppFlow

    print(
        f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
        "Starting OAuth flow — a browser window will open."
    )

    flow = InstalledAppFlow.from_client_secrets_file(str(_CLIENT_SECRETS), SCOPES)
    creds = flow.run_local_server(port=0)

    _save_credentials(creds)

    print(
        f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] "
        f"Authorisation complete. Credentials saved to {_TOKEN_FILE}"
    )

if __name__ == "__main__":
    main()
