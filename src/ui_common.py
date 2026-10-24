"""HTTP helpers shared by the Streamlit pages (shared and frozen).

Two kinds of waiting are handled here, so every page gets them:

- A sleeping host (free tiers stop idle apps). Before the first request in a while to a non-local
  API, GET /health is tried with short timeouts; while it fails or is slow, the page shows
  WAKE_MESSAGE and it is retried for up to WAKE_BUDGET_S. Requests themselves are never resent
  because of a slow or failed connection, so a POST runs at most once per attempt below.
- A 503 with Retry-After (e.g. /search while the API's encoder warms up): the request is retried
  up to MAX_RETRY_AFTER_RETRIES times, waiting as long as the API asks, showing a "starting up"
  message. A 503 without Retry-After (e.g. the database is down) is raised at once.

Messages go to `notify` (a function taking one string); by default, when running under Streamlit,
they appear in a placeholder at the point of the call and are cleared afterwards.
"""

import os
import time
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 30

WAKE_MESSAGE = "Waking up the server, about a minute…"
WAKE_BUDGET_S = 90
WAKE_PING_TIMEOUT_S = (3, 5)  # (connect, read) for each /health try while waking
WAKE_PAUSE_S = 3  # between /health tries
AWAKE_FOR_S = 600  # after a successful response, skip the wake check for this long
MAX_RETRY_AFTER_RETRIES = 6
MAX_RETRY_AFTER_S = 60  # cap on one Retry-After wait
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

# Patched in tests.
_sleep = time.sleep
_clock = time.monotonic
_last_ok: float | None = None  # _clock() of the last successful response from API_URL


class APIError(RuntimeError):
    """The API is unreachable or returned an error. status/retry_after are set for HTTP errors."""

    def __init__(self, message: str, status: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _streamlit_notifier():
    """(notify, clear) showing messages in a Streamlit placeholder; no-ops outside Streamlit."""
    try:
        import streamlit as st
        from streamlit.runtime import exists

        if exists():
            slot = {}

            def notify(message: str) -> None:
                if "box" not in slot:
                    slot["box"] = st.empty()
                slot["box"].info(message, icon=":material/hourglass_top:")

            return notify, lambda: slot["box"].empty() if "box" in slot else None
    except Exception:
        pass
    return (lambda message: None), (lambda: None)


def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


def _retry_after(response) -> int | None:
    value = response.headers.get("Retry-After")
    try:
        return max(0, min(int(value), MAX_RETRY_AFTER_S)) if value is not None else None
    except ValueError:
        return None


def _ensure_awake(notify) -> None:
    """Wait for a sleeping (non-local) host to answer /health, showing WAKE_MESSAGE meanwhile."""
    global _last_ok
    if _is_local(API_URL) or (_last_ok is not None and _clock() - _last_ok < AWAKE_FOR_S):
        return
    deadline = _clock() + WAKE_BUDGET_S
    while True:
        try:
            if requests.get(f"{API_URL}/health", timeout=WAKE_PING_TIMEOUT_S).ok:
                _last_ok = _clock()
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        if _clock() >= deadline:
            raise APIError(f"The API at {API_URL} didn't wake up within {WAKE_BUDGET_S}s. Try again in a minute.")
        notify(WAKE_MESSAGE)
        _sleep(WAKE_PAUSE_S)


def _request(method: str, path: str, notify=None, **kwargs):
    global _last_ok
    url = f"{API_URL}/{path.lstrip('/')}"
    clear = lambda: None  # noqa: E731
    if notify is None:
        notify, clear = _streamlit_notifier()
    try:
        _ensure_awake(notify)
        for attempt in range(MAX_RETRY_AFTER_RETRIES + 1):
            try:
                response = requests.request(method, url, timeout=TIMEOUT_S, **kwargs)
            except requests.ConnectionError as e:
                raise APIError(
                    f"Can't reach the API at {API_URL}. Is it running? Start it with: uvicorn src.api.main:app"
                ) from e
            except requests.Timeout as e:
                raise APIError(f"The API at {API_URL} didn't answer {method} {path} within {TIMEOUT_S}s.") from e
            _last_ok = _clock()  # it answered, so it's awake
            retry_after = _retry_after(response) if response.status_code == 503 else None
            if retry_after is None or attempt == MAX_RETRY_AFTER_RETRIES:
                break
            what = "Search" if path.lstrip("/").startswith("search") else "The server"
            notify(f"{what} is starting up, retrying...")
            _sleep(retry_after)
        if not response.ok:
            raise APIError(
                f"{method} {path} failed with HTTP {response.status_code}: {response.text[:300]}",
                status=response.status_code,
                retry_after=_retry_after(response),
            )
        return response.json()
    finally:
        clear()


def get(path: str, params: dict | None = None, notify=None):
    return _request("GET", path, notify=notify, params=params)


def post(path: str, json: dict | None = None, notify=None):
    return _request("POST", path, notify=notify, json=json)
