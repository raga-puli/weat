"""HTTP helpers shared by the Streamlit pages (shared and frozen)."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 30


class APIError(RuntimeError):
    """The API is unreachable or returned an error."""


def _request(method: str, path: str, **kwargs):
    url = f"{API_URL}/{path.lstrip('/')}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT_S, **kwargs)
    except requests.ConnectionError as e:
        raise APIError(
            f"Can't reach the API at {API_URL}. Is it running? Start it with: uvicorn src.api.main:app"
        ) from e
    except requests.Timeout as e:
        raise APIError(f"The API at {API_URL} didn't answer {method} {path} within {TIMEOUT_S}s.") from e
    if not response.ok:
        raise APIError(f"{method} {path} failed with HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def get(path: str, params: dict | None = None):
    return _request("GET", path, params=params)


def post(path: str, json: dict | None = None):
    return _request("POST", path, json=json)
