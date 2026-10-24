import pytest
import requests

from src import ui_common

REMOTE = "https://weat-api.example.com"


class FakeClock:
    """ui_common's clock and sleep: sleeping advances the clock, nothing really waits."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Response:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body if body is not None else {"ok": True}
        self.headers = headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


@pytest.fixture
def api(monkeypatch):
    """Scripted /health replies (api.health) and request replies (api.replies); records everything."""
    clock = FakeClock()
    monkeypatch.setattr(ui_common, "_clock", clock.clock)
    monkeypatch.setattr(ui_common, "_sleep", clock.sleep)
    monkeypatch.setattr(ui_common, "_last_ok", None)
    monkeypatch.setattr(ui_common, "API_URL", REMOTE)

    class Script:
        health: list = []
        replies: list = []
        calls: list = []
        messages: list = []

    def next_reply(queue, call):
        Script.calls.append(call)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        clock.now += getattr(reply, "takes", 0)
        return reply

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: next_reply(Script.health, ("health", url)))
    monkeypatch.setattr(requests, "request",
                        lambda method, url, timeout=None, **kw: next_reply(Script.replies, (method, url)))
    Script.clock = clock
    Script.health, Script.replies, Script.calls, Script.messages = [], [], [], []
    Script.notify = Script.messages.append
    return Script


# ---------- sleeping host ----------


def test_awake_host_needs_one_health_ping_then_no_more(api):
    api.health = [Response()]
    api.replies = [Response(body={"a": 1}), Response(body={"b": 2})]
    assert ui_common.post("/search", json={}, notify=api.notify) == {"a": 1}
    assert ui_common.get("/group/users/sample", notify=api.notify) == {"b": 2}
    assert [c[0] for c in api.calls] == ["health", "POST", "GET"]
    assert api.messages == [] and api.clock.sleeps == []


def test_sleeping_host_shows_waking_message_and_retries_health_not_the_request(api):
    api.health = [requests.ConnectTimeout("slow"), requests.ConnectionError("502 from proxy"),
                  requests.ReadTimeout("still booting"), Response()]
    api.replies = [Response(body={"ok": 1})]
    assert ui_common.post("/search", json={"text": "x"}, notify=api.notify) == {"ok": 1}
    assert [c[0] for c in api.calls] == ["health"] * 4 + ["POST"]  # the POST ran once
    assert api.messages == [ui_common.WAKE_MESSAGE] * 3
    assert "about a minute" in ui_common.WAKE_MESSAGE


def test_host_that_never_wakes_gives_up_after_the_budget(api):
    api.health = [requests.ConnectionError("down")] * 100
    with pytest.raises(ui_common.APIError, match="didn't wake up"):
        ui_common.get("/health", notify=api.notify)
    assert sum(api.clock.sleeps) <= ui_common.WAKE_BUDGET_S + ui_common.WAKE_PAUSE_S
    assert not [c for c in api.calls if c[0] != "health"]


def test_wake_check_runs_again_after_the_server_may_have_slept(api):
    api.health = [Response(), Response()]
    api.replies = [Response(), Response()]
    ui_common.get("/x", notify=api.notify)
    api.clock.now += ui_common.AWAKE_FOR_S + 1
    ui_common.get("/x", notify=api.notify)
    assert [c[0] for c in api.calls] == ["health", "GET", "health", "GET"]


def test_local_api_skips_the_wake_check_and_fails_fast(api, monkeypatch):
    monkeypatch.setattr(ui_common, "API_URL", "http://localhost:8000")
    api.replies = [requests.ConnectionError("refused")]
    with pytest.raises(ui_common.APIError, match="Can't reach the API"):
        ui_common.get("/search/anchors", notify=api.notify)
    assert api.calls == [("GET", "http://localhost:8000/search/anchors")] and api.messages == []


# ---------- 503 with Retry-After ----------


def test_503_with_retry_after_is_retried_honoring_the_header(api):
    api.health = [Response()]
    api.replies = [Response(503, {"detail": "warming"}, {"Retry-After": "10"}),
                   Response(503, {"detail": "warming"}, {"Retry-After": "4"}),
                   Response(body={"results": []})]
    assert ui_common.post("/search", json={}, notify=api.notify) == {"results": []}
    assert api.clock.sleeps == [10, 4]
    assert api.messages == ["Search is starting up, retrying..."] * 2


def test_503_retries_stop_after_six_and_raise_with_status(api):
    api.health = [Response()]
    api.replies = [Response(503, {"detail": "warming"}, {"Retry-After": "10"})] * 7
    with pytest.raises(ui_common.APIError) as info:
        ui_common.post("/search", json={}, notify=api.notify)
    assert ui_common.MAX_RETRY_AFTER_RETRIES == 6
    assert len([c for c in api.calls if c[0] == "POST"]) == 7 and api.clock.sleeps == [10] * 6
    assert info.value.status == 503 and info.value.retry_after == 10
    assert "HTTP 503" in str(info.value)  # the message format pages already parse


def test_503_without_retry_after_is_raised_at_once(api):
    api.health = [Response()]
    api.replies = [Response(503, {"detail": "Database unavailable."})]
    with pytest.raises(ui_common.APIError) as info:
        ui_common.post("/search", json={}, notify=api.notify)
    assert info.value.status == 503 and info.value.retry_after is None and api.clock.sleeps == []


def test_retry_after_is_capped(api):
    api.health = [Response()]
    api.replies = [Response(503, {}, {"Retry-After": "3600"}), Response()]
    ui_common.get("/x", notify=api.notify)
    assert api.clock.sleeps == [ui_common.MAX_RETRY_AFTER_S]


def test_other_errors_keep_their_messages_and_status(api):
    api.health = [Response()]
    api.replies = [Response(422, {"detail": "bad"}), requests.ReadTimeout("slow LLM")]
    with pytest.raises(ui_common.APIError) as info:
        ui_common.post("/search", json={}, notify=api.notify)
    assert info.value.status == 422 and str(info.value).startswith("POST /search failed with HTTP 422")
    with pytest.raises(ui_common.APIError, match="didn't answer POST /search within 30s"):
        ui_common.post("/search", json={}, notify=api.notify)
    assert [c[0] for c in api.calls] == ["health", "POST", "POST"]  # a timed-out POST is not resent


def test_default_notifier_is_silent_outside_streamlit(api):
    api.health = [requests.ConnectionError("asleep"), Response()]
    api.replies = [Response()]
    assert ui_common.get("/x") == {"ok": True}  # no notify given, no Streamlit runtime: no crash
