import json

import pytest

from src import query_parser as qp
from src.query_parser import ParsedQuery

MODEL = "fake-model:1b"

VALID = {
    "max_price_level": 2,
    "near": "Drexel",
    "max_distance_km": None,
    "dietary": ["vegan"],
    "open_at": "now",
    "categories": ["Thai"],
    "soft": "quiet",
}


class FakeClient:
    """Stands in for ollama.Client; returns the queued contents in order."""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.contents.pop(0)}}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(qp, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(qp, "RESULTS_PATH", tmp_path / "results" / "parser.md")
    return tmp_path


def with_(**changes):
    return json.dumps({**VALID, **changes})


# ---------- validation ----------


def test_validate_accepts_valid_output_and_lowercases_categories():
    parsed = qp.validate(json.dumps(VALID))
    assert parsed == ParsedQuery(2, "Drexel", None, ["vegan"], "now", ["thai"], "quiet")


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[]",
        with_(max_price_level=5),
        with_(max_price_level="2"),
        with_(max_price_level=True),
        with_(near="Kensington"),
        with_(dietary=["dairy_free"]),
        with_(dietary="vegan"),
        with_(open_at="friday at 7"),
        with_(max_distance_km=-1),
        with_(max_distance_km="1.2"),
        with_(soft=None),
        json.dumps({k: v for k, v in VALID.items() if k != "soft"}),
        with_(extra="field"),
    ],
)
def test_validate_rejects_off_spec_output(bad):
    with pytest.raises(ValueError):
        qp.validate(bad)


def test_schema_covers_every_field_and_constrains_enums():
    schema = qp.json_schema()
    assert set(schema["required"]) == set(VALID)
    assert schema["properties"]["near"]["anyOf"][0]["enum"] == list(qp.ANCHORS)
    assert schema["properties"]["dietary"]["items"]["enum"] == list(qp.DIETARY)
    assert schema["properties"]["max_price_level"]["anyOf"][0]["enum"] == [1, 2, 3, 4]


def test_every_eval_expectation_is_valid():
    for line in qp.QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        ParsedQuery.from_dict(json.loads(line)["expected"])


def test_system_prompt_contains_no_eval_queries():
    # One- and two-word queries ("food", "open now") are field vocabulary, not examples.
    prompt = qp.SYSTEM_PROMPT.lower()
    for line in qp.QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if len(row["query"].split()) >= 3:
            assert row["query"].lower() not in prompt
        soft = row["expected"]["soft"]
        if len(soft.split()) >= 3:
            assert soft not in prompt


# ---------- parse: call, retry, fallback, cache ----------


def test_parse_calls_with_schema_format_and_temperature_zero():
    client = FakeClient(json.dumps(VALID))
    qp.parse("anything", client=client, model=MODEL)
    (call,) = client.calls
    assert call["model"] == MODEL
    assert call["format"] == qp.json_schema()
    assert call["options"]["temperature"] == 0
    assert call["keep_alive"] == qp.LLM_KEEP_ALIVE
    assert call["messages"][0] == {"role": "system", "content": qp.SYSTEM_PROMPT}
    assert call["messages"][-1] == {"role": "user", "content": "anything"}


def test_parse_retries_once_with_error_feedback():
    client = FakeClient(with_(near="Kensington"), json.dumps(VALID))
    parsed = qp.parse("thai near drexel", client=client, model=MODEL)
    assert parsed.near == "Drexel"
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "invalid" in retry_messages[-1]["content"]


def test_parse_falls_back_to_soft_after_two_failures_and_does_not_cache():
    client = FakeClient("not json", with_(max_price_level=9))
    parsed = qp.parse("some query", client=client, model=MODEL)
    assert parsed == ParsedQuery(soft="some query")
    assert len(client.calls) == 2
    assert not qp._cache_path(MODEL, "some query").exists()


def test_parse_caches_on_disk_and_skips_the_client_on_a_hit():
    first = FakeClient(json.dumps(VALID))
    parsed = qp.parse("thai near drexel", client=first, model=MODEL)
    assert qp._cache_path(MODEL, "thai near drexel").exists()

    second = FakeClient()  # would raise if called
    assert qp.parse("thai near drexel", client=second, model=MODEL) == parsed
    assert second.calls == []


def test_cache_key_depends_on_model_prompt_version_and_text(monkeypatch):
    base = qp._cache_path(MODEL, "x")
    assert qp._cache_path("other-model", "x") != base
    assert qp._cache_path(MODEL, "y") != base
    monkeypatch.setattr(qp, "PROMPT_VERSION", "v999")
    assert qp._cache_path(MODEL, "x") != base


# ---------- eval scoring ----------


def test_field_matches_compares_lists_as_sets_and_distance_within_10_percent():
    assert qp.field_matches("categories", ["thai", "sushi"], ["Sushi", "thai"])
    assert not qp.field_matches("dietary", ["vegan"], [])
    assert qp.field_matches("max_distance_km", 2.0, 2.19)
    assert not qp.field_matches("max_distance_km", 2.0, 2.3)
    assert not qp.field_matches("max_distance_km", 2.0, None)
    assert qp.field_matches("max_distance_km", None, None)
    assert qp.field_matches("soft", "not too loud, good", "Not too loud good")


def test_evaluate_and_log(tmp_path):
    queries = tmp_path / "queries.jsonl"
    hit = {**VALID, "categories": ["thai"]}
    rows = [
        {"id": 1, "tag": "t", "query": "hit", "expected": hit},
        {"id": 2, "tag": "t", "query": "miss", "expected": {**hit, "near": "Temple"}},
    ]
    queries.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = qp.evaluate(queries, client=FakeClient(json.dumps(VALID), json.dumps(VALID)), model=MODEL)
    assert result["n"] == 2
    assert result["exact_match"] == 0.5
    assert result["per_field"]["near"] == 0.5
    assert result["per_field"]["categories"] == 1.0
    assert [f["wrong"] for f in result["failures"]] == [["near"]]

    qp._append_result(result, "smoke")
    lines = qp.RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("| date | model | prompt_version | exact_match")
    assert MODEL in lines[2] and "smoke" in lines[2]


# ---------- rules engine ----------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("cheap vegetarian near Penn campus, somewhere cozy",
         dict(max_price_level=1, near="Penn campus", dietary=["vegetarian"], soft="cozy")),
        ("affordable sushi by UPenn", dict(max_price_level=2, near="Penn campus", categories=["sushi"])),
        ("$$$ steak near South Philly", dict(max_price_level=3, near="Italian Market", categories=["steak"])),
        ("tacos under $30 a person", dict(max_price_level=2, categories=["tacos"])),
        ("dinner for $70 per person", dict(max_price_level=4, soft="dinner for")),
        ("within 1 mile of Fishtown, halal and kosher",
         dict(near="Fishtown", max_distance_km=1.609, dietary=["halal", "kosher"])),
        ("walking distance from Rittenhouse, quiet", dict(near="Rittenhouse Square", max_distance_km=1.2, soft="quiet")),
        ("gluten-free pho open saturday at 9:30pm", dict(dietary=["gluten_free"], open_at="sat 21:30", categories=["pho"])),
        ("vegan dim sum open right now", dict(dietary=["vegan"], open_at="now", categories=["dim sum"])),
        ("fancy but not too loud", dict(soft="fancy but not too loud")),
        ("pennsylvania dutch food", dict(soft="pennsylvania dutch")),
        ("", dict()),
    ],
)
def test_parse_rules(text, expected):
    assert qp.parse_rules(text) == ParsedQuery(**expected)


def test_parse_rules_output_always_validates():
    for text in ["$$$$$", "open fri at 25pm", "near near near", "under $5 under $100", "Ünïcode café"]:
        qp.validate(json.dumps(qp.asdict(qp.parse_rules(text))))


def test_parse_with_engine_rules_never_calls_the_llm():
    client = FakeClient()
    assert qp.parse_with_engine("vegan near Drexel", "rules", client=client, model=MODEL) == (
        ParsedQuery(near="Drexel", dietary=["vegan"]),
        "rules",
    )
    assert client.calls == []


def test_parse_with_engine_llm_success():
    parsed, engine_used = qp.parse_with_engine("anything", "llm", client=FakeClient(json.dumps(VALID)), model=MODEL)
    assert engine_used == "llm" and parsed.near == "Drexel"


def test_parse_with_engine_falls_back_to_rules_after_one_invalid_output():
    client = FakeClient("not json", json.dumps(VALID))
    assert qp.parse_with_engine("vegan near Drexel", "llm", client=client, model=MODEL) == (
        ParsedQuery(near="Drexel", dietary=["vegan"]),
        "rules",
    )
    assert len(client.calls) == 1  # no retry on the latency-bounded path


def test_parse_with_engine_bounds_the_llm_call_and_falls_back_on_timeout(monkeypatch):
    import httpx

    class SlowClient:
        def chat(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    timeouts = []
    monkeypatch.setattr(qp, "_client", lambda timeout=None: timeouts.append(timeout) or SlowClient())
    parsed, engine_used = qp.parse_with_engine("vegan near Drexel", "llm", model=MODEL)
    assert timeouts == [qp.LLM_TIMEOUT_S] and qp.LLM_TIMEOUT_S < 30  # below ui_common.TIMEOUT_S
    assert (parsed, engine_used) == (ParsedQuery(near="Drexel", dietary=["vegan"]), "rules")


@pytest.mark.parametrize("timeout", [qp.LLM_TIMEOUT_S, None])
def test_client_caps_the_connect_timeout_at_two_seconds(timeout):
    http = qp._client(timeout=timeout)._client  # the underlying httpx.Client
    assert http.timeout.connect == qp.LLM_CONNECT_TIMEOUT_S == 2.0
    assert http.timeout.read == timeout


def test_unreachable_ollama_falls_back_to_rules_within_the_connect_timeout(monkeypatch):
    import time

    # 10.255.255.1 is non-routable: the TCP connect hangs until the connect timeout fires
    monkeypatch.setenv("OLLAMA_HOST", "http://10.255.255.1:11434")
    start = time.perf_counter()
    parsed, engine_used = qp.parse_with_engine("vegan near Drexel", "llm", model=MODEL)
    assert engine_used == "rules" and parsed == ParsedQuery(near="Drexel", dietary=["vegan"])
    assert time.perf_counter() - start < qp.LLM_CONNECT_TIMEOUT_S + 3


@pytest.fixture
def no_dotenv(monkeypatch):
    """default_engine() reads .env; keep the developer's .env out of these tests."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    "value, expected", [("rules", "rules"), ("llm", "llm"), (" RULES ", "rules"), ("gpt", "llm"), (None, "llm")]
)
def test_default_engine_reads_the_environment(no_dotenv, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("DEFAULT_ENGINE", raising=False)
    else:
        monkeypatch.setenv("DEFAULT_ENGINE", value)
    assert qp.default_engine() == expected


def test_parse_with_engine_none_uses_default_engine(no_dotenv, monkeypatch):
    monkeypatch.setenv("DEFAULT_ENGINE", "rules")
    client = FakeClient(json.dumps(VALID))
    assert qp.parse_with_engine("vegan near Drexel", client=client, model=MODEL) == (
        ParsedQuery(near="Drexel", dietary=["vegan"]), "rules")
    assert client.calls == []


def test_parse_with_engine_falls_back_to_rules_when_ollama_is_unreachable():
    class DownClient:
        def chat(self, **kwargs):
            raise ConnectionError("Failed to connect to Ollama")

    parsed, engine_used = qp.parse_with_engine("halal near Temple", "llm", client=DownClient(), model=MODEL)
    assert (parsed, engine_used) == (ParsedQuery(near="Temple", dietary=["halal"]), "rules")


def test_parse_with_engine_rejects_unknown_engine():
    with pytest.raises(ValueError, match="engine"):
        qp.parse_with_engine("x", "gpt")


@pytest.mark.parametrize(
    "soft, cleaned",
    [
        ("near Temple", ""),
        ("food under $15", ""),
        ("cozy, under $40", "cozy"),
        ("within 2 km of Old City", ""),
        ("romantic for two, under $50 per person", "romantic for two"),
        ("near me", "near me"),
        ("in kensington", "in kensington"),
        ("not too spicy", "not too spicy"),
    ],
)
def test_clean_soft_drops_captured_phrases(soft, cleaned):
    assert qp.clean_soft(ParsedQuery(near="Drexel", soft=soft)) == ParsedQuery(near="Drexel", soft=cleaned)


def test_llm_soft_is_cleaned_but_cached_raw():
    client = FakeClient(with_(soft="quiet, near Drexel"))
    assert qp.parse("quiet thai near drexel", client=client, model=MODEL).soft == "quiet"
    cached = json.loads(qp._cache_path(MODEL, "quiet thai near drexel").read_text(encoding="utf-8"))
    assert cached["soft"] == "quiet, near Drexel"
    assert qp.parse("quiet thai near drexel", client=FakeClient(), model=MODEL).soft == "quiet"  # cache hit


def test_preload_llm_loads_model_and_never_raises(monkeypatch):
    calls = []

    class Client:
        def generate(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(qp, "_client", lambda timeout=None: Client())
    monkeypatch.setattr(qp, "_model", lambda: MODEL)
    qp.preload_llm()
    assert calls == [{"model": MODEL, "prompt": "", "keep_alive": qp.LLM_KEEP_ALIVE}]

    monkeypatch.setattr(qp, "_client", lambda timeout=None: (_ for _ in ()).throw(ConnectionError("down")))
    qp.preload_llm()  # logs, doesn't raise
