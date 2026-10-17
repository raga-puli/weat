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
