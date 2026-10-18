import numpy as np
import pandas as pd
import psycopg
import pytest
from fastapi.testclient import TestClient

from src import query_parser
from src import search as s
from src.api import routes_search as rs
from src.api.main import app
from src.query_parser import ANCHORS, ParsedQuery

PENN = ANCHORS["Penn campus"]


def _frame():
    """Four restaurants near Penn campus with 2-dim embeddings; "closed" has is_open = False."""
    rows = [
        # id, name, km north of Penn, price, categories, is_open, embedding
        ("cozy", "Cozy Greens", 0.2, 1, ["Vegetarian", "Cafes"], True, [1.0, 0.0]),
        ("loud", "Loud Salads", 0.4, 1, ["Vegetarian"], None, [0.6, 0.8]),
        ("closed", "Closed Veg", 0.1, 1, ["Vegetarian"], False, [1.0, 0.0]),
        ("pricey", "Fancy Veg", 0.3, 4, ["Vegetarian"], True, [1.0, 0.0]),
    ]
    frame = pd.DataFrame(
        {
            "business_id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "latitude": [PENN[0] + r[2] / 111.0 for r in rows],
            "longitude": [PENN[1]] * len(rows),
            "stars": np.array([4.0] * len(rows), dtype=np.float32),
            "review_count": [100] * len(rows),
            "price_level": [r[3] for r in rows],
            "categories": [r[4] for r in rows],
            "attributes": [{}] * len(rows),
            "hours": [None] * len(rows),
            "is_open": pd.array([r[5] for r in rows], dtype="boolean"),
        }
    )
    return frame, np.array([r[6] for r in rows], dtype=np.float32)


DATA = _frame()
QUERY = "cheap vegetarian near Penn campus, somewhere cozy"
PARSED = ParsedQuery(max_price_level=1, near="Penn campus", dietary=["vegetarian"], soft="cozy")


class FakeEncoder:
    def encode(self, text, normalize_embeddings=True):
        return np.array([1.0, 0.0], dtype=np.float32)


def fake_parser(text, engine):
    return PARSED, engine


def db_down():
    raise psycopg.OperationalError("connection refused")


@pytest.fixture
def client():
    app.dependency_overrides.update(
        {
            rs.get_parser: lambda: fake_parser,
            rs.get_encoder: FakeEncoder,
            rs.search_data_provider: lambda: lambda: DATA,
            rs.db_conn_provider: lambda: db_down,
        }
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sql_on_numpy(monkeypatch, client):
    """backend=sql with search.search running the numpy path on DATA, and a DB that is "up"."""
    monkeypatch.setattr(rs.search, "search", lambda conn, parsed, **kw: s.search_numpy(parsed, data=DATA, **kw))
    app.dependency_overrides[rs.db_conn_provider] = lambda: lambda: object()
    return client


def ids(response):
    return [r["business_id"] for r in response.json()["results"]]


# ---------- POST /search ----------


def test_search_returns_parsed_results_and_timings(sql_on_numpy):
    response = sql_on_numpy.post("/search", json={"text": QUERY})
    assert response.status_code == 200
    body = response.json()
    assert body["engine_used"] == "llm"
    assert body["parsed"] == {
        "max_price_level": 1, "near": "Penn campus", "max_distance_km": None,
        "dietary": ["vegetarian"], "open_at": None, "categories": [], "soft": "cozy",
    }
    assert ids(response) == ["cozy", "loud"]  # closed excluded by default, pricey over budget
    top = body["results"][0]
    assert set(top) == {"business_id", "name", "categories", "price_level", "lat", "lon", "distance_km", "similarity"}
    assert top["categories"] == ["Vegetarian", "Cafes"] and top["price_level"] == 1
    assert top["lat"] == pytest.approx(PENN[0] + 0.2 / 111.0) and top["lon"] == pytest.approx(PENN[1])
    assert top["distance_km"] == pytest.approx(0.2, abs=0.01)
    assert top["similarity"] == pytest.approx(1.0)
    assert set(body["timings_ms"]) == {"parse", "search", "total"}
    assert body["timings_ms"]["total"] >= body["timings_ms"]["parse"] >= 0


def test_numpy_backend_matches_and_never_touches_the_db(client):
    # db_conn_provider still raises here, so a 200 means numpy didn't try to connect
    response = client.post("/search", json={"text": QUERY, "backend": "numpy", "k": 1})
    assert response.status_code == 200
    assert ids(response) == ["cozy"]


def test_include_closed(sql_on_numpy):
    response = sql_on_numpy.post("/search", json={"text": QUERY, "include_closed": True})
    assert ids(response) == ["closed", "cozy", "loud"]  # ties on similarity break by business_id


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 301},
        {},
        {"text": QUERY, "engine": "gpt"},
        {"text": QUERY, "backend": "sqlite"},
        {"text": QUERY, "k": 0},
        {"text": QUERY, "k": 51},
        {"text": QUERY, "engin": "rules"},
    ],
)
def test_invalid_requests_are_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_engine_used_reflects_fallback(sql_on_numpy):
    app.dependency_overrides[rs.get_parser] = lambda: lambda text, engine: (PARSED, "rules")
    response = sql_on_numpy.post("/search", json={"text": QUERY, "engine": "llm"})
    assert response.status_code == 200
    assert response.json()["engine_used"] == "rules"


def test_real_parser_falls_back_to_rules_when_ollama_is_down(sql_on_numpy, monkeypatch):
    def ollama_down(*args, **kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(query_parser, "_parse_llm", ollama_down)
    app.dependency_overrides.pop(rs.get_parser)
    body = sql_on_numpy.post("/search", json={"text": QUERY}).json()
    assert body["engine_used"] == "rules"
    assert body["parsed"]["near"] == "Penn campus" and body["parsed"]["dietary"] == ["vegetarian"]


def test_db_down_is_503(client):
    response = client.post("/search", json={"text": QUERY})
    assert response.status_code == 503
    assert "Database unavailable" in response.json()["detail"]


def test_db_dropping_mid_request_is_503_and_reconnects_next_time(client, monkeypatch):
    def dropped(conn, parsed, **kw):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    cleared = []
    monkeypatch.setattr(rs.search, "search", dropped)
    monkeypatch.setattr(rs.get_db_conn, "cache_clear", lambda: cleared.append(True))
    app.dependency_overrides[rs.db_conn_provider] = lambda: lambda: object()
    assert client.post("/search", json={"text": QUERY}).status_code == 503
    assert cleared == [True]


# ---------- POST /search/parse, GET /search/anchors ----------


def test_parse_only(client):
    response = client.post("/search/parse", json={"text": QUERY, "engine": "rules"})
    assert response.status_code == 200
    assert response.json() == {
        "parsed": {
            "max_price_level": 1, "near": "Penn campus", "max_distance_km": None,
            "dietary": ["vegetarian"], "open_at": None, "categories": [], "soft": "cozy",
        },
        "engine_used": "rules",
    }


def test_parse_only_rejects_empty_text(client):
    assert client.post("/search/parse", json={"text": ""}).status_code == 422


def test_anchors(client):
    response = client.get("/search/anchors")
    assert response.status_code == 200
    assert {a["name"]: (a["lat"], a["lon"]) for a in response.json()} == ANCHORS


# ---------- startup warm-up ----------


def test_warm_up_loads_encoder_and_db_and_preloads_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "get_encoder", lambda: calls.append("encoder") or FakeEncoder())
    monkeypatch.setattr(rs, "get_db_conn", lambda: calls.append("db"))
    monkeypatch.setattr(rs, "preload_llm", lambda: calls.append("llm"))
    rs._warm_up()
    assert set(calls) == {"encoder", "db", "llm"}


def test_warm_up_survives_db_down(monkeypatch):
    monkeypatch.setattr(rs, "get_encoder", FakeEncoder)
    monkeypatch.setattr(rs, "get_db_conn", db_down)
    monkeypatch.setattr(rs, "preload_llm", lambda: None)
    rs._warm_up()  # logs, doesn't raise


def test_lifespan_is_registered_on_the_app(monkeypatch):
    warmed = []
    monkeypatch.setattr(rs, "_warm_up", lambda: warmed.append(True))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
    assert warmed == [True]
