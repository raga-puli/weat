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


@pytest.fixture(autouse=True)
def default_engine_llm(monkeypatch):
    """Requests without `engine` use DEFAULT_ENGINE; pin it so the developer's .env doesn't matter."""
    monkeypatch.setenv("DEFAULT_ENGINE", "llm")


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


def test_numpy_backend_without_data_files_is_400(client):
    def files_missing():
        raise FileNotFoundError(2, "No such file or directory", "data/processed/restaurants.parquet")

    app.dependency_overrides[rs.search_data_provider] = lambda: files_missing
    response = client.post("/search", json={"text": QUERY, "backend": "numpy"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "data/processed/restaurants.parquet" in detail and "backend=sql" in detail


@pytest.mark.parametrize("default", ["rules", "llm"])
def test_engine_defaults_to_DEFAULT_ENGINE(sql_on_numpy, monkeypatch, default):
    monkeypatch.setenv("DEFAULT_ENGINE", default)
    assert sql_on_numpy.post("/search", json={"text": QUERY}).json()["engine_used"] == default
    assert sql_on_numpy.post("/search/parse", json={"text": QUERY}).json()["engine_used"] == default
    # an explicit engine still wins
    assert sql_on_numpy.post("/search", json={"text": QUERY, "engine": "llm"}).json()["engine_used"] == "llm"


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


@pytest.fixture
def no_real_encoder(monkeypatch):
    """Warm-up and get_encoder use FakeEncoder; the warming flag starts and ends cleared."""
    calls = []
    fake = rs.lru_cache(maxsize=1)(lambda: calls.append("encoder") or FakeEncoder())
    monkeypatch.setattr(rs, "_load_encoder", fake)
    rs._encoder_warming.clear()
    yield calls
    rs._encoder_warming.clear()


def test_warm_up_loads_encoder_and_db_and_preloads_llm(monkeypatch, no_real_encoder):
    calls = no_real_encoder
    monkeypatch.setattr(rs, "get_db_conn", lambda: calls.append("db"))
    monkeypatch.setattr(rs, "preload_llm", lambda: calls.append("llm"))
    rs._warm_up()
    assert set(calls) == {"encoder", "db", "llm"}
    assert not rs._encoder_warming.is_set()


def test_warm_up_survives_db_down(monkeypatch, no_real_encoder):
    monkeypatch.setattr(rs, "get_db_conn", db_down)
    monkeypatch.setattr(rs, "preload_llm", lambda: None)
    rs._warm_up()  # logs, doesn't raise


def test_warm_up_survives_encoder_failure_and_search_retries_the_load(monkeypatch, no_real_encoder):
    def broken():
        raise OSError("model download failed")

    monkeypatch.setattr(rs, "_load_encoder", broken)
    monkeypatch.setattr(rs, "get_db_conn", lambda: None)
    monkeypatch.setattr(rs, "preload_llm", lambda: None)
    rs._warm_up()  # logs, doesn't raise
    assert not rs._encoder_warming.is_set()  # /search tries again instead of 503-ing forever


def test_startup_does_not_wait_for_the_warm_up(monkeypatch, no_real_encoder):
    import threading

    release = threading.Event()
    monkeypatch.setattr(rs, "_warm_up", lambda: release.wait(10))
    try:
        with TestClient(app) as c:  # startup returns while the warm-up is still blocked
            assert c.get("/health").status_code == 200
            assert rs._warm_up_thread.is_alive()
    finally:
        release.set()
        rs._warm_up_thread.join(5)


def test_search_is_503_while_the_encoder_warms_up_and_parse_still_works(client, no_real_encoder):
    app.dependency_overrides.pop(rs.get_encoder)
    rs._encoder_warming.set()
    response = client.post("/search", json={"text": QUERY, "backend": "numpy"})
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(rs.ENCODER_RETRY_AFTER_S)
    assert "starting up" in response.json()["detail"]
    assert client.post("/search/parse", json={"text": QUERY}).status_code == 200

    rs._encoder_warming.clear()  # warm-up finished
    assert client.post("/search", json={"text": QUERY, "backend": "numpy"}).status_code == 200
    assert no_real_encoder == ["encoder"]


def test_lifespan_runs_the_warm_up(monkeypatch, no_real_encoder):
    warmed = []
    monkeypatch.setattr(rs, "_warm_up", lambda: warmed.append(True))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
    rs._warm_up_thread.join(5)
    assert warmed == [True]


# ---------- manual overrides ("Edit filters") ----------


def test_parsed_override_skips_the_parser(sql_on_numpy):
    app.dependency_overrides[rs.get_parser] = lambda: lambda text, engine: pytest.fail("parser should not run")
    response = sql_on_numpy.post(
        "/search", json={"text": QUERY, "parsed": {"max_price_level": 4, "near": "Penn campus", "dietary": ["vegetarian"]}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["engine_used"] == "manual"
    assert body["parsed"]["max_price_level"] == 4 and body["parsed"]["soft"] == ""
    assert ids(response) == ["cozy", "loud", "pricey"]  # price 4 lets "pricey" in; no soft -> popularity order


@pytest.mark.parametrize(
    "parsed",
    [{"near": "Narnia"}, {"dietary": ["paleo"]}, {"open_at": "friday night"}, {"max_price_level": 7}, {"bogus": 1}],
)
def test_bad_parsed_override_is_422(sql_on_numpy, parsed):
    assert sql_on_numpy.post("/search", json={"text": QUERY, "parsed": parsed}).status_code == 422


# ---------- Search page (src/pages/2_Search.py), via Streamlit's AppTest ----------

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

from src import ui_common  # noqa: E402

PAGE = str(Path(__file__).resolve().parents[1] / "src" / "pages" / "2_Search.py")
ANCHOR_LIST = [{"name": n, "lat": lat, "lon": lon} for n, (lat, lon) in ANCHORS.items()]
RESULT = {
    "business_id": "cozy", "name": "Cozy Greens", "categories": ["Vegetarian", "Cafes"], "price_level": 1,
    "lat": PENN[0], "lon": PENN[1], "distance_km": 0.2, "similarity": 0.9,
}


def response(engine_used="llm", results=(RESULT,), **parsed):
    full = {"max_price_level": None, "near": None, "max_distance_km": None, "dietary": [],
            "open_at": None, "categories": [], "soft": ""}
    return {"parsed": {**full, **parsed}, "engine_used": engine_used, "results": list(results),
            "timings_ms": {"parse": 12.0, "search": 34.0, "total": 46.0}}


@pytest.fixture
def page(monkeypatch):
    """The page with ui_common faked; set fake.reply (dict or APIError) and read fake.posts."""

    class Fake:
        reply = response(max_price_level=1, near="Penn campus", dietary=["vegetarian"], soft="cozy")
        posts: list = []

    def post(path, json=None):
        Fake.posts.append((path, json))
        if isinstance(Fake.reply, Exception):
            raise Fake.reply
        return Fake.reply

    Fake.posts = []
    monkeypatch.setattr(ui_common, "post", post)
    monkeypatch.setattr(ui_common, "get", lambda path, params=None: ANCHOR_LIST)
    at = AppTest.from_file(PAGE, default_timeout=30)
    at.fake = Fake
    return at.run()


def badges(at):
    """st.badge renders as markdown ':<color>-badge[:material/icon: label]'."""
    return [m.value for m in at.markdown if "-badge[" in m.value]


def text_box(at, label_start):
    return next(w for w in at.text_input if w.label.startswith(label_start))


def search(at, text):
    at.text_input(key="query").input(text)
    at.button[0].click()  # the form's Search button
    return at.run()


def test_page_shows_examples_and_no_results_yet(page):
    assert not page.exception
    labels = [b.label for b in page.button]
    assert len([lbl for lbl in labels if lbl != "Search"]) == 4
    assert page.fake.posts == []


def test_example_click_searches_and_renders_chips_table_timings(page):
    page.button[1].click().run()
    assert not page.exception
    (path, payload), = page.fake.posts
    assert path == "/search"
    assert payload == {"text": page.button[1].label, "engine": "rules", "backend": "sql", "k": 10,
                       "include_closed": False}
    text = " ".join(str(e.value) for e in page.markdown) + " ".join(str(e.value) for e in page.caption)
    assert "parse 12 ms · search 34 ms · total 46 ms" in text
    df = page.dataframe[0].value
    assert list(df.columns) == ["Name", "Categories", "Price", "Distance (km)", "Similarity"]
    assert df.iloc[0].tolist()[:3] == ["Cozy Greens", "Vegetarian, Cafes", "$"]


def test_chips_split_filters_and_vibe(page):
    page = search(page, "cheap vegetarian near Penn campus, somewhere cozy")
    bodies = badges(page)
    assert any(b.startswith(":blue-badge[") and "filter · price ≤ $]" in b for b in bodies)
    assert any(b.startswith(":blue-badge[") and "filter · near Penn campus]" in b for b in bodies)
    assert any(b.startswith(":blue-badge[") and "filter · vegetarian]" in b for b in bodies)
    assert any(b.startswith(":violet-badge[") and "vibe match · cozy]" in b for b in bodies)
    assert any("engine · llm]" in b for b in bodies)


def test_fallback_to_rules_warns(page):
    page.sidebar.radio(key="engine").set_value("llm").run()
    page.fake.reply = response(engine_used="rules", soft="cozy")
    page = search(page, "somewhere cozy")
    assert page.fake.posts[-1][1]["engine"] == "llm"
    assert any("rules** parser ran instead" in w.value for w in page.warning)


def test_empty_results_message(page):
    page.fake.reply = response(results=(), max_price_level=1)
    page = search(page, "cheap")
    assert [i.value for i in page.info] == ["No restaurants match all filters — try loosening price or distance."]
    assert not page.dataframe


@pytest.mark.parametrize(
    "error, expected",
    [
        (ui_common.APIError("POST /search failed with HTTP 503: {\"detail\": \"Database unavailable\"}", status=503),
         "numpy"),
        (ui_common.APIError("Can't reach the API at http://localhost:8000. Is it running? Start it with: "
                            "uvicorn src.api.main:app"), "uvicorn src.api.main:app"),
        (ui_common.APIError("The API at http://localhost:8000 didn't answer POST /search within 30s."), "rules"),
    ],
)
def test_api_errors_are_friendly(page, error, expected):
    page.fake.reply = error
    page = search(page, "anything")
    assert not page.exception
    assert any(expected in e.value for e in [*page.error, *page.warning])
    assert not page.dataframe


def test_backend_and_k_come_from_the_sidebar(page):
    page.sidebar.radio(key="backend").set_value("numpy")
    page.sidebar.slider(key="k").set_value(25)
    page = search(page, "pizza")
    assert page.fake.posts[-1][1]["backend"] == "numpy" and page.fake.posts[-1][1]["k"] == 25


def test_edit_filters_reruns_with_parsed_override(page):
    page = search(page, "cheap vegetarian near Penn campus, somewhere cozy")
    next(w for w in page.selectbox if w.label == "Max price").set_value("$$$")
    text_box(page, "Categories").input("vegan, Thai")
    page.fake.reply = response(engine_used="manual", max_price_level=3)
    rerun = [b for b in page.button if b.label == "Rerun with these filters"][0]
    rerun.click().run()
    assert not page.exception
    (_, payload) = page.fake.posts[-1]
    assert payload["parsed"] == {
        "max_price_level": 3, "near": "Penn campus", "max_distance_km": None, "dietary": ["vegetarian"],
        "open_at": None, "categories": ["vegan", "thai"], "soft": "cozy",
    }
    assert any("engine · manual" in b for b in badges(page))


def test_edit_filters_can_clear_price_and_place(page):
    page = search(page, "cheap vegetarian near Penn campus, somewhere cozy")
    next(w for w in page.selectbox if w.label == "Max price").set_value("Any")
    next(w for w in page.selectbox if w.label == "Near").set_value("Anywhere")
    [b for b in page.button if b.label == "Rerun with these filters"][0].click().run()
    parsed = page.fake.posts[-1][1]["parsed"]
    assert parsed["max_price_level"] is None and parsed["near"] is None


def test_failed_manual_rerun_keeps_the_previous_results(page):
    page = search(page, "cheap vegetarian near Penn campus, somewhere cozy")
    text_box(page, "Open at").input("friday night")  # invalid; the API answers 422
    page.fake.reply = ui_common.APIError("POST /search failed with HTTP 422: invalid parsed query: open_at must be ...",
                                         status=422)
    [b for b in page.button if b.label == "Rerun with these filters"][0].click().run()
    assert any("rejected" in w.value for w in page.warning)
    assert page.dataframe  # still showing the last good results


# ---------- warm-up 503 left over after ui_common's retries ----------


def test_page_says_still_starting_up_when_ui_common_gives_up(page):
    page.fake.reply = ui_common.APIError(f"POST /search failed with HTTP 503: {rs.ENCODER_WARMING}",
                                         status=503, retry_after=rs.ENCODER_RETRY_AFTER_S)
    at = search(page, QUERY)
    assert "still starting up" in at.warning[0].value and not at.error
