import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src import interfaces, query_parser
from src import search as s
from src.query_parser import ANCHORS, ParsedQuery

FRI_19 = "fri 19:00"
OPEN_LATE = {"Friday": "17:0-2:0"}


def _frame():
    """Six restaurants around City Hall with 3-dim embeddings."""
    hall_lat, hall_lon = ANCHORS["City Hall"]
    rows = [
        # id, name, km north of City Hall, stars, reviews, price, categories, attributes, hours
        ("a", "Thai Palace", 0.1, 4.5, 100, 1, ["Thai", "Restaurants"], {}, {"Friday": "11:0-22:0"}),
        ("b", "Vegan Tacos", 0.5, 4.0, 400, 2, ["Vegan", "Tacos"], {}, OPEN_LATE),
        ("c", "Steak Room", 1.0, 4.5, 900, 4, ["Steakhouses"], {}, {"Friday": "17:0-21:0"}),
        ("d", "Corner Cafe", 1.5, 3.5, 50, None, ["Cafes"], {}, None),
        ("e", "Far Pizza", 8.0, 5.0, 1000, 1, ["Pizza"], {}, {"Friday": "0:0-0:0"}),
        ("f", "GF Bakery", 0.2, 4.0, 80, 2, ["Bakeries"],
         {"DietaryRestrictions": "{'gluten-free': True, 'vegan': False}"}, None),
    ]
    frame = pd.DataFrame(
        {
            "business_id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "latitude": [hall_lat + r[2] / 111.0 for r in rows],
            "longitude": [hall_lon] * len(rows),
            "stars": np.array([r[3] for r in rows], dtype=np.float32),
            "review_count": [r[4] for r in rows],
            "price_level": [r[5] for r in rows],
            "categories": [r[6] for r in rows],
            "attributes": [r[7] for r in rows],
            "hours": [r[8] for r in rows],
        }
    )
    emb = np.eye(6, 3, dtype=np.float32)  # a=x, b=y, c=z, d/e/f = 0
    emb[3:] = [[0.6, 0.8, 0.0], [0.0, 0.6, 0.8], [0.8, 0.0, 0.6]]
    return frame, emb


DATA = _frame()


def ids(parsed, **kw):
    return [r["business_id"] for r in s.search_numpy(parsed, data=DATA, **kw)]


@pytest.fixture
def fake_embed(monkeypatch):
    monkeypatch.setattr(s, "embed", lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32))


# ---------- numpy filters ----------


def test_no_constraints_ranks_by_stars_times_log_reviews():
    assert ids(ParsedQuery()) == ["e", "c", "b", "a", "f", "d"]


def test_price_filter_excludes_pricier_and_null_by_default():
    assert set(ids(ParsedQuery(max_price_level=2))) == {"a", "b", "e", "f"}
    assert "d" in ids(ParsedQuery(max_price_level=2), include_null_price=True)


def test_near_uses_given_distance_or_default():
    assert set(ids(ParsedQuery(near="City Hall", max_distance_km=0.6))) == {"a", "b", "f"}
    results = s.search_numpy(ParsedQuery(near="City Hall"), data=DATA)
    assert {r["business_id"] for r in results} == {"a", "b", "c", "d", "f"}  # default 2 km drops e
    assert all(r["distance_km"] <= s.DEFAULT_NEAR_KM for r in results)


def test_distance_without_anchor_is_ignored():
    assert len(ids(ParsedQuery(max_distance_km=0.1))) == 6


def test_dietary_matches_category_or_attribute_and_all_tags_must_hold():
    assert ids(ParsedQuery(dietary=["vegan"])) == ["b"]
    assert ids(ParsedQuery(dietary=["gluten_free"])) == ["f"]
    assert ids(ParsedQuery(dietary=["vegan", "gluten_free"])) == []


def test_categories_match_any_by_substring_of_category_or_name():
    assert set(ids(ParsedQuery(categories=["thai", "steakhouse"]))) == {"a", "c"}
    assert ids(ParsedQuery(categories=["taco"])) == ["b"]
    assert ids(ParsedQuery(categories=["korean bbq"])) == []


def test_soft_ranks_by_similarity(fake_embed):
    results = s.search_numpy(ParsedQuery(soft="anything"), data=DATA, k=3)
    assert [r["business_id"] for r in results] == ["a", "f", "d"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_open_at_filters_on_hours():
    assert set(ids(ParsedQuery(open_at=FRI_19))) == {"a", "b", "c", "e"}
    assert set(ids(ParsedQuery(open_at="sat 01:00"))) == {"b"}  # Friday's late hours run past midnight


def test_k_limits_results():
    assert len(ids(ParsedQuery(), k=2)) == 2


# ---------- hours ----------


@pytest.mark.parametrize(
    "hours, when, expected",
    [
        ({"Friday": "11:0-22:0"}, "fri 21:59", True),
        ({"Friday": "11:0-22:0"}, "fri 22:00", False),
        (OPEN_LATE, "fri 23:30", True),
        (OPEN_LATE, "sat 01:59", True),
        (OPEN_LATE, "sat 02:00", False),
        ({"Monday": "0:0-0:0"}, "mon 03:00", True),
        ({"Monday": "9:0-17:0"}, "tue 10:00", False),
        (None, "fri 19:00", False),
    ],
)
def test_is_open(hours, when, expected):
    assert s.is_open(hours, *s.resolve_open_at(when)) is expected


def test_resolve_open_at_now_uses_local_time():
    assert s.resolve_open_at("now", datetime(2026, 10, 16, 19, 5)) == (4, 19 * 60 + 5)  # a Friday


# ---------- SQL builder ----------


def test_build_query_is_parameterized():
    parsed = ParsedQuery(max_price_level=2, near="Drexel", dietary=["halal"], categories=["50%_off"])
    sql, params = s.build_query(parsed, k=10)
    assert "50%_off" not in sql and "Halal" not in sql and "Drexel" not in sql
    assert sql.count("%s") == len(params)
    assert "%50\\%\\_off%" in params
    assert params[-1] == 10


def test_build_query_overfetches_for_open_at_and_needs_vector_for_soft():
    _, params = s.build_query(ParsedQuery(open_at="now"), k=10)
    assert params[-1] == 10 * s.OPEN_AT_OVERFETCH
    with pytest.raises(ValueError):
        s.build_query(ParsedQuery(soft="cozy"), k=10)


# ---------- SQL vs numpy on real data ----------

EVAL_IDS = [2, 3, 5, 6, 7, 9, 15, 16, 21, 24]


def _db_conn():
    if not interfaces.RESTAURANTS.exists() or not interfaces.EMBEDDINGS.exists():
        return None
    try:
        from src.db import get_conn

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM restaurants")
            if cur.fetchone()[0] == 0:
                return None
        return conn
    except Exception:
        return None


@pytest.fixture(scope="module")
def conn():
    connection = _db_conn()
    if connection is None:
        pytest.skip("database not reachable or not loaded")
    yield connection
    connection.close()


def _eval_queries():
    rows = [json.loads(line) for line in query_parser.QUERIES_PATH.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["id"] in EVAL_IDS]


@pytest.mark.parametrize("row", _eval_queries(), ids=lambda r: f"q{r['id']}")
def test_sql_and_numpy_return_the_same_ids(conn, row):
    parsed = ParsedQuery.from_dict(row["expected"])
    sql_ids = [r["business_id"] for r in s.search(conn, parsed)]
    np_ids = [r["business_id"] for r in s.search_numpy(parsed)]
    assert sql_ids == np_ids


def test_cli_prints_parsed_sql_and_results(conn, monkeypatch, capsys):
    monkeypatch.setattr(query_parser, "parse", lambda text: ParsedQuery(max_price_level=1, categories=["pizza"]))
    s.main(["cheap pizza", "--k", "3"])
    out = capsys.readouterr().out
    assert '"max_price_level": 1' in out
    assert "price_level <= %s" in out
    assert "3 results:" in out


# ---------- _as_dict (shared with src.db) ----------


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"Friday": "11:0-22:0", "Monday": None}, {"Friday": "11:0-22:0"}),
        ('{"Friday": "11:0-22:0", "Monday": null}', {"Friday": "11:0-22:0"}),
        (None, None),
        (float("nan"), None),
        ("", None),
        ("  ", None),
        ("null", None),
        ({"Monday": None}, None),
    ],
)
def test_as_dict(value, expected):
    from src.db import _as_dict

    assert _as_dict(value, "biz1") == expected


@pytest.mark.parametrize("value", ['{"Friday": "11:0-22:0"', "[1, 2]"])
def test_as_dict_rejects_bad_json_naming_business(value):
    from src.db import _as_dict

    with pytest.raises(ValueError, match="business_id biz1"):
        _as_dict(value, "biz1")


def test_load_frame_parses_json_string_hours(monkeypatch):
    restaurants = pd.DataFrame(
        {
            "business_id": ["x", "y"],
            "name": ["X", "Y"],
            "latitude": [39.95, 39.96],
            "longitude": [-75.16, -75.17],
            "stars": [4.0, 3.5],
            "review_count": [30, 40],
            "categories": ["Thai, Restaurants", None],
            "attributes": [{"RestaurantsPriceRange2": "2"}, None],
            "hours": ['{"Friday": "17:0-2:0"}', None],
        }
    )
    emb = np.eye(2, 3, dtype=np.float32)
    monkeypatch.setattr(s, "_frame", None)
    monkeypatch.setattr(s.pd, "read_parquet", lambda path: restaurants)
    monkeypatch.setattr(s.interfaces, "load_embeddings", lambda: (emb, ["y", "x"]))
    frame, frame_emb = s.load_frame()
    assert list(frame["hours"]) == [OPEN_LATE, None]
    assert frame["price_level"].iat[0] == 2 and pd.isna(frame["price_level"].iat[1])
    assert frame_emb.tolist() == [emb[1].tolist(), emb[0].tolist()]


def test_db_rows_load_is_open_and_json_hours(monkeypatch):
    from src import db

    restaurants = pd.DataFrame(
        {
            "business_id": ["w", "x", "y", "z"],
            "name": ["W", "X", "Y", "Z"],
            "latitude": [39.95] * 4,
            "longitude": [-75.16] * 4,
            "stars": [4.0] * 4,
            "review_count": [30] * 4,
            "categories": ["Thai"] * 4,
            "attributes": [{"RestaurantsPriceRange2": "3"}, None, None, None],
            "hours": ['{"Friday": "17:0-2:0"}', None, "", None],
            "is_open": pd.array([True, False, None, pd.NA], dtype="boolean"),
        }
    )
    emb = np.eye(4, 3, dtype=np.float32)
    monkeypatch.setattr(db.pd, "read_parquet", lambda path: restaurants)
    monkeypatch.setattr(db.interfaces, "load_embeddings", lambda: (emb, ["w", "x", "y", "z"]))
    rows = [dict(zip(db.COLUMNS, row)) for row in db._rows()]
    assert [r["is_open"] for r in rows] == [True, False, None, None]
    assert rows[0]["hours"].obj == OPEN_LATE and rows[0]["price_level"] == 3
    assert [r["hours"] for r in rows[1:]] == [None, None, None]


# ---------- encoder injection, include_closed ----------


class _Encoder:
    def __init__(self, vec):
        self.vec, self.calls = vec, []

    def encode(self, text, normalize_embeddings=True):
        self.calls.append((text, normalize_embeddings))
        return self.vec


def test_injected_encoder_is_used_instead_of_the_module_model(monkeypatch):
    monkeypatch.setattr(s, "embed", lambda text: pytest.fail("module model should not be used"))
    encoder = _Encoder([0.0, 0.0, 1.0])
    results = s.search_numpy(ParsedQuery(soft="steak"), data=DATA, k=1, encoder=encoder)
    assert [r["business_id"] for r in results] == ["c"]
    assert encoder.calls == [("steak", True)]


def test_results_carry_categories_and_coordinates():
    (top,) = s.search_numpy(ParsedQuery(categories=["thai"]), data=DATA)
    assert top["categories"] == ["Thai", "Restaurants"]
    assert top["latitude"] == pytest.approx(DATA[0]["latitude"].iat[0])
    assert top["longitude"] == pytest.approx(DATA[0]["longitude"].iat[0])


def test_include_closed_false_drops_only_is_open_false():
    frame, emb = DATA
    frame = frame.assign(is_open=pd.array([False, None, True, True, True, True], dtype="boolean"))
    open_ids = [r["business_id"] for r in s.search_numpy(ParsedQuery(), data=(frame, emb), include_closed=False)]
    all_ids = [r["business_id"] for r in s.search_numpy(ParsedQuery(), data=(frame, emb))]
    assert set(all_ids) - set(open_ids) == {"a"}


def test_build_query_include_closed():
    sql, _ = s.build_query(ParsedQuery(), 10, include_closed=False)
    assert "is_open IS NOT FALSE" in sql
    assert "is_open" not in s.build_query(ParsedQuery(), 10)[0]
