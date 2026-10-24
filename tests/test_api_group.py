import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.aggregate import average, least_misery, normalize
from src.api import routes_group as rg
from src.api.main import app
from src.recommend import ContentBased

IDS = [f"b{i}" for i in range(8)]


class FakeRecommender(ContentBased):
    """8 restaurants in 4-d; u1/u2 like the b0-b3 side, u3/u4 the b4-b7 side."""

    def __init__(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(8, 4))
        m[:4, 0] += 3
        m[4:, 1] += 3
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.business_ids = list(IDS)
        self.index = {b: i for i, b in enumerate(IDS)}
        self.name = "fake"
        self.taste = {
            "u1": self.taste_vector_from(["b0", "b1"]),
            "u2": self.taste_vector_from(["b1", "b2"]),
            "u3": self.taste_vector_from(["b4", "b5"]),
            "u4": self.taste_vector_from(["b5", "b6"]),
        }


FAKE_MODEL = FakeRecommender()
FAKE_RESTAURANTS = pd.DataFrame(
    {
        "name": [f"Restaurant {i}" for i in range(8)],
        "categories": ["Pizza"] * 8,
        "price_level": [1, 2, None, 3, 1, 2, 4, None],
        "latitude": np.linspace(39.9, 40.0, 8),
        "longitude": np.linspace(-75.2, -75.1, 8),
        "is_open": [True, True, True, False, True, True, True, True],  # b3 closed
    },
    index=pd.Index(IDS, name="business_id"),
)
FAKE_REVIEWS = {
    "u1": pd.DataFrame({"business_id": ["b0", "b1"], "stars": [5.0, 4.0]}),
    "u2": pd.DataFrame({"business_id": ["b1", "b2"], "stars": [5.0, 5.0]}),
    "u3": pd.DataFrame({"business_id": ["b4", "b5"], "stars": [4.0, 5.0]}),
    "u4": pd.DataFrame({"business_id": ["b5", "b6"], "stars": [5.0, 3.0]}),
}
FAKE_GROUPS = pd.DataFrame(
    {
        "group_id": ["g_same", "g_mixed"],
        "members": [["u1", "u2", "u3"], ["u1", "u3", "u4"]],
        "target_business_id": ["b7", "b7"],
        "split": ["val", "val"],
    }
)


@pytest.fixture
def client():
    app.dependency_overrides = {
        rg.get_recommender: lambda: FAKE_MODEL,
        rg.get_restaurants: lambda: FAKE_RESTAURANTS,
        rg.get_user_reviews: lambda: FAKE_REVIEWS,
        rg.get_val_users: lambda: ("u1", "u2", "u3", "u4"),
        rg.get_val_groups: lambda: FAKE_GROUPS,
    }
    yield TestClient(app)
    app.dependency_overrides = {}


def _post(client, path, **body):
    return client.post(path, json={"members": ["u1", "u3"], **body})


def test_recommend_valid_request(client):
    r = _post(client, "/group/recommend", strategy="average", k=3)
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [x["business_id"] for x in body["restaurants"]]
    assert len(ids) == 3
    # visited (b0, b1, b4, b5) and closed (b3) restaurants are excluded by default
    assert not {"b0", "b1", "b4", "b5", "b3"} & set(ids)
    assert body["n_candidates"] == 3  # b2, b6, b7
    # matches src.aggregate on the same scores and exclusions
    scores = np.stack([FAKE_MODEL.scores_for(FAKE_MODEL.taste[u]) for u in ["u1", "u3"]])
    excluded = np.array([b in {"b0", "b1", "b3", "b4", "b5"} for b in IDS])
    assert ids == [IDS[i] for i in average(scores, excluded, 3, norm="rank")]
    S = normalize(scores, excluded, "rank")
    first = body["restaurants"][0]
    assert set(first["member_scores"]) == {"u1", "u3"}
    assert first["member_scores"]["u1"] == pytest.approx(S[0, FAKE_MODEL.index[first["business_id"]]])
    assert first["score"] == pytest.approx(np.mean(list(first["member_scores"].values())))
    assert {"name", "categories", "price_level", "lat", "lon"} <= set(first)
    prices = {x["business_id"]: x["price_level"] for x in body["restaurants"]}
    assert prices.get("b2", "absent") in (None, "absent")  # missing level -> null, not NaN
    assert all(p is None or (isinstance(p, int) and 1 <= p <= 4) for p in prices.values())
    assert body["cohesion"] == pytest.approx(float(FAKE_MODEL.taste["u1"] @ FAKE_MODEL.taste["u3"]))
    assert {"group_utility_predicted", "worst_member_predicted", "fairness_gap_predicted"} <= set(body["summary"])
    assert "PREDICTED" in body["summary"]["note"]
    assert "history" not in body


def test_include_closed_and_visited_flags(client):
    body = _post(client, "/group/recommend", strategy="average", k=8,
                 include_closed=True, exclude_visited=False).json()
    assert {x["business_id"] for x in body["restaurants"]} == set(IDS)


def test_fairness_repeated_includes_history(client):
    body = _post(client, "/group/recommend", strategy="fairness_repeated", k=3).json()
    assert len(body["history"]) == 3
    assert [h["business_id"] for h in body["history"]] == [x["business_id"] for x in body["restaurants"]]
    assert sum(body["history"][0]["member_weights"].values()) == pytest.approx(1.0)


def test_unknown_user_is_422_naming_it(client):
    r = client.post("/group/recommend", json={"members": ["u1", "ghost", "nobody"]})
    assert r.status_code == 422
    assert "ghost" in r.json()["detail"] and "nobody" in r.json()["detail"]


@pytest.mark.parametrize("members", [["u1"], ["u1", "u2", "u3", "u4", "u5", "u6", "u7"], []])
def test_member_count_out_of_range_is_422(client, members):
    assert client.post("/group/recommend", json={"members": members}).status_code == 422


def test_duplicate_members_is_422(client):
    r = client.post("/group/recommend", json={"members": ["u1", "u1"]})
    assert r.status_code == 422 and "u1" in r.text


def test_unknown_strategy_is_422(client):
    assert _post(client, "/group/recommend", strategy="borda").status_code == 422


def test_compare_overlap(client):
    r = _post(client, "/group/compare", strategies=["average", "least_misery"], k=2,
              include_closed=True, exclude_visited=False)
    assert r.status_code == 200, r.text
    body = r.json()
    lists = [[x["business_id"] for x in res["restaurants"]] for res in body["results"]]
    assert [res["strategy"] for res in body["results"]] == ["average", "least_misery"]
    assert body["overlap"] == len(set(lists[0]) & set(lists[1]))
    # independently: the same two top-2 lists from src.aggregate
    scores = np.stack([FAKE_MODEL.scores_for(FAKE_MODEL.taste[u]) for u in ["u1", "u3"]])
    none = np.zeros(8, dtype=bool)
    expected = [set(IDS[i] for i in f(scores, none, 2, norm="rank")) for f in (average, least_misery)]
    assert body["overlap"] == len(expected[0] & expected[1])
    same = _post(client, "/group/compare", strategies=["average", "average"], k=2).json()
    assert same["overlap"] == 2


def test_compare_needs_exactly_two_strategies(client):
    assert _post(client, "/group/compare", strategies=["average"]).status_code == 422


def test_sample_users_any(client):
    body = client.get("/group/users/sample", params={"n": 3}).json()
    users = [u["user_id"] for u in body["users"]]
    assert len(users) == 3 and set(users) <= {"u1", "u2", "u3", "u4"}
    u2 = next((u for u in body["users"] if u["user_id"] == "u2"), None)
    if u2:
        assert [p["stars"] for p in u2["profile"]] == [5.0, 5.0]
    assert client.get("/group/users/sample", params={"n": 3}).json() == body  # seeded


def test_sample_users_by_cohesion(client):
    high = client.get("/group/users/sample", params={"n": 3, "cohesion": "high"}).json()
    low = client.get("/group/users/sample", params={"n": 3, "cohesion": "low"}).json()
    assert {high["group_id"], low["group_id"]} == {"g_same", "g_mixed"}
    assert high["cohesion"] > low["cohesion"]
    for body in (high, low):
        for u in body["users"]:
            assert 1 <= len(u["profile"]) <= rg.PROFILE_SIZE
            assert all(p["name"].startswith("Restaurant") for p in u["profile"])


def test_sample_users_n_out_of_range(client):
    assert client.get("/group/users/sample", params={"n": 1}).status_code == 422


# --- the API's model getter: taste cache vs. fresh fit -------------------------------

from pathlib import Path  # noqa: E402

from src import recommend  # noqa: E402
from src.recommend import ContentBasedDebiased  # noqa: E402


class FakeDebiased(ContentBasedDebiased):
    def __init__(self):
        self.matrix = FAKE_MODEL.matrix.astype(np.float32)
        self.business_ids = list(IDS)
        self.index = {b: i for i, b in enumerate(IDS)}
        self.name = "fake_debiased"
        self.embeddings_path = Path("fake.npy")
        self.taste = {}
        self.hub = np.zeros(len(IDS), dtype=np.float32)


FAKE_TRAIN = pd.concat(
    [df.assign(user_id=u) for u, df in FAKE_REVIEWS.items()], ignore_index=True
)[["user_id", "business_id", "stars"]]


def test_api_results_identical_from_cache_and_fresh_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(rg, "RECOMMENDER", FakeDebiased)
    monkeypatch.setattr(rg, "_serving_train", lambda: FAKE_TRAIN)
    monkeypatch.setattr(recommend, "cache_path", lambda name: tmp_path / f"taste_cache_{name}.npz")
    monkeypatch.setattr(recommend, "cache_fingerprint", lambda model, user_filter=None: "fp-test")
    fits = []
    monkeypatch.setattr(FakeDebiased, "fit", lambda self, train: (fits.append(1), ContentBasedDebiased.fit(self, train))[1])

    rg.get_recommender.cache_clear()
    try:
        fresh = rg.get_recommender()  # cold: fits and writes the cache
        assert fits == [1] and (tmp_path / "taste_cache_fake_debiased.npz").exists()
        rg.get_recommender.cache_clear()
        cached = rg.get_recommender()  # warm: loads the cache, no fit
        assert fits == [1] and cached is not fresh
    finally:
        rg.get_recommender.cache_clear()

    bodies = []
    for model in (fresh, cached):
        app.dependency_overrides = {
            rg.get_recommender: lambda m=model: m,
            rg.get_restaurants: lambda: FAKE_RESTAURANTS,
            rg.get_user_reviews: lambda: FAKE_REVIEWS,
        }
        try:
            client = TestClient(app)
            bodies.append([
                client.post("/group/recommend", json={"members": ["u1", "u3"], "strategy": s, "k": 3}).json()
                for s in ("average", "least_misery", "fairness_repeated")
            ])
        finally:
            app.dependency_overrides = {}
    assert bodies[0] == bodies[1]
    assert all(b["restaurants"] for b in bodies[0])


def test_user_profiles_endpoint(client):
    body = client.get("/group/users/profiles", params=[("user_ids", "u2"), ("user_ids", "u1")]).json()
    assert [u["user_id"] for u in body["users"]] == ["u2", "u1"]
    assert [p["stars"] for p in body["users"][0]["profile"]] == [5.0, 5.0]
    assert body["users"][1]["profile"][0]["name"] == "Restaurant 0"  # u1's 5-star b0 first
    bad = client.get("/group/users/profiles", params=[("user_ids", "u1"), ("user_ids", "ghost")])
    assert bad.status_code == 422 and "ghost" in bad.json()["detail"]
    assert client.get("/group/users/profiles").status_code == 422


# --- Group Dinner page helpers (pure functions; no Streamlit runtime) ---------------

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "group_dinner_page", Path(__file__).resolve().parents[1] / "src" / "pages" / "1_Group_Dinner.py"
)
page = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(page)  # __name__ != "__main__", so the UI doesn't run


def _r(bid, name, score, scores, price=2, cats="Mexican, Restaurants"):
    return {"business_id": bid, "name": name, "score": score, "member_scores": scores,
            "price_level": price, "categories": cats}


def test_collapse_chains_merges_same_name_keeping_best_rank():
    rows = page.collapse_chains([
        _r("a", "Taco Place", 0.9, {}), _r("b", "QDOBA", 0.8, {}), _r("c", "Pho", 0.7, {}),
        _r("d", "QDOBA", 0.6, {}), _r("e", "QDOBA", 0.5, {}),
    ])
    assert [r["display_name"] for r in rows] == ["Taco Place", "QDOBA (3 locations)", "Pho"]
    assert [r["rank"] for r in rows] == [1, 2, 3]  # original rank of the best location
    assert rows[1]["business_ids"] == ["b", "d", "e"] and rows[1]["score"] == 0.8
    assert page.collapse_chains([]) == []


def test_display_rows_formatting_and_overlap():
    result = {"restaurants": [
        _r("a", "Taco Place", 0.91234, {"u1": 0.95, "u2": 0.871}, price=3),
        _r("b", "QDOBA", 0.8, {"u1": 0.8, "u2": 0.8}, price=None),
        _r("d", "QDOBA", 0.6, {"u1": 0.6, "u2": 0.6}),
    ]}
    labels = page.member_labels(["u1", "u2"])
    assert labels == {"u1": "M1", "u2": "M2"}
    rows = page.display_rows(result, other_ids={"d"}, labels=labels)
    assert rows[0] == {"rank": 1, "api_rank": 1, "name": "Taco Place", "categories": "Mexican", "price": "$$$",
                       "score": 0.912, "M1": 0.95, "M2": 0.871, "in_both": False}
    assert rows[1]["name"] == "QDOBA (2 locations)" and rows[1]["price"] == "–"
    assert rows[1]["in_both"]  # its second location is in the other list


def test_format_price():
    assert [page.format_price(x) for x in (1, 2, 4, None, 0)] == ["$", "$$", "$$$$", "–", "–"]


def test_parse_member_ids():
    assert page.parse_member_ids("u1, u2\nu3") == (["u1", "u2", "u3"], None)
    ids, err = page.parse_member_ids("u1")
    assert ids == ["u1"] and "at least 2" in err
    _, err = page.parse_member_ids(" ".join(f"u{i}" for i in range(7)))
    assert "at most 6" in err
    _, err = page.parse_member_ids("u1 u2 u1")
    assert "repeated: u1" in err


def test_dinner_rows():
    result = {
        "restaurants": [_r("a", "Taco Place", 0.9, {}), _r("b", "Pho", 0.8, {})],
        "history": [
            {"dinner": 1, "business_id": "a", "member_scores": {"u1": 0.9, "u2": 0.5},
             "member_weights": {"u1": 0.5, "u2": 0.5}},
            {"dinner": 2, "business_id": "b", "member_scores": {"u1": 0.4, "u2": 0.95},
             "member_weights": {"u1": 0.01, "u2": 0.99}},
        ],
    }
    rows = page.dinner_rows(result, page.member_labels(["u1", "u2"]))
    assert rows[1] == {"dinner": 2, "pick": "Pho", "M1 score": 0.4, "M1 weight": 0.01,
                       "M2 score": 0.95, "M2 weight": 0.99}


def test_format_api_error():
    ours = 'POST /group/compare failed with HTTP 422: {"detail":"unknown user_ids or users without a taste profile: [\'ghost\']"}'
    assert page.format_api_error(ours) == "unknown user_ids or users without a taste profile: ['ghost']"
    pydantic = ('POST /group/compare failed with HTTP 422: '
                '{"detail":[{"loc":["body","members"],"msg":"List should have at least 2 items"}]}')
    assert page.format_api_error(pydantic) == "members: List should have at least 2 items"
    down = "Can't reach the API at http://localhost:8000. Is it running?"
    assert page.format_api_error(down) == down


def test_short_categories():
    assert page.short_categories("Restaurants, Mexican, Food, Tacos") == "Mexican, Tacos"
    assert page.short_categories("A, B, C, D, Restaurants") == "A, B, C, …"
    assert page.short_categories(None) == "" and page.short_categories("") == ""


def test_cohesion_reading_uses_val_terciles():
    assert (page.COHESION_LOW, page.COHESION_HIGH) == (0.18, 0.32)
    assert page.cohesion_reading(0.05).startswith("Very different tastes")
    assert page.cohesion_reading(0.18).startswith("Partly overlapping")
    assert page.cohesion_reading(0.319).startswith("Partly overlapping")
    assert page.cohesion_reading(0.32).startswith("Similar tastes")


def test_metric_deltas_b_minus_a():
    a = {"group_utility_predicted": 0.86, "worst_member_predicted": 0.76, "fairness_gap_predicted": 0.18}
    b = {"group_utility_predicted": 0.84, "worst_member_predicted": 0.80, "fairness_gap_predicted": 0.09}
    d = page.metric_deltas(a, b)
    assert d == pytest.approx({"group_utility_predicted": -0.02, "worst_member_predicted": 0.04,
                               "fairness_gap_predicted": -0.09})


def test_top_member_rows_collapses_and_prefixes_rank():
    result = {"restaurants": [
        _r("a", "Taco Place", 0.9, {"u1": 0.9, "u2": 0.7}),
        _r("b", "QDOBA", 0.8, {"u1": 0.8, "u2": 0.6}),
        _r("c", "QDOBA", 0.7, {"u1": 0.1, "u2": 0.1}),
        *[_r(f"x{i}", f"Place {i}", 0.5, {"u1": 0.5, "u2": 0.5}) for i in range(5)],
    ]}
    rows = page.top_member_rows(result, page.member_labels(["u1", "u2"]))
    assert len(rows) == 5
    assert rows[0] == {"restaurant": "1. Taco Place", "M1": 0.9, "M2": 0.7}
    assert rows[1]["restaurant"] == "2. QDOBA (2 locations)" and rows[1]["M1"] == 0.8  # best location's scores
    assert rows[2]["restaurant"] == "4. Place 0"


def test_map_points_marks_list_membership_and_skips_missing_coords():
    def loc(bid, lat=39.95, lon=-75.16):
        return {"business_id": bid, "name": bid, "lat": lat, "lon": lon}

    a = {"restaurants": [loc("a"), loc("both"), loc("nocoords", lat=None)]}
    b = {"restaurants": [loc("both"), loc("b")]}
    points = {p["name"]: p for p in page.map_points([a, b])}
    assert set(points) == {"a", "both", "b"}
    assert {n: p["list"] for n, p in points.items()} == {"a": "A", "both": "both", "b": "B"}
    assert points["both"]["color"] == page.MAP_COLORS["both"]
    assert {p["list"] for p in page.map_points([a])} == {"A"}


def test_member_score_bounds():
    res = {"restaurants": [{"member_scores": {"u1": -1.2, "u2": 0.4}}, {"member_scores": {"u1": 2.5}}]}
    assert page.member_score_bounds([res], "rank") == (0.0, 1.0)
    assert page.member_score_bounds([res], "zscore") == (-1.2, 2.5)


@pytest.mark.parametrize("tau", [0.1, 3.0])
def test_tau_reaches_fairness_repeated(client, tau):
    from src.aggregate import fairness_repeated

    body = _post(client, "/group/recommend", strategy="fairness_repeated", k=4, tau=tau,
                 include_closed=True, exclude_visited=False).json()
    assert body["params"]["tau"] == tau
    scores = np.stack([FAKE_MODEL.scores_for(FAKE_MODEL.taste[u]) for u in ["u1", "u3"]])
    _, _, weights = fairness_repeated(scores, np.zeros(8, dtype=bool), 4, tau=tau, return_weights=True)
    got = [[h["member_weights"][u] for u in ["u1", "u3"]] for h in body["history"]]
    np.testing.assert_allclose(got, weights, atol=1e-9)


def test_tau_default_and_validation(client):
    from src.aggregate import TAU

    body = _post(client, "/group/recommend", strategy="fairness_repeated", k=3).json()
    assert body["params"]["tau"] == TAU == 0.1
    assert _post(client, "/group/recommend", strategy="fairness_repeated", tau=0).status_code == 422


def test_member_colors_distinct_and_ordered():
    labels = page.member_labels([f"u{i}" for i in range(6)])
    colors = page.member_colors(labels)
    assert list(colors) == ["M1", "M2", "M3", "M4", "M5", "M6"]
    assert len(set(colors.values())) == 6
    assert colors["M1"] == page.MEMBER_PALETTE[0]
    assert page.hex_to_rgba("#0072B2") == [0, 114, 178, 220]


def test_member_bars_svg_heights_colors_and_labels():
    import base64

    uri = page.member_bars_svg([0.0, 0.5, 1.0], ["#111111", "#222222", "#333333"], height=41)
    assert uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(uri.split(",", 1)[1]).decode()
    assert svg.count("<rect") == 3 and svg.count("<text") == 3
    for color in ("#111111", "#222222", "#333333"):
        assert f'fill="{color}"' in svg
    assert ">0.00<" in svg and ">0.50<" in svg and ">1.00<" in svg
    # honest fixed axis: bar height is proportional to the value (41 px minus an 11 px label band)
    assert 'height="0.0"' in svg and 'height="15.0"' in svg and 'height="30.0"' in svg
    # values outside the axis are clipped, not rescaled
    clipped = base64.b64decode(page.member_bars_svg([1.7], ["#000000"], height=41).split(",", 1)[1]).decode()
    assert 'height="30.0"' in clipped and ">1.70<" in clipped


def test_tau_ui_options():
    assert page.TAU_UI_DEFAULT == 0.3 and page.TAU_UI_DEFAULT in page.TAU_OPTIONS
    assert page.TAU_OPTIONS[0] == 0.05 and page.TAU_OPTIONS[-1] == 3.0
    assert page.TAU_OPTIONS == sorted(page.TAU_OPTIONS) and 0.1 in page.TAU_OPTIONS


def test_map_points_ranks_and_tooltips():
    def loc(bid):
        return {"business_id": bid, "name": f"N-{bid}", "lat": 39.95, "lon": -75.16}

    a = {"restaurants": [loc("a"), loc("both")]}
    b = {"restaurants": [loc("x"), loc("y"), loc("both")]}
    points = {p["name"]: p for p in page.map_points([a, b])}
    assert points["N-both"]["tooltip"] == "N-both (A #2 · B #3)"
    assert points["N-a"]["tooltip"] == "N-a (A #1)" and points["N-y"]["tooltip"] == "N-y (B #2)"
    assert points["N-both"]["rgba"] == page.hex_to_rgba(page.MAP_COLORS["both"])
    single = page.map_points([a])
    assert [p["tooltip"] for p in single] == ["N-a (#1)", "N-both (#2)"]


def test_fit_view_contains_points_and_zooms_out_for_spread():
    tight = [{"lat": 39.95, "lon": -75.17}, {"lat": 39.96, "lon": -75.16}]
    wide = [{"lat": 39.90, "lon": -75.25}, {"lat": 40.05, "lon": -75.05}]
    vt, vw = page.fit_view(tight), page.fit_view(wide)
    assert vw["zoom"] < vt["zoom"]
    assert 39.90 <= vw["latitude"] <= 40.05 and -75.25 <= vw["longitude"] <= -75.05
    assert page.fit_view([{"lat": 39.95, "lon": -75.16}])["zoom"] == 15.0  # single point: capped
    assert page.fit_view([])["zoom"] == 11.0
    # the padded extent fits the map at deck.gl's 512-px zoom convention (wide case)
    assert page.TILE_PX == 512
    lon_px = (0.20 * 1.3) * page.TILE_PX * 2 ** vw["zoom"] / 360
    assert lon_px <= 1000 + 1


def _api_result(names, members=("u1", "u2"), history=False):
    """An API-shaped result whose member scores encode position: u1 = 1 - i/10, u2 = i/10."""
    restaurants = [
        {"business_id": f"id{i}", "name": n, "score": 0.0, "price_level": 1, "categories": "Food",
         "lat": 39.9 + i / 100, "lon": -75.1, "member_scores": {"u1": 1 - i / 10, "u2": i / 10}}
        for i, n in enumerate(names)
    ]
    result = {"strategy": "average", "members": list(members), "restaurants": restaurants,
              "summary": {"note": "PREDICTED", "group_utility_predicted": -1.0,
                          "worst_member_predicted": -1.0, "fairness_gap_predicted": -1.0}}
    if history:
        result["history"] = [{"dinner": i + 1, "business_id": f"id{i}", "member_scores": {"u1": 0.5, "u2": 0.5},
                              "member_weights": {"u1": 0.5, "u2": 0.5}} for i in range(len(names))]
    return result


def test_trim_result_keeps_k_distinct_names_and_renumbers():
    # API order: A, Chain, B, Chain, Chain, C, D  -> distinct: A, Chain(3), B, C, D
    res = page.trim_result(_api_result(["A", "Chain", "B", "Chain", "Chain", "C", "D"]), k=4)
    rows = res["restaurants"]
    assert [r["display_name"] for r in rows] == ["A", "Chain (3 locations)", "B", "C"]
    assert [r["rank"] for r in rows] == [1, 2, 3, 4]
    assert [r["api_rank"] for r in rows] == [1, 2, 3, 6]
    assert rows[1]["business_ids"] == ["id1", "id3", "id4"] and rows[1]["business_id"] == "id1"
    assert res["api_k"] == 7
    # summary recomputed over the kept rows (best locations id0, id1, id2, id5), as the API does
    u1 = (1.0 + 0.9 + 0.8 + 0.5) / 4
    u2 = (0.0 + 0.1 + 0.2 + 0.5) / 4
    s = res["summary"]
    assert s["group_utility_predicted"] == pytest.approx((u1 + u2) / 2)
    assert s["worst_member_predicted"] == pytest.approx(min(u1, u2))
    assert s["fairness_gap_predicted"] == pytest.approx(abs(u1 - u2))
    assert s["note"] == "PREDICTED"
    # display: collapse again is a no-op that keeps chain counts and new ranks
    shown = page.display_rows(res, set(), page.member_labels(["u1", "u2"]))
    assert [(r["rank"], r["api_rank"], r["name"]) for r in shown][:2] == [(1, 1, "A"), (2, 2, "Chain (3 locations)")]


def test_trim_result_fairness_repeated_keeps_the_dinner_sequence():
    raw = _api_result(["A", "Chain", "B", "Chain", "C", "D", "E"], history=True)
    raw["strategy"] = "fairness_repeated"
    res = page.trim_result(raw, k=5)
    rows = res["restaurants"]
    # first k dinners, in order, nothing collapsed or renumbered; repeat chain location labelled
    assert [r["business_id"] for r in rows] == ["id0", "id1", "id2", "id3", "id4"]
    assert [r["rank"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["display_name"] for r in rows] == ["A", "Chain", "B", "Chain (another location)", "C"]
    assert [h["dinner"] for h in res["history"]] == [1, 2, 3, 4, 5]
    # summary over those k dinners
    u1 = sum(1 - i / 10 for i in range(5)) / 5
    u2 = sum(i / 10 for i in range(5)) / 5
    assert res["summary"]["group_utility_predicted"] == pytest.approx((u1 + u2) / 2)
    # display and per-dinner view keep the sequence and the label
    labels = page.member_labels(["u1", "u2"])
    shown = page.display_rows(res, set(), labels)
    assert [r["name"] for r in shown] == ["A", "Chain", "B", "Chain (another location)", "C"]
    assert [r["pick"] for r in page.dinner_rows(res, labels)][3] == "Chain (another location)"
    assert [r["restaurant"] for r in page.top_member_rows(res, labels)][3] == "4. Chain (another location)"


def test_trim_result_when_fewer_distinct_than_k():
    res = page.trim_result(_api_result(["A", "A", "B"]), k=10)
    assert [r["display_name"] for r in res["restaurants"]] == ["A (2 locations)", "B"]


def test_overlap_counts_any_chain_location_in_trimmed_lists():
    a = page.trim_result(_api_result(["A", "Chain", "B"]), k=3)["restaurants"]
    b_raw = _api_result(["X", "Chain", "A"])
    for r, bid in zip(b_raw["restaurants"], ["x", "id9", "id0"]):  # B kept a different Chain location
        r["business_id"] = bid
    b = page.trim_result(b_raw, k=3)["restaurants"]
    a[1]["business_ids"].append("id9")  # A's Chain row also covers location id9
    assert page.overlap_count(a, b) == 2  # A (id0) and Chain (via id9)
    points = {p["name"]: p["list"] for p in page.map_points([{"restaurants": a}, {"restaurants": b}])}
    assert points == {"A": "both", "Chain": "both", "B": "A", "X": "B"}


# --- DATA_SOURCE=db: getters built from database tables (no Postgres needed) -------------

from src import deploy_load  # noqa: E402

RAW = np.random.default_rng(1).normal(size=(8, 6)).astype(np.float32)
RAW /= np.linalg.norm(RAW, axis=1, keepdims=True)
DB_RESTAURANTS = pd.DataFrame({
    "business_id": IDS,
    "name": FAKE_RESTAURANTS["name"].tolist(),
    "latitude": FAKE_RESTAURANTS["latitude"].tolist(),
    "longitude": FAKE_RESTAURANTS["longitude"].tolist(),
    "price_level": [1, 2, None, 3, 1, 2, 4, None],
    "categories": [["Pizza"]] * 8,
    "is_open": FAKE_RESTAURANTS["is_open"].tolist(),
    "embedding": list(RAW),
})


def _served_model():
    """The model files mode would serve: debiased, fitted on the fake train."""
    model = ContentBasedDebiased(matrix=RAW, business_ids=IDS)
    model.fit(FAKE_TRAIN)
    return model


def _db_tables_for(model, users=("u1", "u2", "u3", "u4")):
    sample = FAKE_GROUPS.assign(cohesion=[0.9, 0.1], cohesion_bucket=["high", "low"])
    tables = deploy_load.build_tables(model, FAKE_REVIEWS, list(users), sample)
    tables["restaurants"] = DB_RESTAURANTS
    # what the DB returns: vectors and arrays as plain lists
    tables["user_taste"] = tables["user_taste"].assign(embedding=tables["user_taste"]["embedding"].map(list))
    tables["model_arrays"] = tables["model_arrays"].assign(vals=tables["model_arrays"]["vals"].map(list))
    return tables


@pytest.fixture
def db_mode(monkeypatch):
    model = _served_model()
    tables = _db_tables_for(model)
    monkeypatch.setenv("DATA_SOURCE", "db")
    monkeypatch.setattr(rg, "_db_tables", lambda: tables)
    getters = (rg.get_recommender, rg.get_restaurants, rg.get_user_reviews, rg.get_val_users, rg.get_val_groups)
    for g in getters:
        g.cache_clear()
    app.dependency_overrides = {}
    yield model, tables
    for g in getters:
        g.cache_clear()


def _files_client(model):
    app.dependency_overrides = {
        rg.get_recommender: lambda: model,
        rg.get_restaurants: lambda: FAKE_RESTAURANTS.assign(categories="Pizza"),
        rg.get_user_reviews: lambda: FAKE_REVIEWS,
        rg.get_val_users: lambda: ("u1", "u2", "u3", "u4"),
        rg.get_val_groups: lambda: FAKE_GROUPS.assign(cohesion_bucket=["high", "low"]),
    }
    return TestClient(app)


def test_data_source_env(monkeypatch):
    monkeypatch.delenv("DATA_SOURCE", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert rg.data_source() == "files"
    monkeypatch.setenv("DATA_SOURCE", "DB")
    assert rg.data_source() == "db"
    monkeypatch.setenv("DATA_SOURCE", "s3")
    with pytest.raises(RuntimeError, match="files.*db"):
        rg.data_source()


def test_db_getters_rebuild_the_served_state(db_mode):
    model, tables = db_mode
    served = rg.get_recommender()
    assert served is not model and type(served) is type(model)
    np.testing.assert_array_equal(served.matrix, model.matrix)  # same centering of the same raw vectors
    np.testing.assert_array_equal(served.hub, model.hub)
    assert list(served.taste) == list(model.taste)
    for u in model.taste:
        np.testing.assert_array_equal(served.taste[u], model.taste[u])
    r = rg.get_restaurants()
    assert r.loc["b3", "is_open"] == False and pd.isna(r.loc["b2", "price_level"])  # noqa: E712
    assert r.loc["b0", "categories"] == "Pizza"
    assert rg.get_val_users() == ("u1", "u2", "u3", "u4")
    reviews = rg.get_user_reviews()
    assert set(reviews["u4"]["business_id"]) == {"b5", "b6"}


@pytest.mark.parametrize("path,body", [
    ("/group/recommend", {"members": ["u1", "u3"], "strategy": "average", "k": 4}),
    ("/group/recommend", {"members": ["u1", "u2", "u4"], "strategy": "fairness_repeated", "k": 3, "tau": 0.3}),
    ("/group/compare", {"members": ["u2", "u3"], "strategies": ["average", "least_misery"], "k": 3,
                        "include_closed": True}),
])
def test_db_mode_responses_identical_to_files_mode(db_mode, path, body):
    model, _ = db_mode
    db_resp = TestClient(app).post(path, json=body).json()
    files_resp = _files_client(model).post(path, json=body).json()
    app.dependency_overrides = {}
    assert db_resp == files_resp


def test_db_mode_profiles_and_sample(db_mode):
    model, _ = db_mode
    client = TestClient(app)
    params = [("user_ids", "u2"), ("user_ids", "u4")]
    db_profiles = client.get("/group/users/profiles", params=params).json()
    for cohesion in ("high", "low"):
        db_sample = client.get("/group/users/sample", params={"n": 3, "cohesion": cohesion}).json()
        assert db_sample["group_id"] == {"high": "g_same", "low": "g_mixed"}[cohesion]  # stored buckets
    files = _files_client(model)
    assert db_profiles == files.get("/group/users/profiles", params=params).json()
    app.dependency_overrides = {}
    unknown = client.post("/group/recommend", json={"members": ["u1", "not_a_demo_user"]})
    assert unknown.status_code == 422 and "not_a_demo_user" in unknown.json()["detail"]


def test_model_from_tables_rejects_missing_hub(db_mode):
    _, tables = db_mode
    with pytest.raises(RuntimeError, match="no hub"):
        rg.model_from_tables({**tables, "model_arrays": tables["model_arrays"].iloc[0:0]})


def test_select_demo_balances_cohesion_and_respects_size():
    model = _served_model()
    groups = pd.DataFrame({
        "group_id": [f"g{i}" for i in range(6)],
        "members": [["u1", "u2"], ["u3", "u4"], ["u1", "u3"], ["u2", "u4"], ["u1", "u4"], ["u2", "u3"]],
    })
    users, sample = deploy_load.select_demo(model, groups, ("u1", "u2", "u3", "u4"), n_users=4, n_groups=4)
    assert len(users) == len(set(users)) <= 4
    assert set(sample["cohesion_bucket"]) <= {"high", "low"}
    assert all(set(m) <= set(users) for m in sample["members"])  # sample groups use demo users only
    counts = sample["cohesion_bucket"].value_counts()
    assert abs(counts.get("high", 0) - counts.get("low", 0)) <= 1
    again, sample2 = deploy_load.select_demo(model, groups, ("u1", "u2", "u3", "u4"), n_users=4, n_groups=4)
    assert again == users and sample2.equals(sample)  # seeded


def test_build_tables_contents_and_no_text():
    model = _served_model()
    tables = deploy_load.build_tables(model, FAKE_REVIEWS, ["u2", "u4"], FAKE_GROUPS.iloc[:0].assign(
        cohesion=[], cohesion_bucket=[]))
    assert set(tables) == set(deploy_load.GROUP_TABLES)
    p = tables["user_profiles"]
    assert p[p.user_id == "u4"][["rank", "business_id", "stars"]].values.tolist() == [[1, "b5", 5.0], [2, "b6", 3.0]]
    assert sorted(tables["user_visited"].itertuples(index=False, name=None)) == [
        ("u2", "b1"), ("u2", "b2"), ("u4", "b5"), ("u4", "b6")]
    hub = tables["model_arrays"].iloc[0]
    assert (hub["name"], hub["model"], hub["n"]) == ("hub", "content_debiased", 8)
    np.testing.assert_array_equal(hub["vals"], model.hub)
    columns = {c for t in tables.values() for c in t.columns}
    assert not columns & {"text", "review_text", "review_id"}


def test_db_mode_accepts_pgvector_vector_objects(db_mode):
    from pgvector import Vector

    model, tables = db_mode
    as_vectors = {**tables,
                  "restaurants": tables["restaurants"].assign(embedding=[Vector(v) for v in RAW]),
                  "user_taste": tables["user_taste"].assign(
                      embedding=[Vector(list(v)) for v in tables["user_taste"]["embedding"]])}
    rebuilt = rg.model_from_tables(as_vectors)
    np.testing.assert_array_equal(rebuilt.matrix, model.matrix)
    for u in model.taste:
        np.testing.assert_array_equal(rebuilt.taste[u], model.taste[u])
