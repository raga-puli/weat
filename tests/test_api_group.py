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
