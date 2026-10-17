import math

import pandas as pd
import pytest

from src import interfaces
from src.baselines import Popularity, Random

EMPTY_TRAIN = pd.DataFrame({"user_id": [], "business_id": [], "stars": []})


@pytest.fixture
def fake_restaurants(tmp_path, monkeypatch):
    restaurants_df = pd.DataFrame(
        {
            "business_id": ["b1", "b2", "b3", "b4"],
            "review_count": [100, 1, 50, 10],
        }
    )
    path = tmp_path / "restaurants.parquet"
    restaurants_df.to_parquet(path)
    monkeypatch.setattr(interfaces, "RESTAURANTS", path)
    return path


def test_random_excludes_businesses(fake_restaurants):
    model = Random()
    model.fit(EMPTY_TRAIN)
    exclude = {"u1": {"b1", "b2", "b3"}}
    recs = model.recommend(["u1"], k=2, exclude=exclude)
    assert recs["u1"] == ["b4"]


def test_random_is_deterministic_across_instances(fake_restaurants):
    exclude = {"u1": set()}
    model_a = Random()
    model_a.fit(EMPTY_TRAIN)
    model_b = Random()
    model_b.fit(EMPTY_TRAIN)
    assert model_a.recommend(["u1"], k=3, exclude=exclude) == model_b.recommend(
        ["u1"], k=3, exclude=exclude
    )


def test_random_score_is_deterministic(fake_restaurants):
    model = Random()
    model.fit(EMPTY_TRAIN)
    assert model.score("u1", ["b1", "b2"]) == model.score("u1", ["b1", "b2"])


def test_popularity_excludes_businesses():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "business_id": ["b1", "b1", "b2", "b3"],
            "stars": [5, 4, 3, 2],
        }
    )
    model = Popularity()
    model.fit(train)
    recs = model.recommend(["u1"], k=2, exclude={"u1": {"b1"}})
    assert "b1" not in recs["u1"]
    assert recs["u1"] == ["b2", "b3"]


def test_popularity_uses_only_passed_in_train_frame():
    # b_high has the most train reviews but would rank last by a real
    # review_count column; b_low has one train review but would rank
    # first. Popularity must ignore any such column entirely.
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "business_id": ["b_high", "b_high", "b_high", "b_low", "b_mid"],
            "stars": [5, 4, 3, 2, 3],
        }
    )
    model = Popularity()
    model.fit(train)
    assert model.counts == {"b_high": 3, "b_low": 1, "b_mid": 1}
    recs = model.recommend(["u1"], k=3, exclude={"u1": set()})
    assert recs["u1"] == ["b_high", "b_low", "b_mid"]


def test_popularity_score_is_log1p_of_train_counts():
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "business_id": ["b1", "b1"],
            "stars": [5, 4],
        }
    )
    model = Popularity()
    model.fit(train)
    scores = model.score("u1", ["b1", "b2"])
    assert math.isclose(scores[0], math.log1p(2))
    assert math.isclose(scores[1], math.log1p(0))
