import numpy as np
import pandas as pd
import pytest

import src.interfaces as interfaces
from src.recommend import ContentBased

BUSINESS_IDS = ["b1", "b2", "b3", "b4", "b5"]

_RAW = np.array(
    [
        [1.0, 0.0, 0.0],  # b1
        [0.9, 0.1, 0.0],  # b2 - close to b1
        [0.0, 1.0, 0.0],  # b3
        [0.0, 0.0, 1.0],  # b4
        [0.5, 0.5, 0.0],  # b5 - halfway between b1 and b3
    ]
)
EMBEDDINGS = _RAW / np.linalg.norm(_RAW, axis=1, keepdims=True)

TRAIN = pd.DataFrame(
    [
        # u1 likes b1 (>=4 stars) and dislikes b3 -> taste == b1's vector.
        {"user_id": "u1", "business_id": "b1", "stars": 5},
        {"user_id": "u1", "business_id": "b3", "stars": 2},
        # u2 has no >=4-star reviews -> falls back to mean of b3, b4.
        {"user_id": "u2", "business_id": "b3", "stars": 3},
        {"user_id": "u2", "business_id": "b4", "stars": 2},
        # u4 only rated a business absent from the embedding index.
        {"user_id": "u4", "business_id": "ghost_biz", "stars": 5},
    ]
)


@pytest.fixture
def model(tmp_path, monkeypatch):
    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "business_ids.txt"
    np.save(emb_path, EMBEDDINGS.astype("float32"))
    ids_path.write_text("\n".join(BUSINESS_IDS) + "\n")

    monkeypatch.setattr(interfaces, "BUSINESS_IDS", ids_path)

    m = ContentBased(embeddings_path=emb_path)
    m.fit(TRAIN)
    return m


def test_taste_vector_uses_only_liked_businesses(model):
    assert np.allclose(model.taste["u1"], EMBEDDINGS[0])


def test_taste_vector_falls_back_when_nothing_liked(model):
    expected = EMBEDDINGS[2] + EMBEDDINGS[3]
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(model.taste["u2"], expected)


def test_fit_skips_businesses_missing_from_embedding_index(model):
    assert "u4" not in model.taste


def test_recommend_orders_by_similarity_and_excludes(model):
    result = model.recommend(["u1"], k=2, exclude={"u1": {"b1"}})
    assert result["u1"] == ["b2", "b5"]


def test_recommend_without_exclusions_ranks_self_first(model):
    result = model.recommend(["u1"], k=3, exclude={})
    assert result["u1"] == ["b1", "b2", "b5"]


def test_recommend_returns_empty_list_for_users_without_taste_vector(model):
    result = model.recommend(["u3", "u4"], k=2, exclude={})
    assert result == {"u3": [], "u4": []}


def test_score_matches_cosine_similarity(model):
    scores = model.score("u1", ["b2", "b1"])
    assert scores[0] == pytest.approx(float(EMBEDDINGS[0] @ EMBEDDINGS[1]))
    assert scores[1] == pytest.approx(1.0)


def test_score_is_nan_for_unknown_business(model):
    scores = model.score("u1", ["unknown_biz"])
    assert np.isnan(scores[0])


def test_score_is_nan_for_unknown_user(model):
    scores = model.score("u3", ["b1"])
    assert np.isnan(scores[0])
