import numpy as np
import pandas as pd
import pytest

import src.interfaces as interfaces
from src.recommend import ContentBased, ContentBasedCentered, hubness_diagnostics

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


def test_centered_matrix_has_near_zero_mean_and_unit_norms(tmp_path, monkeypatch):
    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "business_ids.txt"
    np.save(emb_path, EMBEDDINGS.astype("float32"))
    ids_path.write_text("\n".join(BUSINESS_IDS) + "\n")
    monkeypatch.setattr(interfaces, "BUSINESS_IDS", ids_path)

    plain = ContentBased(embeddings_path=emb_path)
    centered = ContentBasedCentered(embeddings_path=emb_path)

    row_norms = np.linalg.norm(centered.matrix, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)

    plain_bias = np.abs(plain.matrix.mean(axis=0)).mean()
    centered_bias = np.abs(centered.matrix.mean(axis=0)).mean()
    assert centered_bias < plain_bias


def test_centered_recommender_still_fits_and_recommends(tmp_path, monkeypatch):
    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "business_ids.txt"
    np.save(emb_path, EMBEDDINGS.astype("float32"))
    ids_path.write_text("\n".join(BUSINESS_IDS) + "\n")
    monkeypatch.setattr(interfaces, "BUSINESS_IDS", ids_path)

    model = ContentBasedCentered(embeddings_path=emb_path)
    model.fit(TRAIN)

    assert model.name == "content_centered"
    result = model.recommend(["u1"], k=2, exclude={"u1": {"b1"}})
    assert len(result["u1"]) == 2


class TestHubnessDiagnostics:
    RECS = {
        "u1": ["a", "b", "c"],
        "u2": ["a", "b", "d"],
        "u3": ["a", "b", "c"],
    }
    CATALOG = ["a", "b", "c", "d", "e"]  # "e" is never recommended

    def test_top10_counts_and_share_of_users(self):
        result = hubness_diagnostics(self.RECS, self.CATALOG, seed=42)

        top = {row["business_id"]: row for row in result["top10_most_frequent"]}
        assert top["a"]["count"] == 3
        assert top["a"]["share_of_users"] == pytest.approx(1.0)
        assert top["c"]["count"] == 2
        assert top["c"]["share_of_users"] == pytest.approx(2 / 3)
        assert top["d"]["count"] == 1

    def test_unique_items_recommended_excludes_never_recommended(self):
        result = hubness_diagnostics(self.RECS, self.CATALOG, seed=42)
        assert result["n_unique_items_recommended"] == 4  # a, b, c, d -- not e

    def test_occurrence_skewness_is_finite(self):
        result = hubness_diagnostics(self.RECS, self.CATALOG, seed=42)
        assert np.isfinite(result["occurrence_skewness"])

    def test_mean_top10_overlap_matches_hand_computed_expectation(self):
        # only 3 possible unordered user pairs; average of their exact
        # overlaps is (2/3 + 1.0 + 2/3) / 3 = 0.7778
        result = hubness_diagnostics(self.RECS, self.CATALOG, seed=42, n_pairs=500)
        assert result["mean_top10_overlap"] == pytest.approx(7 / 9, abs=0.05)

    def test_identical_lists_have_full_overlap(self):
        recs = {"u1": ["a", "b"], "u2": ["a", "b"], "u3": ["a", "b"]}
        result = hubness_diagnostics(recs, ["a", "b", "c"], seed=1, n_pairs=50)
        assert result["mean_top10_overlap"] == pytest.approx(1.0)

    def test_disjoint_lists_have_zero_overlap(self):
        recs = {"u1": ["a"], "u2": ["b"], "u3": ["c"]}
        result = hubness_diagnostics(recs, ["a", "b", "c"], seed=1, n_pairs=50)
        assert result["mean_top10_overlap"] == pytest.approx(0.0)
