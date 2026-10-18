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


# --- ContentBasedDebiased: column-centered (hub-corrected) scores ------------------

import src.recommend as recommend_module  # noqa: E402
from src.recommend import ContentBasedDebiased  # noqa: E402


def _debiased_with(monkeypatch, matrix, ids):
    monkeypatch.setattr(recommend_module, "load_embeddings", lambda path: (matrix.astype(np.float32), list(ids)))
    return ContentBasedDebiased(), ContentBasedCentered()


def _likes(pairs):
    return pd.DataFrame([{"user_id": u, "business_id": b, "stars": 5} for u, b in pairs])


def _hub_world():
    """4 users who each like their own item (e0..e3 directions); a "hub"
    restaurant along e0+e1+e2+e3 is moderately close to all of them; each user
    also has a near-own candidate."""
    rng = np.random.default_rng(3)
    eye = np.eye(8)
    rows, ids = [], []
    for i in range(4):
        rows.append(eye[i]); ids.append(f"own{i}")                                  # liked
        rows.append(eye[i] + 0.8 * eye[4 + i] + 0.05 * rng.normal(size=8)); ids.append(f"near{i}")
        rows.append(0.3 * eye[i] + eye[4 + (i + 1) % 4]); ids.append(f"far{i}")
    rows.append(eye[:4].sum(axis=0) + 0.1 * eye[4:].sum(axis=0)); ids.append("hub")
    m = np.array(rows)
    return m / np.linalg.norm(m, axis=1, keepdims=True), ids


def test_debiased_hub_uses_only_fitted_users(monkeypatch):
    matrix, ids = _hub_world()
    debiased, _ = _debiased_with(monkeypatch, matrix, ids)
    np.testing.assert_array_equal(debiased.hub, 0)  # before fit: plain cosine

    debiased.fit(_likes([("u0", "own0"), ("u1", "own1")]))
    assert set(debiased.taste) == {"u0", "u1"}
    expected = np.mean([debiased.matrix @ debiased.taste[u] for u in ("u0", "u1")], axis=0)
    np.testing.assert_allclose(debiased.hub, expected, atol=1e-6)

    # refitting on different users recomputes hub from those users only
    debiased.fit(_likes([("u2", "own2")]))
    np.testing.assert_allclose(debiased.hub, debiased.matrix @ debiased.taste["u2"], atol=1e-6)


def test_debiased_scores_are_cosine_minus_hub(monkeypatch):
    matrix, ids = _hub_world()
    debiased, centered = _debiased_with(monkeypatch, matrix, ids)
    train = _likes([(f"u{i}", f"own{i}") for i in range(4)])
    debiased.fit(train)
    centered.fit(train)
    for u in debiased.taste:
        cosine = centered.scores_for(centered.taste[u])
        np.testing.assert_allclose(debiased.scores_for(debiased.taste[u]), cosine - debiased.hub, atol=1e-6)
        np.testing.assert_allclose(debiased.score(u, ids), cosine - debiased.hub, atol=1e-6)
    np.testing.assert_allclose(
        debiased._score_matrix(np.stack([debiased.taste[u] for u in ("u0", "u1")])),
        np.stack([debiased.scores_for(debiased.taste[u]) for u in ("u0", "u1")]),
        atol=1e-6,
    )
    assert debiased.name == "content_debiased"


def test_debiased_hub_restaurant_drops_in_rank(monkeypatch):
    matrix, ids = _hub_world()
    debiased, centered = _debiased_with(monkeypatch, matrix, ids)
    train = _likes([(f"u{i}", f"own{i}") for i in range(4)])
    debiased.fit(train)
    centered.fit(train)
    users = [f"u{i}" for i in range(4)]
    exclude = {u: {f"own{u[1]}"} for u in users}
    k = len(ids) - 1
    before = centered.recommend(users, k, exclude)
    after = debiased.recommend(users, k, exclude)
    for u in users:
        assert before[u].index("hub") <= 1  # the hub is near the top for everyone
        assert after[u].index("hub") > before[u].index("hub")
    # it is the restaurant with the largest hub value
    assert ids[int(np.argmax(debiased.hub))] == "hub"
