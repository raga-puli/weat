import numpy as np
import pandas as pd
import pytest

import src.recommend as recommend
from src.recommend import ContentBased, ContentBasedCentered


# --- embed.py: --limit isolation and guard-before-save ---------------------
# (Lives here because tests/test_embed.py isn't assigned to Person A in week 2.)


@pytest.fixture
def fake_embed(tmp_path, monkeypatch):
    import src.embed as embed

    ids = [f"b{i}" for i in range(4)]
    reviews = pd.DataFrame(
        [
            {"review_id": f"{b}-{s}", "business_id": b, "stars": s, "text": "x" * 300}
            for b in ids
            for s in (5, 2)
        ]
    )
    restaurants = pd.DataFrame({"name": ids, "categories": "Food"}, index=pd.Index(ids, name="business_id"))

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setattr(embed, "_load_eligible_reviews", lambda: (reviews, restaurants, list(ids)))
    monkeypatch.setattr(embed, "SentenceTransformer", FakeModel)
    for name in ("EMBEDDINGS", "EMBEDDINGS_BLOB", "BUSINESS_IDS"):
        suffix = ".txt" if name == "BUSINESS_IDS" else ".npy"
        monkeypatch.setattr(embed, name, tmp_path / (name.lower() + suffix))
    return embed, ids


@pytest.mark.parametrize("mode", ["chunked", "blob"])
def test_limit_run_never_touches_real_outputs(fake_embed, mode):
    embed, ids = fake_embed
    real = [embed.EMBEDDINGS, embed.EMBEDDINGS_BLOB, embed.BUSINESS_IDS]
    run = embed.run_chunked if mode == "chunked" else embed.run_blob

    run(limit=2)

    assert not any(p.exists() for p in real)
    emb_path, ids_path = embed._output_paths(
        embed.EMBEDDINGS if mode == "chunked" else embed.EMBEDDINGS_BLOB, 2
    )
    assert emb_path.name.endswith("_limit2.npy") and ids_path.name.endswith("_limit2.txt")
    assert np.load(emb_path).shape == (2, 3)
    assert ids_path.read_text().split() == ids[:2]


def test_guard_fails_before_any_npy_is_saved(fake_embed):
    embed, ids = fake_embed
    embed.BUSINESS_IDS.write_text("\n".join(reversed(ids)) + "\n")

    with pytest.raises(ValueError, match="ordering would change"):
        embed.run_chunked()

    assert not embed.EMBEDDINGS.exists()
    assert embed.BUSINESS_IDS.read_text().split() == list(reversed(ids))


def test_full_run_writes_real_outputs(fake_embed):
    embed, ids = fake_embed
    embed.run_chunked()
    assert np.load(embed.EMBEDDINGS).shape == (4, 3)
    assert embed.BUSINESS_IDS.read_text().split() == ids


# --- ContentBased.taste_vector_from / scores_for --------------------------
# (Lives here because tests/test_recommend.py isn't assigned to Person A in week 2.)

TINY_IDS = ["r0", "r1", "r2", "r3"]
TINY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.6, 0.8, 0.0],
        [0.0, 0.6, 0.8],
    ],
    dtype="float32",
)


@pytest.fixture
def tiny_embeddings(monkeypatch):
    monkeypatch.setattr(recommend, "load_embeddings", lambda path: (TINY.copy(), list(TINY_IDS)))


def test_taste_vector_skips_unknown_ids(tiny_embeddings):
    model = ContentBased()
    np.testing.assert_allclose(
        model.taste_vector_from(["r0", "nope", "r1"]), model.taste_vector_from(["r0", "r1"])
    )


def test_taste_vector_none_when_nothing_known(tiny_embeddings):
    model = ContentBased()
    assert model.taste_vector_from([]) is None
    assert model.taste_vector_from(["nope", "also-nope"]) is None


def test_taste_vector_is_unit_norm_mean(tiny_embeddings):
    model = ContentBased()
    taste = model.taste_vector_from(["r0", "r1"])
    np.testing.assert_allclose(np.linalg.norm(taste), 1.0, rtol=1e-6)
    np.testing.assert_allclose(taste, np.array([1, 1, 0]) / np.sqrt(2), rtol=1e-6)


def test_centered_model_uses_centered_matrix(tiny_embeddings):
    model = ContentBasedCentered()
    centered = TINY - TINY.mean(axis=0)
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    np.testing.assert_allclose(model.matrix, centered, rtol=1e-6)

    expected = centered[[0, 2]].mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.taste_vector_from(["r0", "r2"]), expected, rtol=1e-5)
    # and differs from the uncentered model's vector
    assert not np.allclose(model.taste_vector_from(["r0", "r2"]), ContentBased().taste_vector_from(["r0", "r2"]))


def test_scores_for_follows_business_ids_order(tiny_embeddings):
    for model in (ContentBased(), ContentBasedCentered()):
        taste = model.taste_vector_from(["r3"])
        scores = model.scores_for(taste)
        assert scores.shape == (len(TINY_IDS),)
        for i, b in enumerate(model.business_ids):
            assert b == TINY_IDS[i]
            np.testing.assert_allclose(scores[i], model.matrix[model.index[b]] @ taste, rtol=1e-6)
        assert int(np.argmax(scores)) == TINY_IDS.index("r3")
