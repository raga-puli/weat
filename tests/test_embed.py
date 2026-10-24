import numpy as np
import pandas as pd
import pytest

import src.embed as embed


def test_write_business_ids_creates_new_file(tmp_path, monkeypatch):
    path = tmp_path / "business_ids.txt"
    monkeypatch.setattr(embed, "BUSINESS_IDS", path)

    embed._write_business_ids(["b1", "b2", "b3"])

    assert path.read_text() == "b1\nb2\nb3\n"


def test_write_business_ids_allows_identical_rewrite(tmp_path, monkeypatch):
    path = tmp_path / "business_ids.txt"
    monkeypatch.setattr(embed, "BUSINESS_IDS", path)

    embed._write_business_ids(["b1", "b2", "b3"])
    embed._write_business_ids(["b1", "b2", "b3"])  # should not raise

    assert path.read_text() == "b1\nb2\nb3\n"


def test_write_business_ids_rejects_reordering(tmp_path, monkeypatch):
    path = tmp_path / "business_ids.txt"
    monkeypatch.setattr(embed, "BUSINESS_IDS", path)

    embed._write_business_ids(["b1", "b2", "b3"])

    with pytest.raises(ValueError, match="index 1"):
        embed._write_business_ids(["b1", "b3", "b2"])

    # the on-disk file must be untouched by the rejected write
    assert path.read_text() == "b1\nb2\nb3\n"


def test_write_business_ids_rejects_different_id_at_same_position(tmp_path, monkeypatch):
    path = tmp_path / "business_ids.txt"
    monkeypatch.setattr(embed, "BUSINESS_IDS", path)

    embed._write_business_ids(["b1", "b2", "b3"])

    with pytest.raises(ValueError, match="index 2"):
        embed._write_business_ids(["b1", "b2", "b4"])


def test_write_business_ids_rejects_length_change(tmp_path, monkeypatch):
    path = tmp_path / "business_ids.txt"
    monkeypatch.setattr(embed, "BUSINESS_IDS", path)

    embed._write_business_ids(["b1", "b2", "b3"])

    with pytest.raises(ValueError, match="index 3"):
        embed._write_business_ids(["b1", "b2", "b3", "b4"])


# --- embed.py: --limit isolation and guard-before-save ---------------------


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
