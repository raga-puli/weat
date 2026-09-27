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
