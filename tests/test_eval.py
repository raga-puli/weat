import math

import numpy as np
import pandas as pd
import pytest

from src import interfaces
from src.eval import (
    catalog_coverage,
    intra_list_diversity,
    main,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_hit_at_rank_1():
    recs = {"u1": ["b1", "b2", "b3"]}
    relevant = {"u1": "b1"}
    assert recall_at_k(recs, relevant, k=3) == 1.0


def test_recall_at_k_hit_at_rank_k():
    recs = {"u1": ["b2", "b3", "b1"]}
    relevant = {"u1": "b1"}
    assert recall_at_k(recs, relevant, k=3) == 1.0
    assert recall_at_k(recs, relevant, k=2) == 0.0


def test_recall_at_k_miss():
    recs = {"u1": ["b2", "b3", "b4"]}
    relevant = {"u1": "b1"}
    assert recall_at_k(recs, relevant, k=3) == 0.0


def test_recall_at_k_empty_list():
    recs = {"u1": []}
    relevant = {"u1": "b1"}
    assert recall_at_k(recs, relevant, k=3) == 0.0


def test_ndcg_at_k_hit_at_rank_1():
    recs = {"u1": ["b1", "b2", "b3"]}
    relevant = {"u1": "b1"}
    assert ndcg_at_k(recs, relevant, k=3) == 1.0


def test_ndcg_at_k_hit_at_rank_k():
    recs = {"u1": ["b2", "b3", "b1"]}
    relevant = {"u1": "b1"}
    expected = 1.0 / math.log2(3 + 1)
    assert math.isclose(ndcg_at_k(recs, relevant, k=3), expected)


def test_ndcg_at_k_miss():
    recs = {"u1": ["b2", "b3", "b4"]}
    relevant = {"u1": "b1"}
    assert ndcg_at_k(recs, relevant, k=3) == 0.0


def test_ndcg_at_k_empty_list():
    recs = {"u1": []}
    relevant = {"u1": "b1"}
    assert ndcg_at_k(recs, relevant, k=3) == 0.0


def test_mrr_hit_at_rank_1():
    recs = {"u1": ["b1", "b2", "b3"]}
    relevant = {"u1": "b1"}
    assert mrr(recs, relevant) == 1.0


def test_mrr_hit_at_rank_k():
    recs = {"u1": ["b2", "b3", "b1"]}
    relevant = {"u1": "b1"}
    assert math.isclose(mrr(recs, relevant), 1 / 3)


def test_mrr_miss():
    recs = {"u1": ["b2", "b3", "b4"]}
    relevant = {"u1": "b1"}
    assert mrr(recs, relevant) == 0.0


def test_mrr_empty_list():
    recs = {"u1": []}
    relevant = {"u1": "b1"}
    assert mrr(recs, relevant) == 0.0


def test_catalog_coverage_basic():
    recs = {"u1": ["b1", "b2"], "u2": ["b2", "b3"]}
    assert catalog_coverage(recs, n_catalog=4) == 0.75


def test_catalog_coverage_empty_list():
    recs = {"u1": []}
    assert catalog_coverage(recs, n_catalog=4) == 0.0


def test_intra_list_diversity_basic():
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    id_to_row = {"b1": 0, "b2": 1, "b3": 2}
    recs = {"u1": ["b1", "b2"], "u2": ["b1", "b3"]}
    # u1: cos(b1, b2) = 0 -> diversity 1.0
    # u2: cos(b1, b3) = 1 -> diversity 0.0
    assert math.isclose(intra_list_diversity(recs, emb, id_to_row), 0.5)


def test_intra_list_diversity_skips_short_lists():
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    id_to_row = {"b1": 0, "b2": 1}
    recs = {"u1": ["b1"], "u2": []}
    assert intra_list_diversity(recs, emb, id_to_row) == 0.0


class FakeRecommender:
    """Toy recommender for exercising the src.eval runner end to end."""

    name = "fake"

    def fit(self, train: pd.DataFrame) -> None:
        self.catalog = sorted(train["business_id"].unique())

    def recommend(self, user_ids, k, exclude):
        recs = {}
        for user_id in user_ids:
            excluded = exclude.get(user_id, set())
            candidates = [b for b in self.catalog if b not in excluded]
            recs[user_id] = candidates[:k]
        return recs

    def score(self, user_id, business_ids):
        return [float(bid[-1]) for bid in business_ids]


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    train_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4"],
            "business_id": ["b1", "b2", "b2", "b1", "b3"],
            "stars": [5, 3, 4, 2, 5],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "business_id": ["b3", "b1", "b3"],
            "stars": [5, 2, 4],
        }
    )
    restaurants_df = pd.DataFrame({"business_id": ["b1", "b2", "b3", "b4"]})

    train_path = tmp_path / "train_reviews.parquet"
    val_path = tmp_path / "val_reviews.parquet"
    test_path = tmp_path / "test_reviews.parquet"
    restaurants_path = tmp_path / "restaurants.parquet"
    results_path = tmp_path / "results.md"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    val_df.to_parquet(test_path)
    restaurants_df.to_parquet(restaurants_path)
    results_path.write_text(
        "# Results\n\n"
        "| date | split | model | config | recall@10 | NDCG@10 | MRR | coverage | "
        "diversity | spearman | latency_ms/user | n_users | note |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )

    monkeypatch.setattr(interfaces, "TRAIN_REVIEWS", train_path)
    monkeypatch.setattr(interfaces, "VAL_REVIEWS", val_path)
    monkeypatch.setattr(interfaces, "TEST_REVIEWS", test_path)
    monkeypatch.setattr(interfaces, "RESTAURANTS", restaurants_path)
    monkeypatch.setattr(interfaces, "EMBEDDINGS", tmp_path / "embeddings.npy")
    monkeypatch.setattr(interfaces, "RESULTS_MD", results_path)

    return {"results_path": results_path}


def test_runner_end_to_end_val_split(runner_env):
    result = main(["--model", "tests.test_eval:FakeRecommender", "--split", "val", "--k", "2"])

    assert result["split"] == "val"
    assert result["model"] == "fake"
    assert result["n_users"] == 3
    assert math.isclose(result["recall@10"], 1.0)
    assert math.isclose(result["MRR"], 0.75)
    assert math.isclose(result["NDCG@10"], (1.0 + 1.0 / math.log2(3)) / 2)
    assert math.isclose(result["coverage"], 0.75)
    assert result["diversity"] == "n/a"
    assert result["spearman"] is not None
    assert math.isclose(result["spearman"], 1.5 / math.sqrt(3), rel_tol=1e-6)
    assert result["latency_ms/user"] >= 0.0


def test_runner_sample_limits_users(runner_env):
    result = main(
        ["--model", "tests.test_eval:FakeRecommender", "--split", "val", "--sample", "2"]
    )
    assert result["n_users"] == 2


def test_runner_logs_row_to_results_md(runner_env):
    main(
        [
            "--model",
            "tests.test_eval:FakeRecommender",
            "--split",
            "val",
            "--log",
            "smoke test",
        ]
    )
    lines = runner_env["results_path"].read_text().splitlines()
    assert len(lines) == 5  # title, blank, header, separator, 1 logged row
    assert "smoke test" in lines[-1]


def test_runner_refuses_test_split_without_flag(runner_env):
    with pytest.raises(SystemExit):
        main(["--model", "tests.test_eval:FakeRecommender", "--split", "test"])


def test_runner_allows_test_split_with_flag(runner_env):
    result = main(
        [
            "--model",
            "tests.test_eval:FakeRecommender",
            "--split",
            "test",
            "--i-mean-it",
        ]
    )
    assert result["split"] == "test"
