import math

import numpy as np

from src.eval import (
    catalog_coverage,
    intra_list_diversity,
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
