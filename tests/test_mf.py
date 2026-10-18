import builtins

import numpy as np
import pandas as pd
import pytest

from src import mf
from src.mf import ALS, ALS32, ALS128, als_numpy, confidence_matrix


def _train():
    """Two taste clusters. Restaurants a0..a5 are liked by "a" users, b0..b5 by
    "b" users. The focus user liked a0..a3 and disliked b0 (2 stars)."""
    rows = []
    for u in range(12):
        for j in range(6):
            if (u + j) % 3:  # each cluster user rates 4 of 6 cluster restaurants
                rows.append({"user_id": f"ua{u}", "business_id": f"a{j}", "stars": 5})
                rows.append({"user_id": f"ub{u}", "business_id": f"b{j}", "stars": 5})
    rows += [{"user_id": "focus", "business_id": f"a{j}", "stars": 5} for j in range(4)]
    rows.append({"user_id": "focus", "business_id": "b0", "stars": 2})
    return pd.DataFrame(rows)


@pytest.fixture(params=["implicit", "numpy"])
def fitted(request):
    model = ALS(factors=4, iterations=15, force_numpy=request.param == "numpy")
    model.fit(_train())
    return model


def test_confidence_matrix_values_and_dedup():
    train = pd.DataFrame([
        {"user_id": "u", "business_id": "x", "stars": 2},
        {"user_id": "u", "business_id": "x", "stars": 5},  # repeat visit: one entry, liked
        {"user_id": "u", "business_id": "y", "stars": 3},
        {"user_id": "v", "business_id": "y", "stars": 4},
    ])
    C, users, items = confidence_matrix(train, alpha=10)
    dense = pd.DataFrame(C.toarray(), index=users, columns=items)
    assert C.nnz == 3
    assert dense.loc["u", "x"] == 11 and dense.loc["u", "y"] == 1 and dense.loc["v", "y"] == 11
    assert dense.loc["v", "x"] == 0


def test_recommend_excludes_visited_and_prefers_own_cluster(fitted):
    visited = {"a0", "a1", "a2", "a3", "b0"}
    recs = fitted.recommend(["focus", "nobody"], 4, {"focus": visited})
    assert recs["nobody"] == []
    assert not visited & set(recs["focus"]) and len(recs["focus"]) == 4
    assert set(recs["focus"][:2]) == {"a4", "a5"}  # the unvisited restaurants of the focus user's cluster


def test_score_ordering_and_unknowns(fitted):
    s = dict(zip(["a4", "a5", "b3", "b4"], fitted.score("focus", ["a4", "a5", "b3", "b4"])))
    assert min(s["a4"], s["a5"]) > max(s["b3"], s["b4"])
    assert np.isnan(fitted.score("nobody", ["a4"])).all()
    assert np.isnan(fitted.score("focus", ["unknown"])[0])
    recs = fitted.recommend(["focus"], 2, {})["focus"]
    scores = fitted.score("focus", recs)
    assert scores == sorted(scores, reverse=True)  # recommend order matches score order


def test_fallback_runs_when_implicit_import_fails(monkeypatch, capsys):
    real_import = builtins.__import__

    def no_implicit(name, *args, **kwargs):
        if name == "implicit" or name.startswith("implicit."):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_implicit)
    model = ALS(factors=4, iterations=10)
    model.fit(_train())
    assert model.backend == "numpy"
    assert "ALS backend: numpy" in capsys.readouterr().out
    assert set(model.recommend(["focus"], 2, {"focus": {"a0", "a1", "a2", "a3"}})["focus"]) == {"a4", "a5"}


def test_numpy_als_solves_the_normal_equations():
    C, _, _ = confidence_matrix(_train(), alpha=10)
    X, Y = als_numpy(C, factors=3, regularization=0.05, iterations=5)
    # the last half-step solves for Y given X, so re-solving it must reproduce Y exactly
    Y2 = mf._als_half_step(C.T.tocsr(), X.astype(np.float64), 0.05)
    np.testing.assert_allclose(Y, Y2, rtol=1e-4, atol=1e-5)


def test_config_classes():
    assert (ALS().factors, ALS32().factors, ALS128().factors) == (64, 32, 128)
    assert (ALS().name, ALS32().name, ALS128().name) == ("als_f64", "als_f32", "als_f128")
    m = ALS128(alpha=5.0)
    assert (m.alpha, m.regularization, m.iterations) == (5.0, mf.REGULARIZATION, mf.ITERATIONS)
    assert (mf.ALPHA, mf.REGULARIZATION, mf.ITERATIONS) == (10.0, 0.05, 20)
