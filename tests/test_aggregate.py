import numpy as np
import pytest

from src.aggregate import (
    LAMBDA,
    TAU,
    average,
    fairness,
    fairness_repeated,
    least_misery,
    normalize,
)

NONE = lambda n: np.zeros(n, dtype=bool)  # noqa: E731  (no exclusions)
ONE_SHOT = (average, least_misery, fairness)


def _all_strategies(scores, excluded, k, norm="rank"):
    out = {f.__name__: f(scores, excluded, k, norm=norm) for f in ONE_SHOT}
    out["fairness_repeated"] = fairness_repeated(scores, excluded, k, norm=norm)[0]
    return out


# Columns: X wins average, Y wins least-misery, Z wins fairness (raw norm,
# so the hand-computed values apply directly).
#   X: mean .683, min .05, fairness .683 - .5 * .95 = .208
#   Y: mean .600, min .60, fairness .600
#   Z: mean .637, min .59, fairness .637 - .5 * .07 = .602
S3 = np.array(
    [
        [1.00, 0.60, 0.66],
        [1.00, 0.60, 0.66],
        [0.05, 0.60, 0.59],
    ]
)
X, Y, Z = 0, 1, 2


def test_one_shot_strategies_pick_different_winners():
    assert LAMBDA == 0.5 and TAU == 0.1
    assert average(S3, NONE(3), 1, norm="raw")[0] == X
    assert least_misery(S3, NONE(3), 1, norm="raw")[0] == Y
    assert fairness(S3, NONE(3), 1, norm="raw")[0] == Z


def test_3x5_average_and_least_misery_disagree():
    # Default rank norm. Members 0 and 1 love item 0; member 2 dislikes it.
    scores = np.array(
        [
            [1.00, 0.90, 0.10, 0.20, 0.30],
            [1.00, 0.90, 0.20, 0.10, 0.30],
            [0.10, 0.05, 0.50, 0.60, 0.70],
        ]
    )
    # rank pct: m0 [1, .75, 0, .25, .5]; m1 [1, .75, .25, 0, .5]; m2 [.25, 0, .5, .75, 1]
    # average      [.75, .5, .25, .333, .667] -> item 0 (X)
    # least-misery [.25, 0, 0, 0, .5]         -> item 4 (Y)
    assert average(scores, NONE(5), 1)[0] == 0
    assert least_misery(scores, NONE(5), 1)[0] == 4


def test_excluded_items_never_returned():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(4, 30))
    excluded = rng.random(30) < 0.5
    # make excluded items everyone's favorites, so a leak would show
    scores[:, excluded] += 10
    for norm in ("raw", "zscore", "rank"):
        for name, picks in _all_strategies(scores, excluded, 10, norm).items():
            assert len(picks) == 10, name
            assert not excluded[picks].any(), (name, norm)
            assert len(set(picks)) == len(picks), name


def test_fewer_candidates_than_k():
    excluded = np.array([True, False, True, False, True])
    for picks in _all_strategies(np.ones((2, 5)), excluded, 10).values():
        assert sorted(picks) == [1, 3]


def test_fairness_lam_zero_equals_average():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(3, 50))
    excluded = rng.random(50) < 0.2
    for norm in ("raw", "zscore", "rank"):
        np.testing.assert_array_equal(
            fairness(scores, excluded, 10, norm=norm, lam=0.0),
            average(scores, excluded, 10, norm=norm),
        )


def test_rank_and_zscore_are_per_member():
    # Same ordering, wildly different scales: normalized rows must match.
    scores = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.80, 0.81, 0.82, 0.83],
            [-5.0, 0.00, 5.00, 10.0],
        ]
    )
    for norm in ("rank", "zscore"):
        n = normalize(scores, NONE(4), norm)
        np.testing.assert_allclose(n[1], n[0], atol=1e-9)
        np.testing.assert_allclose(n[2], n[0], atol=1e-9)
    np.testing.assert_allclose(normalize(scores, NONE(4), "rank")[0], [0, 1 / 3, 2 / 3, 1])


def test_rank_and_zscore_ignore_excluded_items():
    scores = np.array([[0.1, 0.2, 0.3, 100.0], [5.0, 0.3, 0.2, -100.0]])
    excluded = np.array([False, False, False, True])
    for norm in ("rank", "zscore"):
        full = normalize(scores, excluded, norm)
        alone = normalize(scores[:, :3], NONE(3), norm)
        assert np.isnan(full[:, 3]).all()
        np.testing.assert_allclose(full[:, :3], alone)
    np.testing.assert_allclose(normalize(scores, excluded, "rank")[0, :3], [0, 0.5, 1])
    z = normalize(scores, excluded, "zscore")[0, :3]
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)


def test_rank_ties_share_rank():
    np.testing.assert_allclose(normalize(np.array([[0.2, 0.2, 0.9]]), NONE(3)), [[0.25, 0.25, 1.0]])


def test_raw_least_misery_obeys_lowest_scale_member():
    # Member 1's cosines are all tiny, so raw least-misery just follows
    # member 1's ordering and ignores member 0, who hates candidate 0.
    raw = np.array(
        [
            [0.10, 0.90, 0.80],
            [0.03, 0.02, 0.01],
        ]
    )
    assert least_misery(raw, NONE(3), 1, norm="raw")[0] == 0
    assert least_misery(raw, NONE(3), 1, norm="rank")[0] != 0


def test_ties_broken_by_lower_index():
    assert list(average(np.array([[0.5, 0.5, 0.9]]), NONE(3), 3, norm="raw")) == [2, 0, 1]


def test_unknown_norm_raises():
    with pytest.raises(ValueError):
        average(S3, NONE(3), 1, norm="minmax")


def test_fairness_repeated_shape_and_no_repeats():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(4, 40))
    excluded = rng.random(40) < 0.3
    picks, history = fairness_repeated(scores, excluded, 10)
    assert history.shape == (10, 4)
    assert len(set(picks)) == 10 and not excluded[picks].any()
    np.testing.assert_allclose(history, normalize(scores, excluded)[:, picks].T)
    # first dinner: equal weights -> the average strategy's top pick
    assert picks[0] == average(scores, excluded, 1)[0]


def test_fairness_repeated_serves_the_outvoted_member():
    # Members 0 and 1 share a taste; member 2's taste is opposite, so under
    # average member 2 is the worst-served at every dinner.
    rng = np.random.default_rng(3)
    base = rng.normal(size=60)
    scores = np.stack([base, base + 0.01 * rng.normal(size=60), -base])
    excluded = NONE(60)
    k = 10

    pct = normalize(scores, excluded)
    avg_picks = average(scores, excluded, k)
    avg_totals = pct[:, avg_picks].sum(axis=1)
    assert np.argmin(avg_totals) == 2
    assert (pct[2, avg_picks] <= pct[:2, avg_picks].min(axis=0)).all()

    _, history = fairness_repeated(scores, excluded, k)
    fr_totals = history.sum(axis=0)
    assert fr_totals[2] > avg_totals[2]


def test_fairness_repeated_weights_are_diagnostic_only():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(3, 40))
    excluded = rng.random(40) < 0.2
    picks, history = fairness_repeated(scores, excluded, 10)
    picks_w, history_w, weights = fairness_repeated(scores, excluded, 10, return_weights=True)
    np.testing.assert_array_equal(picks, picks_w)
    np.testing.assert_array_equal(history, history_w)
    assert weights.shape == (10, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[0], 1 / 3)  # all totals start at 0
    # the top-weighted member is the one with the lowest total so far
    cumulative = np.cumsum(history, axis=0)[:-1]
    np.testing.assert_array_equal(weights[1:].argmax(axis=1), cumulative.argmin(axis=1))


def test_fairness_repeated_huge_tau_is_average():
    rng = np.random.default_rng(5)
    scores = rng.normal(size=(4, 80))
    excluded = rng.random(80) < 0.25
    fr = fairness_repeated(scores, excluded, 10, tau=1e6)[0]
    avg = average(scores, excluded, 10)
    # Same lists up to float-level ties: w @ S with near-uniform weights can
    # round an exact tie in the mean differently than S.mean().
    group_score = normalize(scores, excluded).mean(axis=0)
    np.testing.assert_allclose(group_score[fr], group_score[avg], rtol=0, atol=1e-12)
    assert set(fr) == set(avg)
