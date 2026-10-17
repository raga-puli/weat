"""Group aggregation strategies over the full restaurant catalog.

Every strategy takes
    scores:   (n_members, n_restaurants) raw member scores (cosine) over the
              whole catalog,
    excluded: bool (n_restaurants,), True for restaurants that must never be
              recommended (e.g. anything a member already reviewed),
    k:        list length,
and returns the top-k restaurant indices, best first, never excluded ones.
Ties are broken by lower restaurant index.

Normalization (`norm`) is applied per member before aggregating:
    "raw":    the cosine as-is (ablation only);
    "zscore": (x - mean) / std per member;
    "rank":   per-member percentile rank, 0 (worst) .. 1 (best); the default.
zscore and rank are computed over the non-excluded restaurants only, so they
describe each member's preferences over the actual candidate set.

Why normalization matters for least-misery: raw cosine scales differ across
members -- a member with a diffuse history has all cosines bunched near
some low value -- so the min over members on raw scores just follows
whoever has the lowest scale, and the average is dominated by whoever has
the widest spread. The default was fixed to "rank" up front, not tuned.
"""

import numpy as np
from scipy.stats import rankdata

LAMBDA = 0.5
TAU = 0.1
NORMS = ("raw", "zscore", "rank")


def normalize(scores: np.ndarray, excluded: np.ndarray, norm: str = "rank") -> np.ndarray:
    """Per-member normalized scores, same shape as `scores`; excluded
    columns are NaN."""
    scores = np.asarray(scores, dtype=float)
    excluded = np.asarray(excluded, dtype=bool)
    if norm not in NORMS:
        raise ValueError(f"unknown norm {norm!r}; expected one of {NORMS}")

    out = np.full_like(scores, np.nan)
    cand = scores[:, ~excluded]
    if cand.shape[1] == 0:
        return out
    if norm == "raw":
        normed = cand
    elif norm == "zscore":
        std = cand.std(axis=1, keepdims=True)
        normed = np.divide(cand - cand.mean(axis=1, keepdims=True), std,
                           out=np.zeros_like(cand), where=std > 0)
    else:
        n = cand.shape[1]
        normed = (
            np.ones_like(cand) if n == 1
            else (rankdata(cand, method="average", axis=1) - 1) / (n - 1)
        )
    out[:, ~excluded] = normed
    return out


def _top_k(group_scores: np.ndarray, excluded: np.ndarray, k: int) -> np.ndarray:
    gs = np.where(excluded, -np.inf, group_scores)
    n_available = int((~excluded).sum())
    return np.argsort(-gs, kind="stable")[: min(k, n_available)]


def average(scores, excluded, k, norm="rank"):
    S = normalize(scores, excluded, norm)
    return _top_k(S.mean(axis=0), excluded, k)


def least_misery(scores, excluded, k, norm="rank"):
    S = normalize(scores, excluded, norm)
    return _top_k(S.min(axis=0), excluded, k)


def fairness(scores, excluded, k, norm="rank", lam=LAMBDA):
    """Mean minus lam * (max - min) over members.

    A one-shot proxy: it penalizes disagreement within a single list. The
    sequential version, fairness across repeated dinners, is
    `fairness_repeated`.
    """
    S = normalize(scores, excluded, norm)
    return _top_k(S.mean(axis=0) - lam * (S.max(axis=0) - S.min(axis=0)), excluded, k)


def fairness_repeated(scores, excluded, k, norm="rank", tau=TAU, return_weights=False):
    """A sequence of k dinners that takes turns favoring whoever has been
    served worst so far.

    At each step the member weights are softmax(-cumulative / tau), where
    cumulative is each member's summed normalized score over the dinners so
    far; the pick is the non-excluded, not-yet-picked restaurant maximizing
    the weighted mean of normalized scores. The first pick (all totals 0) is
    therefore the average strategy's top pick.

    tau = 0.1 was fixed before any results were seen: with rank
    normalization, members' cumulative totals differ by tenths, so tau = 0.1
    makes a gap of 0.1 shift a member's weight by a factor of e.

    Returns (picks, history): picks is (k,) restaurant indices in dinner
    order; history is (k, n_members), each member's normalized score for
    each dinner. Fewer than k rows if fewer than k candidates exist. With
    return_weights=True (diagnostics only; picks are unchanged), also
    returns weights, (k, n_members), the member weights used at each dinner.
    """
    S = normalize(scores, excluded, norm)
    filled = np.nan_to_num(S, nan=0.0)
    blocked = np.asarray(excluded, dtype=bool).copy()
    cumulative = np.zeros(S.shape[0])
    picks, history, weights = [], [], []
    for _ in range(min(k, int((~blocked).sum()))):
        logits = -cumulative / tau
        w = np.exp(logits - logits.max())
        w /= w.sum()
        gs = np.where(blocked, -np.inf, w @ filled)
        pick = int(np.argmax(gs))
        picks.append(pick)
        history.append(S[:, pick])
        weights.append(w)
        cumulative += S[:, pick]
        blocked[pick] = True
    shape = (len(picks), S.shape[0])
    picks, history = np.array(picks, dtype=int), np.array(history).reshape(shape)
    if return_weights:
        return picks, history, np.array(weights).reshape(shape)
    return picks, history


STRATEGIES = {"average": average, "least_misery": least_misery, "fairness": fairness}
