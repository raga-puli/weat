"""Leakage-free, time-respecting evaluation of group recommendation
strategies on the synthetic dinner groups from src/groups.py.

For a group with window_start T and target t, each member's profile is
built only from their TRAIN_REVIEWS rows dated strictly before T, never
including t. Popularity likewise counts only reviews dated before T.

Known gap: the restaurant embeddings were built from train reviews of all
dates, so the target's embedding text can include members' own (in-window)
reviews of it. `target_leak_flags` measures how many groups that affects.
"""

import argparse
import hashlib
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.aggregate import NORMS, TAU, fairness_repeated, normalize
from src.aggregate import STRATEGIES as AGGREGATORS
from src.groups import GROUPS, MIN_PRIOR_LIKES
from src.interfaces import DATA, RELEVANT_MIN_STARS, SEED, TRAIN_REVIEWS
from src.recommend import ContentBasedCentered

K = 10
N_BOOTSTRAP = 1000
CONTENT_STRATEGIES = ("average", "least_misery", "fairness", "fairness_repeated")
ALL_STRATEGIES = ("random", "popularity", *CONTENT_STRATEGIES)
METRICS = ("hit", "ndcg", "group_score", "worst_member", "fairness_gap")
BUCKET_ORDER = ["low", "mid", "high"]


@dataclass
class History:
    business_ids: np.ndarray
    stars: np.ndarray
    dates: np.ndarray  # datetime64[ns], sorted ascending


def index_histories(train: pd.DataFrame, user_ids: set[str]) -> dict[str, History]:
    rows = train[train["user_id"].isin(user_ids)].sort_values(["user_id", "date"])
    return {
        user_id: History(
            g["business_id"].to_numpy(),
            g["stars"].to_numpy(),
            pd.to_datetime(g["date"]).to_numpy("datetime64[ns]"),
        )
        for user_id, g in rows.groupby("user_id", sort=False)
    }


def pre_window(history: History | None, window_start) -> History:
    """A member's train rows dated strictly before window_start."""
    if history is None:
        empty = np.array([], dtype=object)
        return History(empty, np.array([]), np.array([], dtype="datetime64[ns]"))
    cut = np.searchsorted(history.dates, np.datetime64(pd.Timestamp(window_start), "ns"), side="left")
    return History(history.business_ids[:cut], history.stars[:cut], history.dates[:cut])


class PreWindowPopularity:
    """Per-restaurant TRAIN_REVIEWS counts dated strictly before a cutoff."""

    def __init__(self, train: pd.DataFrame, business_ids: list[str]):
        index = {b: i for i, b in enumerate(business_ids)}
        rows = train[train["business_id"].isin(index)].sort_values("date")
        self.dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
        self.bidx = rows["business_id"].map(index).to_numpy()
        self.n = len(business_ids)

    def counts(self, window_start) -> np.ndarray:
        cut = np.searchsorted(self.dates, np.datetime64(pd.Timestamp(window_start), "ns"), side="left")
        return np.bincount(self.bidx[:cut], minlength=self.n)


def _seed_for(group_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16)


def _list_metrics(top: np.ndarray, target_idx: int, pct: np.ndarray) -> dict:
    """top: catalog indices; pct: (n_members, n_restaurants) per-member
    percentile over the candidate set (NaN for excluded, never in top)."""
    hits = np.flatnonzero(top == target_idx)
    member_means = pct[:, top].mean(axis=1)
    return {
        "hit": float(len(hits) > 0),
        "ndcg": float(1 / np.log2(hits[0] + 2)) if len(hits) else 0.0,
        "group_score": float(member_means.mean()),
        "worst_member": float(member_means.min()),
        "fairness_gap": float(member_means.max() - member_means.min()),
    }


def _repeated_diagnostics(picks, history, weights, S, excluded) -> dict:
    """Per-dinner diagnostics for one fairness_repeated run:
    fr_max_weight:    the largest member weight at each dinner;
    fr_top_is_lowest: the top-weighted member has the (strictly) lowest
                      cumulative total so far; None when totals tie (dinner 1);
    fr_dictator:      the pick is also the top-weighted member's own best
                      available restaurant, i.e. the dinner effectively
                      served that one member.
    """
    cumulative = np.vstack([np.zeros(history.shape[1]), np.cumsum(history, axis=0)[:-1]])
    available = ~np.asarray(excluded, dtype=bool)
    max_w, lowest, dictator = [], [], []
    for t, pick in enumerate(picks):
        top = int(np.argmax(weights[t]))
        cum = cumulative[t]
        max_w.append(float(weights[t].max()))
        lowest.append(None if np.isclose(cum.min(), np.sort(cum)[1]) else bool(top == int(np.argmin(cum))))
        dictator.append(bool(S[top, pick] >= np.nanmax(np.where(available, S[top], np.nan))))
        available[pick] = False
    return {"fr_max_weight": max_w, "fr_top_is_lowest": lowest, "fr_dictator": dictator}


def evaluate_groups(
    groups: pd.DataFrame,
    train: pd.DataFrame,
    model,
    k: int = K,
    min_prior_likes: int = MIN_PRIOR_LIKES,
    normalization: str = "rank",
    seed: int = SEED,
    extra_taus: tuple[float, ...] = (),
    diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """One row per (surviving group, strategy). Returns (rows, drop counts).

    `model` needs business_ids, taste_vector_from() and scores_for().

    Sensitivity analysis only: `extra_taus` adds fairness_repeated runs as
    strategies named "fairness_repeated_tau<tau>". `diagnostics` adds
    per-dinner columns to the default fairness_repeated rows (see
    _repeated_diagnostics). Neither changes any other strategy's output.
    """
    n_items = len(model.business_ids)
    index = {b: i for i, b in enumerate(model.business_ids)}
    members_all = set(u for m in groups["members"] for u in m)
    histories = index_histories(train, members_all)
    popularity = PreWindowPopularity(train, model.business_ids)

    drops = {
        "too_few_prior_likes": 0,
        "target_in_prior_history": 0,
        "target_unknown": 0,
        "no_embeddable_likes": 0,
    }
    rows = []
    for g in groups.itertuples(index=False):
        target = g.target_business_id
        if target not in index:
            drops["target_unknown"] += 1
            continue

        priors = [pre_window(histories.get(u), g.window_start) for u in g.members]
        target_seen = any((p.business_ids == target).any() for p in priors)
        likes = [p.business_ids[(p.stars >= RELEVANT_MIN_STARS) & (p.business_ids != target)] for p in priors]
        too_few = any(len(l) < min_prior_likes for l in likes)
        # Count each reason independently; a group can fail both.
        drops["target_in_prior_history"] += target_seen
        drops["too_few_prior_likes"] += too_few
        if target_seen or too_few:
            continue

        tastes = [model.taste_vector_from(list(l)) for l in likes]
        if any(t is None for t in tastes):
            drops["no_embeddable_likes"] += 1
            continue
        tastes = np.stack(tastes)

        seen = set(b for p in priors for b in p.business_ids)
        excluded = np.zeros(n_items, dtype=bool)
        excluded[[index[b] for b in seen if b in index]] = True
        cand = np.flatnonzero(~excluded)
        target_idx = index[target]
        assert not excluded[target_idx], f"target {target} excluded from candidates of {g.group_id}"

        # Full-catalog member scores; the aggregators apply the exclusion.
        scores = np.stack([model.scores_for(t) for t in tastes])
        pct = normalize(scores, excluded, "rank")

        counts = popularity.counts(g.window_start)
        # Ignore the members' own pre-window reviews in the popularity counts.
        for p in priors:
            np.subtract.at(counts, [index[b] for b in p.business_ids if b in index], 1)

        rng = np.random.default_rng(_seed_for(g.group_id, seed))
        tops = {
            "random": cand[rng.permutation(len(cand))[:k]],
            "popularity": cand[np.argsort(-counts[cand], kind="stable")[:k]],
        }
        diag = {}
        for strategy in CONTENT_STRATEGIES:
            if strategy == "fairness_repeated":
                picks, history, weights = fairness_repeated(
                    scores, excluded, k, norm=normalization, return_weights=True
                )
                tops[strategy] = picks
                if diagnostics:
                    S = normalize(scores, excluded, normalization)
                    diag[strategy] = _repeated_diagnostics(picks, history, weights, S, excluded)
            else:
                tops[strategy] = AGGREGATORS[strategy](scores, excluded, k, norm=normalization)
        for tau in extra_taus:
            tops[f"fairness_repeated_tau{tau:g}"], _ = fairness_repeated(
                scores, excluded, k, norm=normalization, tau=tau
            )

        iu = np.triu_indices(len(tastes), k=1)
        cohesion = float((tastes @ tastes.T)[iu].mean())
        base = {
            "group_id": g.group_id,
            "size": g.size,
            "target_business_id": target,
            "window_start": g.window_start,
            "n_candidates": len(cand),
            "cohesion": cohesion,
            "target_prior_reviews": int(counts[index[target]]),
        }
        for strategy, top in tops.items():
            rows.append(
                {
                    **base,
                    "strategy": strategy,
                    **_list_metrics(top, target_idx, pct),
                    "top_k": [model.business_ids[i] for i in top],
                    **diag.get(strategy, {}),
                }
            )

    return pd.DataFrame(rows), drops


def add_buckets(results: pd.DataFrame) -> pd.DataFrame:
    per_group = results.drop_duplicates("group_id").set_index("group_id")
    median = per_group["cohesion"].median()
    cohesion_bucket = np.where(per_group["cohesion"] > median, "high", "low")
    # rank first so heavy ties on review count can't collapse the terciles.
    pop_bucket = pd.qcut(
        per_group["target_prior_reviews"].rank(method="first"), 3, labels=BUCKET_ORDER
    )
    buckets = pd.DataFrame(
        {
            "cohesion_bucket": pd.Categorical(cohesion_bucket, categories=BUCKET_ORDER, ordered=True),
            "popularity_bucket": pd.Categorical(pop_bucket, categories=BUCKET_ORDER, ordered=True),
        },
        index=per_group.index,
    )
    return results.join(buckets, on="group_id")


def _strategy_order(present) -> list[str]:
    """ALL_STRATEGIES order first, then any extra (sensitivity) strategies."""
    present = list(dict.fromkeys(present))
    return [s for s in ALL_STRATEGIES if s in present] + [s for s in present if s not in ALL_STRATEGIES]


def _ordered_strategies(results: pd.DataFrame) -> pd.DataFrame:
    strategies = _strategy_order(results["strategy"])
    return results.assign(
        strategy=pd.Categorical(results["strategy"], categories=strategies, ordered=True)
    )


def summarize(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = _ordered_strategies(results).groupby(by, observed=True)
    table = grouped[list(METRICS)].mean()
    table.insert(0, "n", grouped.size())
    return table


def _per_group(results: pd.DataFrame, column: str) -> pd.DataFrame:
    """group x strategy table of `column`, strategies in ALL_STRATEGIES order."""
    table = results.pivot(index="group_id", columns="strategy", values=column)
    return table[_strategy_order(table.columns)]


def _targets(results: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Target business_id per group, aligned with a _per_group index."""
    return results.drop_duplicates("group_id").set_index("group_id").loc[index, "target_business_id"].to_numpy()


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                         n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile CI of mean(values) under a cluster bootstrap: resample
    clusters (targets) with replacement, keeping all of a cluster's groups
    together. The same seed gives the same cluster draws for any `values`
    over the same clusters, so CIs of differences are paired.
    """
    codes, uniques = pd.factorize(clusters, sort=True)
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniques), size=(n_boot, len(uniques)))
    weights = np.stack([np.bincount(d, minlength=len(uniques)) for d in draws])
    boot = (weights @ sums) / (weights @ counts)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def main_table_with_ci(results: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    hit, ndcg = _per_group(results, "hit"), _per_group(results, "ndcg")
    clusters = _targets(results, hit.index)
    rows = []
    for s in hit.columns:
        rows.append(
            {
                "strategy": s,
                "n": len(hit),
                "hits": int(hit[s].sum()),
                "hit@10": hit[s].mean(),
                "hit_ci": _fmt_ci(*cluster_bootstrap_ci(hit[s].to_numpy(), clusters, n_boot, seed)),
                "ndcg@10": ndcg[s].mean(),
                "ndcg_ci": _fmt_ci(*cluster_bootstrap_ci(ndcg[s].to_numpy(), clusters, n_boot, seed)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def paired_tests(results: pd.DataFrame) -> pd.DataFrame:
    """McNemar exact test on hit@10 for every strategy pair: a two-sided
    binomial test of the discordant pairs against p = 0.5. ASSUMES GROUPS
    ARE INDEPENDENT, which they aren't (groups share targets and members),
    so these p-values are optimistic; prefer the cluster-bootstrap CIs."""
    hit = _per_group(results, "hit").astype(bool)
    rows = []
    for a, b in combinations(hit.columns, 2):
        a_only = int((hit[a] & ~hit[b]).sum())
        b_only = int((~hit[a] & hit[b]).sum())
        n = a_only + b_only
        p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
        rows.append({"a": a, "b": b, "a_hit_b_miss": a_only, "a_miss_b_hit": b_only, "p_mcnemar_indep": p})
    return pd.DataFrame(rows)


def bootstrap_vs(results: pd.DataFrame, reference: str = "average", metric: str = "hit",
                 n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    """Paired cluster-bootstrap (by target) 95% CI of
    metric(strategy) - metric(reference)."""
    table = _per_group(results, metric)
    clusters = _targets(results, table.index)
    rows = []
    for s in table.columns:
        if s == reference:
            continue
        diff = (table[s] - table[reference]).to_numpy(dtype=float)
        lo, hi = cluster_bootstrap_ci(diff, clusters, n_boot, seed)
        rows.append({"strategy": s, f"{metric}_diff_vs_{reference}": diff.mean(), "ci_lo": lo,
                     "ci_hi": hi, "ci_excludes_0": lo > 0 or hi < 0})
    return pd.DataFrame(rows).set_index("strategy")


def leak_by_popularity(results: pd.DataFrame, buckets=("low", "mid"),
                       n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    """Flagged vs unflagged hit@10 for each content strategy within target-
    popularity terciles, so target popularity can't confound the comparison."""
    rows = []
    for bucket in buckets:
        for flagged in (True, False):
            part = results[(results["popularity_bucket"] == bucket) & (results["target_leak"] == flagged)]
            if part.empty:
                continue
            hit = _per_group(part, "hit")
            clusters = _targets(part, hit.index)
            for s in [c for c in CONTENT_STRATEGIES if c in hit.columns]:
                rows.append(
                    {
                        "popularity_bucket": bucket,
                        "strategy": s,
                        "flagged": flagged,
                        "n": len(hit),
                        "n_targets": len(set(clusters)),
                        "hit@10": hit[s].mean(),
                        "ci": _fmt_ci(*cluster_bootstrap_ci(hit[s].to_numpy(), clusters, n_boot, seed)),
                    }
                )
    return pd.DataFrame(rows).set_index(["popularity_bucket", "strategy", "flagged"])


def modeled_diffs_by_cohesion(results: pd.DataFrame, n_boot: int = N_BOOTSTRAP,
                              seed: int = SEED) -> pd.DataFrame:
    """(strategy - average) on the modeled metrics for each non-average
    content strategy, per cohesion bucket, with cluster-bootstrap CIs."""
    rows = []
    for bucket in BUCKET_ORDER:
        part = results[results["cohesion_bucket"] == bucket]
        if part.empty:
            continue
        for metric in ("group_score", "worst_member", "fairness_gap"):
            diffs = bootstrap_vs(part, "average", metric, n_boot, seed)
            for s in [c for c in diffs.index if c not in ("random", "popularity")]:
                d = diffs.loc[s]
                rows.append(
                    {
                        "cohesion_bucket": bucket,
                        "metric": metric,
                        "diff": f"{s} - average",
                        "n": part["group_id"].nunique(),
                        "value": d[f"{metric}_diff_vs_average"],
                        "ci": _fmt_ci(d["ci_lo"], d["ci_hi"]),
                        "ci_excludes_0": d["ci_excludes_0"],
                    }
                )
    return pd.DataFrame(rows).set_index(["cohesion_bucket", "metric", "diff"])


def list_overlap(results: pd.DataFrame) -> pd.DataFrame:
    """Mean per-group Jaccard overlap of top-k lists, strategy x strategy."""
    lists = _per_group(results, "top_k")
    names = list(lists.columns)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for a, b in combinations(names, 2):
        j = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(lists[a], lists[b])]
        out.loc[a, b] = out.loc[b, a] = float(np.mean(j))
    return out


def target_leak_flags(groups: pd.DataFrame, train_reviews: pd.DataFrame,
                      embedded: dict[str, set[str]]) -> pd.Series:
    """group_id -> True if any member's train review of the target is one of
    the reviews whose text went into the target's embedding."""
    pairs = groups[["group_id", "members", "target_business_id"]].explode("members")
    pairs = pairs.rename(columns={"members": "user_id", "target_business_id": "business_id"})
    hits = pairs.merge(train_reviews[["user_id", "business_id", "review_id"]], on=["user_id", "business_id"])
    hits["leak"] = [rid in embedded.get(b, ()) for rid, b in zip(hits["review_id"], hits["business_id"])]
    flags = hits.groupby("group_id")["leak"].any()
    return flags.reindex(groups["group_id"], fill_value=False)


# --- fairness_repeated tau sensitivity (analysis only; tau = TAU is the
# headline and the default; nothing here selects a tau) ----------------------

SENSITIVITY_TAUS = (0.1, 0.3, 1.0, 3.0)
DIAGNOSTIC_DINNERS = (1, 2, 3, 5, 10)


def _repeated_name(tau: float) -> str:
    return "fairness_repeated" if tau == TAU else f"fairness_repeated_tau{tau:g}"


def repeated_diagnostics_table(results: pd.DataFrame, bucket: str = "low",
                               dinners=DIAGNOSTIC_DINNERS) -> pd.DataFrame:
    """Mean per-dinner diagnostics of the default fairness_repeated over the
    groups in one cohesion bucket."""
    rows = results[(results["strategy"] == "fairness_repeated") & (results["cohesion_bucket"] == bucket)]
    out = []
    for d in dinners:
        max_w = [r[d - 1] for r in rows["fr_max_weight"] if len(r) >= d]
        lowest = [r[d - 1] for r in rows["fr_top_is_lowest"] if len(r) >= d and r[d - 1] is not None]
        dictator = [r[d - 1] for r in rows["fr_dictator"] if len(r) >= d]
        out.append(
            {
                "dinner": d,
                "n_groups": len(max_w),
                "mean_max_member_weight": float(np.mean(max_w)),
                "top_weighted_is_lowest_cumulative": float(np.mean(lowest)) if lowest else float("nan"),
                "n_untied": len(lowest),
                "pick_is_top_member_own_best": float(np.mean(dictator)),
            }
        )
    return pd.DataFrame(out).set_index("dinner")


def tau_sensitivity(results: pd.DataFrame, taus=SENSITIVITY_TAUS,
                    n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    """Long table: per tau, hit@10 (+ CI, + diff vs average) over all groups,
    and modeled-metric differences vs average per cohesion bucket."""
    hit = _per_group(results, "hit")
    clusters = _targets(results, hit.index)
    diffs = bootstrap_vs(results, "average", "hit", n_boot, seed)
    rows = []
    for tau in taus:
        name = _repeated_name(tau)
        lo, hi = cluster_bootstrap_ci(hit[name].to_numpy(), clusters, n_boot, seed)
        rows.append({"tau": tau, "cohesion_bucket": "all", "metric": "hit@10", "value": hit[name].mean(),
                     "ci_lo": lo, "ci_hi": hi, "n": len(hit)})
        d = diffs.loc[name]
        rows.append({"tau": tau, "cohesion_bucket": "all", "metric": "hit@10 - average",
                     "value": d["hit_diff_vs_average"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "n": len(hit)})
    for bucket in ("low", "high"):
        part = results[results["cohesion_bucket"] == bucket]
        n = part["group_id"].nunique()
        for metric in ("group_score", "worst_member", "fairness_gap"):
            mdiffs = bootstrap_vs(part, "average", metric, n_boot, seed)
            for tau in taus:
                d = mdiffs.loc[_repeated_name(tau)]
                rows.append({"tau": tau, "cohesion_bucket": bucket, "metric": f"{metric} - average",
                             "value": d[f"{metric}_diff_vs_average"], "ci_lo": d["ci_lo"],
                             "ci_hi": d["ci_hi"], "n": n})
    table = pd.DataFrame(rows)
    table.insert(0, "analysis", "tau_sensitivity (not the headline; default tau=%g)" % TAU)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--i-mean-it", action="store_true", help="required for --split test")
    parser.add_argument("--sample", type=int, default=None, help="seeded sample of N groups")
    parser.add_argument("--normalization", choices=NORMS, default="rank",
                        help="'raw' is the raw-cosine ablation")
    parser.add_argument("--tau-sensitivity", action="store_true",
                        help="sensitivity analysis of fairness_repeated's tau (separate CSV)")
    args = parser.parse_args()
    if args.split == "test" and not args.i_mean_it:
        parser.error("refusing to evaluate on the test split without --i-mean-it")

    groups = pd.read_parquet(GROUPS)
    groups = groups[groups["split"] == args.split]
    if args.sample is not None and args.sample < len(groups):
        groups = groups.sample(n=args.sample, random_state=SEED)

    train = pd.read_parquet(
        TRAIN_REVIEWS, columns=["review_id", "user_id", "business_id", "stars", "date"]
    )
    model = ContentBasedCentered()

    extra_taus = (*[t for t in SENSITIVITY_TAUS if t != TAU], 1e6) if args.tau_sensitivity else ()
    results, drops = evaluate_groups(groups, train, model, normalization=args.normalization,
                                     extra_taus=extra_taus, diagnostics=args.tau_sensitivity)
    n_survive = results["group_id"].nunique() if len(results) else 0
    print(f"{args.split} groups: {len(groups)}  surviving: {n_survive}")
    for reason, n in drops.items():
        print(f"  dropped ({reason}): {n}")
    if not n_survive:
        raise SystemExit("no surviving groups")

    if args.tau_sensitivity:
        results = add_buckets(results)
        pd.set_option("display.width", 200)
        print("\n######## SENSITIVITY ANALYSIS: fairness_repeated tau "
              f"(headline/default tau = {TAU:g}; no tau is selected here) ########")
        print("\n== diagnosis: default tau, low-cohesion groups, per dinner ==")
        print(repeated_diagnostics_table(results, "low").round(4).to_string())
        print("\n== same, high-cohesion groups (for contrast) ==")
        print(repeated_diagnostics_table(results, "high").round(4).to_string())

        lists = _per_group(results, "top_k")
        huge, avg = lists["fairness_repeated_tau1e+06"], lists["average"]
        same = [list(a) == list(b) for a, b in zip(huge, avg)]
        same_set = [set(a) == set(b) for a, b in zip(huge, avg)]
        hits = _per_group(results, "hit")
        print(f"\n== sanity: tau=1e6 reproduces average's top-{K} exactly (same order) in "
              f"{sum(same)} of {len(same)} groups; same set in {sum(same_set)}; "
              f"hit@10 identical in {int((hits['fairness_repeated_tau1e+06'] == hits['average']).sum())} ==")

        table = tau_sensitivity(results)
        print(f"\n== tau sensitivity (cluster bootstrap by target, {N_BOOTSTRAP} resamples) ==")
        print(table.drop(columns="analysis").round(4).to_string(index=False))
        out = DATA / f"group_eval_{args.split}_tau_sensitivity.csv"
        table.to_csv(out, index=False)
        print(f"\nwrote {out}")
        raise SystemExit(0)

    # Imported here: embed pulls in sentence-transformers.
    from src.embed import embedded_review_ids

    leak = target_leak_flags(groups, train, embedded_review_ids("chunked"))
    results = add_buckets(results)
    results["target_leak"] = results["group_id"].map(leak).astype(bool)

    n_targets = results["target_business_id"].nunique()
    boot = f"cluster bootstrap over {n_targets} targets, {N_BOOTSTRAP} resamples"
    pd.set_option("display.width", 200)
    print(f"\n== {args.split}, normalization={args.normalization}, k={K} ({boot}) ==")
    print(main_table_with_ci(results).round(4).to_string())
    print(summarize(results, ["strategy"]).round(4).to_string())
    print(f"\n== hit@10 differences vs average ({boot}) ==")
    print(bootstrap_vs(results).round(4).to_string())
    print("\n== paired McNemar exact tests on hit@10 (ASSUME INDEPENDENT GROUPS; optimistic) ==")
    print(paired_tests(results).round(4).to_string(index=False))
    print(f"\n== mean top-{K} Jaccard overlap ==")
    print(list_overlap(results).round(3).to_string())
    print("\n== by cohesion (split at median) ==")
    print(summarize(results, ["cohesion_bucket", "strategy"]).round(4).to_string())
    print("\n== modeled-metric differences vs average by cohesion "
          f"(cluster bootstrap by target within bucket, {N_BOOTSTRAP} resamples) ==")
    print(modeled_diffs_by_cohesion(results).round(4).to_string())
    print("\n== by target pre-window popularity (terciles) ==")
    print(summarize(results, ["popularity_bucket", "strategy"]).round(4).to_string())

    flagged = results.loc[results["target_leak"], "group_id"].nunique()
    print(f"\n== target-embedding leak: {flagged} of {n_survive} surviving groups flagged ==")
    leak_table = pd.DataFrame(
        {
            "hit@10 all": _per_group(results, "hit").mean(),
            "hit@10 unflagged": _per_group(results[~results["target_leak"]], "hit").mean(),
            "hit@10 flagged": _per_group(results[results["target_leak"]], "hit").mean(),
        }
    )
    print(leak_table.round(4).to_string())
    print("\n== leak check within low/mid target-popularity terciles "
          f"(cluster bootstrap by target within cell, {N_BOOTSTRAP} resamples) ==")
    print(leak_by_popularity(results).round(4).to_string())

    suffix = "" if args.normalization == "rank" else f"_{args.normalization}"
    suffix += f"_sample{args.sample}" if args.sample is not None else ""
    out = DATA / f"group_eval_{args.split}{suffix}.csv"
    lists_out = DATA / f"group_eval_{args.split}{suffix}_lists.parquet"
    results[["group_id", "strategy", "top_k"]].to_parquet(lists_out, index=False)
    results.drop(columns="top_k").to_csv(out, index=False)
    print(f"\nwrote {out}\nwrote {lists_out}")
