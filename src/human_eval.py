"""Blind pairwise human evaluation (Person B): least_misery vs. average, rank normalization.

    python -m src.human_eval --select-groups   # once: 12 fixed test-split groups -> data/human_eval/groups.csv
    python -m src.human_eval                   # print the analysis of data/human_eval/responses.csv
    python -m src.human_eval --log NOTE        # ... and append a row to results/human_eval.md

The protocol is docs/human_eval_protocol.md. Responses are collected by src/pages/3_Blind_Eval.py,
which runs locally only; everything under data/ stays out of git.

Groups: the test split of groups.parquet, restricted to groups whose members all have a taste
profile in the API's served model (so /group/compare accepts them). Cohesion is the mean pairwise
cosine of those taste vectors, as in the API's /group/users/sample; high = above the median over
eligible test groups, low = at or below it. 6 per bucket, seeded with SEED.

Randomization: each participant's trial order and per-trial left/right assignment come from an RNG
seeded by a hash of their participant code, so a session can be resumed and audited; the page
records left_strategy with every response.

Analysis: least_misery's preference rate among judgments with a preference (no-preference
excluded), Wilson 95% CI, exact two-sided binomial test against 0.5; the same per cohesion bucket;
the no-preference rate; n participants and n judgments.
"""

import argparse
import csv
import hashlib
import math
import re
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.interfaces import SEED

STRATEGIES = ("least_misery", "average")  # the first is the one whose preference rate is reported
NORM = "rank"
N_PER_BUCKET = 6
BUCKETS = ("high", "low")

CHOICES = ("List 1", "List 2", "No preference")
NO_PREFERENCE = "none"  # chose_strategy for a "No preference" answer

DIR = Path("data/human_eval")
GROUPS_CSV = DIR / "groups.csv"
RESPONSES_CSV = DIR / "responses.csv"
RESPONSE_COLUMNS = (
    "participant_code", "group_id", "cohesion_bucket", "left_strategy", "choice", "chose_strategy",
    "reason", "timestamp",
)
CODE_RE = re.compile(r"^[A-Za-z0-9_-]{2,32}$")

RESULTS_PATH = Path("results/human_eval.md")
RESULTS_HEADER = (
    "# Human evaluation results\n\n"
    "Note: small convenience sample. Written only by `python -m src.human_eval --log`. Blind pairwise preference, "
    "least_misery vs. average (norm=rank, k=10), 12 fixed test-split groups (6 high / 6 low cohesion); "
    "protocol in docs/human_eval_protocol.md. Rates are least_misery's share of judgments with a "
    "preference (no-preference excluded), with Wilson 95% CIs and exact two-sided binomial p-values vs. 0.5.\n\n"
    "| date | n_participants | n_judgments | no-preference rate | least_misery rate [95% CI] | p | "
    "high cohesion: rate [95% CI] (n), p | low cohesion: rate [95% CI] (n), p | note |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


# ---------- groups ----------


def cohesion(tastes: np.ndarray) -> float:
    """Mean pairwise dot product of (L2-normalized) member taste vectors."""
    pairs = list(combinations(range(len(tastes)), 2))
    return float(np.mean([tastes[i] @ tastes[j] for i, j in pairs]))


def select_groups(groups: pd.DataFrame, taste: dict, n_per_bucket: int = N_PER_BUCKET,
                  seed: int = SEED) -> pd.DataFrame:
    """n_per_bucket seeded test-split groups above the cohesion median and n_per_bucket at or below it.

    groups: groups.parquet (group_id, members, split, ...); taste: user_id -> taste vector.
    Returns group_id, cohesion_bucket, cohesion, n_members, members (";"-joined), high first, each
    bucket sorted by group_id.
    """
    test = groups[groups["split"] == "test"]
    test = test[test["members"].map(lambda ms: all(m in taste for m in ms))]
    test = test.sort_values("group_id").reset_index(drop=True)
    if test.empty:
        raise ValueError("no test-split group has every member in the taste profiles")
    values = test["members"].map(lambda ms: cohesion(np.stack([taste[m] for m in ms])))
    median = float(values.median())
    rng = np.random.default_rng(seed)
    picked = []
    for bucket, mask in (("high", values > median), ("low", values <= median)):
        pool = test[mask]
        if len(pool) < n_per_bucket:
            raise ValueError(f"only {len(pool)} {bucket}-cohesion test groups; need {n_per_bucket}")
        rows = pool.iloc[np.sort(rng.choice(len(pool), size=n_per_bucket, replace=False))]
        picked.append(pd.DataFrame({
            "group_id": rows["group_id"].to_numpy(),
            "cohesion_bucket": bucket,
            "cohesion": values[rows.index].round(4).to_numpy(),
            "n_members": rows["members"].map(len).to_numpy(),
            "members": rows["members"].map(lambda ms: ";".join(ms)).to_numpy(),
        }))
    return pd.concat(picked, ignore_index=True)


def load_groups(path: Path | None = None) -> pd.DataFrame:
    path = path or GROUPS_CSV
    groups = pd.read_csv(path, dtype={"group_id": str})
    groups["members"] = groups["members"].str.split(";")
    return groups


# ---------- randomization ----------


def participant_seed(participant_code: str) -> int:
    return int.from_bytes(hashlib.sha256(participant_code.encode("utf-8")).digest()[:8], "big")


def trial_plan(participant_code: str, group_ids: list[str]) -> list[dict]:
    """The participant's trials: every group once, in a random order, each with a random
    left_strategy (shown as "List 1"). Deterministic per participant code."""
    rng = np.random.default_rng(participant_seed(participant_code))
    order = rng.permutation(len(group_ids))
    lefts = rng.integers(0, 2, size=len(group_ids))
    return [{"group_id": group_ids[i], "left_strategy": STRATEGIES[left]} for i, left in zip(order, lefts)]


def right_strategy(left_strategy: str) -> str:
    return STRATEGIES[1] if left_strategy == STRATEGIES[0] else STRATEGIES[0]


def chose_strategy(choice: str, left_strategy: str) -> str:
    """The strategy behind a choice: List 1 is left_strategy, List 2 the other one."""
    if choice == "List 1":
        return left_strategy
    if choice == "List 2":
        return right_strategy(left_strategy)
    if choice == "No preference":
        return NO_PREFERENCE
    raise ValueError(f"choice must be one of {CHOICES}, got {choice!r}")


def valid_code(code: str) -> bool:
    return bool(CODE_RE.match(code))


# ---------- responses ----------


def response_row(participant_code: str, group_id: str, cohesion_bucket: str, left_strategy: str,
                 choice: str, reason: str = "", timestamp: str | None = None) -> dict:
    return {
        "participant_code": participant_code,
        "group_id": group_id,
        "cohesion_bucket": cohesion_bucket,
        "left_strategy": left_strategy,
        "choice": choice,
        "chose_strategy": chose_strategy(choice, left_strategy),
        "reason": reason.strip(),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_response(row: dict, path: Path | None = None) -> None:
    path = path or RESPONSES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESPONSE_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def load_responses(path: Path | None = None) -> pd.DataFrame:
    path = path or RESPONSES_CSV
    if not path.exists():
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def answered_groups(participant_code: str, path: Path | None = None) -> set[str]:
    responses = load_responses(path)
    return set(responses.loc[responses["participant_code"] == participant_code, "group_id"])


# ---------- analysis ----------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes out of n; (nan, nan) when n == 0."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def preference(judgments: pd.DataFrame) -> dict:
    """least_misery's preference rate among judgments with a preference."""
    from scipy.stats import binomtest

    decided = judgments[judgments["chose_strategy"] != NO_PREFERENCE]
    n = len(decided)
    k = int((decided["chose_strategy"] == STRATEGIES[0]).sum())
    return {
        "k": k,
        "n": n,
        "rate": k / n if n else math.nan,
        "ci": wilson(k, n),
        "p": binomtest(k, n, 0.5, alternative="two-sided").pvalue if n else math.nan,
    }


def analyze(responses: pd.DataFrame) -> dict:
    unknown = set(responses["chose_strategy"]) - {*STRATEGIES, NO_PREFERENCE}
    if unknown:
        raise ValueError(f"unexpected chose_strategy values: {sorted(unknown)}")
    n = len(responses)
    return {
        "n_participants": int(responses["participant_code"].nunique()),
        "n_judgments": n,
        "no_preference_rate": float((responses["chose_strategy"] == NO_PREFERENCE).mean()) if n else math.nan,
        "overall": preference(responses),
        "by_bucket": {b: preference(responses[responses["cohesion_bucket"] == b]) for b in BUCKETS},
    }


def _fmt_pref(pref: dict) -> str:
    lo, hi = pref["ci"]
    return f"{pref['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({pref['k']}/{pref['n']}), p={pref['p']:.3g}"


def report(result: dict) -> str:
    lines = [
        f"participants: {result['n_participants']}  judgments: {result['n_judgments']}",
        f"no-preference rate: {result['no_preference_rate']:.3f}",
        f"{STRATEGIES[0]} preferred over {STRATEGIES[1]} (no-preference excluded): {_fmt_pref(result['overall'])}",
    ]
    lines += [f"  {b} cohesion: {_fmt_pref(result['by_bucket'][b])}" for b in BUCKETS]
    return "\n".join(lines)


def results_row(result: dict, note: str, today: str | None = None) -> str:
    overall = result["overall"]
    lo, hi = overall["ci"]
    buckets = []
    for b in BUCKETS:
        pref = result["by_bucket"][b]
        blo, bhi = pref["ci"]
        buckets.append(f"{pref['rate']:.3f} [{blo:.3f}, {bhi:.3f}] ({pref['n']}), {pref['p']:.3g}")
    return (
        f"| {today or date.today().isoformat()} | {result['n_participants']} | {result['n_judgments']} "
        f"| {result['no_preference_rate']:.3f} | {overall['rate']:.3f} [{lo:.3f}, {hi:.3f}] "
        f"| {overall['p']:.3g} | {buckets[0]} | {buckets[1]} | {note} |\n"
    )


def append_result(result: dict, note: str, path: Path | None = None) -> None:
    path = path or RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(RESULTS_HEADER)
        f.write(results_row(result, note))


# ---------- CLI ----------


def _served_taste() -> dict:
    """The API's served model's taste vectors (the same profiles /group/compare uses)."""
    from src.api.routes_group import get_recommender

    return get_recommender().taste


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--select-groups", action="store_true",
                        help=f"select the 12 fixed test-split groups and write {GROUPS_CSV}")
    parser.add_argument("--log", metavar="NOTE", default=None, help=f"append a row to {RESULTS_PATH}")
    parser.add_argument("--responses", type=Path, default=None, help=f"default {RESPONSES_CSV}")
    args = parser.parse_args(argv)

    if args.select_groups:
        if GROUPS_CSV.exists():
            parser.error(f"{GROUPS_CSV} already exists; the groups are fixed. Delete it only if no session has run.")
        from src.groups import GROUPS

        selected = select_groups(pd.read_parquet(GROUPS), _served_taste())
        DIR.mkdir(parents=True, exist_ok=True)
        selected.to_csv(GROUPS_CSV, index=False)
        print(f"wrote {GROUPS_CSV}: {len(selected)} groups")
        print(selected.drop(columns="members").to_string(index=False))
        return

    responses = load_responses(args.responses)
    if responses.empty:
        parser.error(f"no responses in {args.responses or RESPONSES_CSV}")
    result = analyze(responses)
    print(report(result))
    if args.log is not None:
        append_result(result, args.log)
        print(f"appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
