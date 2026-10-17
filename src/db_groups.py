"""Group scoring in Postgres/pgvector for the evaluated configuration:
ContentBasedCentered embeddings + per-member rank normalization.

    python -m src.db_groups load-taste
    python -m src.db_groups check

load-taste: applies db/groups.sql and, in one transaction, COPYs
- restaurant_centered: the centered catalog ContentBasedCentered uses
  (embeddings minus the catalog mean, L2-normalized);
- user_taste: ContentBasedCentered taste vectors fit on all of TRAIN_REVIEWS
  for every user with 5+ reviews (users.parquet);
- user_reviewed: the (user, restaurant) pairs those users reviewed in train.

These are serving profiles (a user's whole train history). They are not
group_eval's per-group profiles, which use only reviews before each group's
window, so SQL results are not comparable to the group_eval tables.

check: for N_CHECK seeded val groups, compares group_candidates() with a numpy
reference on the same centered embeddings, taste vectors and exclusions.
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import interfaces
from src.aggregate import STRATEGIES as AGGREGATORS
from src.db import get_conn
from src.groups import GROUPS
from src.recommend import ContentBasedCentered

GROUPS_SQL = "db/groups.sql"
K = 10
N_CHECK = 50
SQL_AGGREGATES = {"average": "AVG", "least_misery": "MIN"}


def apply_schema(conn) -> None:
    with open(GROUPS_SQL, encoding="utf-8") as f:
        conn.execute(f.read())


def fit_taste(model: ContentBasedCentered) -> dict[str, np.ndarray]:
    """ContentBasedCentered taste vectors on train for users with 5+ reviews."""
    users = set(pd.read_parquet(interfaces.USERS, columns=["user_id"])["user_id"])
    train = pd.read_parquet(interfaces.TRAIN_REVIEWS, columns=["user_id", "business_id", "stars"])
    model.fit(train[train["user_id"].isin(users)])
    return model.taste


def reviewed_pairs() -> pd.DataFrame:
    users = set(pd.read_parquet(interfaces.USERS, columns=["user_id"])["user_id"])
    train = pd.read_parquet(interfaces.TRAIN_REVIEWS, columns=["user_id", "business_id"])
    return train[train["user_id"].isin(users)].drop_duplicates()


def load_taste() -> None:
    model = ContentBasedCentered()
    taste = fit_taste(model)
    pairs = reviewed_pairs()
    with get_conn() as conn:  # one transaction: commits on success, rolls back on error
        apply_schema(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE restaurant_centered, user_taste, user_reviewed")
            with cur.copy("COPY restaurant_centered (business_id, embedding) FROM STDIN") as copy:
                for business_id, vec in zip(model.business_ids, model.matrix):
                    copy.write_row((business_id, vec.astype(np.float32)))
            with cur.copy("COPY user_taste (user_id, embedding) FROM STDIN") as copy:
                for user_id, vec in taste.items():
                    copy.write_row((user_id, vec.astype(np.float32)))
            with cur.copy("COPY user_reviewed (user_id, business_id) FROM STDIN") as copy:
                for row in pairs.itertuples(index=False):
                    copy.write_row((row.user_id, row.business_id))
    print(f"loaded {len(model.business_ids)} centered restaurants, {len(taste)} taste vectors, "
          f"{len(pairs)} reviewed pairs")


def build_group_query(strategy: str, exclude_reviewed: bool = True, open_only: bool = False) -> str:
    """One query for a group's top k.

    Per member, cosine 1 - (taste <=> embedding) on restaurant_centered, then
    percent_rank() over the candidate set (after exclusions), then AVG/MIN per
    restaurant. open_only assumes restaurants.is_open exists; group_candidates
    checks that before using it.
    """
    if strategy not in SQL_AGGREGATES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(SQL_AGGREGATES)}")
    filters = []
    if exclude_reviewed:
        filters.append(
            "NOT EXISTS (SELECT 1 FROM user_reviewed v"
            " WHERE v.business_id = c.business_id AND v.user_id = ANY(%(members)s))"
        )
    if open_only:
        filters.append(
            "EXISTS (SELECT 1 FROM restaurants o WHERE o.business_id = c.business_id AND o.is_open = 1)"
        )
    where = ("\n  WHERE " + "\n    AND ".join(filters)) if filters else ""
    return f"""
WITH members AS (
  SELECT user_id, embedding FROM user_taste WHERE user_id = ANY(%(members)s)
),
candidates AS (
  SELECT c.business_id, c.embedding FROM restaurant_centered c{where}
),
ranked AS (
  SELECT c.business_id,
         percent_rank() OVER (PARTITION BY m.user_id ORDER BY 1 - (m.embedding <=> c.embedding)) AS pct
  FROM members m CROSS JOIN candidates c
)
SELECT business_id, {SQL_AGGREGATES[strategy]}(pct) AS score, count(*) AS n_members
FROM ranked
GROUP BY business_id
ORDER BY score DESC, business_id COLLATE "C"
LIMIT %(k)s
"""


def _has_column(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns"
            " WHERE table_schema = current_schema() AND table_name = %s AND column_name = %s",
            (table, column),
        )
        return cur.fetchone() is not None


def group_candidates(conn, member_ids, strategy: str = "least_misery", k: int = K,
                     exclude_reviewed: bool = True, open_only: bool = False) -> list[tuple[str, float]]:
    """Top-k (business_id, group score) for a group, best first.

    The group score is the AVG (average) or MIN (least_misery) over members of
    each member's percent_rank of the restaurant among the candidates: the
    evaluated configuration (centered embeddings, rank normalization).
    """
    members = sorted(set(member_ids))
    sql = build_group_query(strategy, exclude_reviewed, open_only)  # validates strategy first
    if open_only and not _has_column(conn, "restaurants", "is_open"):
        raise NotImplementedError(
            "open_only needs a restaurants.is_open column, which db/schema.sql (B's) doesn't have"
        )
    with conn.cursor() as cur:
        cur.execute(sql, {"members": members, "k": k})
        rows = cur.fetchall()
    if not rows or rows[0][2] != len(members):
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM user_taste WHERE user_id = ANY(%s)", (members,))
            found = {r[0] for r in cur.fetchall()}
        missing = sorted(set(members) - found)
        if missing:
            raise ValueError(f"no taste vector for members {missing}; run load-taste")
    return [(business_id, float(score)) for business_id, score, _ in rows]


def percent_rank(scores: np.ndarray) -> np.ndarray:
    """Postgres percent_rank() per row: (rank - 1) / (n - 1) with tied values
    sharing the LOWEST rank (and 0 for a single row). src.aggregate's "rank"
    norm gives ties their AVERAGE rank instead; the two agree when there are
    no ties."""
    n = scores.shape[1]
    if n == 1:
        return np.zeros_like(scores, dtype=float)
    return (rankdata(scores, method="min", axis=1) - 1) / (n - 1)


def reference_candidates(taste, matrix, index, member_ids, strategy="least_misery", k=K,
                         exclude=frozenset()):
    """numpy reference for group_candidates: same cosines, percent_rank over
    the candidates, AVG/MIN, ties by business index (= business_id byte order).
    Returns (top-k indices, group score per catalog index; NaN if excluded)."""
    if strategy not in SQL_AGGREGATES:
        raise ValueError(f"unknown strategy {strategy!r}")
    scores = np.stack([matrix @ taste[u] for u in sorted(set(member_ids))]).astype(np.float64)
    excluded = np.zeros(matrix.shape[0], dtype=bool)
    excluded[[index[b] for b in exclude if b in index]] = True
    cand = np.flatnonzero(~excluded)
    pct = percent_rank(scores[:, cand])
    group = pct.mean(axis=0) if strategy == "average" else pct.min(axis=0)
    full = np.full(matrix.shape[0], np.nan)
    full[cand] = group
    order = cand[np.argsort(-group, kind="stable")]
    return order[:k], full


def lists_match(sql_ids, ref_ids, group_score, index, tol) -> tuple[bool, bool]:
    """(exact, up_to_near_ties): the second allows positions to differ only
    where the reference group scores are within tol."""
    if list(sql_ids) == list(ref_ids):
        return True, True
    if len(sql_ids) != len(ref_ids):
        return False, False
    gaps = [abs(group_score[index[a]] - group_score[index[b]]) for a, b in zip(sql_ids, ref_ids)]
    return False, max(gaps) <= tol


def _ms(values, q):
    return float(np.percentile(np.asarray(values) * 1000, q))


def check(n_groups: int = N_CHECK) -> None:
    model = ContentBasedCentered()
    taste = fit_taste(model)
    matrix, business_ids, index = model.matrix, model.business_ids, model.index
    groups = pd.read_parquet(GROUPS)
    groups = groups[groups["split"] == "val"].sample(n=n_groups, random_state=interfaces.SEED)
    reviewed = reviewed_pairs().groupby("user_id")["business_id"].agg(set)

    with get_conn() as conn:
        for strategy in SQL_AGGREGATES:
            exact = near = same_as_aggregate = 0
            sql_ms, np_ms = [], []
            group_candidates(conn, groups["members"].iloc[0], strategy)  # warm-up
            for members in groups["members"]:
                exclude = set().union(*(reviewed.get(u, set()) for u in members))

                t0 = time.perf_counter()
                sql = group_candidates(conn, members, strategy, K)
                sql_ms.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                top, group_score = reference_candidates(taste, matrix, index, members, strategy, K, exclude)
                np_ms.append(time.perf_counter() - t0)

                ref_ids = [business_ids[i] for i in top]
                # A near-tie between the two engines' cosines can move a member
                # by one rank, i.e. 1/(n-1) in percent_rank.
                one_rank = 1 / (int(np.isfinite(group_score).sum()) - 1)
                e, t = lists_match([b for b, _ in sql], ref_ids, group_score, index, tol=one_rank)
                exact += e
                near += t

                # How often the documented tie difference matters vs src.aggregate's "rank".
                scores = np.stack([matrix @ taste[u] for u in sorted(set(members))])
                excluded = np.isnan(group_score)
                agg_ids = [business_ids[i] for i in AGGREGATORS[strategy](scores, excluded, K, norm="rank")]
                same_as_aggregate += agg_ids == ref_ids

            n = len(groups)
            print(f"{strategy}: SQL top-{K} == reference in {exact}/{n} (exact order), "
                  f"{near}/{n} allowing one-rank near-ties; "
                  f"reference == src.aggregate norm='rank' in {same_as_aggregate}/{n}")
            print(f"  latency ms  SQL p50 {_ms(sql_ms, 50):.2f} p95 {_ms(sql_ms, 95):.2f} | "
                  f"numpy p50 {_ms(np_ms, 50):.2f} p95 {_ms(np_ms, 95):.2f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["load-taste", "check"])
    args = parser.parse_args(argv)
    {"load-taste": load_taste, "check": check}[args.command]()


if __name__ == "__main__":
    main()
