"""Load a hosted database (e.g. Neon) for the group API's DATA_SOURCE=db mode.

    python -m src.deploy_load --database-url URL --demo-users 500

Steps, all against --database-url (nothing is written under deploy/ or to git):
1. Apply db/schema.sql (Person B's, read only) and db/groups.sql (idempotent).
2. restaurants: src.db.load(), the same loader as `python -m src.db load`.
3. Pick --demo-users val users from val groups, alternating high- and low-cohesion
   groups (seeded) so both halves are represented; fill up with random val users
   if needed. Write their ContentBasedDebiased taste vectors (user_taste), the
   model's hub vector (model_arrays), their top-5 train restaurants with stars
   (user_profiles), the IDs of everything they reviewed in train (user_visited),
   and --sample-groups of the val groups drawn entirely from them (sample_groups,
   half high, half low cohesion).
4. Print row counts and on-disk size per table.

No review text is loaded. Taste vectors come from the API's own fitted model
(the taste cache, or a fit), so DB mode serves exactly what files mode serves
for these users.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.interfaces import SEED

SCHEMA_SQL = Path("db/schema.sql")
GROUPS_SQL = Path("db/groups.sql")
GROUP_TABLES = ("user_taste", "model_arrays", "user_profiles", "user_visited", "sample_groups")
PROFILE_SIZE = 5


def cohesion_of(model, members) -> float:
    from src.api.routes_group import _cohesion

    return _cohesion(np.stack([model.taste[m] for m in members]))


def select_demo(model, val_groups: pd.DataFrame, val_users, n_users: int, n_groups: int,
                seed: int = SEED) -> tuple[list[str], pd.DataFrame]:
    """(demo user_ids, sample_groups). Groups are bucketed high/low at the median
    cohesion of all eligible val groups (the files-mode rule in /users/sample).
    High and low groups are taken alternately, in seeded random order, while their
    members fit in n_users; remaining slots are filled with random val users."""
    groups = val_groups[val_groups["members"].map(lambda ms: all(m in model.taste for m in ms))].copy()
    groups["cohesion"] = groups["members"].map(lambda ms: cohesion_of(model, ms))
    median = float(groups["cohesion"].median())
    groups["cohesion_bucket"] = np.where(groups["cohesion"] > median, "high", "low")
    rng = np.random.default_rng(seed)
    queues = {b: list(rng.permutation(groups.index[groups["cohesion_bucket"] == b].to_numpy()))
              for b in ("high", "low")}

    users: list[str] = []
    in_users: set[str] = set()
    taken = {"high": [], "low": []}
    turn = 0
    while len(users) < n_users and (queues["high"] or queues["low"]):
        bucket = ("high", "low")[turn % 2]
        turn += 1
        if not queues[bucket]:
            continue
        idx = queues[bucket].pop(0)
        new = [m for m in groups.at[idx, "members"] if m not in in_users]
        if len(users) + len(new) > n_users:
            continue
        users += new
        in_users.update(new)
        taken[bucket].append(idx)
    pool = sorted(u for u in val_users if u in model.taste and u not in in_users)
    if len(users) < n_users and pool:
        extra = rng.choice(pool, size=min(n_users - len(users), len(pool)), replace=False).tolist()
        users += extra
    per_bucket = n_groups // 2
    chosen = taken["high"][:per_bucket] + taken["low"][: n_groups - min(per_bucket, len(taken["high"]))]
    sample = groups.loc[chosen, ["group_id", "members", "cohesion", "cohesion_bucket"]].reset_index(drop=True)
    sample["members"] = sample["members"].map(list)
    return users, sample


def build_tables(model, user_reviews: dict[str, pd.DataFrame], users: list[str],
                 sample_groups: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The group tables for `users`, as DataFrames in db/groups.sql's column order."""
    taste = pd.DataFrame({"user_id": users, "embedding": [model.taste[u].astype(np.float32) for u in users]})
    profiles, visited = [], []
    for u in users:
        reviews = user_reviews.get(u, pd.DataFrame(columns=["business_id", "stars"]))
        top = reviews.sort_values(["stars", "business_id"], ascending=[False, True]).head(PROFILE_SIZE)
        profiles += [(u, rank, b, float(s)) for rank, (b, s) in
                     enumerate(zip(top["business_id"], top["stars"]), 1)]
        visited += [(u, b) for b in sorted(set(reviews["business_id"]))]
    arrays = []
    if hasattr(model, "hub"):
        arrays.append((("hub", model.name, len(model.hub), model.hub.astype(np.float32))))
    return {
        "user_taste": taste,
        "model_arrays": pd.DataFrame(arrays, columns=["name", "model", "n", "vals"]),
        "user_profiles": pd.DataFrame(profiles, columns=["user_id", "rank", "business_id", "stars"]),
        "user_visited": pd.DataFrame(visited, columns=["user_id", "business_id"]),
        "sample_groups": sample_groups[["group_id", "members", "cohesion", "cohesion_bucket"]],
    }


def write_tables(conn, tables: dict[str, pd.DataFrame]) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE " + ", ".join(GROUP_TABLES))
        for name in GROUP_TABLES:
            df = tables[name]
            with cur.copy(f"COPY {name} ({', '.join(df.columns)}) FROM STDIN") as copy:
                for row in df.itertuples(index=False):
                    copy.write_row([v.tolist() if isinstance(v, np.ndarray) and name == "model_arrays" else v
                                    for v in row])


def table_report(conn, names) -> list[tuple[str, int, int]]:
    out = []
    with conn.cursor() as cur:
        for name in names:
            cur.execute(f"SELECT count(*), pg_total_relation_size('{name}') FROM {name}")
            count, size = cur.fetchone()
            out.append((name, int(count), int(size)))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", default=None,
                        help="target database (default: DATABASE_URL from the environment / .env)")
    parser.add_argument("--demo-users", type=int, default=500)
    parser.add_argument("--sample-groups", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    if args.database_url:
        os.environ["DATABASE_URL"] = args.database_url  # src.db.get_conn reads it; load_dotenv won't override
    if not os.getenv("DATABASE_URL"):
        parser.error("no --database-url and no DATABASE_URL")

    from src import db
    from src.api import routes_group as rg

    model = rg.get_recommender()  # files mode: the API's own fitted model
    user_reviews = rg.get_user_reviews()
    users, sample = select_demo(model, rg.get_val_groups(), rg.get_val_users(), args.demo_users,
                                args.sample_groups, args.seed)
    tables = build_tables(model, user_reviews, users, sample)
    print(f"demo users: {len(users)}; sample groups: {len(sample)} "
          f"({(sample['cohesion_bucket'] == 'high').sum()} high, {(sample['cohesion_bucket'] == 'low').sum()} low)")

    with db.get_conn(connect_timeout=15) as conn:
        conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute(GROUPS_SQL.read_text(encoding="utf-8"))
    db.load()  # restaurants, via Person B's loader
    with db.get_conn(connect_timeout=15) as conn:  # one transaction for the group tables
        write_tables(conn, tables)
    with db.get_conn(connect_timeout=15) as conn:
        report = table_report(conn, ("restaurants", *GROUP_TABLES))
    print(f"\n{'table':15s} {'rows':>8s} {'size':>10s}")
    for name, count, size in report:
        print(f"{name:15s} {count:8d} {size / 1024:8.0f} KB")
    print(f"{'total':15s} {'':8s} {sum(s for *_, s in report) / 1024 / 1024:8.1f} MB")


if __name__ == "__main__":
    main()
