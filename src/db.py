"""Load restaurants + embeddings into Postgres/pgvector, and sanity-check the load.

    python -m src.db load
    python -m src.db check
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from src import interfaces

COLUMNS = [
    "business_id",
    "name",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "price_level",
    "categories",
    "attributes",
    "hours",
    "is_open",
    "embedding",
]
N_EMBEDDING_CHECKS = 20
EMBEDDING_TOLERANCE = 1e-6


def get_conn(connect_timeout: int | None = None) -> psycopg.Connection:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set; copy .env.example to .env")
    conn = psycopg.connect(url, **({} if connect_timeout is None else {"connect_timeout": connect_timeout}))
    register_vector(conn)
    return conn


def _missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def parse_price_level(value) -> int | None:
    """RestaurantsPriceRange2 -> 1-4, or None. Handles 2, "2", "'2'", "u'2'", "None" and NaN."""
    if _missing(value):
        return None
    text = str(value).strip().removeprefix("u").strip("'\"")
    if text in ("", "None"):
        return None
    level = int(text)
    if level not in (1, 2, 3, 4):
        raise ValueError(f"price level out of range: {value!r}")
    return level


def parse_categories(value) -> list[str] | None:
    if _missing(value):
        return None
    return [c.strip() for c in str(value).split(",") if c.strip()]


def _as_dict(value, business_id: str | None = None) -> dict | None:
    """Parquet struct or JSON string -> dict without the keys this business doesn't have, or None.

    prep.py writes hours with to_json(), so it arrives as a str; attributes arrives as a struct (dict).
    Null, "", "null" and dicts with no non-null keys all become None.
    """
    if _missing(value):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"business_id {business_id}: malformed JSON {value[:80]!r} ({e})") from e
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"business_id {business_id}: expected a JSON object, got {type(value).__name__}")
    return {k: v for k, v in dict(value).items() if v is not None} or None


def _rows() -> list[tuple]:
    restaurants = pd.read_parquet(interfaces.RESTAURANTS)
    emb, business_ids = interfaces.load_embeddings()
    row_of = {bid: i for i, bid in enumerate(business_ids)}

    missing = [bid for bid in restaurants["business_id"] if bid not in row_of]
    assert not missing, f"{len(missing)} restaurants have no embedding, e.g. {missing[:3]}"
    extra = len(row_of) - restaurants["business_id"].nunique()
    if extra:
        print(f"note: {extra} embeddings have no restaurant row and are skipped")

    if "hours" not in restaurants.columns:
        print("warning: restaurants.parquet has no hours column; loading hours as NULL (re-run src/prep.py)")
    if "is_open" not in restaurants.columns:
        print("warning: restaurants.parquet has no is_open column; loading is_open as NULL (re-run src/prep.py)")

    rows = []
    for r in restaurants.to_dict("records"):
        attributes = _as_dict(r.get("attributes"), r["business_id"])
        hours = _as_dict(r.get("hours"), r["business_id"])
        rows.append(
            (
                r["business_id"],
                r["name"],
                None if _missing(r["latitude"]) else float(r["latitude"]),
                None if _missing(r["longitude"]) else float(r["longitude"]),
                None if _missing(r["stars"]) else float(r["stars"]),
                None if _missing(r["review_count"]) else int(r["review_count"]),
                parse_price_level((attributes or {}).get("RestaurantsPriceRange2")),
                parse_categories(r["categories"]),
                None if attributes is None else Jsonb(attributes),
                None if hours is None else Jsonb(hours),
                None if _missing(r.get("is_open")) else bool(r["is_open"]),
                emb[row_of[r["business_id"]]],
            )
        )
    return rows


def load() -> None:
    rows = _rows()
    with get_conn() as conn:  # one transaction: commits on success, rolls back on error
        with conn.cursor() as cur:
            cur.execute("TRUNCATE restaurants")
            with cur.copy(f"COPY restaurants ({', '.join(COLUMNS)}) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
    print(f"loaded {len(rows)} restaurants")


def check() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM restaurants")
        (n,) = cur.fetchone()
        print(f"rows: {n}")
        if n == 0:
            return

        counts = ", ".join(f"count({c})" for c in COLUMNS)
        cur.execute(f"SELECT {counts} FROM restaurants")
        print("\nnull rate per column:")
        for column, present in zip(COLUMNS, cur.fetchone()):
            print(f"  {column}: {1 - present / n:.3f}")

        cur.execute("SELECT price_level, count(*) FROM restaurants GROUP BY 1 ORDER BY 1 NULLS LAST")
        print("\nprice_level distribution:")
        for level, count in cur.fetchall():
            print(f"  {level}: {count}")

        cur.execute("SELECT is_open, count(*) FROM restaurants GROUP BY 1 ORDER BY 1 NULLS LAST")
        print("\nis_open distribution:")
        for is_open, count in cur.fetchall():
            print(f"  {is_open}: {count}")

        emb, business_ids = interfaces.load_embeddings()
        row_of = {bid: i for i, bid in enumerate(business_ids)}
        cur.execute("SELECT business_id FROM restaurants ORDER BY business_id")
        sample = random.Random(interfaces.SEED).sample([r[0] for r in cur.fetchall()], min(N_EMBEDDING_CHECKS, n))
        cur.execute("SELECT business_id, embedding FROM restaurants WHERE business_id = ANY(%s)", (sample,))
        worst = 0.0
        for bid, stored in cur.fetchall():
            stored = stored.to_numpy() if hasattr(stored, "to_numpy") else np.asarray(stored)
            worst = max(worst, float(np.max(np.abs(stored - emb[row_of[bid]]))))
        ok = worst <= EMBEDDING_TOLERANCE
        print(f"\nembeddings: {len(sample)} random rows, max abs diff {worst:.2e} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["load", "check"])
    args = parser.parse_args(argv)
    {"load": load, "check": check}[args.command]()


if __name__ == "__main__":
    main()
