"""Search restaurants for a ParsedQuery, in Postgres (search) or in numpy (search_numpy).

    python -m src.search "cheap vegan tacos near Fishtown"

Hard constraints filter; they never rank:
- max_price_level: price_level <= level. Restaurants with no price level are dropped unless
  include_null_price=True.
- near: haversine_km from the ANCHORS point <= max_distance_km, or DEFAULT_NEAR_KM when the query
  names an anchor without a distance. max_distance_km without an anchor is ignored (nothing to
  measure from).
- dietary: every tag must hold, via the matching Yelp category ("Vegan", "Gluten-Free", ...) or
  attributes.DietaryRestrictions marking it True.
- categories: any of them matches, as a case-insensitive substring of a Yelp category or the name
  ("thai" -> "Thai", "brunch" -> "Breakfast & Brunch"). Words with no such category match nothing.
- open_at: "now" (this machine's local time) or "<day> <HH:MM>", a lowercase 3-letter day and a
  24-hour time, e.g. "fri 19:00". Checked against Yelp hours, {"Friday": "11:0-22:0", ...}: a close
  time at or before the open time runs past midnight, and "0:0-0:0" means open all day. Restaurants
  with no hours are dropped. SQL over-fetches k * OPEN_AT_OVERFETCH rows and filters in Python, so
  fewer than k results can come back.

Ranking: by cosine similarity to the embedded soft text when soft is non-empty, otherwise by
stars * ln(1 + review_count). Popularity ties break on business_id (byte order) so both versions
agree; the similarity ORDER BY has no tie-break so pgvector indexes can be used.
"""

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from src import interfaces
from src.query_parser import ANCHORS, ParsedQuery

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_NEAR_KM = 2.0
OPEN_AT_OVERFETCH = 5
EARTH_RADIUS_KM = 6371.0

DIETARY_CATEGORY = {
    "vegetarian": "Vegetarian",
    "vegan": "Vegan",
    "gluten_free": "Gluten-Free",
    "halal": "Halal",
    "kosher": "Kosher",
}
DIETARY_ATTRIBUTE_KEY = {
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "gluten_free": "gluten-free",
    "halal": "halal",
    "kosher": "kosher",
}
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_model = None
_frame = None


def embed(text: str) -> np.ndarray:
    """Encode text with the same model as the restaurant embeddings, loaded once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model.encode(text, normalize_embeddings=True).astype(np.float32)


# ---------- shared pieces ----------


def _dietary_attribute_pattern(tag: str) -> str:
    # DietaryRestrictions is stored as a Python-dict string: "{'vegan': True, ...}"
    return f"%'{DIETARY_ATTRIBUTE_KEY[tag]}': True%"


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_hhmm(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def resolve_open_at(open_at: str, now: datetime | None = None) -> tuple[int, int]:
    """'now' or 'fri 19:00' -> (weekday 0=Monday, minutes since midnight)."""
    if open_at == "now":
        now = now or datetime.now()
        return now.weekday(), now.hour * 60 + now.minute
    day, time = open_at.split()
    return DAYS.index(day), _parse_hhmm(time)


def is_open(hours: dict | None, weekday: int, minute: int) -> bool:
    if not hours:
        return False

    def span(day: int):
        value = hours.get(DAY_NAMES[day % 7])
        if not value:
            return None
        start, end = value.split("-")
        return _parse_hhmm(start), _parse_hhmm(end)

    today = span(weekday)
    if today:
        start, end = today
        if start == end:  # "0:0-0:0": open all day
            return True
        if start < end and start <= minute < end:
            return True
        if end < start and minute >= start:  # runs past midnight
            return True
    yesterday = span(weekday - 1)
    if yesterday:
        start, end = yesterday
        if end < start and minute < end:  # still open from last night
            return True
    return False


def _filter_open(rows: list[dict], open_at: str, k: int, hours_key: str = "hours") -> list[dict]:
    weekday, minute = resolve_open_at(open_at)
    return [r for r in rows if is_open(r.get(hours_key), weekday, minute)][:k]


# ---------- Postgres ----------


def build_query(parsed: ParsedQuery, k: int, query_vec=None, include_null_price: bool = False):
    """Return (sql, params). query_vec must be given when parsed.soft is non-empty."""
    select = ["business_id", "name", "price_level", "hours"]
    select_params: list = []
    where: list[str] = []
    where_params: list = []

    if parsed.max_price_level is not None:
        if include_null_price:
            where.append("(price_level <= %s OR price_level IS NULL)")
        else:
            where.append("price_level <= %s")
        where_params.append(parsed.max_price_level)

    if parsed.near is not None:
        lat, lon = ANCHORS[parsed.near]
        select.append("haversine_km(%s, %s, latitude, longitude) AS distance_km")
        select_params += [lat, lon]
        where.append("haversine_km(%s, %s, latitude, longitude) <= %s")
        where_params += [lat, lon, parsed.max_distance_km or DEFAULT_NEAR_KM]
    else:
        select.append("NULL::double precision AS distance_km")

    for tag in parsed.dietary:
        where.append("(%s = ANY(categories) OR attributes->>'DietaryRestrictions' LIKE %s)")
        where_params += [DIETARY_CATEGORY[tag], _dietary_attribute_pattern(tag)]

    if parsed.categories:
        clauses = []
        for category in parsed.categories:
            pattern = f"%{_like_escape(category)}%"
            clauses.append("(EXISTS (SELECT 1 FROM unnest(categories) c WHERE c ILIKE %s) OR name ILIKE %s)")
            where_params += [pattern, pattern]
        where.append("(" + " OR ".join(clauses) + ")")

    order_params: list = []
    if parsed.soft:
        if query_vec is None:
            raise ValueError("query_vec is required when parsed.soft is non-empty")
        select.append("1 - (embedding <=> %s) AS similarity")
        select_params.append(query_vec)
        # A bare distance ORDER BY lets pgvector indexes serve the query; a second sort key
        # forces a seq scan + sort. Exact cosine ties are vanishingly rare, so no tie-break.
        order = "embedding <=> %s"
        order_params.append(query_vec)
    else:
        select.append("NULL::double precision AS similarity")
        order = 'stars * ln(1 + review_count) DESC, business_id COLLATE "C"'

    limit = k * OPEN_AT_OVERFETCH if parsed.open_at else k
    sql = f"SELECT {', '.join(select)}\nFROM restaurants"
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)
    sql += f"\nORDER BY {order}\nLIMIT %s"
    return sql, select_params + where_params + order_params + [limit]


def search(conn, parsed: ParsedQuery, k: int = 10, include_null_price: bool = False) -> list[dict]:
    query_vec = embed(parsed.soft) if parsed.soft else None
    sql, params = build_query(parsed, k, query_vec, include_null_price)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        names = [d.name for d in cur.description]
        rows = [dict(zip(names, row)) for row in cur.fetchall()]
    if parsed.open_at:
        rows = _filter_open(rows, parsed.open_at, k)
    return [_result(r) for r in rows]


def _result(row: dict) -> dict:
    return {
        "business_id": row["business_id"],
        "name": row["name"],
        "price_level": None if row["price_level"] is None or pd.isna(row["price_level"]) else int(row["price_level"]),
        "distance_km": None if row["distance_km"] is None else float(row["distance_km"]),
        "similarity": None if row["similarity"] is None else float(row["similarity"]),
    }


# ---------- numpy ----------


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Restaurants (one row each, parsed like the DB load) and their embeddings, cached per process."""
    global _frame
    if _frame is None:
        from src.db import _as_dict, parse_categories, parse_price_level

        restaurants = pd.read_parquet(interfaces.RESTAURANTS)
        emb, business_ids = interfaces.load_embeddings()
        row_of = {bid: i for i, bid in enumerate(business_ids)}
        attributes = restaurants["attributes"].map(_as_dict)
        frame = pd.DataFrame(
            {
                "business_id": restaurants["business_id"],
                "name": restaurants["name"],
                "latitude": restaurants["latitude"],
                "longitude": restaurants["longitude"],
                "stars": restaurants["stars"].astype(np.float32),  # Postgres stores real
                "review_count": restaurants["review_count"],
                "price_level": [parse_price_level((a or {}).get("RestaurantsPriceRange2")) for a in attributes],
                "categories": restaurants["categories"].map(lambda c: parse_categories(c) or []),
                "attributes": attributes,
                "hours": restaurants["hours"].map(_as_dict) if "hours" in restaurants else None,
            }
        )
        _frame = (frame.reset_index(drop=True), emb[[row_of[b] for b in frame["business_id"]]])
    return _frame


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


_prepared: dict = {}

# Joins strings for vectorized matching; it can't appear in a category, a name, or a query word.
_SEP = "\x1f"


def _prepare(frame: pd.DataFrame, emb: np.ndarray) -> dict:
    """Per-dataset arrays search_numpy reuses on every query, built once."""
    key = (id(frame), id(emb))
    if key not in _prepared:
        cats = frame["categories"]
        dietary_attr = [str((a or {}).get("DietaryRestrictions") or "") for a in frame["attributes"]]
        _prepared[key] = {
            "ids": frame["business_id"].to_numpy(),
            "price": frame["price_level"].astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan),
            "lat": frame["latitude"].to_numpy(dtype=np.float64),
            "lon": frame["longitude"].to_numpy(dtype=np.float64),
            "popularity": frame["stars"].to_numpy(dtype=np.float64) * np.log1p(frame["review_count"].to_numpy()),
            # exact (case-sensitive) category membership, like `%s = ANY(categories)`
            "cat_exact": np.array([_SEP + _SEP.join(c) + _SEP for c in cats]),
            # case-insensitive substring of any category or the name, like ILIKE
            "cat_or_name": np.array([(_SEP.join(c) + _SEP + n).lower() for c, n in zip(cats, frame["name"])]),
            "dietary_attr": np.array(dietary_attr),
            "emb64": emb.astype(np.float64),
        }
    return _prepared[key]


def search_numpy(
    parsed: ParsedQuery, k: int = 10, include_null_price: bool = False, data=None
) -> list[dict]:
    """Same filters and ranking as search(), on the parquet + .npy. data=(frame, emb) overrides the files."""
    frame, emb = data if data is not None else load_frame()
    arr = _prepare(frame, emb)
    mask = np.ones(len(frame), dtype=bool)

    if parsed.max_price_level is not None:
        price = arr["price"]
        mask &= np.where(np.isnan(price), include_null_price, price <= parsed.max_price_level)

    distance = np.full(len(frame), np.nan)
    if parsed.near is not None:
        lat, lon = ANCHORS[parsed.near]
        distance = haversine_km(lat, lon, arr["lat"], arr["lon"])
        mask &= distance <= (parsed.max_distance_km or DEFAULT_NEAR_KM)

    for tag in parsed.dietary:
        in_category = np.char.find(arr["cat_exact"], _SEP + DIETARY_CATEGORY[tag] + _SEP) >= 0
        in_attribute = np.char.find(arr["dietary_attr"], _dietary_attribute_pattern(tag).strip("%")) >= 0
        mask &= in_category | in_attribute

    if parsed.categories:
        any_match = np.zeros(len(frame), dtype=bool)
        for category in parsed.categories:
            any_match |= np.char.find(arr["cat_or_name"], category.lower()) >= 0
        mask &= any_match

    idx = np.flatnonzero(mask)
    ids = arr["ids"][idx]
    if parsed.soft:
        # scoring every row and then indexing avoids copying the embedding rows
        similarity = (arr["emb64"] @ embed(parsed.soft).astype(np.float64))[idx]
        order = np.lexsort((ids, -similarity))
    else:
        similarity = np.full(len(idx), np.nan)
        order = np.lexsort((ids, -arr["popularity"][idx]))

    limit = k * OPEN_AT_OVERFETCH if parsed.open_at else k
    rows = []
    for j in order[:limit]:
        i = idx[j]
        rows.append(
            {
                "business_id": ids[j],
                "name": frame["name"].iat[i],
                "price_level": frame["price_level"].iat[i],
                "hours": frame["hours"].iat[i] if "hours" in frame else None,
                "distance_km": None if math.isnan(distance[i]) else float(distance[i]),
                "similarity": None if math.isnan(similarity[j]) else float(similarity[j]),
            }
        )
    if parsed.open_at:
        rows = _filter_open(rows, parsed.open_at, k)
    return [_result(r) for r in rows]


# ---------- CLI ----------


def _show_sql(sql: str, params: list) -> str:
    shown = ["<384-dim query vector>" if isinstance(p, np.ndarray) else repr(p) for p in params]
    return f"{sql}\n-- params: [{', '.join(shown)}]"


def main(argv=None) -> None:
    from src.db import get_conn
    from src.query_parser import parse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--include-null-price", action="store_true")
    args = parser.parse_args(argv)

    parsed = parse(args.query)
    print("parsed:")
    print(json.dumps(asdict(parsed), indent=2))

    query_vec = embed(parsed.soft) if parsed.soft else None
    sql, params = build_query(parsed, args.k, query_vec, args.include_null_price)
    print("\nsql:")
    print(_show_sql(sql, params))

    with get_conn() as conn:
        results = search(conn, parsed, args.k, args.include_null_price)
    print(f"\n{len(results)} results:")
    for i, r in enumerate(results, 1):
        price = "$" * r["price_level"] if r["price_level"] else "-"
        dist = f"{r['distance_km']:.2f} km" if r["distance_km"] is not None else ""
        sim = f"sim {r['similarity']:.3f}" if r["similarity"] is not None else ""
        print(f"{i:>2}. {r['name']}  {price}  {dist}  {sim}".rstrip())


if __name__ == "__main__":
    main()
