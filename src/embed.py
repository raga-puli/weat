import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.interfaces import (
    BUSINESS_IDS,
    EMBEDDINGS,
    EMBEDDINGS_BLOB,
    RELEVANT_MIN_STARS,
    RESTAURANTS,
    SEED,
    TRAIN_REVIEWS,
    load_embeddings,
)

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_LEN = 200
MAX_LEN = 2000
MAX_PER_SIDE = 15
MIN_ELIGIBLE = 5


def _select_side(df, mode, cap=MAX_PER_SIDE, seed=SEED):
    """Map business_id -> up to `cap` review texts, longest-first.

    blob mode always takes the `cap` longest reviews (deterministic,
    matches the original ablation). chunked mode takes a seeded
    random sample of up to `cap` reviews, then orders them
    longest-first for a stable, readable output order.
    """
    if df.empty:
        return {}

    df = df.assign(_len=df["text"].str.len())
    selected = {}
    for business_id, group in df.groupby("business_id"):
        if len(group) > cap:
            if mode == "chunked":
                group = group.sample(n=cap, random_state=seed)
            else:
                group = group.nlargest(cap, "_len")
        selected[business_id] = group.sort_values("_len", ascending=False)[
            "text"
        ].tolist()
    return selected


def _load_eligible_reviews():
    reviews = pd.read_parquet(TRAIN_REVIEWS)
    reviews = reviews[reviews["text"].str.len().between(MIN_LEN, MAX_LEN)]
    restaurants = pd.read_parquet(RESTAURANTS).set_index("business_id")

    business_ids = sorted(set(reviews["business_id"]) & set(restaurants.index))
    reviews = reviews[reviews["business_id"].isin(business_ids)]

    counts = reviews.groupby("business_id").size().reindex(business_ids, fill_value=0)
    n_low = int((counts < MIN_ELIGIBLE).sum())
    print(f"restaurants with < {MIN_ELIGIBLE} eligible reviews: {n_low}")

    return reviews, restaurants, business_ids


def _write_business_ids(business_ids):
    if BUSINESS_IDS.exists():
        with open(BUSINESS_IDS) as f:
            existing = [line.strip() for line in f if line.strip()]
        if existing != business_ids:
            first_diff = next(
                i
                for i in range(max(len(existing), len(business_ids)))
                if i >= len(existing)
                or i >= len(business_ids)
                or existing[i] != business_ids[i]
            )
            old = existing[first_diff] if first_diff < len(existing) else "<missing>"
            new = (
                business_ids[first_diff]
                if first_diff < len(business_ids)
                else "<missing>"
            )
            raise ValueError(
                f"business_ids.txt ordering would change at index {first_diff}: "
                f"existing={old!r} new={new!r}. Refusing to overwrite; the two "
                f"embedding files would no longer share one consistent row order."
            )

    with open(BUSINESS_IDS, "w") as f:
        f.write("\n".join(business_ids) + "\n")


def run_blob(limit=None):
    reviews, restaurants, business_ids = _load_eligible_reviews()
    if limit is not None:
        business_ids = business_ids[:limit]

    pos = reviews[reviews["stars"] >= RELEVANT_MIN_STARS]
    neg = reviews[reviews["stars"] < RELEVANT_MIN_STARS]
    pos_sel = _select_side(pos, mode="blob")
    neg_sel = _select_side(neg, mode="blob")

    texts = []
    for business_id in tqdm(business_ids, desc="building blobs"):
        name = restaurants.loc[business_id, "name"]
        categories = restaurants.loc[business_id, "categories"]
        side_texts = pos_sel.get(business_id, []) + neg_sel.get(business_id, [])
        texts.append(f"{name}. {categories}. {' '.join(side_texts)}")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    np.save(EMBEDDINGS_BLOB, emb)
    _write_business_ids(business_ids)
    print(f"{EMBEDDINGS_BLOB}: {emb.shape} {emb.dtype}")


def run_chunked(limit=None):
    reviews, restaurants, business_ids = _load_eligible_reviews()
    if limit is not None:
        business_ids = business_ids[:limit]

    pos = reviews[reviews["stars"] >= RELEVANT_MIN_STARS]
    neg = reviews[reviews["stars"] < RELEVANT_MIN_STARS]
    pos_sel = _select_side(pos, mode="chunked")
    neg_sel = _select_side(neg, mode="chunked")

    flat_texts = []
    spans = {}
    for business_id in tqdm(business_ids, desc="collecting reviews"):
        name = restaurants.loc[business_id, "name"]
        categories = restaurants.loc[business_id, "categories"]
        spans[business_id] = {}
        for side, sel in (("pos", pos_sel), ("neg", neg_sel)):
            side_texts = sel.get(business_id, [])
            start = len(flat_texts)
            flat_texts.extend(f"{name}. {categories}. {t}" for t in side_texts)
            spans[business_id][side] = (start, len(flat_texts))

    model = SentenceTransformer(MODEL_NAME)
    review_emb = model.encode(
        flat_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    dim = review_emb.shape[1]
    final = np.zeros((len(business_ids), dim), dtype="float32")
    for row, business_id in enumerate(tqdm(business_ids, desc="averaging")):
        side_means = []
        for side in ("pos", "neg"):
            start, end = spans[business_id][side]
            if end > start:
                side_means.append(review_emb[start:end].mean(axis=0))
        vec = np.mean(side_means, axis=0)
        norm = np.linalg.norm(vec)
        final[row] = vec / norm if norm > 0 else vec

    np.save(EMBEDDINGS, final)
    _write_business_ids(business_ids)
    print(f"{EMBEDDINGS}: {final.shape} {final.dtype}")


def print_neighbors(name, mode, k=10):
    path = EMBEDDINGS_BLOB if mode == "blob" else EMBEDDINGS
    matrix, business_ids = load_embeddings(path)
    restaurants = pd.read_parquet(RESTAURANTS).set_index("business_id")

    matches = restaurants[restaurants["name"].str.lower() == name.lower()]
    if matches.empty:
        print(f"no restaurant named {name!r} found")
        return
    if len(matches) > 1:
        print(f"warning: {len(matches)} restaurants named {name!r}; using the first match")
    target_id = matches.index[0]

    if target_id not in business_ids:
        print(f"{matches.loc[target_id, 'name']!r} has no embedding in {path.name}")
        return

    idx = business_ids.index(target_id)
    sims = matrix @ matrix[idx]
    order = [i for i in np.argsort(-sims) if i != idx][:k]

    print(f"nearest to {matches.loc[target_id, 'name']}:")
    for rank, i in enumerate(order, 1):
        business_id = business_ids[i]
        row_name = (
            restaurants.loc[business_id, "name"]
            if business_id in restaurants.index
            else business_id
        )
        print(f"{rank:2d}. {row_name}  (sim={sims[i]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["blob", "chunked"], default="chunked")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--neighbors", type=str, default=None)
    args = parser.parse_args()

    if args.neighbors:
        print_neighbors(args.neighbors, args.mode)
    elif args.mode == "blob":
        run_blob(args.limit)
    else:
        run_chunked(args.limit)
