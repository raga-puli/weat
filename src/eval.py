import argparse
import importlib
import math
import random
import sys
import time
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import interfaces

COLUMNS = [
    "date",
    "split",
    "model",
    "config",
    "recall@10",
    "NDCG@10",
    "MRR",
    "coverage",
    "diversity",
    "spearman",
    "latency_ms/user",
    "n_users",
    "note",
]

BATCH_SIZE = 1000


def recall_at_k(recs, relevant, k):
    if not relevant:
        return 0.0
    hits = 0
    for user_id, target in relevant.items():
        top_k = recs.get(user_id, [])[:k]
        if target in top_k:
            hits += 1
    return hits / len(relevant)


def ndcg_at_k(recs, relevant, k):
    if not relevant:
        return 0.0
    total = 0.0
    for user_id, target in relevant.items():
        top_k = recs.get(user_id, [])[:k]
        if target in top_k:
            rank = top_k.index(target) + 1
            total += 1.0 / math.log2(rank + 1)
    return total / len(relevant)


def mrr(recs, relevant):
    if not relevant:
        return 0.0
    total = 0.0
    for user_id, target in relevant.items():
        items = recs.get(user_id, [])
        if target in items:
            rank = items.index(target) + 1
            total += 1.0 / rank
    return total / len(relevant)


def catalog_coverage(recs, n_catalog):
    if n_catalog == 0:
        return 0.0
    unique_items = set()
    for items in recs.values():
        unique_items.update(items)
    return len(unique_items) / n_catalog


def intra_list_diversity(recs, emb, id_to_row):
    per_user_scores = []
    for items in recs.values():
        if len(items) < 2:
            continue
        rows = [id_to_row[i] for i in items]
        vectors = emb[rows]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        pair_scores = [
            1.0 - float(np.dot(normalized[i], normalized[j]))
            for i, j in combinations(range(len(items)), 2)
        ]
        per_user_scores.append(sum(pair_scores) / len(pair_scores))
    if not per_user_scores:
        return 0.0
    return sum(per_user_scores) / len(per_user_scores)


def _load_model(spec: str):
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def _batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MODULE:CLASS to import")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--log", default=None, help="note; also triggers appending to results.md")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--i-mean-it", action="store_true", dest="i_mean_it")
    return parser.parse_args(argv)


def run(args) -> dict:
    if args.split == "test":
        print("WARNING: evaluating on the test split.")
        if not args.i_mean_it:
            print("Refusing --split test without --i-mean-it.")
            sys.exit(1)

    split_path = {
        "val": interfaces.VAL_REVIEWS,
        "test": interfaces.TEST_REVIEWS,
    }[args.split]

    model = _load_model(args.model)

    train_df = pd.read_parquet(interfaces.TRAIN_REVIEWS)
    model.fit(train_df)

    eval_df = pd.read_parquet(split_path)

    user_ids = eval_df["user_id"].tolist()
    if args.sample is not None:
        rng = random.Random(interfaces.SEED)
        user_ids = rng.sample(user_ids, min(args.sample, len(user_ids)))
        eval_df = eval_df[eval_df["user_id"].isin(user_ids)]

    relevant = {}
    excluded = 0
    for row in eval_df.itertuples():
        if row.stars >= interfaces.RELEVANT_MIN_STARS:
            relevant[row.user_id] = row.business_id
        else:
            excluded += 1
    print(
        f"Excluded {excluded} users with held-out stars below "
        f"RELEVANT_MIN_STARS={interfaces.RELEVANT_MIN_STARS}"
    )

    train_by_user = (
        train_df[train_df["user_id"].isin(user_ids)]
        .groupby("user_id")["business_id"]
        .apply(set)
        .to_dict()
    )
    exclude = {uid: train_by_user.get(uid, set()) for uid in user_ids}

    recs = {}
    start = time.perf_counter()
    for batch in _batched(user_ids, BATCH_SIZE):
        batch_exclude = {uid: exclude[uid] for uid in batch}
        recs.update(model.recommend(batch, args.k, batch_exclude))
    elapsed_ms = (time.perf_counter() - start) * 1000
    ms_per_user = elapsed_ms / len(user_ids) if user_ids else 0.0

    recall = recall_at_k(recs, relevant, args.k)
    ndcg = ndcg_at_k(recs, relevant, args.k)
    mrr_score = mrr(recs, relevant)

    restaurants_df = pd.read_parquet(interfaces.RESTAURANTS)
    n_catalog = len(restaurants_df)
    coverage = catalog_coverage(recs, n_catalog)

    if interfaces.EMBEDDINGS.exists():
        emb, business_ids = interfaces.load_embeddings()
        id_to_row = {bid: i for i, bid in enumerate(business_ids)}
        diversity = intra_list_diversity(recs, emb, id_to_row)
    else:
        diversity = "n/a"

    pred_scores = []
    actual_stars = []
    for row in eval_df.itertuples():
        scores = model.score(row.user_id, [row.business_id])
        if scores and not math.isnan(scores[0]):
            pred_scores.append(scores[0])
            actual_stars.append(row.stars)

    spearman_corr = None
    if len(pred_scores) >= 2:
        corr, _ = spearmanr(pred_scores, actual_stars)
        if not math.isnan(corr):
            spearman_corr = corr

    return {
        "date": date.today().isoformat(),
        "split": args.split,
        "model": getattr(model, "name", args.model),
        "config": args.config or "",
        "recall@10": recall,
        "NDCG@10": ndcg,
        "MRR": mrr_score,
        "coverage": coverage,
        "diversity": diversity,
        "spearman": spearman_corr,
        "latency_ms/user": ms_per_user,
        "n_users": len(user_ids),
        "note": args.log or "",
    }


def _print_table(result: dict) -> None:
    for key in COLUMNS:
        print(f"{key}: {result[key]}")


def _append_result(result: dict) -> None:
    row = "| " + " | ".join(str(result[col]) for col in COLUMNS) + " |\n"
    with open(interfaces.RESULTS_MD, "a") as f:
        f.write(row)


def main(argv=None) -> dict:
    args = _parse_args(argv)
    result = run(args)
    _print_table(result)
    if args.log is not None:
        _append_result(result)
    return result


if __name__ == "__main__":
    main()
