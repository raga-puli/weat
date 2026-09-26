import math
from itertools import combinations

import numpy as np


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
