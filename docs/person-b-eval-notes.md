# WEat: evaluation and baselines (Person B) — notes for discussion/interviews

Status as of 2026-10-16. Rows marked PENDING were not yet run when this was written; fill them from `results.md`.

## What I owned
`src/eval.py`, `src/baselines.py`, `results.md`, `tests/test_eval.py`, `tests/test_baselines.py`.
Person A owned data prep, the split, embeddings and the recommenders. Shared files (`CLAUDE.md`, `src/interfaces.py`) were frozen.

## The evaluation setup
- **Task:** one held-out restaurant per user (leave-one-out style). A model returns a top-10 list; we ask whether the held-out restaurant is in it.
- **Relevance:** only held-out reviews with stars >= `RELEVANT_MIN_STARS` (4) count. On val, 7,405 of 25,439 users were excluded from recall/NDCG/MRR for that reason. They are still used for Spearman.
- **Exclusion:** each user's train businesses are excluded from their recommendations, so a model can't "win" by re-suggesting places they've already reviewed.
- **Metrics:** recall@10, NDCG@10, MRR, catalog coverage, intra-list diversity (mean pairwise 1 - cosine over embeddings), Spearman (model score vs. actual held-out stars), latency per user.
- **Pure metric functions** (no I/O) are separate from the runner, so they're unit-tested with hand-computed toy cases (hit at rank 1, hit at rank k, miss, empty list).

## Design decisions and why
| Decision | Why |
| --- | --- |
| Models loaded by `MODULE:CLASS` string via `importlib` | `eval.py` never imports model code, so two people can work in parallel and any model plugs in with no eval changes. |
| `results.md` written only through `--log` | One code path writes the log, so rows are reproducible and can't be hand-edited by accident. |
| `--split test` refused without `--i-mean-it` | Protects the test set from being tuned against. All numbers so far are on val. |
| Popularity counts come only from the train frame passed to `fit` | `reviews.parquet` and `restaurants.review_count` include held-out reviews. Using them would leak the answer into the baseline. A test uses a fake frame whose ordering differs from `review_count` to prove it. |
| Random is seeded per user with `SEED` | Reproducible, so reruns give identical rows. |
| Random and Popularity baselines before any content model | They set the floor. A model that can't beat Popularity isn't adding value. |
| Evaluation in batches of 1000 users, seeded `--sample N` | Lets a quick sample run before a full run, with a reproducible subset. |

## Data-integrity checks worth mentioning
- **Train leakage fix (A's commit 6c424ab):** if a user revisited their held-out restaurant on another date, the duplicate visit stayed in train, which leaked the answer. The split now masks every (user, business) pair of a held-out item. It removed 716 extra train reviews across 676 users.
- **I re-verified it independently:** 0 val (user, business) pairs and 0 test pairs appear in train.
- **Embedding row order:** `business_ids.txt` has a SHA256 guard (commit e5ac53c). We checked that the new `embeddings.npy` (3902 x 384, float32, no NaNs, unit norms) matched our hash `57F9534C…D4036E9` before using it. A wrong row order would silently mismatch embeddings to restaurants.
- **Stale artifacts:** `embeddings_blob.npy` was built from the old split, so it was rebuilt before use.
- **Known caveat from A:** 1,515 users still violate strict date ordering (test >= val >= max train) because of unrelated repeat visits; the fix doesn't target that.

## Results so far (val split, 25,439 users, k=10)
| model | recall@10 | NDCG@10 | MRR | coverage | diversity |
| --- | --- | --- | --- | --- | --- |
| random | 0.0024 | 0.0011 | 0.0007 | 1.00 | 0.411 |
| popularity | 0.0505 | 0.0270 | 0.0201 | 0.0097 | 0.355 |
| content_chunked | 0.0100 | 0.0047 | 0.0032 | 0.795 | 0.164 |
| content_blob | 0.0062 | 0.0032 | 0.0023 | 0.811 | 0.275 |

Relative to Popularity: chunked is 0.20x on recall@10 and 0.18x on NDCG@10; blob is 0.12x and 0.12x. Chunked beats blob (1.6x recall, 1.5x NDCG), which supports per-review encoding. Blob has the higher Spearman (0.055 vs 0.028 chunked, 0.044 popularity), so blob scores track held-out star ratings a bit better even though it ranks the held-out item worse.

### How to talk about the result
- Content-based beats random (~4x on recall@10) but **loses to popularity (~5x lower)**. This is a real finding, not a bug we could find: the split is leakage-free, row order is verified, and embeddings are valid.
- Likely explanation: popular restaurants dominate held-out visits, and a similarity-only recommender ignores popularity. Coverage shows the tradeoff: popularity recommends from ~1% of the catalog, content-based from ~80%.
- Hypotheses to test, in order: (1) hubness, where a few restaurants are nearest neighbors of everyone (A built `ContentBasedCentered` for this); (2) blending content similarity with a popularity prior; (3) richer user profiles than the mean of train embeddings.
- Be honest that recall@10 near 0.01-0.05 is low in absolute terms. With 3,902 items and one held-out item per user, random is ~0.26%, so these are meaningful lifts over chance, but the task is hard.

## Things I would do differently / open questions
- The first chunked run was logged without `--config` and with a different note than the schedule specified, which left a duplicate chunked row. `--log` should probably require `--config`.
- Test split has not been touched, by design. It should be run once, at the end, with the chosen model.
- Diversity is only defined for models with embeddings, so baselines have "n/a" until rerun after embeddings existed.
