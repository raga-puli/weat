# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A group restaurant recommender built from the Yelp Academic Dataset, scoped to Philadelphia restaurants (see README.md). The pipeline: raw Yelp JSON -> filtered/joined Parquet tables (DuckDB) -> sentence embeddings per restaurant (SentenceTransformers) -> recommendation models -> offline evaluation against a held-out split.

## File ownership (WEEK 3)

Supersedes the week 2 table; week 3 = everything each person owned in week 2 plus the new files.

| Person | Owns |
| --- | --- |
| A | `src/prep.py`, `src/split.py`, `src/embed.py`, `src/recommend.py`, `src/groups.py`, `src/aggregate.py`, `src/group_eval.py`, `src/db_groups.py`, `db/groups.sql`, `results/groups.md`, `tests/test_groups.py`, `tests/test_aggregate.py`, `tests/test_group_eval.py`, `tests/test_embed.py`, `tests/test_recommend.py`, `README.md`, `src/api/routes_group.py`, `src/pages/1_Group_Dinner.py`, `src/mf.py`, `tests/test_api_group.py`, `tests/test_mf.py` |
| B | `src/eval.py`, `src/baselines.py`, `results.md`, `src/query_parser.py`, `src/search.py`, `src/db.py`, `src/bench.py`, `docker-compose.yml`, `eval/queries.jsonl`, `results/parser.md`, `results/latency.md`, `tests/test_query_parser.py`, `tests/test_search.py`, `src/api/routes_search.py`, `src/pages/2_Search.py`, `Dockerfile`, `.dockerignore`, `tests/test_api_search.py` |
| Shared, FROZEN | `CLAUDE.md`, `src/interfaces.py`, `requirements.txt`, `.gitignore`, `.env.example`, `db/schema.sql`, `src/api/main.py`, `src/app.py`, `src/ui_common.py` |

### Rules

- Only edit files owned by the current person.
- Never edit frozen files; if a change seems necessary, stop and tell the user instead.
- Results logging: `results.md` is written only by `python -m src.eval --log`; `results/groups.md` (Person A) is written only by `python -m src.group_eval --log`. Never edit either by hand.
- Never evaluate on the test split unless the user explicitly says so.
- Never commit anything under `data/`.
- Group evaluation uses the val group split unless the user explicitly says test.
- Never commit secrets; read config from .env.
- Only write results files through their owning script's --log flag.
- LLM calls go through Ollama (OLLAMA_HOST, LLM_MODEL from .env). Parser calls use temperature 0 and a JSON-schema `format`. Pin the exact model tag in every results/parser.md row.
- Scripts may take CLI flags (argparse); the current ones are listed under "CLI flags" below. Keep that list in sync when adding or changing a flag.
- Heavy resources (embeddings, models, DB connections) are loaded through functools.lru_cache getter functions used as FastAPI dependencies, so tests can override them; never load them at import time.
- API tests use FastAPI TestClient with dependency_overrides and must not need Postgres, Ollama, or real data files.

## Environment

- Windows, PowerShell, CPU-only torch.
- Python virtualenv lives in `.venv`. Activate it before running anything: `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Postgres + pgvector run in Docker (`docker compose up -d`, see docker-compose.yml); copy `.env.example` to `.env`.

## Pipeline (run in order)

Run everything from the repo root as `python -m src.<module>` (paths are relative to the root; the data paths live in `src/interfaces.py`).

1. `src/prep.py` — DuckDB over `data/raw/`: businesses in Philadelphia (city spelling variants, case/whitespace-insensitive) whose categories include "Restaurants" and with `review_count >= 20`. Writes to `data/processed/`:
   - `restaurants.parquet`: business_id, name, latitude, longitude, stars, review_count, categories, attributes, hours (JSON string, null if missing)
   - `reviews.parquet`: review_id, user_id, business_id, stars, text, date for those restaurants
   - `users.parquet`: user_id, n_reviews, avg_stars for users with >= 5 reviews
   `--stage restaurants` rebuilds only restaurants.parquet.
2. `src/split.py` — time-based per-user holdout. Users in users.parquet with >= 5 distinct restaurants (keeping each user's earliest review per restaurant) get their latest review in `test_reviews.parquet` and the second-latest in `val_reviews.parquet`; `train_reviews.parquet` is everything else, minus every review a split user wrote of their val/test restaurant.
3. `src/embed.py` — restaurant embeddings from train reviews of 200–2000 characters, up to 15 per side (>= 4 stars, < 4 stars), encoded with `all-MiniLM-L6-v2` (384-dim, normalized).
   - `--mode chunked` (default): a seeded random sample per side; each review is encoded as "name. categories. text"; per-side means, then the mean of the sides, L2-normalized -> `embeddings.npy`.
   - `--mode blob`: the 15 longest reviews per side concatenated into one text per restaurant -> `embeddings_blob.npy`.
   - `business_ids.txt`: one business_id per line (sorted), the row order of both .npy files. A run refuses, before saving anything, if it would change an existing order.
   - `--limit N` writes `*_limitN.npy` / `business_ids_limitN.txt` only; `--neighbors NAME` prints a restaurant's nearest neighbors.
4. `src/recommend.py` — `ContentBased` (taste vector = L2-normalized mean embedding of a user's >= 4-star train restaurants, or all of them if none; cosine ranking), `ContentBasedBlob` (blob embeddings), `ContentBasedCentered` (catalog-mean-centered, re-normalized embeddings); `taste_vector_from` / `scores_for` for arbitrary histories; `hubness_diagnostics`. `src/baselines.py` — `Random`, `Popularity`. Models follow the `Recommender` protocol in `src/interfaces.py`.
5. `src/eval.py` — individual-user evaluation of `--model MODULE:CLASS` (fit on train) on val, or test with `--i-mean-it`: recall@10, NDCG@10, MRR, coverage, diversity, Spearman, latency. `--log` appends a row to `results.md`.
6. `src/groups.py` — synthetic co-rating groups -> `groups.parquet`: 3–4 val-split users who each liked (>= 4 stars) the same target restaurant within 30 days, each with >= 3 likes before the window; at most 20 groups per target, 10 per user, 12,000 total; split 70/30 val/test by a hash of the group id.
7. `src/group_eval.py` — time-respecting group evaluation: member profiles from train likes strictly before each group's window (target excluded); candidates exclude anything a member reviewed before the window. Strategies: random, popularity, and `src/aggregate.py`'s average, least_misery, fairness, fairness_repeated. Cluster-bootstrap CIs over targets. Writes `group_eval_<split>*.csv` and `*_lists.parquet` to `data/processed/`; `--log` appends to `results/groups.md`; `--tau-sensitivity` writes `group_eval_val_tau_sensitivity.csv` instead.
8. Postgres/pgvector — `db/schema.sql` (restaurants with `embedding vector(384)`); `src/db.py` loads restaurants.parquet + `embeddings.npy` into it and checks the load. `src/db_groups.py` + `db/groups.sql` (group scoring in SQL) are on branch `a/db-groups`, not yet on main.
9. Search — `src/query_parser.py` turns a query into a `ParsedQuery` via Ollama (responses cached under `data/cache/llm/`; `--eval` scores `eval/queries.jsonl`, `--log` appends to `results/parser.md`); `src/search.py` filters and ranks restaurants for a `ParsedQuery` in Postgres (`search`) or numpy (`search_numpy`); `src/bench.py` benchmarks latency and pgvector indexes (`--log` appends to `results/latency.md`).
10. API and UI (week 3) — `uvicorn src.api.main:app` (`src/api/main.py`, routers in `src/api/routes_group.py` and `src/api/routes_search.py`); `streamlit run src/app.py` (pages in `src/pages/`, API helpers in `src/ui_common.py`).

### CLI flags

| Script | Flags |
| --- | --- |
| `src.prep` | `--stage {all,restaurants}` (default all) |
| `src.split` | none |
| `src.embed` | `--mode {blob,chunked}` (default chunked), `--limit N`, `--neighbors NAME` |
| `src.eval` | `--model MODULE:CLASS` (required), `--split {val,test}`, `--k N` (default 10), `--log NOTE`, `--config TEXT`, `--sample N`, `--i-mean-it` |
| `src.groups` | none |
| `src.group_eval` | `--split {val,test}`, `--i-mean-it`, `--sample N`, `--normalization/--norm {raw,zscore,rank}` (default rank), `--strategy NAME` or `--all` (default), `--lam X` (default 0.5), `--log NOTE`, `--tau-sensitivity` |
| `src.db_groups` (branch `a/db-groups`) | command `load-taste` or `check` |
| `src.db` | command `load` or `check` |
| `src.search` | `QUERY`, `--k N` (default 10), `--include-null-price` |
| `src.query_parser` | `[TEXT]`, `--eval`, `--log NOTE` |
| `src.bench` | `--log NOTE`, `--runs N` (default 200), `--warmup N` (default 10), `--seconds S` (default 30) |

## Data layout

- `data/raw/` — original Yelp JSON dumps (gitignored, not present unless downloaded manually): `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_tip.json`
- `data/processed/` — pipeline outputs described above (gitignored; regenerate via the pipeline)
- Parquet/`.npy` files and `data/raw/`, `data/processed/` are gitignored — treat them as regeneratable build artifacts, never commit them.

## Working conventions in this codebase

- Scripts use inline SQL strings against DuckDB rather than pandas transformations — prefer extending queries in that style rather than pulling data into pandas early.
- Scripts currently have print-based sanity checks after each major step (row counts, sample rows) rather than assertions or tests. Follow this pattern if extending `prep.py`/`embed.py`.
