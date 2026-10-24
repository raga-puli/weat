# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A group restaurant recommender built from the Yelp Academic Dataset, scoped to Philadelphia restaurants (see README.md). The pipeline: raw Yelp JSON -> filtered/joined Parquet tables (DuckDB) -> sentence embeddings per restaurant (SentenceTransformers) -> recommendation models -> offline evaluation against a held-out split.

## File ownership (WEEK 4)

Supersedes the week 3 table; week 4 = everything each person owned in week 3 plus the new files. Amended by the week 4 scaffolding PR (both people agreed): `tests/test_human_eval.py` and the rest of `docs/` to B, `requirements-api.txt` and `tests/test_ui_common.py` shared.

| Person | Owns |
| --- | --- |
| A | `src/prep.py`, `src/split.py`, `src/embed.py`, `src/recommend.py`, `src/groups.py`, `src/aggregate.py`, `src/group_eval.py`, `src/db_groups.py`, `db/groups.sql`, `results/groups.md`, `tests/test_groups.py`, `tests/test_aggregate.py`, `tests/test_group_eval.py`, `tests/test_embed.py`, `tests/test_recommend.py`, `README.md`, `src/api/routes_group.py`, `src/pages/1_Group_Dinner.py`, `src/mf.py`, `tests/test_api_group.py`, `tests/test_mf.py`, `src/deploy_load.py`, `deploy/` (all files), `docs/search.md` |
| B | `src/eval.py`, `src/baselines.py`, `results.md`, `src/query_parser.py`, `src/search.py`, `src/db.py`, `src/bench.py`, `docker-compose.yml`, `eval/queries.jsonl`, `results/parser.md`, `results/latency.md`, `tests/test_query_parser.py`, `tests/test_search.py`, `src/api/routes_search.py`, `src/pages/2_Search.py`, `Dockerfile`, `.dockerignore`, `tests/test_api_search.py`, `src/pages/3_Blind_Eval.py`, `src/human_eval.py`, `docs/human_eval_protocol.md`, `docs/groups.md`, `results/human_eval.md`, `tests/test_human_eval.py`, `docs/` (every file not assigned elsewhere in this table, including `docs/screenshots/`) |
| Shared, FROZEN | `CLAUDE.md`, `src/interfaces.py`, `requirements.txt`, `.gitignore`, `.env.example`, `db/schema.sql`, `src/api/main.py`, `src/app.py`, `src/ui_common.py`, `docs/frozen_configs.md`, `requirements-api.txt`, `tests/test_ui_common.py` |

### Rules

- Only edit files owned by the current person.
- Never edit frozen files; if a change seems necessary, stop and tell the user instead.
- Results logging: `results.md` is written only by `python -m src.eval --log`; `results/groups.md` (Person A) is written only by `python -m src.group_eval --log`. Never edit either by hand.
- Never evaluate on the test split unless the user explicitly says so.
- The test split is run exactly once per model, on Day 1, using docs/frozen_configs.md. Never rerun test to get a better number.
- Never commit anything under `data/`.
- No Yelp data (parquet, npy, review text, embeddings) is ever committed or copied into deploy/; the hosted API reads everything from DATABASE_URL.
- Writing in docs/ must be drafted by the person, not generated; Claude may outline, quiz, and fact-check.
- Group evaluation uses the val group split unless the user explicitly says test.
- Never commit secrets; read config from .env.
- Only write results files through their owning script's --log flag.
- LLM calls go through Ollama (OLLAMA_HOST, LLM_MODEL from .env). Parser calls use temperature 0 and a JSON-schema `format`. Pin the exact model tag in every results/parser.md row.
- Scripts may take CLI flags (argparse); the current ones are listed under "CLI flags" below. Keep that list in sync when adding or changing a flag.
- Closed restaurants (is_open = false) STAY in all offline evaluation (eval.py, group_eval.py, baselines, splits, embeddings), because held-out reviews are historical. User-facing paths (API endpoints, Streamlit pages) exclude them by default via an include_closed=false parameter.
- Heavy resources (embeddings, models, DB connections) are loaded through functools.lru_cache getter functions used as FastAPI dependencies, so tests can override them; never load them at import time.
- API tests use FastAPI TestClient with dependency_overrides and must not need Postgres, Ollama, or real data files.

## Environment

- Windows, PowerShell, CPU-only torch.
- Python virtualenv lives in `.venv`. Activate it before running anything: `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt` (the full pipeline, including torch for `src/embed.py`). The API/UI Docker image installs `requirements-api.txt` instead: no torch; the search query encoder runs on onnxruntime (`ENCODER=onnx`).
- Query encoder: `ENCODER=torch` (default) or `ENCODER=onnx` in `.env`. Both give the same query vectors (cosine >= 0.9999999 on `eval/queries.jsonl`). Locally, `onnx` needs the pinned model files: `python -m src.search --download-onnx` (writes `models/`, gitignored; the image downloads its own copy).
- Streamlit pages call the API only through `src/ui_common.py`, which waits for a sleeping host ("Waking up the server, about a minute") and retries 503s that carry Retry-After; pages don't add their own retry loops.
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
8. Postgres/pgvector — `db/schema.sql` (restaurants with `embedding vector(384)` and a nullable `is_open boolean`); `src/db.py` loads restaurants.parquet + `embeddings.npy` into it and checks the load.
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
| `src.db` | command `load` or `check` |
| `src.search` | `QUERY`, `--k N` (default 10), `--include-null-price`; or `--download-onnx` alone |
| `src.recommend` | command `build-cache`, `--model {ContentBased,ContentBasedBlob,ContentBasedCentered,ContentBasedDebiased}` (default ContentBasedDebiased) |
| `src.deploy_load` | `--database-url URL` (default DATABASE_URL from the environment / .env), `--demo-users N` (default 500), `--sample-groups N` (default 100), `--seed N` (default SEED) |
| `src.human_eval` | `--select-groups`, `--log NOTE`, `--responses PATH` (default data/human_eval/responses.csv) |
| `src.query_parser` | `[TEXT]`, `--eval`, `--log NOTE` |
| `src.bench` | `--log NOTE`, `--runs N` (default 200), `--warmup N` (default 10), `--seconds S` (default 30) |

## Data layout

- `data/raw/` — original Yelp JSON dumps (gitignored, not present unless downloaded manually): `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_tip.json`
- `data/processed/` — pipeline outputs described above (gitignored; regenerate via the pipeline)
- Parquet/`.npy` files and `data/raw/`, `data/processed/` are gitignored — treat them as regeneratable build artifacts, never commit them.

## Working conventions in this codebase

- Tests: pytest, under `tests/` in `test_<module>.py` files. New code needs tests, and `python -m pytest tests` must pass before committing. Tests that need a loaded Postgres (and the real data files) skip themselves when it isn't available; all other tests run on synthetic data.
- CLI flags are allowed (argparse) and are listed in the "CLI flags" table above.
- Both DuckDB and pandas are used: `src/prep.py` builds the Parquet tables with inline DuckDB SQL over the raw JSON; the other modules read Parquet with pandas. Extend prep in the DuckDB style; elsewhere, pandas is fine.
- Data paths, `SEED`, and `RELEVANT_MIN_STARS` are defined in `src/interfaces.py`; import them rather than redefining them (`src/prep.py`'s SQL still spells out its paths).
- Scripts print sanity checks (row counts, sample rows) after each major step, in addition to tests.
