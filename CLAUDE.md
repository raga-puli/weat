# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A group restaurant recommender built from the Yelp Academic Dataset, scoped to Philadelphia restaurants (see README.md). The pipeline: raw Yelp JSON -> filtered/joined Parquet tables (DuckDB) -> sentence embeddings per restaurant (SentenceTransformers) -> recommendation models -> offline evaluation against a held-out split.

## File ownership (WEEK 2)

| Person | Owns |
| --- | --- |
| A | `src/prep.py`, `src/split.py`, `src/embed.py`, `src/recommend.py`, `src/groups.py`, `src/aggregate.py`, `src/group_eval.py`, `src/db_groups.py`, `db/groups.sql`, `results/groups.md`, `tests/test_groups.py`, `tests/test_aggregate.py`, `tests/test_group_eval.py`, `README.md` |
| B | `src/eval.py`, `src/baselines.py`, `results.md`, `src/query_parser.py`, `src/search.py`, `src/db.py`, `src/bench.py`, `docker-compose.yml`, `eval/queries.jsonl`, `results/parser.md`, `results/latency.md`, `tests/test_query_parser.py`, `tests/test_search.py` |
| Shared, FROZEN | `CLAUDE.md`, `src/interfaces.py`, `requirements.txt`, `.gitignore`, `.env.example`, `db/schema.sql` |

### Rules

- Only edit files owned by the current person.
- Never edit frozen files; if a change seems necessary, stop and tell the user instead.
- Never edit `results.md` except via `python -m src.eval --log`.
- Never evaluate on the test split unless the user explicitly says so.
- Never commit anything under `data/`.
- Group evaluation uses the val group split unless the user explicitly says test.
- Never commit secrets; read config from .env.
- Only write results files through their owning script's --log flag.
- LLM calls go through Ollama (OLLAMA_HOST, LLM_MODEL from .env). Parser calls use temperature 0 and a JSON-schema `format`. Pin the exact model tag in every results/parser.md row.

## Environment

- Windows, PowerShell, CPU-only torch.
- Python virtualenv lives in `.venv`. Activate it before running anything: `.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt` (duckdb, pandas, pyarrow, sentence-transformers, numpy, streamlit)

## Pipeline (run in order)

Each script is a standalone, run-top-to-bottom script (no `if __name__ == "__main__"`, no CLI args, no functions) — run with `python src/<script>.py` from the repo root, since paths are hardcoded relative to it.

1. `src/prep.py` — reads the raw Yelp JSON dumps from `data/raw/` via `duckdb.read_json_auto`, filters businesses to Philadelphia restaurants with `review_count >= 20`, and writes three Parquet files to `data/processed/`:
   - `restaurants.parquet`: business_id, name, latitude, longitude, stars, review_count, categories, attributes
   - `reviews.parquet`: reviews joined down to only the filtered restaurants
   - `users.parquet`: reviewers with >= 5 reviews (id, review count, avg stars)
2. `src/embed.py` — for each restaurant, pulls up to 15 of the longest reviews (split evenly between >=4-star and <4-star, via `row_number() OVER (PARTITION BY business_id, stars >= 4)`) with text length between 200-2000 chars, concatenates them with the name/categories into one text blob, and encodes each blob with `all-MiniLM-L6-v2` (384-dim, normalized). Outputs:
   - `data/processed/embeddings.npy` — float32 array, shape (n_restaurants, 384)
   - `data/processed/business_ids.txt` — one business_id per line, same row order as the embeddings array (this is the join key back to `restaurants.parquet`; order must stay in sync)
3. `src/recommend.py` — not yet implemented (empty file). Intended to consume the embeddings to produce similarity-based recommendations.
4. `src/app.py` — not yet implemented (empty file). Intended as the Streamlit front end (`streamlit run src/app.py` once built).

## Data layout

- `data/raw/` — original Yelp JSON dumps (gitignored, not present unless downloaded manually): `yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`, `yelp_academic_dataset_tip.json`
- `data/processed/` — pipeline outputs described above (gitignored; regenerate via `prep.py` then `embed.py`)
- Parquet/`.npy` files and `data/raw/`, `data/processed/` are gitignored — treat them as regeneratable build artifacts, never commit them.

## Working conventions in this codebase

- Scripts use inline SQL strings against DuckDB rather than pandas transformations — prefer extending queries in that style rather than pulling data into pandas early.
- Scripts currently have print-based sanity checks after each major step (row counts, sample rows) rather than assertions or tests. Follow this pattern if extending `prep.py`/`embed.py`.
