# Deploying the demo

Three pieces: a hosted Postgres with pgvector (Neon), the API container (`Dockerfile`), and the Streamlit UI (`src/app.py`) on Streamlit Community Cloud. The hosted API reads everything from `DATABASE_URL` (`DATA_SOURCE=db`). No Yelp data (parquet, npy, review text, embeddings) is committed or copied into `deploy/`; it goes into the database from your machine with `src.deploy_load`.

Steps marked **(verify on the provider's site)** depend on a provider's current UI, plans or free-tier limits, which change; check them before relying on them.

Before making the demo public, check that the Yelp Open Dataset terms allow it for your purpose **(verify on Yelp's site)**.

## (a) Neon: database and the vector extension

1. Create a Neon account and a new project **(verify on the provider's site)**. Pick a region close to where the API will run.
2. Copy the project's connection string (Postgres URL, with `sslmode=require`) from the project's connection details **(verify on the provider's site)**. Treat it as a secret: never commit it, and don't paste it into `.env.example`.
3. Enable pgvector. In Neon's SQL editor, or with `psql "<connection string>"`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   `src.deploy_load` also runs this, so this step only confirms the extension is available on your plan **(verify on the provider's site)**.
4. Load the data from your machine (repo root, `.venv` active, `data/processed/` built, ideally with the taste cache from `python -m src.recommend build-cache --model ContentBasedDebiased`):
   ```
   python -m src.deploy_load --database-url "<connection string>" --demo-users 500
   ```
   It applies `db/schema.sql` and `db/groups.sql`, loads `restaurants` through `src.db`, then the demo users' taste vectors, top-5 profiles (restaurant IDs and stars only), visited-restaurant IDs, the model's hub vector and 100 sample groups. At the end it prints the row count and size of each table. Check the total against your plan's storage limit **(verify on the provider's site)**.
5. Optional check: `DATABASE_URL="<connection string>" python -m src.db check`.

Re-running `deploy_load` replaces the restaurants and group tables. Only the 500 demo users have taste vectors in the database; the API returns 422 for any other user ID.

## (b) The API container

The image is built from the repo's `Dockerfile` (Person B's): CPU-only torch, `requirements.txt`, `src/` and `db/`, running `uvicorn src.api.main:app` on port 8000. Host: **TBD (fill in)**. The steps below fit any service that builds or runs a Dockerfile.

1. Create a web service from this GitHub repository (or push an image built with `docker build -t weat-api .` to the host's registry) **(verify on the provider's site)**.
2. Set the container port to 8000. If the host injects a `PORT` variable instead, override the start command to `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` **(verify on the provider's site)**.
3. Add these as secrets / environment variables, not in the image or the repo **(verify on the provider's site for where secrets go)**:
   | name | value |
   | --- | --- |
   | `DATABASE_URL` | the Neon connection string from (a) |
   | `DATA_SOURCE` | `db` |
   | `DEFAULT_ENGINE` | `rules` (no Ollama on the host) |
4. Set the health check path to `/health` **(verify on the provider's site)**.
5. Deploy, then check `https://<api host>/health` returns `{"status": "ok"}` and `https://<api host>/docs` loads. The first group request after a cold start reads the database tables, which takes a few seconds.
6. Memory: the image includes torch and sentence-transformers, which search uses to embed query text. Check that the instance size has enough memory, and how the free tier handles idle sleep and cold starts **(verify on the provider's site)**.

## (c) The UI on Streamlit Community Cloud

1. In Streamlit Community Cloud, create a new app from this GitHub repository, branch `main`, main file `src/app.py` **(verify on the provider's site)**. The pages in `src/pages/` are picked up automatically.
2. Choose the Python version used by the team's environment (3.12) in the app's advanced settings **(verify on the provider's site)**.
3. Add the secret in the app's secrets settings (TOML) **(verify on the provider's site)**:
   ```toml
   API_URL = "https://<api host>"
   ```
   `src/ui_common.py` reads `API_URL` from the environment; Streamlit Community Cloud exposes root-level secrets as environment variables **(verify on the provider's site)**.
4. Deploy. The landing page shows a green "API online" badge when it can reach the API.
5. Dependencies: Streamlit Community Cloud installs `requirements.txt`, which includes torch and sentence-transformers even though the UI only calls the API. If the build exceeds its size or memory limits **(verify on the provider's site)**, a UI-only requirements file is a change to frozen files (requirements) that both people need to agree on.

## Checklist

- [ ] Neon project created, `vector` extension available
- [ ] `src.deploy_load` printed the expected row counts
- [ ] API deployed with `DATABASE_URL`, `DATA_SOURCE=db`, `DEFAULT_ENGINE=rules` as secrets; `/health` OK
- [ ] UI deployed with `API_URL` as a secret; the badge shows "API online"
- [ ] No connection strings or data files committed (`git status`, `git log -p`)
