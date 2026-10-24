# Deploying the demo

Three pieces, all on free tiers: **Neon** (Postgres + pgvector), the API container on a **Render** free web service, and the Streamlit UI (`src/app.py`) on **Streamlit Community Cloud**. The hosted API reads everything from `DATABASE_URL` (`DATA_SOURCE=db`). No Yelp data (parquet, npy, review text, embeddings) is committed or copied into `deploy/`; it goes into the database from your machine with `src.deploy_load`.

Steps marked **(verify on the provider's site)** depend on a provider's current UI, plans or free-tier limits, which change; check them before relying on them.

Before making the demo public, check that the Yelp Open Dataset terms allow it for your purpose **(verify on Yelp's site)**.

Never commit `DATABASE_URL` (or any connection string) to the repo, `.env.example`, `deploy/`, or the image; set it only as a secret on Render and in your local `.env` (gitignored).

### Free-tier limits (checked 2026-10-24)

| provider | limit | source |
| --- | --- | --- |
| Render free web service | 0.1 CPU, 512 MB RAM (plan id `free`) | [render.com/docs/compute-plans](https://render.com/docs/compute-plans) |
| | spins down after 15 minutes without inbound traffic; spinning back up "takes about one minute" | [render.com/docs/free](https://render.com/docs/free) |
| | 750 free instance hours per workspace per calendar month; outbound bandwidth and build minutes count against the workspace's included amounts (no figures on that page) | [render.com/docs/free](https://render.com/docs/free) |
| Neon Free | 1 GB storage per project (20 GB across projects), 100 CU-hours per project per month | [neon.com/docs/introduction/plans](https://neon.com/docs/introduction/plans) |
| | compute scales to zero after 5 minutes of inactivity; can't be disabled on Free | [neon.com/docs/introduction/plans](https://neon.com/docs/introduction/plans) |
| | pgvector "is available on every Neon plan" | [neon.com/docs/extensions/pgvector](https://neon.com/docs/extensions/pgvector) |

These change; recheck them before relying on them **(verify on the provider's site)**. The local dry run of `src.deploy_load` (500 demo users) used 16.2 MB including indexes, well under Neon's 1 GB.

## (a) Neon: database and the vector extension

1. Create a Neon account and a new project **(verify on the provider's site)**. Pick a region close to where the API will run.
2. Copy the project's connection string (Postgres URL, with `sslmode=require`) from the project's connection details **(verify on the provider's site)**. Treat it as a secret: never commit it, and don't paste it into `.env.example`.
3. Enable pgvector. In Neon's SQL editor, or with `psql "<connection string>"`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   `src.deploy_load` also runs this; pgvector is available on Neon Free (see the table above).
4. Load the data from your machine (repo root, `.venv` active, `data/processed/` built, ideally with the taste cache from `python -m src.recommend build-cache --model ContentBasedDebiased`):
   ```
   python -m src.deploy_load --database-url "<connection string>" --demo-users 500
   ```
   It applies `db/schema.sql` and `db/groups.sql`, loads `restaurants` through `src.db`, then the demo users' taste vectors, top-5 profiles (restaurant IDs and stars only), visited-restaurant IDs, the model's hub vector and 100 sample groups. At the end it prints the row count and size of each table (about 16 MB in the dry run).
5. Optional check: `DATABASE_URL="<connection string>" python -m src.db check`.

Re-running `deploy_load` replaces the restaurants and group tables. Only the 500 demo users have taste vectors in the database; the API returns 422 for any other user ID.

## (b) The API on a Render free web service

The image is built from the repo's `Dockerfile` (Person B's), running `uvicorn src.api.main:app`. **No torch in the API image:** the image installs `requirements-api.txt`, and the search encoder runs MiniLM via onnxruntime (`ENCODER=onnx` is set in the image, which downloads its own pinned model files). The API image must not include sentence-transformers or torch: with them it won't fit in the free instance's 512 MB. The group API needs neither (its endpoints run with both blocked).

1. In the Render dashboard, create a new **Web Service** from this GitHub repository, branch `main`, runtime **Docker**, instance type **Free** **(verify on the provider's site)**.
2. Port: Render routes to the port the service listens on and provides a `PORT` environment variable. Set the start command (Docker command override) to `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`, or confirm Render detects port 8000 **(verify on the provider's site)**.
3. Under **Environment**, add these as environment variables / secrets, never in the image or the repo **(verify on the provider's site for where secrets go)**:
   | name | value |
   | --- | --- |
   | `DATABASE_URL` | the Neon connection string from (a) |
   | `DATA_SOURCE` | `db` |
   | `DEFAULT_ENGINE` | `rules` (no Ollama on the host) |
4. Set the health check path to `/health` **(verify on the provider's site)**.
5. Deploy, then check `https://<service>.onrender.com/health` returns `{"status": "ok"}` and `/docs` loads. The first group request after a start reads the database tables, which takes a few seconds.

### Cold starts

- Render spins the free service down after 15 minutes without traffic; the next request wakes it, which takes about a minute.
- Neon suspends idle compute after 5 minutes; the first query afterwards also waits for it to resume.
- The UI waits it out: `src/ui_common.py` shows "Waking up the server, about a minute…" and retries for up to 90 s, and retries 503s that carry `Retry-After` (e.g. search while the encoder warms up).

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
6. After an idle period the first page load shows the wake-up message for about a minute; see "Cold starts" above.

## Checklist

- [ ] Neon project created, `vector` extension available
- [ ] `src.deploy_load` printed the expected row counts
- [ ] API image built from `requirements-api.txt` (no torch or sentence-transformers)
- [ ] API deployed on Render (Free) with `DATABASE_URL`, `DATA_SOURCE=db`, `DEFAULT_ENGINE=rules` as secrets; `/health` OK
- [ ] UI deployed with `API_URL` as a secret; the badge shows "API online"
- [ ] No connection strings or data files committed (`git status`, `git log -p`)
