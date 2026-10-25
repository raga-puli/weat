# 🍽️ weat — Group Restaurant Recommender

**Whose taste wins when five friends pick one dinner?**

A Beli-inspired restaurant recommender for *groups*, built on 666K Yelp reviews of Philadelphia restaurants. Ask in plain English — *"cozy date spot, under $40, walkable from campus"* — and get recommendations that balance everyone at the table, not just the loudest eater.

![Python](https://img.shields.io/badge/python-3.11-blue)
![Postgres](https://img.shields.io/badge/postgres-pgvector-336791)
![Streamlit](https://img.shields.io/badge/demo-streamlit-FF4B4B)
![FastAPI](https://img.shields.io/badge/api-fastapi-009688)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

<img src="assets/weat.jpg" width="70%">

**[Live demo →](#)** <!-- TODO: Streamlit Community Cloud URL -->

---

## ✨ What it does

| | |
|---|---|
| 👥 **Group recommendations** | Score every restaurant for each member, then aggregate with one of three strategies — average utility, least-misery, or fairness-weighted across repeated dinners — and compare what each one picks. |
| 💬 **Natural-language search** | A local LLM (via Ollama) parses hard constraints — price, distance, dietary, open now — into structured filters. Soft constraints like "cozy" are matched by embedding similarity. |
| ⚡ **Hybrid retrieval** | Filters and vector search run in a single SQL query on Postgres + pgvector. |

### Why hard and soft constraints are kept separate

"Under $40" is a fact, not a vibe. Stuffing it into an embedding means the model *might* respect it. Parsing it into a SQL `WHERE` clause means it *always* does. Only the genuinely fuzzy part of the query ("cozy") goes through semantic search.

---

## 🏗️ Architecture

```
                 ┌──────────────┐
  "cozy, <$40" → │  Streamlit   │
                 └──────┬───────┘
                        │ HTTP
                 ┌──────▼───────┐      ┌───────────────┐
                 │   FastAPI    │ ───► │ Ollama (LLM)  │  query → JSON filters
                 └──────┬───────┘      └───────────────┘
                        │ SQL: WHERE filters + vector <=> taste
                 ┌──────▼───────┐
                 │  Postgres +  │
                 │   pgvector   │
                 └──────────────┘

  Offline:  Yelp JSON ──DuckDB──► parquet ──MiniLM──► embeddings ──► Postgres
```

**DuckDB offline, Postgres online — on purpose.** DuckDB scans 5GB of raw JSON without loading it into memory; Postgres handles transactional, filtered serving. Different access patterns, different tools.

---

## 📊 Results

All numbers come from scripts in this repo and are logged run-by-run in [`results.md`](results.md). Accuracy is measured on a time-based holdout: each user's last 1–2 reviews were held out *before* building any taste profile.

### Individual recommendations

| Model | Recall@10 | NDCG@10 | Catalog coverage |
|---|---|---|---|
| Random | — | — | — |
| Most-reviewed (popularity) | — | — | — |
| Content-based (embeddings) | — | — | — |
| Matrix factorization | — | — | — |

<!-- TODO: fill from results.md. Headline claim = lift over the popularity baseline. -->

**Calibration:** Spearman ρ between predicted scores and held-out star ratings = **—** <!-- TODO -->

### Group aggregation

Evaluated on synthetic groups of 3–4 real users who reviewed the same restaurant within a few weeks of each other, stratified by how similar the members' tastes are.

| Strategy | Mean predicted score | Worst-member predicted score | Fairness gap |
|---|---|---|---|
| Average utility | — | — | — |
| Least-misery | — | — | — |
| Fairness-weighted | — | — | — |

<!-- TODO: fill, and add the cohesion-bucket comparison (high vs. low pairwise similarity). -->

**Key finding:** <!-- TODO: e.g. "strategy choice barely matters for like-minded groups and matters a lot for divergent ones" — only if the numbers show it. -->

### System performance

| Backend | p95 latency (unfiltered) | p95 latency (filtered) |
|---|---|---|
| numpy dot product | — | — |
| pgvector (`ivfflat`) | — | — |

At 3,894 restaurants, numpy may well be faster. Postgres was chosen for filtering and persistence, not raw speed.

> **A note on language.** We have no live users, so we never *measure* satisfaction — we predict it and validate the predictor. Group scores are **predicted satisfaction**, built from individual predictions that *are* checked against real held-out ratings. They are meaningful as comparisons between strategies on identical inputs, not as claims about how happy anyone was.

---

## 🚀 Quickstart

```bash
git clone <repo-url> && cd rnjsjffoodw
```

**1. Get the data.** Download the [Yelp Open Dataset](https://www.yelp.com/dataset) and extract it to `data/raw/`.

**2. Set up Python.**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**3. Build the offline artifacts.**
```bash
python src/prep.py     # Yelp JSON → parquet (Philadelphia only)
python src/embed.py    # → data/processed/embeddings.npy  (3894, 384)
```

**4. Run the stack.**
```bash
ollama pull <model>    # TODO: model name used by the query parser
docker compose up
```

- App: http://localhost:8501
- API docs: http://localhost:8000/docs

> ⚠️ **Windows users:** keep the project out of OneDrive. Cloud-placeholder files break torch's DLL loading (`WinError 1114 ... c10.dll`).

---

## 🗂️ Data

[Yelp Open Dataset](https://www.yelp.com/dataset), filtered to Philadelphia.

| Artifact | Count |
|---|---|
| Restaurants (`review_count ≥ 20`) | 3,894 |
| Reviews of those restaurants | 666,113 |
| Users with 5+ reviews | 26,061 |
| Reviews by those users | 399,448 |

All 666K reviews are used to describe *restaurants* — a one-time reviewer still describes a place fine. The 5+ review filter only applies to users we build *taste profiles* for.

**Restaurant embeddings:** for each restaurant, take the 15 longest positive (≥4★) and 15 longest negative reviews (200–2000 chars), prepend name and categories, and encode with `all-MiniLM-L6-v2`. Long reviews carry atmosphere and crowd detail that the `categories` field never will.

We use the Open Dataset rather than the Yelp Fusion API, which caps reviews at three excerpts per business.

---

## 📁 Repo layout

```
src/
  prep.py        DuckDB extraction → parquet
  embed.py       review blobs + sentence-transformers
  recommend.py   taste profiles, scoring, group aggregation
  app.py         Streamlit UI
  api/           FastAPI service          # TODO: confirm path
notebooks/       exploration + eval
results.md       every experiment, every run
docker-compose.yml
```

---

## 🚧 Known limitations

- **Truncated embeddings.** MiniLM reads only the first 256 tokens of each review blob. Chunk-and-average would use the whole thing.
- **No ground truth for groups.** Nobody logs "we all agreed on Thai." Synthetic groups of real co-raters are a reasonable proxy, not the real thing.
- **Small human eval.** <!-- TODO: N -->  people compared unlabeled top-5 lists from two strategies. Real signal, small sample.
- **City spelling.** Yelp's `city` field is free text, so a few Philadelphia restaurants listed under variants may be missing.
- **Experiment count.** <!-- TODO: be honest about how many configurations were tried before picking the reported one. -->

---

## 👥 Team

Built by two people, split by **vertical feature slice** rather than by layer — each of us owns a slice from data through model, SQL, and UI.

| | Owns |
|---|---|
| **<!-- name -->** | Group aggregation strategies & evaluation |
| **<!-- name -->** | Natural-language query parser & hybrid search |

Each of us wrote up the *other's* component — if you can't explain it, you don't understand it.

---

<sub>Data © Yelp, used under the Yelp Dataset Terms of Use for non-commercial academic purposes.</sub>
