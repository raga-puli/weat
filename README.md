# 🍽️ weat — Group Restaurant Recommender

**Whose taste wins when five friends pick one dinner?**

A Beli-inspired restaurant recommender for *groups*, built on Yelp reviews of 3,902 Philadelphia restaurants. Pick a group of diners, choose how to balance their tastes, and get one list for the table. Or search in plain English, like *"cozy date spot, under $40, walkable from campus."*

![Python](https://img.shields.io/badge/python-3.11-blue)
![Postgres](https://img.shields.io/badge/postgres-pgvector-336791)
![Streamlit](https://img.shields.io/badge/demo-streamlit-FF4B4B)
![FastAPI](https://img.shields.io/badge/api-fastapi-009688)
![Docker](https://img.shields.io/badge/docker-compose-2496ED)

<!-- TODO: add a screenshot or GIF of the Group Dinner page -->
<img src="assets/weat.jpg" width="70%">

**[Live demo →](#)** [https://thorfinn.streamlit.app/](url)

---

## ✨ Headline result

> **Pre-registered and confirmed on a held-out test set.** In groups with divergent tastes, *sequential fairness* (τ=0.3) raised the worst-off member's predicted satisfaction **1.6× more than least-misery** (+0.047 vs +0.029 over average), at **a third less group-utility cost** (−0.012 vs −0.018). It showed no detectable loss in hit rate.

Everything here is *predicted* satisfaction (see [What we can and can't claim](#-what-we-can-and-cant-claim)).

---

## 🧭 What it does

| | |
|---|---|
| 👥 **Group recommendations** | Score every restaurant for each member, then aggregate with one of four strategies: **average**, **least-misery**, **fairness**, or **sequential fairness** (built for groups that eat together repeatedly and take turns). |
| 💬 **Natural-language search** | Hard constraints like price, distance, dietary needs and "open now" are parsed into structured filters by a rule-based parser or a local LLM via Ollama. Soft constraints like "cozy" are matched by embedding similarity. |
| ⚡ **Hybrid retrieval** | Filters and vector search run as one SQL query on Postgres + pgvector. |
| 🚪 **Closed-restaurant handling** | Closed restaurants stay in offline evaluation, because held-out reviews are historical, but are hidden from user-facing results by default. |

### Why hard and soft constraints are kept separate

"Under $40" is a fact, not a vibe. If you put it into an embedding, the model *might* respect it. If you parse it into a SQL `WHERE` clause, it *always* does. Only the genuinely fuzzy part of a query goes through semantic search.

---

## 🏗️ Architecture

```
                 ┌──────────────┐
  "cozy, <$40" → │  Streamlit   │   Group Dinner · Search
                 └──────┬───────┘
                        │ HTTP
                 ┌──────▼───────┐      ┌──────────────────────┐
                 │   FastAPI    │ ───► │ Query parser          │
                 │  /docs       │      │ rules  |  Ollama LLM  │
                 └──────┬───────┘      └──────────────────────┘
                        │ SQL: WHERE filters + vector similarity
                 ┌──────▼───────┐
                 │  Postgres +  │
                 │   pgvector   │
                 └──────────────┘

  Offline:  Yelp JSON ─DuckDB─► parquet ─► time-based split ─► MiniLM per-review
            embeddings (train reviews only) ─► taste vectors / ALS ─► Postgres
```

**DuckDB offline, Postgres online, on purpose.** DuckDB scans the raw Yelp JSON without loading it into memory. Postgres handles filtered, transactional serving. Different access patterns call for different tools.

---

## 📊 Results

Splits are time-based: each user's most recent reviews are held out as validation and test. Restaurant embeddings and taste profiles use **training-split reviews only**. Configurations were frozen in `docs/frozen_configs.md` before test was touched, and each test invocation ran exactly once. Every run is logged in [`results.md`](results.md) and [`results/groups.md`](results/groups.md).

### Group aggregation (test set)

~3,000 synthetic dining groups of real users who reviewed the same restaurant within a short window. The groups were split at the median taste cohesion (0.119) into high- and low-cohesion halves.

**Pre-registered hypothesis** (committed 2026-10-17, a week before the test run): *in low-cohesion groups, sequential fairness with τ=0.3 yields a higher worst-member predicted percentile than least-misery, at a smaller group-score cost.*

| Low-cohesion groups (1,497) | vs. least-misery [95% CI] | Verdict |
|---|---|---|
| Worst-member predicted score | **+0.019** [+0.018, +0.019] | ✅ held |
| Group-score cost | **+0.005** [+0.005, +0.006] smaller | ✅ held |

Change vs. **average** aggregation, low-cohesion groups:

| Strategy | Worst member | Group score | Hits@10 |
|---|---|---|---|
| Average | — | — | 19 |
| Least-misery | +0.029 | −0.018 | 12 |
| **Sequential fairness (τ=0.3)** | **+0.047** | **−0.012** | 16 |

Hit@10 for sequential fairness (τ=0.3) by cohesion half:

| Half | hit@10 [95% CI] | Δ vs. average [95% CI] |
|---|---|---|
| High cohesion | 0.0227 [0.0140, 0.0318] | 0.000 [−0.004, +0.004] |
| Low cohesion | 0.0107 [0.0052, 0.0168] | −0.002 [−0.007, +0.003] |
| All | 0.0167 [0.0109, 0.0229] | −0.001 [−0.004, +0.002] |

Confidence intervals are paired cluster bootstraps over target restaurants.

**What we learned:**
- **Strategy barely matters for like-minded groups and matters a lot for divergent ones.** Least-misery and fairness lift the worst member about 2.6× more in low-cohesion groups than in high-cohesion ones. This replicated on test, matching validation almost exactly.
- **Per-dinner fairness and cumulative fairness are different goals.** Least-misery makes *each* dinner OK for everyone. Sequential fairness takes turns, so each dinner suits someone and it evens out over many dinners. For a one-off dinner, use least-misery. For a recurring group, take turns.
- **Our first pre-registered setting failed, and we report it.** τ=0.1 was fixed in advance, failed on validation, and failed again on test. A sensitivity analysis on validation (τ ∈ {0.1, 0.3, 1.0, 3.0}, all reported) found τ=0.3, which was then pre-registered and confirmed.

### Individual recommendations (test set, 25,439 users)

Each frozen configuration ran once on test.

| Model | Recall@10 | NDCG@10 | MRR | Coverage | Spearman ρ |
|---|---|---|---|---|---|
| **ALS, 64 factors** (frozen default) | **0.0559** | **0.0289** | **0.0207** | <!-- TODO --> | 0.058 |
| Popularity (train counts only) | 0.0482 | 0.0250 | 0.0181 | 0.010 | 0.040 |
| Random | 0.0020 | 0.0008 | 0.0005 | 1.000 | −0.004 |
| Content-based, per-review chunked | 0.0089 | 0.0042 | 0.0028 | 0.795 | 0.035 |
| Content-based, single blob (ablation) | 0.0065 | 0.0029 | 0.0019 | 0.811 | 0.058 |

**ALS factor sizes (validation; only the 64-factor default ran on test):**

| Model | Recall@10 | NDCG@10 | Coverage | Spearman ρ |
|---|---|---|---|---|
| Popularity (val, for reference) | 0.0505 | 0.0270 | 0.010 | 0.044 |
| ALS, 32 factors | 0.0745 | 0.0390 | 0.163 | 0.070 |
| **ALS, 64 factors (pre-set default)** | **0.0695** | **0.0364** | 0.238 | 0.063 |
| ALS, 128 factors | 0.0666 | 0.0347 | 0.340 | 0.047 |

- **Collaborative filtering beats popularity on test: +16% recall@10, +16% NDCG@10, +14% MRR** (ALS-64 vs. popularity). ALS-64 was frozen before the test run and run once. The 32-factor model scored highest on validation but was never run on test, because 64 was the default fixed in advance.
- **The lift shrank from validation to test:** +38% recall@10 on validation, +16% on test. We report the test number as the result.
- **Popularity is a strong baseline:** 24× random on test. Every model has to clear it.
- **Content embeddings lose to popularity at predicting the next visit, and we say so.** On test, popularity's recall@10 is about 5× the content model's. What embeddings offer instead is reach: popularity recommends the same 38 restaurants to nearly everyone, while content-based covers 80% of the catalog. Embeddings are also what make queries like "cozy" possible.
- **Per-review encoding beats single-blob encoding:** +37% recall@10 and +43% NDCG@10 on test. The blob version truncated at 256 tokens, before it reached most reviews.
- **ALS trades accuracy for coverage.** Going from 32 to 128 factors, recall falls from 0.0745 to 0.0666 while coverage doubles.
- **Hubness correction.** Raw embeddings pushed a few "generic" restaurants to everyone. Centering the user-by-restaurant score matrix cut the top restaurant's share of group lists from 12% to 7.5%, at a recall cost within noise.

### System performance

- **API latency:** ~0.05 s per request against Postgres. DB mode returned identical responses to file mode on 81/81 real requests.
- **pgvector `ivfflat`:** with `probes=1`, recall against exact search is only 0.485. This is the approximate-search tradeoff in action, and at ~3,900 restaurants exact search is cheap enough anyway.
- **ONNX encoder** *(pending merge)*: same top-10 lists on 40/40 queries, with startup going from 38 s / 568 MB to 1.3 s / 196 MB.

---

## 🧪 What we can and can't claim

We have no live users, so we never **measure** satisfaction. We **predict** it and validate the predictor.

- **The individual models are checked against real held-out ratings** from real people.
- **Group scores are constructed from those predictions.** "Worst-member satisfaction" means *the lowest predicted percentile in the group*. These numbers are meaningful as comparisons between strategies on identical inputs, not as claims about how happy anyone was.
- **Calibration is weak, and we lead with that.** Spearman ρ between predicted scores and actual held-out stars is ≈0.01 for content-based and ≈0.06 for ALS. Part of this is restricted range, since most held-out reviews are 4–5★, but every group metric above rests on these scores.
- **Effects are a few percentile points.** They are statistically clear but small.

---

## 🚀 Quickstart

```bash
git clone https://github.com/raga-puli/weat.git && cd weat
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
python src/prep.py       # Yelp JSON → parquet (Philadelphia only)
python src/split.py      # time-based train / val / test
python src/embed.py      # per-review embeddings, train reviews only
python -m src.groups     # synthetic dining groups
```

**4. Run the stack.**
```bash
ollama pull qwen2.5:3b   # optional: LLM parser (the rule-based parser works without it)
docker compose up
```

- App: http://localhost:8501
- API docs: http://localhost:8000/docs

**Evaluate:**
```bash
python -m src.eval --model src.baselines:Popularity
python -m src.group_eval --tau 0.3
```

> ⚠️ **Windows users:** keep the project out of OneDrive. Cloud-placeholder files break torch's DLL loading (`WinError 1114 ... c10.dll`).

---

## 🗂️ Data

The [Yelp Open Dataset](https://www.yelp.com/dataset), filtered to Philadelphia restaurants with 20+ reviews: **3,902 restaurants** and ~666K reviews. We use the Open Dataset rather than the Yelp Fusion API, which caps reviews at three excerpts per business.

Every training-split review is used to describe *restaurants*, since a one-time reviewer still describes a place fine. Taste profiles are built only for users with 5+ reviews.

Restaurant vectors are built by encoding reviews individually with `all-MiniLM-L6-v2` and taking a star-balanced mean. The earlier approach encoded one long blob per restaurant, which got truncated at 256 tokens before the negative reviews were even read. That version is kept as an ablation.

---

## 📁 Repo layout

```
src/
  prep.py          DuckDB extraction → parquet
  split.py         leakage-free time-based splits
  embed.py         per-review sentence-transformers encoding
  recommend.py     content-based models, taste profiles
  baselines.py     random, popularity
  groups.py        synthetic group construction
  group_eval.py    aggregation strategies + bootstrap evaluation
  eval.py          individual-model evaluation runner
  deploy_load.py   load artifacts into Postgres
  app.py           Streamlit (Group Dinner, Search)
docs/
  frozen_configs.md   configs fixed before any test run
results.md            individual-model runs
results/              group and parser runs
tests/                366 tests
```

<!-- TODO: confirm paths for the ALS model, API, and parser modules -->

---

## 🚧 Known limitations

- **No ground truth for groups.** Nobody logs "we all agreed on Thai." Groups of real co-raters are a reasonable proxy, not the real thing.
- **Possible leak in some test groups.** 730 of 2,994 test groups were flagged for a possible target-embedding leak. They are reported, not dropped.
- **Few hits per cell.** Low-cohesion hit-rate comparisons rest on 10–19 hits per cell, so treat them as fragile.
- **Validation configurations tried.** <!-- TODO: total count from the val log -->. All of them are reported in `results/`. Test was run once per frozen configuration.
- **City spelling.** Yelp's `city` field is free text, so a few Philadelphia restaurants listed under spelling variants may be missing.
- **Hosted demo uses the rule-based parser.** The LLM parser needs a local Ollama instance. <!-- TODO: rules-vs-LLM accuracy from results/parser.md -->

---


<sub>Data © Yelp, used under the Yelp Dataset Terms of Use for non-commercial academic purposes.</sub>
