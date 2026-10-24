# Frozen test configurations

Frozen on: <date>

Each model is run on the test split exactly once, on Day 1, with the invocation below. Never rerun test to get a better number. Val rows cited are from `results.md` (individual) and `results/groups.md` (groups).

## Individual users (`src.eval`, logs to `results.md`)

| model | exact CLI invocation | why this config (val rows) |
| --- | --- | --- |
| random | `python -m src.eval --model src.baselines:Random --split test --i-mean-it --log "test, frozen"` | Floor baseline. `results.md` 2026-10-02 val random (recall@10 0.0024) and 2026-10-16 rerun (same numbers, plus diversity). No parameters. |
| popularity | `python -m src.eval --model src.baselines:Popularity --split test --i-mean-it --log "test, frozen"` | Strongest val baseline: `results.md` 2026-10-02 / 2026-10-16 val popularity, recall@10 0.0505, NDCG@10 0.0270. Train counts only. |
| content_chunked | `python -m src.eval --model src.recommend:ContentBased --split test --i-mean-it --config "chunked, star-balanced mean" --log "test, frozen"` | Default embedding mode. `results.md` 2026-10-16 val content_chunked (both rows identical): recall@10 0.0100, NDCG@10 0.0047, beats blob on recall/NDCG. Needs `embeddings.npy` from `python -m src.embed` (chunked). |
| content_blob | `python -m src.eval --model src.recommend:ContentBasedBlob --split test --i-mean-it --config "blob, truncated at 256 tokens" --log "test, frozen"` | Ablation, reported regardless. `results.md` 2026-10-16 val content_blob: recall@10 0.0062, NDCG@10 0.0032. Needs `embeddings_blob.npy` from `python -m src.embed --mode blob`. |
| ALS | **TBD** — one of `src.mf:ALS32` / `src.mf:ALS` (64) / `src.mf:ALS128` | **No ALS val row exists in `results.md`** (or in README). Person A must log val runs for the configs tried, then freeze the best one here. |

## Groups (`src.group_eval`, logs to `results/groups.md`)

All use `--model ContentBasedCentered` (the README headline and pre-registration model), `--norm rank`, k=10, groups v3.

| strategy | norm / lam | exact CLI invocation | why this config (val rows) |
| --- | --- | --- | --- |
| average | rank / – | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy average --log "test, frozen"` | `results/groups.md` 2026-10-17 val average: hit@10 0.0191 [0.0154, 0.0235], highest of the content strategies. |
| least_misery | rank / – | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy least_misery --log "test, frozen"` | `results/groups.md` 2026-10-17 val least_misery: hit@10 0.0160, worst member 0.8790 (highest), fairness gap 0.0470. Pre-registration comparator. |
| fairness | rank / lam=0.5 | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --lam 0.5 --strategy fairness --log "test, frozen"` | `results/groups.md` 2026-10-17 val fairness, lam=0.5 (the only lam run on val): hit@10 0.0163, worst member 0.8789. |
| fairness_repeated | rank / – (tau=0.1) | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy fairness_repeated --log "test, frozen"` | `results/groups.md` 2026-10-17 val fairness_repeated, tau=0.1: fairness gap 0.0283 (lowest). tau=0.1 is the original pre-registered default, reported regardless of outcome (README "Pre-registered for test"). |
