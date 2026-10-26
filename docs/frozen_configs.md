# Frozen test configurations

Frozen on: 2026-10-24

Each model is run on the test split exactly once, on Day 1, with the invocation below. Never rerun test to get a better number. Val rows cited are from `results.md` (individual) and `results/groups.md` (groups).

## Individual users (`src.eval`, logs to `results.md`)

| model | exact CLI invocation | why this config (val rows) |
| --- | --- | --- |
| random | `python -m src.eval --model src.baselines:Random --split test --i-mean-it --log "test, frozen"` | Floor baseline. `results.md` 2026-10-02 val random (recall@10 0.0024) and 2026-10-16 rerun (same numbers, plus diversity). No parameters. |
| popularity | `python -m src.eval --model src.baselines:Popularity --split test --i-mean-it --log "test, frozen"` | Strongest val baseline: `results.md` 2026-10-02 / 2026-10-16 val popularity, recall@10 0.0505, NDCG@10 0.0270. Train counts only. |
| content_chunked | `python -m src.eval --model src.recommend:ContentBased --split test --i-mean-it --config "chunked, star-balanced mean" --log "test, frozen"` | Default embedding mode. `results.md` 2026-10-16 val content_chunked (both rows identical): recall@10 0.0100, NDCG@10 0.0047, beats blob on recall/NDCG. Needs `embeddings.npy` from `python -m src.embed` (chunked). |
| content_blob | `python -m src.eval --model src.recommend:ContentBasedBlob --split test --i-mean-it --config "blob, truncated at 256 tokens" --log "test, frozen"` | Ablation, reported regardless. `results.md` 2026-10-16 val content_blob: recall@10 0.0062, NDCG@10 0.0032. Needs `embeddings_blob.npy` from `python -m src.embed --mode blob`. |
| ALS | `python -m src.eval --model src.mf:ALS --split test --i-mean-it --config "implicit ALS, 64 factors, reg 0.05, 20 iters, seed 42" --log "FINAL TEST"` (`src.mf:ALS`: 64 factors, reg 0.05, 20 iterations, seed 42, confidence 1 + 10*[stars >= 4]) | Fixed as default before any val run; 32 and 128 reported on val only. |

## Groups (`src.group_eval`, logs to `results/groups.md`)

All use `--model ContentBasedCentered` (the README headline and pre-registration model), `--norm rank`, k=10, groups v3.

| strategy | norm / lam | exact CLI invocation | why this config (val rows) |
| --- | --- | --- | --- |
| average | rank / – | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy average --log "test, frozen"` | `results/groups.md` 2026-10-17 val average: hit@10 0.0191 [0.0154, 0.0235], highest of the content strategies. |
| least_misery | rank / – | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy least_misery --log "test, frozen"` | `results/groups.md` 2026-10-17 val least_misery: hit@10 0.0160, worst member 0.8790 (highest), fairness gap 0.0470. Pre-registration comparator. |
| fairness | rank / lam=0.5 | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --lam 0.5 --strategy fairness --log "test, frozen"` | `results/groups.md` 2026-10-17 val fairness, lam=0.5 (the only lam run on val): hit@10 0.0163, worst member 0.8789. |
| fairness_repeated | rank / – (tau=0.1) | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy fairness_repeated --log "test, frozen"` | `results/groups.md` 2026-10-17 val fairness_repeated, tau=0.1: fairness gap 0.0283 (lowest). tau=0.1 is the original pre-registered default, reported regardless of outcome (README "Pre-registered for test"). |
| fairness_repeated (τ=0.3) | rank / – (tau=0.3) | `python -m src.group_eval --split test --i-mean-it --model ContentBasedCentered --norm rank --strategy fairness_repeated --tau 0.3 --log "test, frozen"` | Pre-registered hypothesis (README "Pre-registered for test", 2026-10-17): τ=0.3 gives a higher worst-member predicted percentile than least_misery in low-cohesion groups, at a smaller group-score cost. Val evidence: τ sensitivity run (README val run 6, `group_eval_val_tau_sensitivity.csv`; not logged to `results/groups.md`), where in low-cohesion groups the per-restaurant worst member is 0.753 [0.749, 0.757] at τ=0.3 vs. 0.645 [0.638, 0.652] at τ=0.1. |

**Amendment, 2026-10-24:** the τ=0.3 row was left out of the original freeze and added after the four group test runs above. τ=0.3 comes from the 2026-10-17 pre-registration, not from the test results. Requires a `--tau` flag in `src.group_eval`, which does not exist yet (Person A will add it). Run once.
