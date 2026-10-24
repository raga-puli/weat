# weat

<img src="assets/weat.jpg" width="70%">



A restaurant recommendation system built on Yelp-style business, review, and embedding data.

The app is called WEat — a combination of "we" and "eat".

## Counts

| metric | value |
| --- | --- |
| restaurants (review_count >= 20) | 3902 |
| their reviews | 666849 |
| users (5+ reviews) | 26093 |
| those users' reviews | 399881 |

restaurants.parquet
  business_id (str), name (str), latitude (float), longitude (float),
  stars (float), review_count (int), categories (str),
  attributes (struct of Yelp attribute strings; price level is derived from RestaurantsPriceRange2 when loading Postgres),
  hours (JSON str): object keyed by weekday, values "H:M-H:M" (minutes not zero-padded);
    days the restaurant is closed or unlisted are null; whole field null for 318 restaurants with no hours,
  is_open (bool): open in the dataset snapshot; 2,453 open, 1,449 closed. Not filtered: closed restaurants stay in
    every offline step (embeddings, splits, evaluation)
  Rows are sorted by business_id, the same order as business_ids.txt.

reviews.parquet
  review_id (str), user_id (str), business_id (str),
  stars (int), text (str), date (timestamp)

embeddings.npy    -> (n_restaurants, 384) float32
business_ids.txt  -> one business_id per line, same order as embeddings

groups.parquet (src/groups.py, built from train_reviews only)
  group_id (str): 12-char content hash of members + target
  members (list[str]): 3-4 distinct user_ids, all from the val split, sorted; each has >= 3 train
    reviews with >= 4 stars dated strictly before window_start; no user is in more than 10 groups
  target_business_id (str): held-out restaurant every member rated >= 4 stars within the window
    (at most 20 groups per target)
  window_start (timestamp): earliest member's review of the target; all members' reviews fall within 30 days of it
  size (int): number of members (3 or 4)
  split (str): "val" or "test", 70/30, deterministic from SEED + group_id

## Group evaluation

Setup: synthetic co-rating groups of 3–4 users who each liked (≥ 4 stars) the same target restaurant within 30 days of each other (`src/groups.py`). Each member's profile is built only from their 4+ star train reviews dated strictly before the group's window start, with the target excluded; every member has at least 3 such prior likes, and no user is in more than 10 groups. Candidates are all restaurants minus anything a member reviewed before the window, and the target is always a candidate. The model is `ContentBasedCentered`; each member's scores are converted to rank percentiles over the candidate set before aggregation, so all metrics below are in terms of predicted satisfaction. Results are on the val group split only (n = 7,008 groups over 1,310 target restaurants, k = 10); 95% CIs are cluster bootstraps over targets (1,000 resamples; all groups for a sampled target are kept together). Source: `python -m src.group_eval` (`data/processed/group_eval_val.csv`). fairness_repeated's top-10 is a sequence of 10 dinners, so its hit@10 means "the target came up within 10 dinners."

| strategy | hits | hit@10 [95% CI] | NDCG@10 [95% CI] | group score | worst member | fairness gap |
| --- | --- | --- | --- | --- | --- | --- |
| random | 14 | 0.0020 [0.0010, 0.0032] | 0.0008 [0.0004, 0.0013] | 0.499 | 0.428 | 0.141 |
| popularity | 190 | 0.0271 [0.0159, 0.0400] | 0.0120 [0.0062, 0.0186] | 0.579 | 0.484 | 0.186 |
| average | 134 | 0.0191 [0.0154, 0.0235] | 0.0094 [0.0073, 0.0120] | 0.914 | 0.860 | 0.096 |
| least_misery | 112 | 0.0160 [0.0122, 0.0198] | 0.0079 [0.0059, 0.0100] | 0.901 | 0.879 | 0.047 |
| fairness | 114 | 0.0163 [0.0125, 0.0200] | 0.0081 [0.0061, 0.0102] | 0.904 | 0.879 | 0.049 |
| fairness_repeated (τ=0.1) | 119 | 0.0170 [0.0130, 0.0213] | 0.0086 [0.0065, 0.0111] | 0.885 | 0.872 | 0.028 |

Group score (mean over the list of the members' mean predicted percentile), worst member (lowest member's mean predicted percentile across the list), and fairness gap (highest minus lowest member mean) use the same predicted percentiles the content strategies optimize, so they compare content strategies with each other; only hit@10 and NDCG@10 are external measures. On hit@10 vs. average (cluster bootstrap), least_misery (−0.0031 [−0.0053, −0.0009]) and fairness (−0.0029 [−0.0049, −0.0009]) are lower; popularity (+0.0080 [−0.0040, +0.0217]) and fairness_repeated (−0.0021 [−0.0045, +0.0003]) are not distinguishable from average.

Modeled-metric differences vs. average by group cohesion (mean pairwise cosine of member taste vectors, split at the val median):

| cohesion | strategy − average | group score [95% CI] | worst member [95% CI] | fairness gap [95% CI] |
| --- | --- | --- | --- | --- |
| low (n=3504) | least_misery | −0.0179 [−0.0184, −0.0175] | +0.0279 [0.0265, 0.0293] | −0.0731 [−0.0755, −0.0707] |
| low (n=3504) | fairness | −0.0153 [−0.0156, −0.0149] | +0.0277 [0.0264, 0.0291] | −0.0719 [−0.0744, −0.0696] |
| low (n=3504) | fairness_repeated (τ=0.1) | −0.0496 [−0.0515, −0.0478] | +0.0074 [0.0057, 0.0090] | −0.0980 [−0.1005, −0.0954] |
| high (n=3504) | least_misery | −0.0071 [−0.0074, −0.0068] | +0.0101 [0.0095, 0.0109] | −0.0258 [−0.0270, −0.0246] |
| high (n=3504) | fairness | −0.0050 [−0.0052, −0.0048] | +0.0101 [0.0095, 0.0108] | −0.0235 [−0.0247, −0.0224] |
| high (n=3504) | fairness_repeated (τ=0.1) | −0.0082 [−0.0087, −0.0077] | +0.0160 [0.0153, 0.0167] | −0.0382 [−0.0399, −0.0367] |

hit@10 by the target's pre-window review count (terciles):

| target popularity | n | random | popularity | average | least_misery | fairness | fairness_repeated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | 2336 | 0.0021 | 0.0021 | 0.0231 | 0.0214 | 0.0210 | 0.0188 |
| mid | 2336 | 0.0021 | 0.0060 | 0.0188 | 0.0141 | 0.0154 | 0.0150 |
| high | 2336 | 0.0017 | 0.0732 | 0.0154 | 0.0124 | 0.0124 | 0.0171 |

Calibration: Spearman correlation between predicted scores and held-out star ratings is ~0.01 on val. Predicted scores rank restaurants by similarity to a user's history but barely predict how many stars they'll give, partly because held-out ratings are concentrated at 4–5 stars. All 'predicted satisfaction' figures should be read in that light.

### τ sensitivity (val, all values reported)

fairness_repeated with each τ tested; differences vs. average with cluster-bootstrap 95% CIs (n = 7,008 overall, 3,504 per cohesion bucket). Worst member and group score use the primary (member-mean) definitions; the "per restaurant" columns are the secondary per-dinner definitions (see "Per-dinner vs. cumulative fairness"). Source: `python -m src.group_eval --tau-sensitivity` (`data/processed/group_eval_val_tau_sensitivity.csv`).

| τ | hit@10 − average | low cohesion: worst member − average | low cohesion: group score − average | low cohesion: per-restaurant worst member − average | low cohesion: per-restaurant fairness gap − average | high cohesion: worst member − average | high cohesion: group score − average | high cohesion: per-restaurant worst member − average | high cohesion: per-restaurant fairness gap − average |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 (default) | −0.0021 [−0.0045, +0.0003] | +0.0074 [+0.0057, +0.0090] | −0.0496 [−0.0515, −0.0478] | −0.1234 [−0.1278, −0.1190] | +0.1386 [+0.1335, +0.1436] | +0.0160 [+0.0153, +0.0167] | −0.0082 [−0.0087, −0.0077] | −0.0162 [−0.0174, −0.0150] | +0.0159 [+0.0146, +0.0172] |
| 0.3 | −0.0020 [−0.0042, +0.0000] | +0.0467 [+0.0453, +0.0482] | −0.0119 [−0.0124, −0.0115] | −0.0155 [−0.0164, −0.0145] | +0.0105 [+0.0093, +0.0117] | +0.0177 [+0.0168, +0.0186] | −0.0033 [−0.0035, −0.0031] | −0.0006 [−0.0010, −0.0002] | −0.0019 [−0.0024, −0.0014] |
| 1.0 | −0.0016 [−0.0029, −0.0003] | +0.0386 [+0.0372, +0.0400] | −0.0047 [−0.0049, −0.0045] | +0.0057 [+0.0051, +0.0064] | −0.0135 [−0.0145, −0.0125] | +0.0120 [+0.0113, +0.0128] | −0.0011 [−0.0012, −0.0010] | +0.0030 [+0.0027, +0.0034] | −0.0050 [−0.0055, −0.0045] |
| 3.0 | −0.0006 [−0.0015, +0.0003] | +0.0235 [+0.0225, +0.0246] | −0.0014 [−0.0014, −0.0013] | +0.0083 [+0.0077, +0.0089] | −0.0131 [−0.0140, −0.0123] | +0.0063 [+0.0059, +0.0068] | −0.0003 [−0.0003, −0.0002] | +0.0025 [+0.0022, +0.0027] | −0.0035 [−0.0038, −0.0031] |

τ=0.1 was fixed before any results. In low-cohesion groups it reduced the member gap about as much as τ=0.3 (and the most of any τ in high-cohesion groups), but mainly by lowering best-served members: its worst-member gain was the smallest of the four values tested. τ=0.3 gave the largest worst-member gain in low-cohesion groups. A possible cause, not verified, is that weights overcorrect as the worst-off member changes between dinners. Because the gap metric rewards leveling down, we treat worst-member predicted satisfaction as the primary fairness metric.

### Per-dinner vs. cumulative fairness

Worst member and fairness gap under both definitions (val, n = 7,008; predicted percentiles). Primary: each member's mean across the top-10 list, then the minimum / max − min over members. Secondary: the minimum / max − min over members for each restaurant, averaged over the list.

| strategy | worst member (primary, member mean) | worst member (secondary, per restaurant) | fairness gap (primary, member mean) | fairness gap (secondary, per restaurant) |
| --- | --- | --- | --- | --- |
| average | 0.860 | 0.834 | 0.096 | 0.140 |
| least_misery | 0.879 | 0.860 | 0.047 | 0.086 |
| fairness | 0.879 | 0.858 | 0.049 | 0.087 |
| fairness_repeated (τ=0.1) | 0.872 | 0.764 | 0.028 | 0.217 |

fairness_repeated's per-restaurant fairness gap by cohesion tercile (2,336 groups each): 0.387 in the bottom (least cohesive) tercile vs. 0.078 in the top tercile; its member-mean gap is 0.039 and 0.017 respectively.

The two definitions measure different things. Per-restaurant metrics ask whether each dinner works for everyone; member-mean metrics ask whether everyone is served equally across a sequence of dinners. fairness_repeated is designed for the second: it spreads its picks across members' tastes, so individual restaurants favor some members over others while the totals even out. It has the smallest cumulative gap and the largest per-dinner gap of the content strategies, and the difference grows as groups diverge. For a one-off dinner, least_misery is the better fit; for a group that eats together repeatedly, spreading picks across members is. τ=0.1 is not just more uneven across dinners; it is worse at each dinner. In low-cohesion groups its per-restaurant worst member is 0.645 [0.638, 0.652] vs. 0.753 [0.749, 0.757] at τ=0.3, and its per-restaurant fairness gap is 0.330 vs. 0.202 (paired difference +0.128 [0.123, 0.133], cluster bootstrap). From τ=0.3 upward, per-dinner metrics are close to or slightly better than average, so the per-dinner cost of sequential fairness comes mostly from setting τ too low.

### Compromise picks in low-cohesion groups

In low-cohesion groups, aggregation surfaces compromise restaurants that bridge members' tastes (e.g. a sample group of ramen, café, Chinese and vegan diners received mostly Mexican/Latin places that each member ranked in their top ~20%). This is aggregation working as intended, not an individual-model artifact.

### Hubness correction

`ContentBasedDebiased` subtracts each restaurant's mean similarity across all users from every score (column-centering the user-by-restaurant score matrix); it has no parameters and was fixed in advance.

Individual-user val (25,439 users; `python -m src.eval`, not logged to `results.md`):

| model | recall@10 | NDCG@10 | coverage | Spearman | k-occurrence skewness | top item (share of users) |
| --- | --- | --- | --- | --- | --- | --- |
| ContentBasedCentered | 0.0126 | 0.0065 | 0.989 | 0.011 | 4.42 | Talk 5.0% |
| ContentBasedDebiased | 0.0121 | 0.0064 | 0.992 | 0.013 | 3.50 | Urban Saloon 3.6% |

Group val (7,008 groups; `python -m src.group_eval --model ContentBasedDebiased`). hit@10, with the paired difference and its cluster-bootstrap 95% CI over targets, and concentration across the 7,008 top-10 lists:

| strategy | hit@10 Centered | hit@10 Debiased | Debiased − Centered [95% CI] | unique restaurants (C → D) | skewness (C → D) | top hub share (C → D) |
| --- | --- | --- | --- | --- | --- | --- |
| average | 0.0191 | 0.0171 | −0.0020 [−0.0042, +0.0001] | 3,208 → 3,639 | 7.18 → 5.12 | Root 12.2% → Root 7.5% |
| least_misery | 0.0160 | 0.0140 | −0.0020 [−0.0039, −0.0001] | 3,413 → 3,728 | 5.47 → 3.93 | Root 8.9% → Root 5.2% |
| fairness | 0.0163 | 0.0153 | −0.0010 [−0.0028, +0.0008] | 3,381 → 3,722 | 5.79 → 4.11 | Root 9.4% → Root 5.5% |
| fairness_repeated (τ=0.1) | 0.0170 | 0.0163 | −0.0007 [−0.0029, +0.0016] | 3,254 → 3,583 | 7.25 → 5.41 | Root 12.6% → Root 8.0% |

The headline results above and the pre-registration use `ContentBasedCentered`. The API (`/group`) serves `ContentBasedDebiased`, because it cuts list concentration by about 30% (k-occurrence skewness −25% to −29% in group lists, −21% in individual lists) at an accuracy cost within or near noise: only least_misery's hit@10 drop has a CI that excludes zero, and only barely. Hubs are reduced, not eliminated: Root, Talk and Jamonera still lead the group lists, and some bars rise in the individual lists. The debiased group run's hub vector is fit on all train users, a single global statistic rather than a per-group pre-window one; the API's is fit on the users it serves (5+ reviews).

### Pre-registered for test

Pre-registered 2026-10-17, before any test-split evaluation: fairness_repeated with τ=0.3 yields a higher worst-member predicted percentile than least_misery in low-cohesion groups, at a smaller group-score cost. τ=0.1 remains the original pre-registered default and will be reported regardless of outcome.

### Validation configurations tried

Every group-evaluation configuration run on val, with how it can be verified. All used the chunked embeddings and rank-percentile normalization; runs 1–6 used `ContentBasedCentered`, run 7 `ContentBasedDebiased`. No group evaluation has been run on test.

1. Groups v1 (`groups.py` as in commit "Add synthetic dinner groups": no prior-history filter at construction, cap 3,000 groups, no per-user cap), `--sample 500`, strategies random / popularity / average / least_misery / fairness. Verified: `data/processed/group_eval_val_sample500.csv` (119 surviving groups).
2. Groups v1, full val, same five strategies. Not verifiable: its output file was overwritten by later runs.
3. Groups v2 (≥ 3 prior likes required at construction, cap 12,000, no per-user cap), full val, same five strategies, group-level (non-cluster) bootstrap. Not verifiable: this `groups.py` version was never committed and its outputs were overwritten.
4. Groups v3 (current: v2 plus at most 10 groups per user), full val, same five strategies, cluster bootstrap. Not verifiable from `data/processed/` (overwritten by run 5); run 5 reproduced its five strategies' top-10 lists exactly when the aggregation interface changed.
5. Groups v3, full val, the five strategies plus fairness_repeated with τ=0.1. Verified: `data/processed/group_eval_val.csv` and `group_eval_val_lists.parquet` (the tables above).
6. Groups v3, τ sensitivity: fairness_repeated with τ ∈ {0.1, 0.3, 1.0, 3.0}. Verified: `data/processed/group_eval_val_tau_sensitivity.csv`. The same run also evaluated τ=1e6 as a sanity check against average (code path in `src/group_eval.py`; not in the CSV).
7. Groups v3, full val, all six strategies with `--model ContentBasedDebiased`. Verified: `data/processed/group_eval_val_content_debiased.csv` and `group_eval_val_content_debiased_lists.parquet`, and the six "debiased model, val" rows in `results/groups.md`.

The `--normalization raw` and `zscore` options exist in `src/group_eval.py`, but no val output for them exists and they have not been run. Individual-user (non-group) val runs are logged separately in `results.md` (random, popularity, content_chunked, content_blob). `ContentBasedCentered` and `ContentBasedDebiased` were also evaluated individually on full val (the "Hubness correction" table) but not logged, since `results.md` is B's; those numbers come from the run output only and are reproducible with `python -m src.eval --model src.recommend:ContentBasedCentered` / `:ContentBasedDebiased`.

## Test results

Group evaluation on the **test** group split, each configuration run exactly once on 2026-10-24 with the invocations in `docs/frozen_configs.md` (`ContentBasedCentered`, rank normalization, k = 10). 2,994 of 3,095 test groups survive (101 dropped: a member had reviewed the target before the window); 998 targets. 95% CIs are cluster bootstraps over targets. Group score, worst member and fairness gap are **predicted** (member rank percentiles). Rows are in `results/groups.md` ("test, frozen"). Random and popularity were not run on test: they are not in the frozen group configurations.

| strategy | hit@10 [95% CI] | NDCG@10 | group score | worst member | fairness gap |
| --- | --- | --- | --- | --- | --- |
| average | 0.0177 [0.0122, 0.0237] | 0.0086 | 0.913 | 0.858 | 0.098 |
| least_misery | 0.0150 [0.0101, 0.0207] | 0.0072 | 0.901 | 0.878 | 0.048 |
| fairness (λ=0.5) | 0.0147 [0.0095, 0.0208] | 0.0067 | 0.903 | 0.877 | 0.049 |
| fairness_repeated (τ=0.1) | 0.0180 [0.0126, 0.0241] | 0.0089 | 0.884 | 0.871 | 0.029 |
| fairness_repeated (τ=0.3) | 0.0167 [0.0109, 0.0229] | 0.0084 | 0.905 | 0.891 | 0.032 |

No hit@10 difference between strategies over all test groups has a CI that excludes zero.

### Pre-registered hypothesis: held

Low-cohesion half (1,497 groups, 726 targets), fairness_repeated (τ=0.3) minus least_misery, paired cluster bootstrap:

- **(a) Higher worst-member predicted percentile: held.** +0.019 [+0.018, +0.019].
- **(b) Smaller group-score cost: held.** Group score +0.005 [+0.005, +0.006] higher than least_misery's. Relative to average, the cost is −0.012 [−0.013, −0.012] for τ=0.3 against −0.018 [−0.018, −0.017] for least_misery, while the worst-member gain is +0.047 [+0.045, +0.050] against +0.029 [+0.027, +0.031].

Both parts are on predicted scores, which are weakly calibrated to star ratings (see Calibration). On hit@10 in the low-cohesion half, τ=0.3 scores 0.0107 (16 hits) against least_misery's 0.0080 (12 hits) and average's 0.0127 (19 hits); none of these differences is distinguishable.

τ=0.1, the original default reported regardless: in the low-cohesion half it gives the smallest worst-member gain over average (+0.009 [+0.007, +0.011]) at the largest group-score cost (−0.050 [−0.052, −0.047]), the same pattern as on val.

### Post-hoc: does strategy matter more for low-cohesion groups?

Not pre-registered. Differences vs. average by cohesion half:

| cohesion | strategy | worst member Δ | group score Δ | hit@10 (hits) | hit@10 Δ |
| --- | --- | --- | --- | --- | --- |
| high (n=1,497) | average | — (0.916) | — (0.950) | 0.0227 (34) | — |
| | least_misery | +0.011 [+0.010, +0.012] | −0.007 [−0.008, −0.007] | 0.0220 (33) | −0.001 [−0.007, +0.005] |
| | fairness (λ=0.5) | +0.011 [+0.010, +0.012] | −0.005 [−0.005, −0.005] | 0.0227 (34) | 0.000 [−0.005, +0.005] |
| | fairness_repeated (τ=0.1) | +0.017 [+0.016, +0.018] | −0.008 [−0.009, −0.008] | 0.0234 (35) | +0.001 [−0.004, +0.005] |
| | fairness_repeated (τ=0.3) | +0.019 [+0.017, +0.020] | −0.003 [−0.004, −0.003] | 0.0227 (34) | 0.000 [−0.004, +0.004] |
| low (n=1,497) | average | — (0.799) | — (0.876) | 0.0127 (19) | — |
| | least_misery | +0.029 [+0.027, +0.031] | −0.018 [−0.018, −0.017] | 0.0080 (12) | −0.005 [−0.010, 0.000] |
| | fairness (λ=0.5) | +0.028 [+0.027, +0.030] | −0.015 [−0.016, −0.015] | 0.0067 (10) | −0.006 [−0.012, −0.001] |
| | fairness_repeated (τ=0.1) | +0.009 [+0.007, +0.011] | −0.050 [−0.052, −0.047] | 0.0127 (19) | 0.000 [−0.006, +0.006] |
| | fairness_repeated (τ=0.3) | +0.047 [+0.045, +0.050] | −0.012 [−0.013, −0.012] | 0.0107 (16) | −0.002 [−0.007, +0.003] |

On predicted scores, yes: in high-cohesion groups every strategy is within about 0.02 of average, and in low-cohesion groups the worst-member gains and group-score costs are 2–3 times larger. The absolute effects are still small (at most 0.05 of a percentile point). On hit@10, strategy makes no difference in high-cohesion groups; in low-cohesion groups least_misery and fairness lose hits relative to average (fairness's CI excludes zero; least_misery's ends at zero), while both fairness_repeated settings are indistinguishable from average.

### Caveats

- Cohesion halves are a median split within test (median cohesion 0.119, using pre-window profiles), so "high" and "low" are relative, not fixed thresholds.
- 730 of the 2,994 test groups are flagged for a possible target-embedding leak (a member's own review of the target is among those encoded into its embedding), as on val.
- Low-cohesion hit@10 cells rest on 10–19 hits each, so those comparisons are fragile.
- **Amendment:** the τ=0.3 configuration was missing from the original frozen list in error and was added to `docs/frozen_configs.md` after the four other group test runs, then run once. This doesn't involve choosing a configuration from test results: τ=0.3 and the hypothesis it tests were fixed in the 2026-10-17 pre-registration, before any test-split evaluation, and the τ=0.1 results are reported alongside regardless.

## Group API and page

The `/group` API (`src/api/routes_group.py`) serves `ContentBasedDebiased` with each user's whole train history (users with 5+ reviews), scored in numpy through `src/aggregate.py`.

| endpoint | what it does |
| --- | --- |
| `GET /group/users/sample?n=4&cohesion=any\|high\|low&seed=42` | n val users with their top-rated places; high/low draws them from a real val group in that half of the cohesion distribution |
| `GET /group/users/profiles?user_ids=…` | top-rated places for given user IDs |
| `POST /group/recommend` | one strategy's list: members (2–6), strategy, norm (rank/zscore/raw), lam, tau, k, exclude_visited, include_closed |
| `POST /group/compare` | the same for two strategies, plus their overlap |
| `GET /health` | liveness |

To run it (repo root, `.venv` active):

```
python -m src.recommend build-cache --model ContentBasedDebiased   # optional: ~90 s once; otherwise the first request fits and caches (~110 s)
uvicorn src.api.main:app                                           # API on :8000, docs at /docs
streamlit run src/app.py                                           # UI on :8501, "Group Dinner" page
```

- **τ default:** the API's `tau` defaults to 0.1, the pre-registered evaluation setting. The Group Dinner page defaults to 0.3, which performed better per dinner on validation (see "τ sensitivity").
- **Hosted mode:** with `DATA_SOURCE=db` the API reads everything from `DATABASE_URL` instead of `data/processed/` (only the demo users loaded by `python -m src.deploy_load` are served). See `deploy/README.md`.
- **Closed restaurants** are excluded by default (`include_closed=false`), and so is anything a member already reviewed (`exclude_visited=true`).
- **Chains:** the page asks for k + 5 results and collapses repeat locations of a chain into one row, "(N locations)". fairness_repeated is a dinner sequence, so its list is shown uncollapsed, with repeat locations labelled "(another location)".

## Known gaps

- Hours come from the dataset snapshot, so "open now" means open per the hours listed when the data was captured.
- Closed restaurants: 1,449 of the 3,902 restaurants are marked closed (`is_open`) in the dataset snapshot. They stay in all offline evaluation on purpose, because the held-out reviews are historical, so they appear in every evaluation candidate set. User-facing paths exclude them by default: the `/group` API does (`include_closed=false`); the Postgres loader doesn't load `is_open` yet.
- Groups are co-rating groups: strangers who liked the same place within 30 days, not real dining parties. Targets skew toward popular restaurants, and the same targets appear in both val and test groups.
- Possible small target-embedding leak: in 1,672 of the 7,008 val groups, a member's own review of the target is among the reviews encoded into the target's embedding. Within the low and mid target-popularity terciles, flagged and unflagged groups showed no significant hit@10 difference for any content strategy, but an effect of about 0.005 is not ruled out.
- Groups share targets (handled by the cluster bootstrap) and members (up to 10 groups per user, not clustered), so the CIs are slightly optimistic.
- Morea, the only restaurant tagged "Adult" (its categories mix Nightlife, Adult, Bars, Seafood, Sandwiches, Shopping, Breakfast & Brunch and more), was the top hub in the API's lists under `ContentBasedCentered`: it appeared in 21% of 300 sampled API group lists and 15% of individual top-10s (closed restaurants excluded). In the group_eval lists, which include closed restaurants, it ranked third (8.9% of average lists) behind Root and Talk. Under `ContentBasedDebiased` it drops out of the top five hubs in the group lists.
