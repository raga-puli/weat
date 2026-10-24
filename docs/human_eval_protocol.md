# Human evaluation protocol (blind pairwise preference)

<!-- TEMPLATE. Every "_____" and every "TO WRITE" is for Person B to fill in, in their own words.
     Fill this in and commit it BEFORE the first participant session. -->

Written by: _____
Committed (before any session) on: _____ (commit: _____)

## 1. Hypothesis

TO WRITE.

- Direction (one-sided or two-sided): _____
- Stated before any data were collected: yes / no

## 2. Strategies compared

- least_misery vs. average, both with rank normalization (`norm=rank`), k=10.
- Lists come from the API's `POST /group/compare` (served model: `ContentBasedDebiased`), with the API defaults: exclude restaurants members already reviewed, exclude closed restaurants.
- Participants see each pair as "List 1" / "List 2", restaurant names, categories and price only. The left/right assignment is randomized per trial and is never shown. Scores are not shown, since least_misery and average scores would give the strategy away.

Other notes: _____

## 3. Groups

- 12 fixed groups from the **test** split of `groups.parquet`: 6 high-cohesion and 6 low-cohesion.
- Cohesion is the mean pairwise cosine of the members' taste vectors (the API's whole-history profiles), split at the median over eligible test groups. Eligible: every member has a taste profile in the served model.
- Selected once, seeded: `python -m src.human_eval --select-groups` (seed = `SEED` from `src/interfaces.py`). This writes `data/human_eval/groups.csv`. The list below is copied from that file and not changed afterwards.
- Each participant sees all 12 groups, in their own random order (seeded by the participant code).

| # | group_id | cohesion_bucket | cohesion | n_members |
| --- | --- | --- | --- | --- |
| 1 | _____ | high | _____ | _____ |
| 2 | _____ | high | _____ | _____ |
| 3 | _____ | high | _____ | _____ |
| 4 | _____ | high | _____ | _____ |
| 5 | _____ | high | _____ | _____ |
| 6 | _____ | high | _____ | _____ |
| 7 | _____ | low | _____ | _____ |
| 8 | _____ | low | _____ | _____ |
| 9 | _____ | low | _____ | _____ |
| 10 | _____ | low | _____ | _____ |
| 11 | _____ | low | _____ | _____ |
| 12 | _____ | low | _____ | _____ |

## 4. What participants see and the exact question

What each trial shows: the members' profile cards (each member's top-rated restaurants), then List 1 and List 2.

Exact question wording (must match `QUESTION` in `src/pages/3_Blind_Eval.py`):

> _____

Answer options: List 1 / List 2 / No preference, plus an optional free-text reason.

Instructions read or shown to participants before starting:

> _____

## 5. Participants

- Number of participants: _____
- How recruited: _____
- Inclusion / exclusion: _____
- Anonymity: participants enter an anonymous code only; no names are recorded. Code format: _____
- Consent / what participants are told: _____

## 6. Analysis plan

TO WRITE. Decide before data collection:

- Primary outcome: _____
- Unit of analysis (judgments vs. participants) and how repeated judgments per participant are handled: _____
- How "No preference" is handled: _____
- Test and significance level: _____
- Cohesion-bucket comparison: _____
- Exclusions (incomplete sessions, duplicate codes, ...): _____

What `python -m src.human_eval --log` reports (fixed by the code): least_misery preference rate excluding no-preference, with a Wilson 95% CI and an exact two-sided binomial test against 0.5; the same per cohesion bucket; the no-preference rate; n participants and n judgments.

## 7. If the result is null

TO WRITE: what we will report, and how we will word it, if the preference rate's CI includes 0.5.

## 8. Limitations

TO WRITE. (For example: small convenience sample, synthetic co-rating groups, participants judging on behalf of strangers.)

## Deviations from this protocol

Record any change made after the first session here, with the date and reason.

| date | change | reason |
| --- | --- | --- |
| | | |
