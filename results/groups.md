# Group evaluation results

Written only by `python -m src.group_eval --log`. group score, worst member and fairness gap
are PREDICTED (member rank percentiles from the model), not observed satisfaction;
hit@10 and NDCG@10 are against the held-out target. CIs are cluster bootstraps over targets.

| date | split | strategy | config | n_groups | hit@10 [CI] | NDCG@10 | group score (PREDICTED) | worst member (PREDICTED) | fairness gap (PREDICTED) | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-17 | val | random | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0020 [0.0010, 0.0032] | 0.0008 | 0.4987 | 0.4279 | 0.1415 | baseline group run, val |
| 2026-10-17 | val | popularity | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0271 [0.0159, 0.0400] | 0.0120 | 0.5795 | 0.4842 | 0.1862 | baseline group run, val |
| 2026-10-17 | val | average | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0191 [0.0154, 0.0235] | 0.0094 | 0.9139 | 0.8600 | 0.0964 | baseline group run, val |
| 2026-10-17 | val | least_misery | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0160 [0.0122, 0.0198] | 0.0079 | 0.9014 | 0.8790 | 0.0470 | baseline group run, val |
| 2026-10-17 | val | fairness | norm=rank; k=10; groups: >=3 prior likes, <=10/user; lam=0.5 | 7008 | 0.0163 [0.0125, 0.0200] | 0.0081 | 0.9038 | 0.8789 | 0.0487 | baseline group run, val |
| 2026-10-17 | val | fairness_repeated | norm=rank; k=10; groups: >=3 prior likes, <=10/user; tau=0.1 | 7008 | 0.0170 [0.0130, 0.0213] | 0.0086 | 0.8850 | 0.8717 | 0.0283 | baseline group run, val |
