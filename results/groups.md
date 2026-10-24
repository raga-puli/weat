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
| 2026-10-18 | val | random | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0020 [0.0010, 0.0032] | 0.0008 | 0.4988 | 0.4253 | 0.1467 | debiased model, val |
| 2026-10-18 | val | popularity | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0271 [0.0159, 0.0400] | 0.0120 | 0.5290 | 0.4296 | 0.1971 | debiased model, val |
| 2026-10-18 | val | average | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0171 [0.0135, 0.0211] | 0.0089 | 0.9027 | 0.8426 | 0.1079 | debiased model, val |
| 2026-10-18 | val | least_misery | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user | 7008 | 0.0140 [0.0108, 0.0173] | 0.0071 | 0.8888 | 0.8645 | 0.0513 | debiased model, val |
| 2026-10-18 | val | fairness | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user; lam=0.5 | 7008 | 0.0153 [0.0117, 0.0191] | 0.0076 | 0.8911 | 0.8645 | 0.0524 | debiased model, val |
| 2026-10-18 | val | fairness_repeated | model=ContentBasedDebiased; norm=rank; k=10; groups: >=3 prior likes, <=10/user; tau=0.1 | 7008 | 0.0163 [0.0128, 0.0200] | 0.0082 | 0.8666 | 0.8520 | 0.0310 | debiased model, val |
| 2026-10-24 | test | average | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 2994 | 0.0177 [0.0122, 0.0237] | 0.0086 | 0.9129 | 0.8578 | 0.0982 | test, frozen |
| 2026-10-24 | test | least_misery | norm=rank; k=10; groups: >=3 prior likes, <=10/user | 2994 | 0.0150 [0.0101, 0.0207] | 0.0072 | 0.9005 | 0.8778 | 0.0476 | test, frozen |
| 2026-10-24 | test | fairness | norm=rank; k=10; groups: >=3 prior likes, <=10/user; lam=0.5 | 2994 | 0.0147 [0.0095, 0.0208] | 0.0067 | 0.9028 | 0.8774 | 0.0494 | test, frozen |
| 2026-10-24 | test | fairness_repeated | norm=rank; k=10; groups: >=3 prior likes, <=10/user; tau=0.1 | 2994 | 0.0180 [0.0126, 0.0241] | 0.0089 | 0.8840 | 0.8706 | 0.0287 | test, frozen |
| 2026-10-24 | test | fairness_repeated | norm=rank; k=10; groups: >=3 prior likes, <=10/user; tau=0.3 | 2994 | 0.0167 [0.0109, 0.0229] | 0.0084 | 0.9051 | 0.8908 | 0.0317 | test, frozen |
