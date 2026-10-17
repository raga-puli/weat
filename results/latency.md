| date | engine | index | params | filtered | p50_ms | p95_ms | qps | recall@10_vs_exact | build_s | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-17 | parse | - | qwen2.5:7b-instruct v1, cache hit | - | 0.07 | 0.17 |  |  |  | cached responses only; live LLM latency not measured; baseline, 3902 restaurants, no hours |
| 2026-10-17 | embed | - | all-MiniLM-L6-v2 | - | 18.17 | 25.17 |  |  |  | query text -> vector; baseline, 3902 restaurants, no hours |
| 2026-10-17 | numpy | none | exact | no | 5.76 | 9.49 | 163 | 1.000 |  | 40 queries; baseline, 3902 restaurants, no hours |
| 2026-10-17 | numpy | none | exact | yes | 10.86 | 22.26 | 89 |  |  | 30 queries; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | none | exact | no | 18.61 | 48.35 | 53 | 1.000 |  | 40 queries; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | none | exact | yes | 10.17 | 24.24 | 89 |  |  | 30 queries; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=10 probes=1 (default) | no | 1.09 | 1.64 |  | 0.657 | 0.211 | index used; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=10 probes=3 (sqrt(lists)) | no | 16.84 | 49.74 |  | 1.000 | 0.211 | planner did NOT use the index; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=50 probes=1 (default) | no | 1.10 | 2.54 |  | 0.547 | 0.273 | index used; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=50 probes=7 (sqrt(lists)) | no | 1.62 | 3.06 |  | 0.860 | 0.273 | index used; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=100 probes=1 (default) | no | 0.90 | 1.48 |  | 0.485 | 0.441 | index used; baseline, 3902 restaurants, no hours |
| 2026-10-17 | postgres | ivfflat | lists=100 probes=10 (sqrt(lists)) | no | 1.43 | 2.73 |  | 0.878 | 0.441 | index used; baseline, 3902 restaurants, no hours |
