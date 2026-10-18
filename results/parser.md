| date | model | prompt_version | exact_match | per-field accuracies | n | note |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-16 | qwen2.5:7b-instruct | v1 | 0.350 | max_price_level 0.825, near 0.950, max_distance_km 1.000, dietary 0.975, open_at 0.975, categories 0.825, soft 0.475 | 40 | v1 baseline |
| 2026-10-17 | qwen2.5:7b-instruct | v2 | 0.300 | max_price_level 0.875, near 0.975, max_distance_km 0.975, dietary 0.975, open_at 0.975, categories 0.850, soft 0.350 | 40 | v2: soft must not repeat captured fields; worked example; somewhere is filler |
| 2026-10-17 | qwen2.5:7b-instruct | v1 | 0.550 | max_price_level 0.825, near 0.950, max_distance_km 1.000, dietary 0.975, open_at 0.975, categories 0.825, soft 0.800 | 40 | v1 prompt + rules-based soft cleanup (clean_soft); v2 reverted |
