# Fear-Free Night Navigator - Evaluation Report

## Model Comparison

| Model | CV AUC | Train AUC | Precision | Recall | F1 | Brier |
|-------|--------|-----------|-----------|--------|-----|-------|
| GBM (our model) | 0.8705 | 0.8713 | 0.7832 | 0.7299 | 0.7556 | 0.1440 |
| LogisticRegression | 0.8599 | 0.8600 | 0.8199 | 0.6413 | 0.7197 | 0.1533 |
| RandomForest | 0.8663 | 0.8667 | 0.7948 | 0.7016 | 0.7453 | 0.1475 |

## Ablation Study

| Removed Group | AUC Without | AUC Delta | Features |
|---------------|-------------|-----------|---------|
| safe_POI | 0.8304 | -0.0401 | safe_poi_count_100m, safe_poi_count_300m, night_x_safe_poi |
| dead_end_flag | 0.8451 | -0.0255 | dead_end_flag |
| neutral_POI | 0.8560 | -0.0146 | neutral_poi_count_100m, neutral_poi_count_300m |
| road_struct | 0.8665 | -0.0040 | road_type_encoded, length_m |
| risky_POI | 0.8681 | -0.0025 | risky_poi_count_100m, risky_poi_count_300m, night_x_risky_poi |
| temporal | 0.8682 | -0.0023 | is_night, time_band, is_weekend, night_x_road, night_x_safe_poi, night_x_risky_poi |

## Target Checks

| Metric | Target | Achieved | Pass? |
|--------|--------|----------|-------|
| CV AUC    | >0.80 | 0.8705    | Yes |
| Train AUC | >0.82 | 0.8713       | Yes |
| Precision | >0.75 | 0.7832  | Yes |
| Recall    | >0.70 | 0.7299     | Yes |
| F1        | >0.74 | 0.7556         | Yes |
| Brier     | <0.20 | 0.1440      | Yes |