# Metrics Report

## Latest metrics

| timestamp | label | config | N | recall | mrr | faithfulness | citation_rate | instrument_hit_rate | llm_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-04 17:44:16 | llm-enabled-eval | expanded_k4 | 40 | 0.7 | 0.7 | 0.6 | 1.0 | 1.0 | 0 |
| 2026-03-04 17:44:19 | llm-enabled-eval | expanded_k8 | 40 | 0.7 | 0.7 | 0.6 | 1.0 | 1.0 | 0 |
| 2026-03-04 17:11:46 | after-structured-lookup-fix | small_k4 | 3 | 1.0 | 1.0 | 0.3333333333333333 | 1.0 | 1.0 | 0 |

## Before vs After

| config | N | recall_before | recall_after | mrr_before | mrr_after | faithfulness_before | faithfulness_after | instrument_before | instrument_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded_k4 | 40 | 0.4 | 0.7 | 0.4 | 0.7 | 0.525 | 0.6 | 0.8888888889 | 1.0 |
| expanded_k8 | 40 | 0.4 | 0.7 | 0.4 | 0.7 | 0.525 | 0.6 | 0.8888888889 | 1.0 |
| small_k4 | 3 | 0.6666666667 | 1.0 | 0.6666666667 | 1.0 | 0.3333333333 | 0.3333333333333333 | 1.0 | 1.0 |

## Plot

![Metrics history](metrics_history.png)