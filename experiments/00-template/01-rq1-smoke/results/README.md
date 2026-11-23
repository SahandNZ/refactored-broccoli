# Results: 01-rq1-smoke

Plan: [../README.md](../README.md). Runs: 2026-11-16T1400.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| 2026-11-16T1400 | 2026-11-16 13:57 / 14:57 | 24.1 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 16.95 / 1.46 GB | - | 0ab19a1c2124f342092a411d81882db67d82fec3 |

## Q1 (RQ-smoke)

Does every zero-shot model load and give valid forecasts for real hourly contexts?

Primary metric: share of models that pass all four checks.

**Q1**: yes. 10 of 10 models pass every check (share 1.00).

Limitations: 24 recent contexts of 3 assets only; a smoke check of the wrappers, not of accuracy

Files: [2026-11-16T1400/tables/checks.csv](2026-11-16T1400/tables/checks.csv), [2026-11-16T1400/tables/timing.csv](2026-11-16T1400/tables/timing.csv), [2026-11-16T1400/figures/timing.png](2026-11-16T1400/figures/timing.png)
