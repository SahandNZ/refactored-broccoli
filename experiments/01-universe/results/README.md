# Results: 01-universe

Plan: [../README.md](../README.md). Runs: 2026-11-17T0005-random-1000.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| 2026-11-17T0005-random-1000 | 2026-11-16 22:59 / 23:59 | 142.0 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 1.14 / 0.01 GB | - | e5ade0d0fc6e5de5030c7be3aaeccc11ce5e4a52 |

## Q1 (RQ-universe)

Does the rule-based universe cover more independent market behaviour than simple alternatives of the same size?

Primary metric: effective rank of the weekly-return correlation matrix on the held-out window.

**Q1**: yes. Held-out window 2024-01-01 to 2024-11-01 (the decision): effective rank of the universe 20.70 (noise level 31.43: independent series with its missing data), Liquid 17.29, random 90th percentile 20.35 (median 19.47, 1000 draws), 72 assets each, the same number in every pool. In-sample, on the selection window 2020-01-01 to 2024-01-01 where the universe was picked (not the decision): effective rank of the universe 28.76 (noise level 60.33: independent series with its missing data), Liquid 18.71, random 90th percentile 22.69 (median 21.24, 1000 draws); gap to the random median 7.52 (paired block-bootstrap normal interval 6.36 to 8.68, in-sample), 72 assets each, the same number in every pool. The rule would answer yes there.

Limitations: the candidate lists were drawn up in 2026 (crypto perpetuals still trading then, today's large-cap stocks), a survivorship bias of the candidate set that mostly affects return-based results; coverage to the end of the selection window drops assets that stopped trading inside it (it uses only data available at selection time); the held-out window is short (10 months, about 43 weekly returns, so its matrix has rank at most 43 and effective-rank levels are compared only within a window) and the bootstrap interval of the gap is that of the selection window, in-sample, and the normal interval (gap +- z times the standard deviation of the paired bootstrap gaps; its Random median uses the first 20 draws), because the percentile interval is shifted below the point gap by the in-sample optimism and the repeated weeks (resampling lowers the tuned universe more); trading hours differ by market; 117 stock candidates pass the filters, 69 of them non-US

Files: [2026-11-17T0005-random-1000/tables/universe.csv](2026-11-17T0005-random-1000/tables/universe.csv), [2026-11-17T0005-random-1000/tables/diversity.csv](2026-11-17T0005-random-1000/tables/diversity.csv), [2026-11-17T0005-random-1000/tables/universe_correlation.csv](2026-11-17T0005-random-1000/tables/universe_correlation.csv), [2026-11-17T0005-random-1000/figures/universe_correlation.pdf](2026-11-17T0005-random-1000/figures/universe_correlation.pdf), [2026-11-17T0005-random-1000/tables/sector_mix.csv](2026-11-17T0005-random-1000/tables/sector_mix.csv), [2026-11-17T0005-random-1000/tables/comparison_universes.csv](2026-11-17T0005-random-1000/tables/comparison_universes.csv)

## Q3 (RQ-universe)

Does the ranking of the models by mean CRPS stay the same across parts of the universe?

Primary metric: median Kendall tau of the split halves' rankings by mean CRPS.

**Q3**: yes. Computed by analyze.py from the test forecast cache test-2026-11-17 of 02-forecasts (same data version as this run), not by the selection run whose commit the table above shows: test quarters t1, t2, t3, t4, horizon 1, mean CRPS per asset averaged over the assets, on the decisions every compared model scored; decided by the hourly group's median split tau against 0.7. Hourly models (11 models, 296,830 decisions: Kendall tau of the two halves of 1000 random splits median 1.00 (central 95 % of splits 0.96 to 1.00); the same first model in both halves in 100.0 % of splits and the same first three in order in 100.0 % (first on the whole universe: climatology); each market against the other four 0.85 to 0.96; each test quarter against the other three 0.89 to 1.00.) All models on Kronos's decision points (13 models, 3,944 decisions: Kendall tau of the two halves of 1000 random splits median 0.90 (central 95 % of splits 0.85 to 0.95); the same first model in both halves in 75.3 % of splits and the same first three in order in 74.5 % (first on the whole universe: linear); each market against the other four 0.67 to 0.85; each test quarter against the other three 0.82 to 0.87.)

Limitations: exploratory, amended after the plan lock and after RQ1's test results were seen (Plan changes, 2026-11-18); parts of the one universe, not other universes (the Liquid and Random comparison needs new forecasts and is left to later work); with 11 models a tau of 0.7 allows 8 of the 55 pairs to swap (the plan's four models needed the same order), and most pairs are far apart, so the first-model and first-three shares are the sharper check; the halves share the test hours and the market-wide moves, so their rankings are not independent; the range is the spread over random splits, not a confidence interval

Files: [2026-11-17T0005-random-1000/tables/ranking_scores.csv](2026-11-17T0005-random-1000/tables/ranking_scores.csv), [2026-11-17T0005-random-1000/tables/ranking_parts.csv](2026-11-17T0005-random-1000/tables/ranking_parts.csv), [2026-11-17T0005-random-1000/tables/ranking_splits.csv](2026-11-17T0005-random-1000/tables/ranking_splits.csv)
