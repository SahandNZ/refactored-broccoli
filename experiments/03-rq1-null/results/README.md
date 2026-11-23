# Results: 03-rq1-null

Plan: [../README.md](../README.md). Runs: test-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-17 | 2026-11-17 18:46 / 19:46 | 171.0 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 3.18 / 0.0 GB | - | 0538aba606e493317430ea7f260f96dd09cb2e40 |

## Q1 (RQ-null)

Does any zero-shot foundation model beat the best simple baseline on any market at 1, 4 or 24 bars, on data after the model's scoring start?

Primary metric: CRPS of the vol-scaled return.

**Q1**: no. Main verdict, main family (chronos-2, chronos-bolt-base, chronos-bolt-small, kronos-base, kronos-small, moirai-2-small, timesfm-2.5, tirex-1.1): 0 of 119 scored cells beat the best baseline: none. Separate verdict, late family (t0-gamma, timesfm-3, tirex-2, toto-2-22m, toto-2-313m): 0 of 68 scored cells beat the best baseline: none; the late models' MDE at 1 bar(s): t0-gamma: 1 quarter(s), the rule cannot detect it; timesfm-3: 1 quarter(s), the rule cannot detect it; tirex-2: 2 quarter(s), the rule cannot detect it; toto-2-22m 0.47 % (5 of 5 markets reach 80 % power); toto-2-313m: 2 quarter(s), the rule cannot detect it. Cells not scored (8, in the table with their status): 1 untested: 169 decision times, fewer than 5 blocks of 236; 1 untested: 228 decision times, fewer than 5 blocks of 60; 1 untested: 250 decision times, fewer than 5 blocks of 112; 1 untested: 250 decision times, fewer than 5 blocks of 164; 1 untested: 261 decision times, fewer than 5 blocks of 92; 1 untested: 384 decision times, fewer than 5 blocks of 92; 1 untested: 55 decision times, fewer than 5 blocks of 19; 1 untested: 916 decision times, fewer than 5 blocks of 292.

Limitations: test folds; only decisions after each model's scoring start, so a model released inside the window has fewer quarters (late family); a 'no' rules out only edges the rule can detect: at 1 bar(s) its MDE (80 % power) is 0.90 % of the baseline's CRPS in the median cell that reaches it, and 29 cells never reach it (Q2); a late model with fewer than 3 quarters can never pass the quarter condition; the cells at 24 bars, whose labels overlap most, may reject more often than the nominal level (the size check of the shared bootstrap tests with overlapping 24-bar labels at the adopted block floor of 4 x the overlap, 09-mde-threshold dev-2026-11-17 tables/bootstrap_size.csv: the mean test 5.0%-7.3%, the difference test 6.0%-6.7% at a nominal 5%, series of 600, 2400 decision times, 300 draws each (standard error about 1.3%))

Files: [test-2026-11-17/tables/verdict.csv](test-2026-11-17/tables/verdict.csv), [test-2026-11-17/tables/skill.csv](test-2026-11-17/tables/skill.csv), [test-2026-11-17/tables/quarters.csv](test-2026-11-17/tables/quarters.csv)

## Q2 (RQ-null)

Could the Q1 rule detect a small but real edge, i.e. is a "no" in Q1 adequately powered?

Primary metric: MDE per cell: the smallest planted CRPS gain over the best baseline that the
Q1 rule detects in at least 80 % of block-bootstrap resamples, in % of the baseline's CRPS.

**Q2**: no. 35 of the 45 scored cells at 1 bar(s) with at least 3 quarters (78%) have an MDE at or below X = 2 % of the best baseline's CRPS, against K = 80% (sensitivity, not deciding: K = 50%: yes; K = 90%: no); 20 cells with fewer quarters, which the rule cannot detect, are left out. Q2 counts the cells the Q1 rule can pass (at least 3 quarters with data), so toto-2-22m (late family: fewer than 3 whole quarters after the scoring start) counts for Q2; counted by whole quarters the share would be 75% (30 of 40), with the same verdict. Under no edge the rule fired in at most 1.5% of the resamples of any cell (median 0; one-sided level alpha / m at most 0.00074); the largest is kronos-base stock 24 bar(s), 0.015 with 52 decision points. 93 of 187 cells fire at least once, which at the resolution of 0.001 is above their level; 5 exceed ten times it. The real tests: the smallest one-sided p of a scored cell is 0.024, 1 after Holm.

Limitations: the planted edge is a constant CRPS gain added to the real model's centred loss difference, so it has the real noise and autocorrelation but not the shape of a real edge; Holm is replaced by its threshold for the most significant cell of each family (conservative); X comes from 09-mde-threshold

Files: [test-2026-11-17/tables/mde_threshold.csv](test-2026-11-17/tables/mde_threshold.csv), [test-2026-11-17/figures/mde_threshold.pdf](test-2026-11-17/figures/mde_threshold.pdf), [test-2026-11-17/tables/power_rule.csv](test-2026-11-17/tables/power_rule.csv), [test-2026-11-17/tables/mde.csv](test-2026-11-17/tables/mde.csv), [test-2026-11-17/tables/power.csv](test-2026-11-17/tables/power.csv)
