# Results: 09-mde-threshold

Plan: [../README.md](../README.md). Runs: dev-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| dev-2026-11-17 | 2026-11-17 16:12 / 17:12 | 68.3 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 1.99 / 0.0 GB | - | 5ef31cb2ba41fd56e1e0ae47f0f15f61273e1980 |

## Q1 (RQ-power)

Which minimum detectable effect (MDE) threshold should the power question of 03-rq1-null (its Q2) use?

Primary metric: share of the scored rq1 cells at 1 bar with MDE at or below X.

**Q1**: yes. threshold 2 % of the best baseline's CRPS, the smallest of the grid at which at least 80% of the 10 cells at 1 bar(s) have an MDE at or below it (90%); set it as mde_threshold_pct of 03-rq1-null. 28 cells at 1 bar(s) with fewer than 3 quarters, which the rule cannot detect, are left out.

Limitations: development folds d1-d4 only, which overlap some releases (each model is scored after its release, so late models have fewer decisions, and a model released after d4 has no cell); the planted edge is a constant CRPS gain on the real model's centred loss difference; MDEs between grid gains are interpolated linearly; the shared bootstrap tests reject somewhat more often than their nominal level with overlapping labels and few blocks (size check: tables/bootstrap_size.csv)

Files: [dev-2026-11-17/tables/mde_threshold.csv](dev-2026-11-17/tables/mde_threshold.csv), [dev-2026-11-17/figures/mde_threshold.pdf](dev-2026-11-17/figures/mde_threshold.pdf), [dev-2026-11-17/tables/mde.csv](dev-2026-11-17/tables/mde.csv), [dev-2026-11-17/figures/mde.pdf](dev-2026-11-17/figures/mde.pdf), [dev-2026-11-17/tables/power.csv](dev-2026-11-17/tables/power.csv), [dev-2026-11-17/tables/bootstrap_size.csv](dev-2026-11-17/tables/bootstrap_size.csv)
