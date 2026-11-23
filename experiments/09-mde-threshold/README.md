# 09 MDE threshold: which edge can the rq1 rule resolve, and which threshold should rq1 use?

## Questions

**Q1** (RQ-power). Which minimum detectable effect (MDE) threshold should the power question of
03-rq1-null (its Q2) use?
Decision rule: on the development folds d1-d4 only, over all scored rq1 cells at 1 bar
(`mde_horizon`, the horizon 03-rq1-null's Q2 judges; every model, both Holm families), the
share of cells whose MDE is at or below X, for X = 0, 0.5, ..., 10 % of the best baseline's
CRPS (`mde_threshold_grid_pct`); the threshold is the smallest X whose share reaches K = 80 %
(`power_share`). Yes (that X is the threshold, copied into `mde_threshold_pct` of
03-rq1-null) if some X of the grid reaches K; no (there is no usable threshold) otherwise.
Primary metric: share of the scored rq1 cells at 1 bar with MDE at or below X.

## Hypothesis

- H1: at 1 bar at least 80 % of the cells reach an MDE at or below the owner's default of 2 %;
  the Kronos cells (one decision point every 167 hours) form most of the cells that need a
  larger X.

## Design

The MDE is the one of 03-rq1-null's Q2 (see [its Design](../03-rq1-null/README.md#design)),
computed here on the development folds so that rq1's threshold is fixed before any test data
is scored. The code is `src.benchmark` (`_power.py`) and `src.figures` (`_mde.py`), shared with rq1:
`run.py` calls the same function as 03-rq1-null's `run.py`, `power_tables`.

- **One place for the rule.** The rule whose power is measured and every setting of it
  (`alpha`, `min_winning_quarters`, `power_improvements_pct`, `power_target`, the bootstrap
  options, `mde_horizon`, `mde_threshold_grid_pct`, `power_share`) and the models are read
  from 03-rq1-null's `study.toml`; this study's `study.toml` holds only `forecast_cache` (its
  `[models]` is empty). `run.py`
  copies the file it read into the run as `rq1-config.toml`, and `analyze.py` reads that copy.
- **Cells** (`null_tables`): the rq1 cells on d1-d4, whatever 03-rq1-null's `[folds] use`;
  `run.py` refuses any `[folds] use` of this study other than `development`. Every zero-shot
  model of `[models]` is scored on its valid decisions in d1-d4 after its release, and each
  fold's best baseline is chosen on the other three development folds (leave one fold out),
  as rq1 does while its plan is open. A model released after d4 has no scored cell (reported
  with its `status` in `cells.csv`).
- **Power** (`power_table`): per scored cell, a planted model with the real model's centred
  loss difference to the best baseline and a gain of g % of the baseline's CRPS, for g in
  `power_improvements_pct` (0 to 10 in steps of 0.5); the detection rate of the rq1 rule (the
  one-sided Diebold-Mariano test with Newey-West lags max(overlap, Andrews) and t(n - 1), the
  cell's Politis-White block, as in rq1; p x m against `alpha` with m the number of scored
  cells of the cell's Holm family) over `bootstrap_resamples` circular-block bootstrap
  resamples.
- **MDE** (`minimum_detectable_effects`): the smallest g with detection rate at least
  `power_target` (0.8), linearly interpolated; never reached above 10 %.
- **Threshold sweep** (`threshold_shares`): for each X of the grid, the share of cells whose
  MDE is reached and at most X, over all cells and per horizon; the share never falls as X
  grows.
- **Threshold** (`powered_threshold`): the smallest X whose share of the 1-bar cells is at
  least `power_share` (0.8); none when no X of the grid reaches it.

`run.py` writes `tables/cells.csv`, `tables/power.csv` and `tables/mde.csv`; `analyze.py`
writes `tables/mde_threshold.csv` (one row per X: `share` and `share_1h`, `share_4h`,
`share_24h`), `figures/mde_threshold.pdf` (the share against X, one line per horizon and one
over all, the chosen X marked; drawn from `tables/mde_threshold.csv`), `figures/mde.pdf` (the
MDE of every cell per market, one panel per horizon, a line at the chosen X; drawn from
`tables/mde.csv`) and the results README.

### Size check

The shared bootstrap tests (`src.stats.bootstrap_mean_test` and `bootstrap_difference`, a
circular-block bootstrap standard error with a t reference) are checked for their size with
overlapping 24-bar labels (`lib/_size.py`, `bootstrap_size_table`): pairs of series without an
edge (24-bar sums of white noise, so neighbours share 23 terms), the block as in 03-rq1-null
(Politis-White, at least the floor times the overlap), one-sided at a nominal 5 %, for series
of 600 and 2,400 decision times and floors of 2 and 4 (`size_*` options: 300 draws, 300
bootstrap draws each, seed 0). `run.py` writes `tables/bootstrap_size.csv`; reproduce with
`.venv/bin/python experiments/01-benchmark/09-mde-threshold/run.py`. Measured before the run
with the same code (2026-11-17), rejection rate of the mean test / the difference test:

| Decision times | Floor 2: blocks, size | Floor 4: blocks, size |
|---|---|---|
| 600 | 13; 8.7 % / 6.0 % | 6; 7.3 % / 6.0 % |
| 2,400 | 40; 7.3 % / 8.0 % | 26; 5.0 % / 6.7 % |

The tests stay somewhat liberal with few, short blocks; the floor of 4 (owner's decision,
2026-11-17) brings them closer to 5 %. The RQ1 verdict uses the Newey-West Diebold-Mariano
test, not these bootstrap tests; RQ1's hourly cells have about 8,800 decision times.

## Data

The universe of 01-universe (72 assets over five markets), 1h bars; the development folds
d1-d4 of the paper's shared forecast cache (02-forecasts), read through
`src.benchmark.read_cache` from the run named by `forecast_cache`.

## Metrics

- MDE per cell in % of the best baseline's CRPS, and whether it is reached within 10 %.
- Share of cells with MDE at or below X, over all cells and per horizon, for every X.
- The detection rate without an edge per cell, against its expected one-sided level
  0.05 / m (m: the scored cells of the cell's Holm family).

## Decision rule

Development folds d1-d4 only (2024-11 .. 2025-10); no test fold is read, so rq1's threshold
is fixed before its test run. The rule (the smallest X of the grid at which at least
`power_share` = 80 % of the 1-bar cells have an MDE at or below X) is fixed in this plan before
the run, so the threshold is not picked by eye. The
folds are defined in [the README](../../README.md#folds). The question's
own rule is under [Questions](#questions).

## Compute

Minutes on CPU once the shared forecasts exist (the power analysis of rq1 on the development
folds: each resample's long-run variance is computed once for all 21 planted gains); no GPU.

## Status

Added on 2026-11-16 (owner's rule: choosing the threshold is its own question). Plan waiting
for the owner's approval; code tested on a synthetic cache (`tests/benchmark/test_power.py`,
`tests/experiments/01-benchmark/09-mde-threshold/test_mde_threshold_study.py`); not run, the
shared forecast cache does not exist yet (`forecast_cache` is empty).

Changed on 2026-11-16 after an independent code review, before any run: the knee is taken on
the 1 h cells (the horizon 03-rq1-null's Q2 judges), the rule and its settings are read from
03-rq1-null's `study.toml` (no copies), and the code is the one function shared with
03-rq1-null.

Changed on 2026-11-17 (owner's decision, before any run): the threshold is the smallest X at
which at least K = 80 % of the 1-bar cells of d1-d4 have an MDE at or below X (an absolute
rule, the one 03-rq1-null's Q2 applies), in place of the knee; the planted gains are 0 to 10 %
in steps of 0.5; the size check is at the one-sided level alpha / m within each Holm family.
