# 10 rq1 checks: rq1 on Kronos's decision points

Exploratory checks of rq1 (03-rq1-null), planned after rq1's test results were seen (owner's
approval, 2026-11-18). They never change rq1's answer; the paper reports them in the
appendix as exploratory.

## Questions

**Q1** (RQ-null). On Kronos's own decision points, does any zero-shot model of rq1's main family beat the best baseline by rq1's rule?
Decision rule: rq1's rule (03-rq1-null Q1: lower mean CRPS in at least 3 of the 4 test quarters and a one-sided Diebold-Mariano test significant after Holm, adjusted p < 0.05), with every zero-shot model scored only on the decisions Kronos-small forecast and on Kronos's 167-hour grid; one Holm family over the tested cells of the eight models; yes if any cell passes, no otherwise.
Primary metric: CRPS of the vol-scaled return.

## Hypothesis

- H1: no hourly zero-shot model beats the best baseline on Kronos's decision points either.
- H1b (reported, not the decision): on the same decision points the hourly models have a
  lower CRPS than Kronos (`kronos_gaps.csv`), so Kronos's weaker cells in rq1 do not come
  only from where it decides.
- H2: on the markets that close, Kronos's decisions follow a closure more often than the
  hourly decisions do (`reopening.csv`).

## Design

- **Cache**: the test forecast cache of 02-forecasts named by `forecast_cache`; no forecast
  is made. The rule and its settings are read from 03-rq1-null's `study.toml` (copied into
  the run as `rq1-config.toml`), so the test, its blocks, its skill interval and the best
  baseline are rq1's.
- **Kronos's decision points** (`grid_cache`): every zero-shot model of `[models]` keeps only
  the (time, asset, horizon) decisions that `grid_model` (Kronos-small; Kronos-base decides at
  the same points) forecast (a model also loses its own invalid forecasts there: Chronos-2
  4), and every model decides on Kronos's 167-hour grid, so
  `src.benchmark.null_tables` averages each cell over the same decision points as Kronos.
  The baselines keep every forecast, so each window's best baseline is chosen on all
  development decisions as in rq1; it is paired with each model on that model's decisions.
  Kronos's own cells are therefore rq1's cells.
- **Holm**: one family, the tested cells of the eight zero-shot models (rq1's main family);
  the late models are not compared (most have no decision on Kronos's points in enough
  quarters).
- **Gain over Kronos** (`kronos_gaps`, reported, not the decision): per hourly model, Kronos
  checkpoint and market, at `gap_horizon` (1 bar), both models' losses averaged over the
  market's assets at each Kronos decision point; the gain `100 (1 - CRPS_model /
  CRPS_kronos)` on the points both scored, with a 95 % circular-block bootstrap interval of
  the paired points (a percentile interval over `bootstrap_resamples` draws with rq1's seed,
  resampling Kronos's decision points in circular blocks of the Politis-White length of the
  difference, at least 2; no interval with fewer than five blocks).
- **Reopenings** (`reopening_shares`): per market, over all labelled test decisions and over
  Kronos's, the share of horizon-1 decisions made at the close of an asset's first own bar
  after a closure of at least `min_closure_hours` (12) hours without a bar (the gap lies in
  the last bar of their input), and the share made at the last close before such a closure
  (their target is the reopening bar, 08-closures' first target). The closure finder is
  08-closures' (`src.utils.closures`); 08's targets are the first three bars after the
  reopening, while here only the decision whose target is the reopening bar counts.

## Data

The test folds t1-t4 (2025-11-01 to 2026-10-31) of the 02-forecasts cache `test-2026-11-17`,
the 72 assets of 01-universe's run `2026-11-17T0005-random-1000`, horizons 1, 4 and 24 bars.

## Metrics

- Per cell (model, market, horizon): rq1's `cells` columns (mean CRPS of the model and of the
  best baseline, skill with its interval, Diebold-Mariano statistic and p-value, quarters won).
- `adjusted_p` (Holm over the family) and `beats_baseline` per cell (`analyze.py`).
- `kronos_gaps.csv`: per hourly model, Kronos checkpoint and market, the points, the gain and
  its interval.
- `reopening.csv`: decisions and reopening share per market, all decisions and Kronos's.

## Decision rule

Q1 as under [Questions](#questions). Nothing here changes rq1's answer.

## Compute

CPU only, minutes (reads the test cache).

## Status

Q1 planned and run on 2026-11-18 (exploratory, after rq1's test results).

### Plan changes

- 2026-11-18 (before the final run, after a review): the gain over Kronos with its paired
  interval is added (`kronos_gaps.csv`), the reopening count uses `src.utils.closures`, and
  the planned question on baselines fitted on the models' context leaves this plan until its
  rule is written.
- 2026-11-18, second review: `reopening.csv` also counts the decisions whose target spans a
  closure (the first share puts the gap in the input, not the target); rerun as
  `test-2026-11-18-v2`, the run the results cite.
- 2026-11-18: the first run, under the same id `test-2026-11-18` and before these review
  fixes, was deleted and run again (an error against experiments rule 8; its Q1 answer was
  the same: no cell passes, 119 tested cells). `runs/test-log.csv` keeps both starts.
