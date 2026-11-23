# 11 Closures length: the context's length after a closure, at the same encoding

Exploratory decomposition of [08-closures](../08-closures/README.md), planned after 08's test
results were seen (owner's approval, 2026-11-18). It never changes 08's answers; the paper
reports it in the appendix as exploratory.

## Questions

**Q1** (RQ-closures). On the markets that close, does a zero-shot model forecast the first bars after a reopening better from the last 512 own bars than from the own bars of the last 512 calendar hours, both real closes only?
Decision rule: for each model of 08's on-time family, the one-sided Diebold-Mariano test of 08-closures Q1 (H1: the last 512 own bars have the lower CRPS) on the series of closure days, Holm-adjusted over the tested on-time models; yes if at least one model is better at adjusted p < 0.05, no otherwise; the late models form their own Holm family, reported only.
Primary metric: CRPS of the first 3 bars after closures of 12 hours or more.

## Hypothesis

- H1: most models forecast better from the longer context, so part of what 08-closures shows
  as an effect of the representation is an effect of how many bars reach the model.

## Design

- **Inputs**: 08-closures' benchmark input (the last 512 own bars, the main results' input)
  and its reference, trading hours only (the own bars inside the last 512 calendar hours,
  about 100 for a US stock). Both show real closes only and no closed hour, so they share the
  encoding and differ in the context's length (and in how far back in calendar time it
  reaches). 08's forward-filled and missing arms against trading hours only change the
  encoding at the same bars; that comparison is 08's Q1 and is not repeated here.
- **Scores**: the per-decision CRPS of the 08-closures run named by `closures_run`
  (`metrics.parquet`, its `is_scored` decisions); no forecast is made. Each input drops its own
  invalid forecasts (Chronos-2: 30,611 benchmark-input against 30,680 trading-hours decisions;
  Moirai 2: 23,522 against 23,525), so the two series share their closure days, which the test
  checks, but not every decision inside a day. The settings (`alpha`, `min_closure_days`, `bootstrap_resamples`,
  `bootstrap_seed`) are read from that run's `config.toml` (copied as `closures-config.toml`)
  and the Holm families from its committed `verdict.csv`.
- **Test** (`length_tests`, as 08-closures' `closure_tests`; the code mirrors 08's
  `closure_day_means`, `skill_interval` and `_test` because a study's `lib` cannot import
  another study's, and the seed has no alternative index): each input's losses averaged per
  closure day (over the assets at each time, then over the day's times), the one-sided
  Diebold-Mariano test of the day series with overlap 0, the skill
  `1 - CRPS_benchmark / CRPS_trading` with a moving-block bootstrap interval (Politis-White
  block, `bootstrap_resamples` draws, level 1 - `alpha`); a model with fewer than
  `min_closure_days` days or fewer than five blocks is not tested.
- **Per market** (`length_by_market`, descriptive): the same skill over each market's scored
  decisions.

## Data

The test quarters t1-t4 of 08-closures' run `test-2026-11-17` (closed markets: forex,
commodities, indices, stocks; 72-asset universe of 01-universe).

## Metrics

`length.csv`: per model, closure days, both mean CRPS, skill with interval, Diebold-Mariano
statistic, p-value and Holm-adjusted p; `length_by_market.csv`: per model and market, the
decisions and the skill.

## Decision rule

Q1 as under [Questions](#questions). Nothing here changes 08-closures' answers.

## Compute

CPU only, minutes (reads 08's per-decision scores).

## Status

Q1 planned and run on 2026-11-18 (exploratory, after 08's test results).

### Plan changes

- 2026-11-18, after a review (no change of the test): `length_by_market.csv` also counts the
  benchmark input's decisions; rerun as `test-2026-11-18-v2`, the run the results cite
  (`test-2026-11-18` stays as the first run).
