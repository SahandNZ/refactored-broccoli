# 12 Horizon spans: how much calendar time a horizon of h own bars covers

Descriptive, planned on 2026-11-18 (owner's approval) so the paper can say what its horizons
mean in calendar time. It changes no result.

## Questions

**Q1** (RQ-forecasts). How much calendar time does a horizon of h own bars (1, 4, 24) span in each market on the test quarters?
Decision rule: descriptive; yes when every market and horizon has its span measured on the labelled decisions of the test quarters, no otherwise.
Primary metric: median calendar hours from a decision to the close of its label's last own bar.

## Hypothesis

- H1: a horizon of 24 own bars spans about one day in crypto and several days on the markets
  that close (a US stock has 7 own bars a day: the 09:30-10:00 half bar and six full bars).

## Design

- **Labels**: the test labels of the 02-forecasts cache named by `forecast_cache`; no model.
- **Span** (`horizon_spans`): per decision, its label time (the close of the label's last own
  bar) minus the decision time, in hours, so closed hours count; per market and horizon the
  median and the quartiles over the decisions of `windows`.

## Data

The test quarters t1-t4 (2025-11-01 to 2026-10-31) of the cache `test-2026-11-17`, the 72
assets of 01-universe's run `2026-11-17T0005-random-1000`.

## Metrics

`spans.csv`: per market and horizon, the decisions and the median, 25th and 75th percentile
of the span in hours.

## Decision rule

Q1 as under [Questions](#questions).

## Compute

CPU only, seconds.

## Status

Q1 planned and run on 2026-11-18.
