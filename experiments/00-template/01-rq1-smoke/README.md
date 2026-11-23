# 01 rq1 Smoke: do all zero-shot models run and give valid forecasts?

A real, small study kept as the template: copy this folder to start a study, then replace the
question, the plan and the code. It runs in a few minutes on the local GPU and fills `runs/`
and `results/` like any study.

## Questions

**Q1** (RQ-smoke). Does every zero-shot model load and give valid forecasts for real hourly
contexts?
Decision rule: yes if every model returns 24 steps x 9 quantiles for every context, all finite,
with ordered quantiles and a median within 5 % of the last price; no otherwise (the answer names
the models that failed).
Primary metric: share of models that pass all four checks.

## Hypothesis

- H1: every installed model passes (they were smoke-tested on 2026-11-16).

## Design

Each model of `study.toml` forecasts 24 hourly steps from 512 hourly closes, for 8 recent
contexts each of BTCUSDT, EURUSD and AAPL (24 contexts). Checks per model: shape, finiteness,
quantile order, distance of the first-step median from the last price; also load time, forecast
time and peak GPU memory. One model is on the GPU at a time.

## Data

1h bars of BTCUSDT (Binance), EURUSD (Dukascopy, mid) and AAPL (Alpaca, all bars) from `data/`,
the last 2,400 bars of each.

## Metrics

Pass or fail per check and model; load and forecast seconds; peak GPU memory in GB.

## Decision rule

Every choice is made on the development folds; the test folds run once, after every choice is
fixed ([the README](../../../README.md#folds)). This smoke study has no folds:
it uses the most recent bars only to check the models. Each question's own rule is under
[Questions](#questions).

## Compute

Local GPU (RTX 3500, 12 GB), about 5 minutes for all models, no cost.

## Status

Plan approved by the owner as the template example (2026-11-16) and run.
