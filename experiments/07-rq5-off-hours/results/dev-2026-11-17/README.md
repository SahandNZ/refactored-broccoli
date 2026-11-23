# Results: 07-rq5-off-hours

Plan: [../README.md](../README.md). Runs: dev-2026-11-17.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| dev-2026-11-17 | 2026-11-17 20:25 / 21:25 | 126.8 s | completed | local | NVIDIA GeForce RTX 4090 | 12.14 / 3.44 GB | - | None |

## Q1 (RQ-off-hours)

Does feeding the weekend crypto move to a zero-shot model as a covariate improve its forecast of the first bar after a weekend reopening?

Primary metric: CRPS of the first bar after the reopening, in the benchmark's label units.

**Q1**: no. the covariate arm beats all four other arms in 0 of 9 tested market and model cells: none. One-sided tests (adjusted p below 0.05): covariate vs calendar: 3 of 9 cells, median skill +5.9 % (cell intervals from -29.1 to +21.9 %); covariate vs event: 0 of 9 cells, median skill -48.3 % (cell intervals from -325.3 to -4.6 %); covariate vs correction: 0 of 9 cells, median skill -75.2 % (cell intervals from -341.6 to -21.7 %); covariate vs model-free: 0 of 9 cells, median skill -73.7 % (cell intervals from -327.8 to -20.1 %); correction vs event: 6 of 9 cells, median skill +6.7 % (cell intervals from -1.1 to +21.9 %); model-free vs event: 6 of 9 cells, median skill +5.3 % (cell intervals from -2.6 to +22.1 %). Median share of the move's crypto slope a model passes on from its covariate (descriptive): chronos-2 -0.34; t0-gamma 2.33; tirex-2 2.02. 0 cells not tested.

Limitations: weekends from 2024-01-01, scored in the d1-d4 folds, about one weekly point per weekend and cell (low power); no cut at a model's release: some or all scored weekends precede a model's release (on d1-d4 all of them, for every model; on t1-t4 none for Chronos-2, until 2026-07-02 for TiRex-2 and almost all for t0, released 2026-10-15), so a model that memorised a reopening could favour its own arms over the correction and the model-free arm; one forecast per weekend and model; the label is the close of the first bar, not the gap of Q2; the correction and the model-free arm are refitted before every weekend on all earlier weekends since 2024-01-01 whose label was known (at least 26), also inside the scored quarter (past labels only); the covariate use is descriptive (no interval)

Files: [dev-2026-11-17/tables/reopening_cells.csv](tables/reopening_cells.csv), [dev-2026-11-17/tables/reopening_tests.csv](tables/reopening_tests.csv), [dev-2026-11-17/tables/covariate_use.csv](tables/covariate_use.csv)

## Q2 (RQ-off-hours)

Does the weekend crypto move explain the reopening gap of closed-market assets beyond Friday's information?

Primary metric: the crypto coefficient's two-sided Holm-adjusted p-value per asset (its
sign, HAC interval, rank correlation and R^2 increment are reported next to it).

**Q2**: yes. 9 of 85 tested assets: EURHUF beta -0.01 [-0.01, -0.00] (negative, adjusted p 6.9e-05), trimmed rank corr -0.15, R^2 +0.06; USDILS beta -0.02 [-0.02, -0.01] (negative, adjusted p 0.01), trimmed rank corr -0.20, R^2 +0.04; AUS200 beta 0.03 [0.01, 0.04] (positive, adjusted p 0.034), trimmed rank corr 0.26, R^2 +0.14; CHE20 beta 0.04 [0.02, 0.05] (positive, adjusted p 0.0015), trimmed rank corr 0.26, R^2 +0.17; ESP35 beta 0.04 [0.02, 0.06] (positive, adjusted p 0.0019), trimmed rank corr 0.30, R^2 +0.15; EUS50 beta 0.05 [0.02, 0.07] (positive, adjusted p 0.0086), trimmed rank corr 0.26, R^2 +0.15; FRA40 beta 0.05 [0.03, 0.07] (positive, adjusted p 0.0013), trimmed rank corr 0.25, R^2 +0.15; NLD25 beta 0.05 [0.02, 0.07] (positive, adjusted p 0.015), trimmed rank corr 0.26, R^2 +0.17; PLN20 beta 0.06 [0.03, 0.08] (positive, adjusted p 0.0033), trimmed rank corr 0.32, R^2 +0.15. Median rank correlation 0.08; on crypto-down weekends 0.03, on crypto-up weekends 0.07; positive in 468 of 760 tested asset half-years. 0 assets not tested.

Limitations: weekends 2020-01-01 .. 2025-11-01, in effect from the listing of the last basket asset on Binance USD-M, before which the basket move is missing (weekends counted as incomplete); assets that passed 01-universe's data checks of 2020-2023, but from candidate lists of 2026; Friday's information is only the last 6 hours of the asset and the basket; two-sided HAC t, Holm over 85 assets, a little liberal with 4 Bartlett lags (about 7 % at 5 % with 4 lags on 100 synthetic null weekends); 95% HAC intervals, not Holm-adjusted; the gap is a mid quote of a CFD, whose spread is wide at the reopening; an in-sample regression over all weekends up to the end of the scored folds, so a test run includes the development months

Files: [dev-2026-11-17/tables/gap_assets.csv](tables/gap_assets.csv), [dev-2026-11-17/tables/gap_half_years.csv](tables/gap_half_years.csv), [dev-2026-11-17/tables/gap_directions.csv](tables/gap_directions.csv)

## Reading the answers (added after review)

An independent review on 2026-11-17 recomputed every number of both runs from
`forecasts.parquet` and `weekends.csv` (CRPS per cell and arm, skills, intervals, the
Diebold-Mariano tests, Holm, the Q2 regressions) and found them exact; it found no look-ahead
in the correction and model-free fits and no implementation error in the contexts. What the
numbers mean:

- **Why the covariate arm loses.** The covariate is given only in calendar time, whose context
  ends in about 49 missing closed hours, so every model forecasts as if about 50 hours ahead:
  its 10-90 % interval is 5 to 12 times wider than the event forecast's, with a coverage of
  84-98 %, and its median drifts by 1-2.5 label units. Covariate against event time, the
  correction and the model-free forecast therefore mostly measures calendar time against
  event time; only covariate against calendar isolates the covariate. 08-closures' "missing"
  arm shows the same loss on its development run. Q1's "no" holds under its rule, but it does
  not show that a zero-shot model cannot use the weekend crypto move: giving the move in
  event time, as a known covariate at the forecast step (known at `reopen_time`), is not
  tested.
- **The covariate-against-calendar wins are Chronos-2's only** (3 of 9 cells on both runs), and
  Chronos-2 does not pass on the crypto direction (`use_share` about 0 or negative): its gain
  comes from a less drifting centre of the forecast, not from following crypto.
- **The event forecast is too narrow at a reopening.** It is worse than the naive
  `ewma-normal` in every cell, with a 10-90 % coverage of 40-63 %, so "correction and
  model-free beat event time" largely repairs that width; the model-free forecast barely
  beats `ewma-normal` on forex.
- **`use_share` is not meaningful for forex**, whose move has no crypto slope (`move_slope`
  about 0): its forex values (e.g. -29.8 for t0 on the test run) enter the per-model medians
  above; read the commodity and index cells.
