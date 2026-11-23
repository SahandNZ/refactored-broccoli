# Results: 11-closures-length

Plan: [../README.md](../README.md). Runs: test-2026-11-18-v2.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-18-v2 | 2026-11-18 03:27 / 04:27 | 2.4 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 1.19 / 0.0 GB | - | 76636803029060a92275ddd4402c828795e3cecb |

## Q1 (RQ-closures)

On the markets that close, does a zero-shot model forecast the first bars after a reopening better from the last 512 own bars than from the own bars of the last 512 calendar hours, both real closes only?

Primary metric: CRPS of the first 3 bars after closures of 12 hours or more.

**Q1**: yes. Closed markets, test quarters, skill of the last 512 own bars (the benchmark input) over the own bars of the last 512 calendar hours (trading hours only), both real closes only: 4 of 6 on-time models are significantly better with the longer context (one-sided Diebold-Mariano on closure days, Holm, 0.05), 2 have the whole interval below zero: chronos-2 1.0 % [0.5, 1.6], adjusted p 0.00034; chronos-bolt-base -2.3 % [-2.8, -1.7], adjusted p 1; chronos-bolt-small -2.0 % [-2.5, -1.4], adjusted p 1; moirai-2-small 0.9 % [0.3, 1.6], adjusted p 0.011; timesfm-2.5 1.7 % [1.4, 2.1], adjusted p 8.2e-22; tirex-1.1 1.1 % [0.8, 1.4], adjusted p 4.9e-13. Late family: 4 of 5 better. The encoding at the same bars (forward-filled and missing against trading hours only) is 08-closures' Q1.

Limitations: exploratory, planned after 08-closures' test results were seen; the benchmark input reaches further back in calendar time as well as holding more bars, so length and calendar reach are not separated; the benchmark input was descriptive and untested in 08's locked plan

Files: [test-2026-11-18-v2/tables/length.csv](test-2026-11-18-v2/tables/length.csv), [test-2026-11-18-v2/tables/length_by_market.csv](test-2026-11-18-v2/tables/length_by_market.csv)
