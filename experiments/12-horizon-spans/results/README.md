# Results: 12-horizon-spans

Plan: [../README.md](../README.md). Runs: test-2026-11-18.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-18 | 2026-11-18 03:13 / 04:13 | 1.1 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 1.14 / 0.0 GB | - | 76636803029060a92275ddd4402c828795e3cecb |

## Q1 (RQ-forecasts)

How much calendar time does a horizon of h own bars (1, 4, 24) span in each market on the test quarters?

Primary metric: median calendar hours from a decision to the close of its label's last own bar.

**Q1**: yes. Test quarters, every labelled decision; a horizon of 24 own bars spans: commodity median 27 h (1.1 days; quartiles 25 to 73 h); crypto median 24 h (1.0 days; quartiles 24 to 24 h); forex median 24 h (1.0 days; quartiles 24 to 24 h); index median 32 h (1.3 days; quartiles 25 to 73 h); stock median 120 h (5.0 days; quartiles 75 to 140 h). Spans of every horizon are in spans.csv.

Limitations: descriptive; the span counts calendar hours, including the closed hours of nights, weekends and holidays, so a market's typical span depends on when its decisions fall

Files: [test-2026-11-18/tables/spans.csv](test-2026-11-18/tables/spans.csv)
