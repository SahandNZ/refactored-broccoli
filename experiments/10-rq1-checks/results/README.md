# Results: 10-rq1-checks

Plan: [../README.md](../README.md). Runs: test-2026-11-18-v2.

| Run | Start (UTC / Oslo) | Duration | Status | Where | GPU | Peak RAM / VRAM | Cost | Commit |
|---|---|---|---|---|---|---|---|---|
| test-2026-11-18-v2 | 2026-11-18 03:26 / 04:26 | 127.0 s | completed | local | NVIDIA RTX 3500 Ada Generation Laptop GPU | 3.21 / 0.0 GB | - | 76636803029060a92275ddd4402c828795e3cecb |

## Q1 (RQ-null)

On Kronos's own decision points, does any zero-shot model of rq1's main family beat the best baseline by rq1's rule?

Primary metric: CRPS of the vol-scaled return.

**Q1**: no. Every zero-shot model of rq1's main family on the decisions of kronos-small and its grid, test quarters t1-t4, rq1's rule, one Holm family of 119 tested cells: 0 cells beat the best baseline. On the same decision points at 1 bar(s), the hourly models' gain over Kronos is 4.8 % to 22.2 % (median 12.3 %); 58 of 60 pairs of hourly model, Kronos checkpoint and market have the whole 95 % interval above zero. Made at the close of the first bar after a closure of at least 12 hours (the gap in the last input bar): commodity 43.2 % of Kronos's decisions against 2.9 % of all; forex 32.7 % of Kronos's decisions against 0.9 % of all; index 45.7 % of Kronos's decisions against 2.4 % of all; stock 73.9 % of Kronos's decisions against 13.7 % of all. Target across such a closure (made at the last close before it): commodity 5.0 % against 2.9 %; forex 3.6 % against 0.9 %; index 4.1 % against 2.3 %; stock 4.5 % against 13.6 %.

Limitations: exploratory, planned after rq1's test results were seen; 52 to 55 decision points per cell, so the tests have little power; the best baseline is chosen on all development decisions, as in rq1, not on Kronos's points; the gain over Kronos has an interval but no multiple-testing correction

Files: [test-2026-11-18-v2/tables/verdict.csv](test-2026-11-18-v2/tables/verdict.csv), [test-2026-11-18-v2/tables/kronos_gaps.csv](test-2026-11-18-v2/tables/kronos_gaps.csv), [test-2026-11-18-v2/tables/reopening.csv](test-2026-11-18-v2/tables/reopening.csv)
