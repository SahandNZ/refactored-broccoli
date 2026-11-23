# Simple Baselines Still Win: A Leakage-Aware Benchmark of Time-Series Foundation Models on Financial Markets

Code for an anonymous submission. The benchmark evaluates 13 checkpoints of ten zero-shot
time-series foundation models (Chronos-Bolt, Chronos-2, TimesFM 2.5 and 3, Moirai 2, TiRex and
TiRex-2, Toto 2, t0 and the finance-pretrained Kronos) against five simple statistical baselines
on 72 assets in five markets (crypto, FX, commodities, equity indices, stocks), forecasting 1, 4
and 24 hourly bars ahead. Every checkpoint is scored only on data after its release, every test
comes with its smallest detectable gain, forecasts are scored with CRPS, and the sign of the
median is traded net of per-market costs. This repository holds code, the locked study plans
and the aggregated result tables, but no prices: the data providers' terms forbid redistributing
them, so you rebuild the data from the original sources.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```
uv sync --frozen --extra dev --python 3.12
```

Moirai 2 runs in its own environment (Python 3.11, `scripts/moirai_worker.py`):

```
uv venv --python 3.11 .venv-moirai
uv pip install --python .venv-moirai/bin/python -r configs/requirements-moirai.txt
```

To rerun the forecasts you also need Kronos and the Dukascopy downloader (Node.js):

```
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
git -C vendor/Kronos checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
(cd vendor/dukascopy && npm ci)
```

Model weights are downloaded from the Hugging Face Hub at the commits pinned in the code into
`.hf/` (the `HF_HOME` the package sets).

## Rebuild the data

The asset lists are in `configs/data.toml`. Download every series from its original source
(Binance Vision for crypto, Dukascopy for FX, commodities, indices and non-US stocks, Alpaca for
US stocks), each under its provider's terms:

```
export APCA_API_KEY_ID=...        # your own Alpaca key (free account)
export APCA_API_SECRET_KEY=...
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/refresh_manifest.py
```

The data are written to `data/` (ignored by git). `configs/data_checksums.csv` lists the SHA-256
of the files we built; `scripts/write_checksums.py` writes yours, so you can diff the two lists.

## Run the studies

Each study in `experiments/NN-name/` has its plan (`README.md`, locked before its test run), its
settings (`study.toml`: pinned model commits, seeds, thresholds) and two scripts:

```
.venv/bin/python experiments/<study>/run.py --run-id <id>
.venv/bin/python experiments/<study>/analyze.py --run-id <id>
```

Run `01-universe`, then `02-forecasts` (the forecast cache; GPU), then `03` to `13`, which read that
cache. `experiments/README.md` lists every study and its research question; each study's
`results/` folder holds the aggregated tables the paper reports.

## Regenerate the paper's tables and figures

The `make_tab_*.py` scripts in the studies and `paper/scripts/` write the table bodies to
`paper/tables/generated/` and the figures to `paper/figures/` (create both folders first:
`mkdir -p paper/tables/generated paper/figures`). Without any data,
`.venv/bin/python scripts/verify_counts.py` recomputes the paper's main counts from the
committed result tables.

## Tests

```
.venv/bin/python -m pytest -q -m "not data and not cuda"
```

runs on synthetic data and the committed tables (CPU only, no prices, weights or network).

## Licence

MIT (`LICENSE`).
