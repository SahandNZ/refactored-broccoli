import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.data import DataConfig, Instrument, bars_file, load_data_config
from src.enums import Market, Source, Timeframe

type WriteBars = Callable[[Path, Instrument, pd.Series, float], None]

STUDY_DIR = LAB_DIR / "experiments" / "01-universe"
LIMITS = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=Path()).quality
STOCK_CALENDARS = tomllib.loads((LAB_DIR / "configs" / "data.toml").read_text())["stock_calendars"]
HOURS = pd.date_range("2024-01-01", "2025-01-01", freq="1h", tz="UTC", inclusive="left")
COMMON = np.random.default_rng(99).normal(0, 0.003, len(HOURS))

# market, symbol, share of the common factor, liquidity
SPECS = [
    (Market.CRYPTO, "AAA", 0.9, 5.0), (Market.CRYPTO, "BBB", 0.9, 4.0),
    (Market.CRYPTO, "CCC", 0.1, 3.0), (Market.CRYPTO, "DDD", 0.5, 2.0),
    (Market.CRYPTO, "EEE", 0.0, 1.0),
    (Market.FX, "EURUSD", 0.2, 5.0), (Market.FX, "GBPUSD", 0.3, 4.0),
    (Market.FX, "USDCHF", 0.0, 3.0), (Market.FX, "AUDUSD", 0.4, 2.0),
    (Market.FX, "USDHKD", 0.0, 9.0),
    (Market.COMMODITY, "GOLD", 0.1, 1.0), (Market.COMMODITY, "COCOA", 0.0, 1.0),
    (Market.STOCK, "AAPL", 0.3, 3.0), (Market.STOCK, "MSFT", 0.2, 2.0),
    (Market.STOCK, "7203.T", 0.1, 1.0), (Market.STOCK, "SAP.DE", 0.0, 1.5),
]  # fmt: skip


def _instrument(symbol: str, market: Market) -> Instrument:
    source = {Market.CRYPTO: Source.BINANCE, Market.STOCK: Source.ALPACA}.get(
        market, Source.DUKASCOPY
    )
    calendar = None
    if market is Market.STOCK:
        calendar = STOCK_CALENDARS[symbol.rpartition(".")[2] if "." in symbol else ""]
    return Instrument(symbol, source, market, symbol.lower(), Timeframe.H1, False, calendar)


def _world(
    data_dir: Path,
    write_bars: WriteBars,
    specs: list[tuple[Market, str, float, float]],
) -> DataConfig:
    """Write hourly bars of every hour of 2024 (past the held-out end) and the manifest.

    The manifest says every candidate failed its quality checks over the whole history, which
    the selection must ignore."""
    rng = np.random.default_rng(0)
    instruments: list[Instrument] = []
    entries: list[dict[str, object]] = []
    for market, symbol, factor, liquidity in specs:
        instrument = _instrument(symbol, market)
        returns = factor * COMMON + (1 - factor) * rng.normal(0, 0.003, len(HOURS))
        write_bars(
            data_dir, instrument, pd.Series(100 * np.exp(np.cumsum(returns)), HOURS), liquidity
        )
        instruments.append(instrument)
        entries.append({
            "symbol": symbol, "market": market.value, "source": instrument.source.value,
            "passed": False, "reason": "gap 99 days > 10",
            "first_bar": "2024-01-01T00:00:00+00:00", "last_bar": "2024-12-31T23:00:00+00:00",
        })  # fmt: skip
    (data_dir / "manifest.json").write_text(json.dumps({"instruments": entries}))
    return DataConfig("2020-01-01", "2026-11-01", {}, LIMITS, data_dir, tuple(instruments))


@pytest.fixture(name="options")
def _weekly_options(options: dict[str, Any]) -> dict[str, Any]:
    """The shared options with windows long enough for weekly returns: 26 selection weeks and
    13 held-out weeks."""
    return {**options, "end": "2024-07-01", "holdout_end": "2024-10-01"}


@pytest.fixture(name="world")
def _world_fixture(tmp_path: Path, write_bars: WriteBars) -> DataConfig:
    return _world(tmp_path, write_bars, SPECS)


def _select(
    universe: ModuleType, data: DataConfig, options: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    return universe.select_universe(data, universe.read_settings(options), seed=0)


# the whole selection on synthetic data


def test_select_universe_end_to_end_on_synthetic_data(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    outputs = _select(universe, world, options)
    table = outputs["universe"].set_index("symbol")
    assert {"AAA", "EURUSD"} <= set(table.index[table["in_universe"]])
    assert table.loc["USDHKD", "dropped"] == "pegged currency HKD"
    # the manifest's "passed" over the whole history is not used
    assert table["dropped"].isna()["EURUSD"]
    assert table["coverage"].notna().all()
    # passed_data_filter: exactly the candidates the data filter kept, in the universe or not
    assert not table.loc["USDHKD", "passed_data_filter"]
    assert table.loc[table["in_universe"], "passed_data_filter"].all()
    data_reasons = ("pegged currency", "no bar in the selection window", "starts ", "quality: ")
    failed = table["dropped"].fillna("").str.startswith(data_reasons)
    assert (table["passed_data_filter"] == ~failed).all()
    assert set(outputs["correlation"].index) <= set(table.index[table["passed_data_filter"]])
    diversity = outputs["diversity"]
    assert set(diversity["universe"]) == {"Universe", "Liquid", *(f"Random {i}" for i in range(5))}
    assert set(diversity["window"]) == {"selection", "holdout"}
    for _, rows in diversity.groupby("window"):
        answer, evidence = universe.universe_beats_alternatives(rows, percentile=90)
        assert answer in ("yes", "no")
        assert "effective rank" in evidence


def test_two_stock_pools_are_picked_separately_with_their_quotas(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    table = _select(universe, world, options)["universe"].set_index("symbol")
    stocks = table[table["market"] == "stock"]
    assert stocks.loc[["AAPL", "MSFT"], "pool"].eq("stock_us").all()
    assert stocks.loc[["7203.T", "SAP.DE"], "pool"].eq("stock_non_us").all()
    for pool in ("stock_us", "stock_non_us"):
        members = stocks[stocks["pool"] == pool]
        assert members["in_universe"].sum() == 1
        assert members["order"].dropna().tolist() == [1]


def test_only_the_universe_has_an_order(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    table = _select(universe, world, options)["universe"]
    chosen = table[table["in_universe"]]
    assert table["order"].notna().tolist() == table["in_universe"].tolist()
    for _, pool in chosen.groupby("pool"):
        assert sorted(pool["order"]) == list(range(1, len(pool) + 1))
    assert chosen["dropped"].isna().all()


def test_a_pool_with_fewer_candidates_than_its_quota_takes_all_and_records_the_count(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    outputs = _select(universe, world, options)
    table = outputs["universe"].set_index("symbol")
    commodities = table[table["market"] == "commodity"]
    assert commodities["in_universe"].tolist() == [True, True]
    diversity = outputs["diversity"]
    sizes = diversity[diversity["window"] == "selection"].set_index("universe")["size"]
    assert sizes["Universe"] == table["in_universe"].sum() == 2 + 2 + 2 + 1 + 1
    assert (sizes == sizes["Universe"]).all()


def test_liquid_and_random_match_a_pool_cut_short_by_near_duplicates(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # four crypto clones: the pick keeps the anchor and drops the others as near-duplicates,
    # so the crypto pool gives 1 asset, not its quota of 3
    clones = [(Market.CRYPTO, name, 1.0, 5.0 - i) for i, name in enumerate(["AAA", "B", "C", "D"])]
    forex = [(Market.FX, "EURUSD", 0.0, 2.0), (Market.FX, "GBPUSD", 0.0, 1.0)]
    data = _world(tmp_path, write_bars, clones + forex)
    settings = {
        **options, "anchors": {"crypto": ["AAA"], "forex": []},
        "quotas": {"crypto": 3, "forex": 2}, "liquid_pool_factor": {"crypto": 2, "forex": 2},
    }  # fmt: skip
    outputs = _select(universe, data, settings)
    table = outputs["universe"].set_index("symbol")
    assert table.loc[table["market"] == "crypto", "in_universe"].sum() == 1
    assert str(table.loc["B", "dropped"]).startswith("near-duplicate of AAA")
    diversity = outputs["diversity"]
    assert (diversity["size"] == 3).all()
    random = diversity[diversity["universe"].str.startswith("Random")]
    assert (random["size"] == 3).all()


def test_near_duplicates_are_checked_across_pools_in_the_fixed_pool_order(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    # GOLD moves exactly like the coin AAA; crypto is picked before commodities whatever the
    # order of the quotas, so AAA is kept and GOLD dropped as its near-duplicate
    specs = [
        (Market.CRYPTO, "AAA", 1.0, 1.0), (Market.CRYPTO, "BBB", 0.0, 0.5),
        (Market.COMMODITY, "GOLD", 1.0, 1.0), (Market.COMMODITY, "COCOA", 0.0, 1.0),
    ]  # fmt: skip
    settings = {
        **options, "anchors": {"commodity": [], "crypto": []},
        "quotas": {"commodity": 2, "crypto": 1},
        "liquid_pool_factor": {"commodity": 2, "crypto": 2},
    }  # fmt: skip
    table = _select(universe, _world(tmp_path, write_bars, specs), settings)["universe"]
    table = table.set_index("symbol")
    assert table.loc[["AAA", "COCOA"], "in_universe"].all()
    assert str(table.loc["GOLD", "dropped"]).startswith("near-duplicate of AAA")
    # an anchor of a later pool counts from the start: with GOLD a commodity anchor, the coin
    # AAA is its near-duplicate and BBB takes the crypto place
    anchored = {**settings, "anchors": {"commodity": ["GOLD"], "crypto": []}}
    table = _select(universe, _world(tmp_path / "anchored", write_bars, specs), anchored)
    table = table["universe"].set_index("symbol")
    assert table.loc[["GOLD", "BBB"], "in_universe"].all()
    assert str(table.loc["AAA", "dropped"]).startswith("near-duplicate of GOLD")


def test_an_unknown_pool_raises(universe: ModuleType, options: dict[str, Any]):
    settings = {
        **options, "quotas": {"bonds": 1}, "anchors": {"bonds": []},
        "liquid_pool_factor": {"bonds": 2},
    }  # fmt: skip
    with pytest.raises(KeyError, match="bonds"):
        universe.read_settings(settings)


def _q1_rows(sizes_crypto: list[int], sizes_forex: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "universe": ["Universe", "Liquid", "Random 0"],
        "size": [a + b for a, b in zip(sizes_crypto, sizes_forex, strict=True)],
        "size_crypto": sizes_crypto, "size_forex": sizes_forex,
        "effective_rank": [10.0, 9.0, 8.0], "noise_effective_rank": [12.0, 12.0, 12.0],
        "gap": [2.0, np.nan, np.nan], "gap_low": [1.5, np.nan, np.nan],
        "gap_high": [2.5, np.nan, np.nan],
    })  # fmt: skip


def test_the_q1_answer_refuses_universes_of_different_sizes(universe: ModuleType):
    with pytest.raises(ValueError, match="differ in size per pool"):
        universe.universe_beats_alternatives(_q1_rows([8, 7, 8], [16, 16, 16]), percentile=90)


def test_the_q1_answer_refuses_the_same_total_with_other_pool_sizes(universe: ModuleType):
    # 24 assets each, but Liquid has one crypto less and one forex pair more
    with pytest.raises(ValueError, match="size_crypto"):
        universe.universe_beats_alternatives(_q1_rows([8, 7, 8], [16, 17, 16]), percentile=90)


def test_the_q1_answer_refuses_a_table_without_pool_sizes(universe: ModuleType):
    rows = _q1_rows([8, 8, 8], [16, 16, 16]).drop(columns=["size_crypto", "size_forex"])
    with pytest.raises(ValueError, match="no size_<pool> columns"):
        universe.universe_beats_alternatives(rows, percentile=90)


def test_the_q1_answer_states_the_gap_its_interval_and_the_noise_level(universe: ModuleType):
    answer, evidence = universe.universe_beats_alternatives(
        _q1_rows([8, 8, 8], [16, 16, 16]), percentile=90
    )
    assert answer == "yes"
    assert "noise level 12.00" in evidence
    assert "random median 2.00 (paired block-bootstrap normal interval 1.50 to 2.50" in evidence
    without = _q1_rows([8, 8, 8], [16, 16, 16]).assign(gap=np.nan)
    assert "interval" not in universe.universe_beats_alternatives(without, percentile=90)[1]


def test_the_universe_row_of_the_selection_window_has_its_gap_and_interval(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    diversity = _select(universe, world, options)["diversity"].set_index(["universe", "window"])
    universe_row = diversity.loc[("Universe", "selection")]
    selection = diversity.xs("selection", level="window")["effective_rank"]
    random_median = selection[selection.index.str.startswith("Random")].median()
    assert universe_row["gap"] == pytest.approx(selection["Universe"] - random_median)
    assert universe_row["gap_low"] <= universe_row["gap_high"]
    others = diversity.drop(index=("Universe", "selection"))
    assert others[["gap", "gap_low", "gap_high"]].isna().all().all()
    noise = diversity["noise_effective_rank"]
    assert ((noise >= 1.0) & (noise <= diversity["size"])).all()
    pools = [column for column in diversity.columns if column.startswith("size_")]
    assert sorted(pools) == sorted(f"size_{pool}" for pool in options["quotas"])
    assert (diversity[pools].sum(axis=1) == diversity["size"]).all()


def test_more_random_draws_than_the_bootstrap_uses_keep_their_rows_without_a_noise_level(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    first = universe.BOOTSTRAP_RANDOM_UNIVERSES
    outputs = _select(universe, world, {**options, "random_universes": first + 2})
    diversity = outputs["diversity"].set_index(["universe", "window"])
    names = diversity.xs("selection", level="window").index
    assert sum(name.startswith("Random") for name in names) == first + 2
    noise = diversity["noise_effective_rank"].unstack("window")
    late = [f"Random {first}", f"Random {first + 1}"]
    assert noise.loc[late].isna().all().all()
    assert noise.drop(index=late).notna().all().all()
    # the point gap uses every draw; its interval is centred on it
    selection = diversity.xs("selection", level="window")["effective_rank"]
    random_median = selection[selection.index.str.startswith("Random")].median()
    row = diversity.loc[("Universe", "selection")]
    assert row["gap"] == pytest.approx(selection["Universe"] - random_median)
    assert (row["gap_low"] + row["gap_high"]) / 2 == pytest.approx(row["gap"])
    members = outputs["comparison_universes"]
    assert members["universe"].nunique() == first + 4


def test_the_members_of_every_compared_universe_are_saved(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    outputs = _select(universe, world, options)
    members = outputs["comparison_universes"]
    assert list(members.columns) == ["universe", "symbol", "pool"]
    diversity = outputs["diversity"]
    sizes = diversity[diversity["window"] == "selection"].set_index("universe")["size"]
    assert members.groupby("universe").size().sort_index().equals(sizes.sort_index())
    table = outputs["universe"].set_index("symbol")
    chosen = members[members["universe"] == "Universe"]
    assert set(chosen["symbol"]) == set(table.index[table["in_universe"]])
    assert (members["pool"] == table.loc[members["symbol"], "pool"].to_numpy()).all()


def test_rounding_noise_is_not_reported_as_a_negative_eigenvalue(
    universe: ModuleType, caplog: pytest.LogCaptureFixture
):
    # 3 weeks of 4 assets: a rank-deficient matrix whose zero eigenvalues come out near -1e-16
    weeks = np.random.default_rng(12).standard_normal((3, 4))
    rounding = pd.DataFrame(
        np.corrcoef(weeks, rowvar=False), index=list("ABCD"), columns=list("ABCD")
    )
    assert universe.diversity(rounding)["min_eigenvalue"] > universe.NEGATIVE_EIGENVALUE_TOLERANCE
    impossible = rounding.copy()
    impossible.loc[["A", "B", "C"], ["A", "B", "C"]] = [
        [1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0],
    ]  # fmt: skip
    sets = {"Universe": ["A", "B", "C"], "Random 0": ["A", "B", "D"]}
    correlations = {"selection": rounding, "holdout": impossible}
    with caplog.at_level("WARNING"):
        table = universe._study._diversity_table(sets, correlations, {"crypto": list("ABCD")})
    eigenvalues = table.set_index(["window", "universe"])["min_eigenvalue"]
    tolerance = universe.NEGATIVE_EIGENVALUE_TOLERANCE
    assert (eigenvalues["selection"] > tolerance).all()
    assert (eigenvalues["selection"] < 0).any()
    assert eigenvalues[("holdout", "Universe")] < -0.5
    below = int((eigenvalues < tolerance).sum())
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{below} correlation matrices have an eigenvalue below -1e-08")


def test_every_candidate_that_is_not_picked_says_why(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    # crypto: quota 2, liquid pool 2 x 2 = 4 of the 5 coins; EEE, the least liquid, is outside
    table = _select(universe, world, options)["universe"].set_index("symbol")
    assert table.loc["EEE", "dropped"] == "outside liquid pool"
    crypto = table[(table["market"] == "crypto") & ~table["in_universe"]]
    assert crypto["dropped"].notna().all()
    in_pool = crypto.drop(index="EEE")["dropped"]
    assert in_pool.str.match("not picked|near-duplicate of").all()
    assert table.loc[table["in_universe"], "dropped"].isna().all()


def test_each_pool_has_its_own_liquid_pool_factor(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    # crypto factor 1 with quota 2: the liquid pool is AAA and BBB only; forex keeps factor 2
    factors = {**options["liquid_pool_factor"], "crypto": 1}
    table = _select(universe, world, {**options, "liquid_pool_factor": factors})["universe"]
    table = table.set_index("symbol")
    outside = table[table["dropped"] == "outside liquid pool"]
    assert sorted(outside.index[outside["market"] == "crypto"]) == ["CCC", "DDD", "EEE"]
    forex = table[table["pool"] == "forex"]
    assert (forex["dropped"] != "outside liquid pool").all()


def test_a_full_sector_is_the_reason_a_stock_is_not_picked(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    banks = ["JPM", "GS", "C", "WFC"]
    specs = [(Market.STOCK, bank, 0.0, 10.0 - i) for i, bank in enumerate(banks)]
    specs += [(Market.STOCK, "XOM", 0.0, 1.0)]
    settings: dict[str, Any] = {
        **options, "quotas": {"stock_us": 4}, "anchors": {"stock_us": []}, "max_per_sector": 2,
        "liquid_pool_factor": {"stock_us": 2},
    }  # fmt: skip
    table = _select(universe, _world(tmp_path, write_bars, specs), settings)["universe"]
    table = table.set_index("symbol")
    assert table["in_universe"].sum() == 3
    capped = table.loc[banks][~table.loc[banks, "in_universe"]]
    assert (capped["dropped"] == "not picked: sector cap (Financials full)").all()


def test_a_rerun_reproduces_the_numbers(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    first = _select(universe, world, options)
    second = _select(universe, world, options)
    for name in ("universe", "diversity", "correlation", "holdout_correlation"):
        pd.testing.assert_frame_equal(first[name], second[name])


def _change_bars(data: DataConfig, after: str) -> None:
    """Scale the prices of three assets after `after` and remove their bars of each 14th day."""
    rng = np.random.default_rng(5)
    for symbol, market in [("CCC", Market.CRYPTO), ("GBPUSD", Market.FX), ("AAPL", Market.STOCK)]:
        path = bars_file(data.data_dir, market, symbol, Timeframe.H1)
        frame = pd.read_parquet(path)
        times = pd.DatetimeIndex(frame.index)
        late = times >= pd.Timestamp(after, tz="UTC")
        frame.loc[late] = frame.loc[late] * rng.uniform(0.5, 2.0, size=(late.sum(), 1))
        frame[~(late & (times.day == 14))].to_parquet(path)


def test_nothing_uses_data_after_the_held_out_end(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    before = _world(tmp_path / "before", write_bars, SPECS)
    after = _world(tmp_path / "after", write_bars, SPECS)
    _change_bars(after, "2024-10-01")
    first, second = _select(universe, before, options), _select(universe, after, options)
    for name in ("universe", "diversity", "correlation", "holdout_correlation"):
        pd.testing.assert_frame_equal(first[name], second[name])


def test_the_selection_uses_nothing_after_its_end(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    before = _world(tmp_path / "before", write_bars, SPECS)
    after = _world(tmp_path / "after", write_bars, SPECS)
    _change_bars(after, "2024-07-01")
    first, second = _select(universe, before, options), _select(universe, after, options)
    pd.testing.assert_frame_equal(first["universe"], second["universe"])
    pd.testing.assert_frame_equal(first["correlation"], second["correlation"])
    selection = [
        outputs["diversity"][outputs["diversity"]["window"] == "selection"]
        for outputs in (first, second)
    ]
    pd.testing.assert_frame_equal(selection[0], selection[1])
    # the change is real: the held-out window sees it
    assert not first["holdout_correlation"].equals(second["holdout_correlation"])


# settings


def test_every_option_of_study_toml_becomes_a_setting(universe: ModuleType):
    options = tomllib.loads((STUDY_DIR / "study.toml").read_text())["options"]
    settings = universe.read_settings(options)
    assert settings.quotas["stock_us"] == 16
    assert settings.anchors["commodity"] == ("XAUUSD", "COPPER", "BRENT")
    assert isinstance(settings.min_overlap, int)
    assert settings.random_universes == 1000
    assert settings.liquid_pool_factor == {
        "crypto": 4, "forex": 2, "commodity": 4, "index": 4, "stock_us": 4, "stock_non_us": 4,
    }  # fmt: skip


def test_the_liquid_pool_factors_must_name_the_pools_of_the_quotas(
    universe: ModuleType, options: dict[str, Any]
):
    factors = {**options["liquid_pool_factor"], "index": 4}
    with pytest.raises(KeyError, match="liquid_pool_factor"):
        universe.read_settings({**options, "liquid_pool_factor": factors})
    del factors["index"], factors["forex"]
    with pytest.raises(KeyError, match="liquid_pool_factor"):
        universe.read_settings({**options, "liquid_pool_factor": factors})


def test_a_missing_or_unknown_option_raises(universe: ModuleType, options: dict[str, Any]):
    for key in options:
        with pytest.raises(KeyError, match=key):
            universe.read_settings({name: value for name, value in options.items() if name != key})
    with pytest.raises(KeyError, match="min_coverage"):
        universe.read_settings({**options, "min_coverage": 0.98})


def test_a_pool_without_a_quota_raises(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    quotas = {pool: quota for pool, quota in options["quotas"].items() if pool != "stock_us"}
    anchors = {pool: names for pool, names in options["anchors"].items() if pool != "stock_us"}
    factors = {
        pool: factor for pool, factor in options["liquid_pool_factor"].items() if pool != "stock_us"
    }
    settings = {**options, "quotas": quotas, "anchors": anchors, "liquid_pool_factor": factors}
    with pytest.raises(ValueError, match="stock_us"):
        _select(universe, world, settings)


# liquidity in the whole selection


@pytest.mark.parametrize(("liquidity", "first"), [((3.0, 2.0), "AAPL"), ((2.0, 3.0), "MSFT")])
def test_the_us_pick_starts_from_the_largest_dollar_volume(
    universe: ModuleType,
    tmp_path: Path,
    write_bars: WriteBars,
    options: dict[str, Any],
    liquidity: tuple[float, float],
    first: str,
):
    specs = [spec for spec in SPECS if spec[1] not in ("AAPL", "MSFT")]
    specs += [(Market.STOCK, "AAPL", 0.3, liquidity[0]), (Market.STOCK, "MSFT", 0.2, liquidity[1])]
    table = _select(universe, _world(tmp_path, write_bars, specs), options)["universe"]
    table = table.set_index("symbol")
    assert table.loc[first, "order"] == 1
    us = table.loc[["AAPL", "MSFT"]]
    assert us["liquidity"].tolist() == pytest.approx(us["dollar_volume"].tolist())
    assert us["spread_bps"].isna().all()
    assert us["liquidity"].idxmax() == first


def test_a_pool_mixing_dollar_volumes_and_spreads_raises(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    # AAPL gets bid and ask bars (those of EURUSD), like a Dukascopy stock
    forex = pd.read_parquet(bars_file(world.data_dir, Market.FX, "EURUSD", Timeframe.H1))
    forex.to_parquet(bars_file(world.data_dir, Market.STOCK, "AAPL", Timeframe.H1))
    with pytest.raises(ValueError, match="stock_us mixes"):
        _select(universe, world, options)


# sectors

GICS_SECTORS = {
    "Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy",
    "Financials", "Health Care", "Industrials", "Information Technology", "Materials",
    "Real Estate", "Utilities",
}  # fmt: skip


def test_every_stock_candidate_has_one_gics_sector(universe: ModuleType):
    sectors = pd.read_csv(universe.SECTORS_FILE)
    groups = tomllib.loads((LAB_DIR / "configs" / "data.toml").read_text())["groups"]
    stocks = [
        s for group in groups.values() if group["market"] == "stock" for s in group["instruments"]
    ]
    assert sorted(sectors["symbol"]) == sorted(stocks)
    assert len(stocks) == 128
    assert set(sectors["sector"]) <= GICS_SECTORS


def test_the_universe_table_labels_stocks_with_their_sector(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    outputs = _select(universe, world, options)
    table = outputs["universe"].set_index("symbol")
    assert table.loc["AAPL", "sector"] == "Information Technology"
    assert table.loc["7203.T", "sector"] == "Consumer Discretionary"
    assert table["sector"].isna()["EURUSD"]
    mix = outputs["sector_mix"]
    assert sorted(mix.columns) == ["stock_non_us", "stock_us"]
    assert mix.to_numpy().sum() == 2


def test_the_last_bar_of_every_candidate_is_recorded(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    table = _select(universe, world, options)["universe"]
    assert (table["last_bar"] == "2024-12-31T23:00:00+00:00").all()


def test_a_stock_without_a_sector_raises(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    specs = [*SPECS, (Market.STOCK, "NEWCO.PA", 0.0, 1.0)]
    with pytest.raises(ValueError, match=r"NEWCO\.PA"):
        _select(universe, _world(tmp_path, write_bars, specs), options)


def test_a_manifest_candidate_outside_data_toml_raises(
    universe: ModuleType, world: DataConfig, options: dict[str, Any]
):
    smaller = DataConfig(
        world.start, world.end, world.workers, world.quality, world.data_dir,
        world.instruments[1:],
    )  # fmt: skip
    with pytest.raises(KeyError, match="AAA"):
        _select(universe, smaller, options)


def test_sector_mix_counts_the_stocks_of_the_universe_per_pool(universe: ModuleType):
    table = pd.DataFrame(
        {
            "market": ["stock", "stock", "stock", "stock", "forex"],
            "pool": ["stock_us", "stock_us", "stock_non_us", "stock_us", "forex"],
            "in_universe": [True, True, True, False, True],
            "sector": ["Energy", "Energy", "Financials", "Financials", None],
        }
    )
    assert universe.sector_mix(table).to_dict() == {
        "stock_non_us": {"Energy": 0, "Financials": 1},
        "stock_us": {"Energy": 2, "Financials": 0},
    }


def test_a_stock_pool_with_many_banks_takes_at_most_three_and_fills_its_quota(
    universe: ModuleType, tmp_path: Path, write_bars: WriteBars, options: dict[str, Any]
):
    banks = ["JPM", "GS", "C", "WFC", "MS", "AXP"]
    specs = [(Market.STOCK, bank, 0.0, 10.0 - i) for i, bank in enumerate(banks)]
    specs += [(Market.STOCK, other, 0.5, 1.0) for other in ("XOM", "KO", "NVDA")]
    settings: dict[str, Any] = {
        **options, "quotas": {"stock_us": 5}, "anchors": {"stock_us": []},
        "liquid_pool_factor": {"stock_us": 4},
    }  # fmt: skip
    outputs = _select(universe, _world(tmp_path, write_bars, specs), settings)
    table = outputs["universe"].set_index("symbol")
    chosen = table[table["in_universe"]]
    assert len(chosen) == 5
    assert (chosen["sector"] == "Financials").sum() == 3
    assert set(chosen.index) - set(banks) <= {"XOM", "KO", "NVDA"}
    assert outputs["sector_mix"].loc["Financials", "stock_us"] == 3
    diversity = outputs["diversity"]
    selection = diversity[diversity["window"] == "selection"].set_index("universe")
    assert (selection["size"] == 5).all()
    # without the cap Liquid would be five independent banks; with it, two of the correlated
    # non-banks join
    assert selection["mean_abs_correlation"]["Liquid"] > 0.04
