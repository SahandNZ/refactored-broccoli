from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src import LAB_DIR
from src.benchmark import (
    Bootstrap,
    CachePart,
    NullRule,
    StudySettings,
    _baselines,
    _build,
    build_cache,
    power_tables,
    read_cache,
    saved_paths,
)
from src.benchmark._build import decision_labels, label_quantiles, training_labels
from src.benchmark._cache import LABEL_COLUMNS, QUANTILE_COLUMNS
from src.benchmark._settings import BASELINES, cache_settings, cache_windows, on_model_grid
from src.enums import Market, Timeframe
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS
from src.models import Forecaster, KronosSettings
from src.panel import hourly_closes, load_panel_bars
from tests.benchmark.synthetic_lab import (
    CUT,
    MODELS,
    OFFSETS,
    OPTIONS,
    STOCK_GAP,
    lab_config,
    plain_build,
    skip_revision_check,
    write_lab,
)

STUDY = LAB_DIR / "experiments" / "02-forecasts"


@pytest.fixture(scope="module")
def changed_lab(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_lab(tmp_path_factory.mktemp("changed"), change_after_cut=True)


def _closes(lab: tuple[Path, Path]) -> dict[str, pd.Series]:
    markets = {"BTCUSDT": Market.CRYPTO, "EURUSD": Market.FX, "AAA": Market.STOCK}
    return hourly_closes(load_panel_bars(lab[0], markets, (Timeframe.H1,)))


def test_tables_follow_the_contract(run: Path) -> None:
    cache = read_cache(run)
    labels = cache.labels
    assert list(labels.columns) == list(LABEL_COLUMNS)
    assert set(labels["window"]) == {"d1", "d2", "d3", "d4"}
    assert set(labels["asset"]) == {"BTCUSDT", "EURUSD", "AAA"}
    assert np.isfinite(labels[["label", "log_return", "sigma"]]).all().all()
    scale = labels["sigma"] * np.sqrt(labels["horizon"])
    np.testing.assert_allclose(labels["label"], labels["log_return"] / scale)
    assert (labels["label_time"] > labels["time"]).all()
    assert not labels.duplicated(["time", "asset", "horizon"]).any()
    assert cache.models["model"].tolist() == list(MODELS)
    assert cache.models["kind"].tolist() == ["baseline"] * 5 + ["zero-shot"] * 2
    assert cache.models["release_date"].isna().sum() == 5
    assert cache.models.set_index("model")["pretraining_cutoff"].notna().tolist() == [False] * 6 + [
        True
    ]
    costs = cache.assets.set_index("asset")["cost_per_side_bps"].to_dict()
    assert costs == pytest.approx({"BTCUSDT": 5.0, "EURUSD": 0.3, "AAA": 1.0})
    for model in MODELS:
        assert (run / "forecasts" / f"{model}.parquet").exists()


def test_decisions_are_hourly_and_closed_hours_take_the_label_of_the_last_close(run: Path) -> None:
    labels = read_cache(run).labels
    for fold in DEVELOPMENT_FOLDS:
        times = labels.loc[labels["window"] == fold.name, "time"]
        assert times.min() >= fold.evaluation_start
        assert times.max() < fold.evaluation_end
        assert (times.dt.minute == 0).all()
    # one decision per own close: crypto keeps every hour, forex (closed from Friday 21:00 to
    # Sunday 21:00) has no decision at the weekend
    crypto = labels[(labels["asset"] == "BTCUSDT") & (labels["horizon"] == 1)]
    assert len(crypto) == sum(
        len(pd.date_range(fold.evaluation_start, fold.evaluation_end, freq="1h", inclusive="left"))
        for fold in DEVELOPMENT_FOLDS
    )
    forex = labels[(labels["asset"] == "EURUSD") & (labels["horizon"] == 1)]
    assert not (forex["time"].dt.day_of_week == 5).any()
    assert len(forex) > 0


def test_no_two_labelled_decisions_of_an_asset_share_their_last_close(run: Path) -> None:
    labels = read_cache(run).labels
    # with one decision per own close, the label times of one asset and horizon are distinct
    assert not labels.duplicated(["asset", "horizon", "label_time"]).any()


def test_zero_shot_quantiles_use_the_label_transform(run: Path) -> None:
    frame = read_cache(run).forecasts("fake")
    assert set(frame["seed"]) == {0}
    expected = 0.01 * OFFSETS[None, :] / frame["sigma"].to_numpy()[:, None]
    np.testing.assert_allclose(frame[list(QUANTILE_COLUMNS)], expected, rtol=1e-5)


def test_a_decision_with_a_stale_context_has_no_label(run: Path) -> None:
    cache = read_cache(run)
    reopened = pd.Timestamp(STOCK_GAP[1], tz="UTC")
    stock = cache.labels[cache.labels["asset"] == "AAA"]
    in_gap = stock["time"].between(
        pd.Timestamp(STOCK_GAP[0], tz="UTC") + pd.Timedelta(days=5), reopened
    )
    assert not in_gap.any()
    before_gap = stock[stock["time"] < pd.Timestamp(STOCK_GAP[0], tz="UTC")]
    assert len(before_gap) > 0


def test_every_labelled_decision_gets_a_forecast_from_every_model(run: Path) -> None:
    cache = read_cache(run)
    for model in ("fake", "climatology", "linear"):
        keys = cache.labels[["time", "asset", "horizon"]]
        assert len(keys.merge(cache.forecasts(model), how="inner")) == len(keys)


def test_kronos_decides_once_at_each_167_hour_grid_point_with_one_pooled_forecast(
    run: Path,
) -> None:
    frame = read_cache(run).forecasts("fake-kronos")
    assert set(frame["seed"]) == {0}
    assert not frame.duplicated(["time", "asset", "horizon"]).any()
    starts = frame["window"].map({fold.name: fold.evaluation_start for fold in DEVELOPMENT_FOLDS})
    hours = (frame["time"] - starts) / pd.Timedelta(hours=1)
    crypto = frame["asset"] == "BTCUSDT"
    assert (hours[crypto] % 167 == 0).all()
    grid_point = (hours // 167).astype(int)
    assert (
        not grid_point.groupby([frame["window"], frame["asset"], frame["horizon"]])
        .apply(lambda points: points.duplicated().any())
        .any()
    )


class _SeededPaths:
    """Returns paths that name the torch seed and count the draws since it was set."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def sample_closes(self, candles: list[pd.DataFrame], horizon: int) -> np.ndarray:

        draw = float(torch.rand(1))
        self.calls.append((torch.initial_seed(), draw))
        paths = 100.0 * torch.initial_seed() + np.arange(3, dtype=float) + draw
        return np.broadcast_to(paths[None, :, None], (len(candles), 3, horizon)).copy()


def _no_candles(contexts: pd.DataFrame, bars: object, lookback: int) -> list[int]:
    # one placeholder candle frame per context (the fake models read none)
    return [0] * len(contexts)


def _window_rows(name: str) -> pd.DataFrame:
    # the labelled rows of a window, as far as the Kronos pooling reads them
    return pd.DataFrame({"window": [name, name], "asset": ["A", "B"]})


def test_kronos_reads_saved_paths_by_window_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # a seed saved for d1 and d2 pools d1 and d2 in any order (the arrays are named by window)
    settings = replace(cache_settings(lab_config()), seeds=(0,))
    monkeypatch.setattr(_build, "candle_windows", _no_candles)
    contexts = pd.DataFrame({"asset": ["A", "B"]})
    d1, d2 = (_window_rows("d1"), contexts), (_window_rows("d2"), contexts)
    samples = tmp_path / "samples"
    model = cast("Forecaster", _SeededPaths())
    pool = _build._pooled_kronos_quantiles  # pyright: ignore[reportPrivateUsage]
    forward = pool(model, [d1, d2], {}, settings, samples=samples, seeds=(0,))
    backward = pool(model, [d2, d1], {}, settings, samples=samples, seeds=(0,))
    assert forward is not None
    assert backward is not None
    np.testing.assert_array_equal(forward[0], backward[1])
    with np.load(samples / "seed0.npz") as saved:
        assert sorted(saved.files) == ["d1", "d2"]


def test_saved_paths_refuse_a_missing_window_and_a_wrong_number_of_contexts(
    tmp_path: Path,
) -> None:
    # named: a window the file lacks stops the reading (a resume with other --windows); legacy
    # (arr_0, arr_1, before the arrays were named): read in order only with as many windows,
    # and only when each array has its window's number of contexts
    named, legacy = tmp_path / "named.npz", tmp_path / "legacy.npz"
    two, three = np.zeros((2, 3, 24)), np.ones((3, 3, 24))
    np.savez(named, d1=two, d2=three)
    np.savez(legacy, two, three)
    read = saved_paths(named, ["d2", "d1"], [3, 2])
    np.testing.assert_array_equal(read[0], three)
    with pytest.raises(ValueError, match=r"lacks windows \['d3'\].*delete .*named\.npz and run"):
        saved_paths(named, ["d1", "d3"], [2, 3])
    np.testing.assert_array_equal(saved_paths(legacy, ["d1", "d2"], [2, 3])[1], three)
    with pytest.raises(
        ValueError, match=r"window d3 holds 2 contexts, its grid 3; delete .*and run"
    ):
        saved_paths(legacy, ["d3", "d4"], [3, 2])
    with pytest.raises(ValueError, match="holds 2 unnamed arrays for 3 windows"):
        saved_paths(legacy, ["d1", "d2", "d3"], [2, 3, 2])
    with pytest.raises(
        ValueError, match=r"window d1 holds 2 contexts, its grid 4; delete .*and run"
    ):
        saved_paths(named, ["d1"], [4])


def test_kronos_pools_the_paths_of_every_seed_and_sets_each_seed_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # two windows, two seeds: the seed is set once per seed (not per window), every window is
    # sampled under it in order, and the quantiles come from the 2 x 3 pooled paths
    settings = cache_settings(lab_config())
    monkeypatch.setattr(
        _build, "candle_windows", lambda contexts, bars, lookback: [0] * len(contexts)
    )
    seeded: list[int] = []
    real_seed = _build.seed_everything

    def counting_seed(seed: int) -> None:
        seeded.append(seed)
        real_seed(seed)

    monkeypatch.setattr(_build, "seed_everything", counting_seed)
    model = _SeededPaths()
    contexts = pd.DataFrame({"asset": ["A", "B"]})
    windows = [(_window_rows("d1"), contexts), (_window_rows("d2"), contexts)]
    pooled = _build._pooled_kronos_quantiles(
        model,  # type: ignore[arg-type]
        windows,
        {},
        settings,
        samples=tmp_path / "together",
        seeds=(0, 1),
    )
    assert pooled is not None
    assert seeded == [0, 1]
    assert [seed for seed, _ in model.calls] == [0, 0, 1, 1]
    assert model.calls[0][1] != model.calls[1][1]
    for index, quantiles in enumerate(pooled):
        draws = [model.calls[index][1], model.calls[2 + index][1]]
        paths = np.concatenate(
            [100.0 * seed + np.arange(3) + draw for seed, draw in zip((0, 1), draws, strict=True)]
        )
        expected = np.quantile(paths, np.linspace(0.1, 0.9, 9))
        assert quantiles.shape == (2, 24, 9)
        np.testing.assert_allclose(quantiles[0, 0], expected)
    # the same seeds in two shards: no forecast until both are sampled, then the same one
    first = _build._pooled_kronos_quantiles(
        _SeededPaths(),
        windows,
        {},
        settings,
        samples=tmp_path / "apart",
        seeds=(1,),  # type: ignore[arg-type]
    )
    assert first is None
    assert sorted(path.name for path in (tmp_path / "apart").iterdir()) == ["seed1.npz"]
    second = _build._pooled_kronos_quantiles(
        _SeededPaths(),
        windows,
        {},
        settings,
        samples=tmp_path / "apart",
        seeds=(0,),  # type: ignore[arg-type]
    )
    assert second is not None
    for together, apart in zip(pooled, second, strict=True):
        np.testing.assert_array_equal(together, apart)


def test_no_forecast_row_is_written_for_a_horizon_without_a_label(run: Path) -> None:
    cache = read_cache(run)
    keys = cache.labels[["time", "asset", "horizon"]]
    for model in MODELS:
        frame = pd.read_parquet(run / "forecasts" / f"{model}.parquet")
        assert len(frame.merge(keys, how="inner")) == len(frame), model
    # the last decisions of a window have a 1-bar label but no 24-bar one
    counts = cache.labels.groupby(["time", "asset"])["horizon"].size()
    assert (counts < 3).any()


def test_an_unknown_baseline_raises() -> None:
    settings = cache_settings(lab_config())
    labels = pd.DataFrame({"window": pd.Series([], dtype=str)})
    training = {fold.name: pd.DataFrame() for fold, _ in cache_windows(settings)}
    with pytest.raises(ValueError, match="nope is not a baseline"):
        _build._baseline_forecasts("nope", labels, training, {}, settings)


def test_a_garch_fit_that_does_not_converge_is_left_out_and_listed(
    lab: tuple[Path, Path],
    built: dict[str, list[Forecaster]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # every fit of BTCUSDT fails with arch's flag 8; its windows get no garch row, the other
    # assets are untouched, and each failure is logged and listed in not_fitted.csv
    original = _baselines._garch_variance

    def failing(
        returns: pd.Series, start: pd.Timestamp, horizon: int
    ) -> tuple[pd.DataFrame | None, int]:
        if returns.name == "BTCUSDT":
            return None, 8
        return original(returns, start, horizon)

    monkeypatch.setattr(_baselines, "_garch_variance", failing)
    config = lab_config()
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("garch-normal",))
    cache = read_cache(tmp_path)
    forecasts = cache.forecasts("garch-normal")
    assert "BTCUSDT" not in set(forecasts["asset"])
    assert set(forecasts["asset"]) == set(cache.labels["asset"]) - {"BTCUSDT"}
    listed = pd.read_csv(tmp_path / "not_fitted.csv")
    assert listed["asset"].tolist() == ["BTCUSDT"] * len(DEVELOPMENT_FOLDS)
    assert listed["window"].tolist() == [fold.name for fold in DEVELOPMENT_FOLDS]
    assert (listed["convergence_flag"] == 8).all()


def test_a_crash_before_the_forecasts_lists_a_fit_twice_never_loses_it(
    lab: tuple[Path, Path],
    built: dict[str, list[Forecaster]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # not_fitted.csv and windows.csv are written before the model's forecasts: a crash in
    # between leaves the rows, the resume builds the model again and lists it again
    original = _baselines._garch_variance  # pyright: ignore[reportPrivateUsage]

    def failing(
        returns: pd.Series, start: pd.Timestamp, horizon: int
    ) -> tuple[pd.DataFrame | None, int]:
        return (None, 8) if returns.name == "BTCUSDT" else original(returns, start, horizon)

    monkeypatch.setattr(_baselines, "_garch_variance", failing)

    def crash(folder: Path, model: str, frame: pd.DataFrame) -> None:
        raise RuntimeError("crash before the forecasts are written")

    with monkeypatch.context() as patch:
        patch.setattr(_build, "write_forecasts", crash)
        with pytest.raises(RuntimeError, match="crash before"):
            build_cache(
                tmp_path, lab_config(), data_dir=lab[0], universe=lab[1], models=("garch-normal",)
            )
    first = pd.read_csv(tmp_path / "not_fitted.csv")
    assert not (tmp_path / "forecasts" / "garch-normal.parquet").exists()
    build_cache(tmp_path, lab_config(), data_dir=lab[0], universe=lab[1], models=("garch-normal",))
    listed = pd.read_csv(tmp_path / "not_fitted.csv")
    assert len(listed) == 2 * len(first)
    pd.testing.assert_frame_equal(listed.drop_duplicates(ignore_index=True), first)
    assert pd.read_csv(tmp_path / "windows.csv")["model"].tolist() == ["garch-normal"] * 2


def test_windows_cannot_be_combined_with_a_shard_of_seeds(
    lab: tuple[Path, Path], built: dict[str, list[Forecaster]], tmp_path: Path
) -> None:
    # a sampling model saves one array per window in order: every shard must see every window
    part = CachePart(last_days=None, seeds=(0,), windows=("d2",))
    with pytest.raises(ValueError, match="--windows cannot be combined with --seeds"):
        build_cache(
            tmp_path, lab_config(), data_dir=lab[0], universe=lab[1], models=("fake",), part=part
        )


def test_last_days_keeps_the_last_days_of_each_development_fold(
    lab: tuple[Path, Path], built: dict[str, list[Forecaster]], tmp_path: Path
) -> None:
    config = lab_config()
    metrics = build_cache(
        tmp_path,
        config,
        data_dir=lab[0],
        universe=lab[1],
        models=("ewma-normal",),
        part=CachePart(2, None, None),
    )
    assert metrics["last_days"].tolist() == [2]
    labels = read_cache(tmp_path).labels
    for fold in DEVELOPMENT_FOLDS:
        times = labels.loc[labels["window"] == fold.name, "time"]
        assert times.min() >= fold.evaluation_end - pd.Timedelta(days=2)
        assert times.max() < fold.evaluation_end
    with pytest.raises(ValueError, match="development folds only"):
        build_cache(
            tmp_path / "test",
            replace(config, folds="test"),
            data_dir=lab[0],
            universe=lab[1],
            models=("ewma-normal",),
            part=CachePart(2, None, None),
        )


def test_windows_forecasts_only_those_windows_and_writes_every_label(
    lab: tuple[Path, Path], built: dict[str, list[Forecaster]], tmp_path: Path
) -> None:
    config = lab_config()
    part = CachePart(last_days=None, seeds=None, windows=("d2",))
    models = ("ewma-normal", "climatology", "fake")
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=models, part=part)
    cache = read_cache(tmp_path)
    assert set(cache.labels["window"]) == {fold.name for fold in DEVELOPMENT_FOLDS}
    for model in models:
        assert set(cache.forecasts(model)["window"]) == {"d2"}
    listed = pd.read_csv(tmp_path / "windows.csv")
    assert listed["model"].tolist() == list(models)
    assert (listed["windows"] == "d2").all()
    (tmp_path / "other").mkdir()
    with pytest.raises(ValueError, match=r"windows \['t9'\] are not windows of the run"):
        build_cache(
            tmp_path / "other",
            config,
            data_dir=lab[0],
            universe=lab[1],
            models=("ewma-normal",),
            part=CachePart(last_days=None, seeds=None, windows=("t9",)),
        )


def test_the_pinned_revisions_are_checked_before_a_model_loads(
    lab: tuple[Path, Path],
    built: dict[str, list[Forecaster]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[tuple[str, str]] = []
    monkeypatch.setattr(
        _build,
        "require_revision",
        lambda repository, revision: checked.append((repository, revision)),
    )
    config = lab_config(models=("fake-kronos",))
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("fake-kronos",))
    assert checked == [("test/kronos", "b" * 40), ("test/tokenizer", "c" * 40)]


def test_kronos_takes_each_assets_first_decision_at_or_after_each_grid_point(run: Path) -> None:
    # independent of the window's step: a market that closes decides at its first own close
    # after the grid point, so every asset gets a decision at almost every grid point
    labels = read_cache(run).labels
    on_grid = on_model_grid(labels, "fake-kronos", cache_settings(lab_config()))
    for (window, asset), rows in on_grid.groupby(["window", "asset"]):
        fold = next(fold for fold in DEVELOPMENT_FOLDS if fold.name == window)
        grid = pd.date_range(
            fold.evaluation_start, fold.evaluation_end, freq="167h", inclusive="left"
        )
        own = (
            labels[(labels["window"] == window) & (labels["asset"] == asset)]["time"]
            .drop_duplicates()
            .sort_values()
        )
        expected = {
            own[own >= point].iloc[0]
            for point in grid
            if (own >= point).any() and own[own >= point].iloc[0] < point + pd.Timedelta(hours=167)
        }
        assert set(rows["time"]) == expected
    counts = (
        on_grid.drop_duplicates(["window", "asset", "time"]).groupby(["window", "asset"]).size()
    )
    assert (counts.xs("AAA", level="asset") >= 12).all()
    assert len(on_grid.groupby(["window", "asset", "horizon"])) == 4 * 3 * 3


def test_kronos_reads_real_candles_ending_at_the_last_close(
    lab: tuple[Path, Path], built: dict[str, list[Forecaster]], tmp_path: Path
) -> None:
    config = lab_config(models=("fake-kronos",))
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("fake-kronos",))
    (kronos,) = built["kronos"]
    assert kronos.settings == KronosSettings("test/tokenizer", 3, 1.0, 0.9, pd.Timedelta(hours=1))
    assert all(
        list(frame.columns) == ["open", "high", "low", "close", "volume", "amount"]
        for frame in kronos.candles
    )
    assert {len(frame) for frame in kronos.candles} == {16}


def test_every_model_forecast_at_t_ignores_data_after_t(
    run: Path, changed_lab: tuple[Path, Path], tmp_path: Path
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_build, "build_forecaster", plain_build)
        patch.setattr(_build, "require_revision", skip_revision_check)
        build_cache(
            tmp_path, lab_config(), data_dir=changed_lab[0], universe=changed_lab[1], models=MODELS
        )
    for model in MODELS:
        first = pd.read_parquet(run / "forecasts" / f"{model}.parquet")
        second = pd.read_parquet(tmp_path / "forecasts" / f"{model}.parquet")
        first, second = (
            frame[frame["time"] <= CUT].reset_index(drop=True) for frame in (first, second)
        )
        assert len(first) > 0
        pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-6)


def test_training_labels_end_before_the_window_starts(lab: tuple[Path, Path]) -> None:
    closes = _closes(lab)
    start = DEVELOPMENT_FOLDS[1].evaluation_start
    train = training_labels(closes, start, cache_settings(lab_config()))
    assert (train["label_time"] <= start).all()
    assert train["time"].min() >= start - pd.Timedelta(days=20)
    crypto = train[train["asset"] == "BTCUSDT"]
    assert (
        crypto.groupby("horizon")["time"].max() == start - pd.to_timedelta([1, 4, 24], unit="h")
    ).all()
    assert (train.groupby("horizon")["label_time"].max() == start).all()


# A stock-like series: 1h bars closing 15:00 .. 21:00 UTC on weekdays only (7 a day)


def _stock_closes(seed: int, *, scale: float = 0.01) -> pd.Series:
    close_times = pd.date_range("2025-01-01", "2025-06-01", freq="1h", tz="UTC")
    close_times = close_times[
        (close_times.hour >= 15) & (close_times.hour <= 21) & (close_times.day_of_week < 5)
    ]
    steps = scale * np.random.default_rng(seed).standard_normal(len(close_times))
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=close_times)


def test_a_label_spans_exactly_h_own_bars_of_a_market_that_closes() -> None:
    closes = _stock_closes(0)
    times = pd.date_range("2025-03-03", periods=24 * 14, freq="1h", tz="UTC")
    labels = decision_labels({"AAA": closes}, times, cache_settings(lab_config()), step_hours=1)
    positions = {time: k for k, time in enumerate(closes.index)}
    for horizon, group in labels.groupby("horizon"):
        ends = closes.index[np.searchsorted(closes.index, group["time"], side="right") - 1]
        spans = np.array(
            [
                positions[later] - positions[end]
                for end, later in zip(ends, group["label_time"], strict=True)
            ]
        )
        assert (spans == horizon).all()
        np.testing.assert_allclose(
            group["log_return"],
            np.log(closes[group["label_time"]].to_numpy() / closes[ends].to_numpy()),
        )
    ends = closes.index[np.searchsorted(closes.index, labels["time"], side="right") - 1]
    assert (
        not pd.Series(ends)
        .groupby(labels["horizon"].to_numpy())
        .apply(lambda end: end.duplicated().any())
        .any()
    )
    assert not (labels["time"].dt.day_of_week >= 5).any()
    assert len(labels.drop_duplicates("time")) == 7 * 10


def test_an_asset_that_stops_trading_is_labelled_up_to_its_last_bar() -> None:
    # no survival condition: a delisted asset keeps its decisions until its bars end, and the
    # other assets are not affected
    full = _stock_closes(0)
    stopped = _stock_closes(1)
    stopped = stopped[stopped.index < pd.Timestamp("2025-03-10", tz="UTC")]
    times = pd.date_range("2025-03-03", periods=24 * 14, freq="1h", tz="UTC")
    settings = cache_settings(lab_config())
    both = decision_labels({"AAA": full, "BBB": stopped}, times, settings, step_hours=1)
    alone = decision_labels({"AAA": full}, times, settings, step_hours=1)
    stopped_labels = both[both["asset"] == "BBB"]
    assert not stopped_labels.empty
    assert (stopped_labels["label_time"] <= stopped.index[-1]).all()
    pd.testing.assert_frame_equal(
        both[both["asset"] == "AAA"].reset_index(drop=True), alone.reset_index(drop=True)
    )


def test_a_calibrated_step_h_forecast_is_calibrated_for_the_label() -> None:
    # i.i.d. normal own-bar returns of known size: the step-h quantiles of the true return
    # distribution, put through the label's transform, must cover the label at their levels
    closes = _stock_closes(1, scale=0.01)
    times = pd.date_range("2025-03-03", "2025-05-30", freq="1h", tz="UTC")
    labels = decision_labels(
        {"AAA": closes}, times, cache_settings(lab_config(span_bars=24)), step_hours=1
    )
    labels = labels.drop_duplicates(["asset", "horizon", "label_time"])
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
    for horizon, group in labels.groupby("horizon"):
        last = closes[closes.index[np.searchsorted(closes.index, group["time"], side="right") - 1]]
        prices = last.to_numpy()[:, None, None] * np.exp(
            0.01 * np.sqrt(np.arange(1, 25))[None, :, None] * z
        )
        quantiles = label_quantiles(
            prices, last.to_numpy(), group["sigma"].to_numpy(), (int(horizon),)
        )[:, 0]
        coverage = (group["label"].to_numpy()[:, None] <= quantiles).mean(axis=0)
        np.testing.assert_allclose(coverage, np.linspace(0.1, 0.9, 9), atol=0.12)


def test_a_resumed_run_skips_models_already_written(
    lab: tuple[Path, Path], built: dict[str, list[Forecaster]], tmp_path: Path
) -> None:
    config = lab_config()
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("ewma-normal",))
    # windows.csv is the run's log of the models built: the resume adds a row, keeps the first
    written = {
        path.name: path.stat().st_mtime_ns
        for path in tmp_path.rglob("*.*")
        if path.name != "windows.csv"
    }
    assert "fake" not in built
    build_cache(tmp_path, config, data_dir=lab[0], universe=lab[1], models=("ewma-normal", "fake"))
    assert len(built["fake"]) == 1
    for name, modified in written.items():
        assert next(tmp_path.rglob(name)).stat().st_mtime_ns == modified
    assert pd.read_csv(tmp_path / "windows.csv")["model"].tolist() == ["ewma-normal", "fake"]
    assert (tmp_path / "forecasts" / "fake.parquet").exists()


def test_unknown_models_and_missing_options_raise(lab: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"models \['nope'\] are not in study.toml"):
        build_cache(tmp_path, lab_config(), data_dir=lab[0], universe=lab[1], models=("nope",))
    with pytest.raises(ValueError, match=r"models \['mystery'\] are neither"):
        cache_settings(lab_config(models=("mystery",)))
    options = {key: value for key, value in OPTIONS.items() if key != "linear_ridge"}
    with pytest.raises(KeyError, match="linear_ridge"):
        cache_settings(replace(lab_config(), options=options))


def test_an_empty_universe_raises(lab: tuple[Path, Path], tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    pd.read_csv(lab[1]).assign(in_universe=False).to_csv(universe, index=False)
    with pytest.raises(ValueError, match="no asset in the universe"):
        build_cache(tmp_path, lab_config(), data_dir=lab[0], universe=universe, models=())


def test_test_folds_add_the_test_and_leakage_windows() -> None:
    names = [fold.name for fold, _ in cache_windows(cache_settings(lab_config()))]
    assert names == ["d1", "d2", "d3", "d4"]
    windows = cache_windows(cache_settings(replace(lab_config(), folds="test")))
    assert [fold.name for fold, _ in windows] == [
        *names,
        "t1",
        "t2",
        "t3",
        "t4",
        *(f"l{k}" for k in range(1, 13)),
    ]
    assert {fold.name[0]: step for fold, step in windows} == {"d": 1, "t": 1, "l": 4}


@settings(max_examples=60, deadline=None)
@given(
    log_prices=arrays(np.float64, (5, 24, 9), elements=st.floats(-0.2, 0.2)),
    last_close=arrays(np.float64, (5,), elements=st.floats(1e-3, 1e5)),
    sigma=arrays(np.float64, (5,), elements=st.floats(1e-4, 0.1)),
)
def test_label_quantiles_invert_to_the_price_quantiles(log_prices, last_close, sigma) -> None:
    prices = last_close[:, None, None] * np.exp(np.sort(log_prices, axis=-1))
    quantiles = label_quantiles(prices, last_close, sigma, (1, 4, 24))
    assert quantiles.shape == (5, 3, 9)
    for column, horizon in enumerate((1, 4, 24)):
        back = last_close[:, None] * np.exp(
            quantiles[:, column] * sigma[:, None] * np.sqrt(horizon)
        )
        np.testing.assert_allclose(back, prices[:, horizon - 1], rtol=1e-9)
    assert (np.diff(quantiles, axis=-1) >= -1e-12).all()


def test_label_quantiles_mark_non_positive_prices_invalid() -> None:
    prices = np.ones((2, 1, 9))
    prices[0, 0, 0], prices[1, 0, 3] = 0.0, -1.0
    quantiles = label_quantiles(prices, np.ones(2), np.ones(2), (1,))
    assert quantiles[0, 0, 0] == -np.inf
    assert np.isnan(quantiles[1, 0, 3])
    assert np.isfinite(np.delete(quantiles.reshape(-1), [0, 12])).all()


# A locked test run end to end: every window (d1-d4, t1-t4, l1-l12) built and scored


def test_a_test_fold_cache_is_built_and_scored_by_every_study(
    tmp_path_factory: pytest.TempPathFactory, load_study_lib: Callable[[Path], ModuleType]
) -> None:
    lab = write_lab(
        tmp_path_factory.mktemp("long-lab"), change_after_cut=False, start="2021-11-15",
        end="2026-11-03",
    )  # fmt: skip
    folder = tmp_path_factory.mktemp("test-run")
    models = (*BASELINES, "fake", "fake-kronos")
    config = replace(lab_config(models=models), folds="test", plan_locked=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_build, "build_forecaster", plain_build)
        patch.setattr(_build, "require_revision", skip_revision_check)
        build_cache(folder, config, data_dir=lab[0], universe=lab[1], models=models)
    cache = read_cache(folder)
    windows = {fold.name for fold, _ in cache_windows(cache_settings(config))}
    assert set(cache.labels["window"]) == windows
    assert not cache.labels.duplicated(["time", "asset", "horizon"]).any()
    ends = {fold.name: fold.evaluation_end for fold, _ in cache_windows(cache_settings(config))}
    assert (cache.labels["label_time"] <= cache.labels["window"].map(ends)).all()
    validity = load_study_lib(STUDY).validity_table(cache, config)
    assert (validity.loc[validity["window"] == "all", "share"] == 1.0).all()
    bootstrap = Bootstrap(level=0.95, n_resamples=20, seed=0, block_hours=24, min_block=4)
    settings = StudySettings(DEVELOPMENT_FOLDS, TEST_FOLDS, 0.05, bootstrap)
    rule = NullRule(3, 3, 50, (0.0, 5.0), 0.8, 1, (0.0, 10.0), 0.8, (0.5, 0.9), 2.0)
    names = list(models)
    null = power_tables(cache, names, settings, rule)
    assert set(null["quarters"]["window"]) == {"t1", "t2", "t3", "t4"}
    assert (null["cells"]["status"] == "scored").any()
    rq2 = load_study_lib(STUDY.parent / "04-rq2-center-vs-scale")
    centre_rule = rq2.CentreScaleRule("ewma-normal", "climatology", 3, 3, 0.05, 1)
    centre = rq2.centre_scale_tables(cache, names, settings, rule=centre_rule)
    assert (centre["cells"]["status"] == "scored").any()
    net = load_study_lib(STUDY.parent / "05-rq3-net").net_tables(cache, names, settings)
    assert set(net["quarters"]["window"]) == {"t1", "t2", "t3", "t4"}
    rq4 = load_study_lib(STUDY.parent / "06-rq4-leakage")
    options = rq4.LeakageOptions(
        grid_hours=4,
        placebo_models=("fake",),
        placebo_split=pd.Timestamp("2024-01-01", tz="UTC"),
        not_built=(),
    )
    leakage = rq4.leakage_tables(
        cache, names, settings, before_folds=LEAKAGE_FOLDS, options=options
    )
    periods = leakage["periods"]
    assert set(periods.loc[periods["model"] == "fake", "period"]) == {
        "before", "after", "after-hourly", "early", "late",
    }  # fmt: skip
    assert set(periods.loc[periods["model"] == "fake-kronos", "period"]) == {
        "before", "between", "after",
    }  # fmt: skip
    assert (leakage["cells"]["test"] == "leakage").any()


def test_the_cost_uses_the_median_spread_of_the_spread_window_only(run: Path) -> None:
    # the synthetic forex pair quotes 0.6 bps in the spread window and 5 bps in every fold, so
    # its half spread is 0.3 bps, not a median over the folds
    costs = read_cache(run).assets.set_index("asset")["cost_per_side_bps"]
    assert costs["EURUSD"] == pytest.approx(0.3)


def test_median_spread_reads_only_bars_opening_inside_the_window() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {"mid_close": 100.0, "spread": [0.01, 0.01, 0.01, 0.5, 0.5, 0.5]}, index=index
    )
    inside = _build.median_spread_bps(bars, start=index[0], end=index[3])
    assert inside == pytest.approx(1.0)
    assert np.isnan(
        _build.median_spread_bps(bars.drop(columns="spread"), start=index[0], end=index[3])
    )
