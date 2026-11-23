"""07-rq5-off-hours, Q1: contexts, the correction, the model-free arm and the forecasts."""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from src.benchmark import QUANTILE_COLUMNS, calendar_windows
from src.enums import Market
from src.models import QUANTILE_LEVELS

Z = ndtri(np.asarray(QUANTILE_LEVELS))


@pytest.fixture
def weekends(
    off_hours: ModuleType,
    settings: Any,
    off_hours_market: Callable[..., Any],
    off_hours_lab: Callable[..., Any],
    tmp_path: Path,
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=3, start="2023-06-01", end="2025-11-01")
    lab = off_hours_lab(tmp_path, {"FX": ("forex", bars)}, crypto_bars)
    return off_hours.read_weekends(lab, {"FX": Market.FX}, settings)


@pytest.fixture
def fake_models(
    off_hours: ModuleType, off_hours_fake: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Any], list[tuple[str, str]]]:
    # every model is the fake; the pinned revisions are recorded, nothing is looked up
    built: list[Any] = []
    pinned: list[tuple[str, str]] = []

    def build(kind: str, **kwargs: Any) -> Any:
        built.append(off_hours_fake())
        return built[-1]

    def pin(name: str, revision: str) -> None:
        pinned.append((name, revision))

    monkeypatch.setattr(off_hours._reopening, "build_forecaster", build)
    monkeypatch.setattr(
        off_hours._reopening,
        "require_revision",
        pin,
    )
    return built, pinned


# --- contexts -----------------------------------------------------------------------------


def test_every_arm_reads_the_same_calendar_hours_up_to_the_reopening(
    off_hours: ModuleType, settings: Any, weekends: Any
):
    table = weekends.table[weekends.table["reopen_time"] >= "2024-01-01"].reset_index(drop=True)
    contexts = off_hours.reopening_contexts(table, weekends, settings.lookback)
    assert contexts.event.shape == contexts.calendar.shape == (len(table), 512)
    assert contexts.covariates.shape == (len(table), 1, 512)
    # the calendar context's last hour closes at the reopening: the market is still closed
    hours_closed = (table["reopen_time"] - table["close_time"]) / pd.Timedelta(hours=1)
    assert (hours_closed == 49).all()
    assert np.isnan(contexts.calendar[:, -49:]).all()
    np.testing.assert_allclose(contexts.calendar[:, -50], table["friday_close"])
    np.testing.assert_allclose(contexts.covariates[:, 0, -1], table["weekend_crypto"])
    np.testing.assert_allclose(contexts.covariates[:, 0, -50], 0.0, atol=1e-12)
    # event time: exactly the own closes of those hours, in order, ending at Friday's close
    own = np.isfinite(contexts.calendar)
    np.testing.assert_array_equal(contexts.own_bars, own.sum(axis=1))
    for row in (0, len(table) - 1):
        length = contexts.own_bars[row]
        np.testing.assert_array_equal(
            contexts.event[row, -length:], contexts.calendar[row][own[row]]
        )
        assert np.isnan(contexts.event[row, :-length]).all()
    np.testing.assert_allclose(contexts.event[:, -1], table["friday_close"])
    assert 300 < contexts.own_bars.min() <= contexts.own_bars.max() < 512


def test_a_weekend_without_enough_hours_has_no_context(
    off_hours: ModuleType, settings: Any, weekends: Any
):
    first = weekends.table.iloc[[0]]
    contexts = off_hours.reopening_contexts(first, weekends, settings.lookback)
    assert contexts.own_bars[0] == 0
    assert np.isnan(contexts.event).all()
    assert np.isnan(contexts.calendar).all()
    assert np.isfinite(contexts.covariates[0, 0, -1])


def test_changing_prices_after_the_cut_leaves_every_context_unchanged(
    off_hours: ModuleType, settings: Any, weekends: Any
):
    # leakage: the asset's closes after its Friday close and the basket after the reopening
    # are scrambled; no context of that weekend may change
    row = weekends.table[weekends.table["reopen_time"] >= "2024-06-01"].iloc[[0]]
    close_time, reopen_time = row["close_time"].iloc[0], row["reopen_time"].iloc[0]
    before = off_hours.reopening_contexts(row, weekends, settings.lookback)
    closes = {asset: series.copy() for asset, series in weekends.closes.items()}
    for series in closes.values():
        series[series.index > close_time] *= 1.5
    crypto = weekends.crypto.copy()
    crypto[crypto.index > reopen_time] += 1.0
    scrambled = dataclasses.replace(weekends, closes=closes, crypto=crypto)
    after = off_hours.reopening_contexts(row, scrambled, settings.lookback)
    for name in ("event", "own_bars", "calendar", "covariates"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


def test_the_calendar_hours_are_the_benchmarks_windows(
    off_hours: ModuleType, settings: Any, weekends: Any
):
    table = weekends.table.iloc[[60]]
    contexts = off_hours.reopening_contexts(table, weekends, settings.lookback)
    hours = calendar_windows(
        weekends.closes["FX"], pd.DatetimeIndex(table["reopen_time"]), settings.lookback
    )
    np.testing.assert_array_equal(
        contexts.calendar, np.where(hours.is_closed, np.nan, hours.values)
    )


# --- the correction and the model-free arm ------------------------------------------------


def test_the_correction_recovers_a_planted_beta_and_width(off_hours: ModuleType, settings: Any):
    rng = np.random.default_rng(0)
    crypto = rng.normal(0, 2, 400)
    labels = 0.6 * crypto + 0.2 * rng.standard_normal(400)
    quantiles = rng.normal(0, 1e-2, (400, 1)) + 0.1 * Z[None, :]
    beta, width = off_hours.fit_correction(quantiles, crypto, labels, settings.width_factors)
    assert beta == pytest.approx(0.6, abs=0.01)
    assert width == pytest.approx(2.0, abs=0.15)


def _event_rows(n: int, rng: np.random.Generator, *, sigma: float = 0.002) -> pd.DataFrame:
    close = pd.date_range("2024-01-05 21:00", periods=n, freq="7D", tz="UTC")
    crypto = rng.normal(0, 0.02, n)
    move = 0.5 * crypto + 0.002 * rng.standard_normal(n)
    frame = pd.DataFrame(
        {"asset": "FX", "close_time": close, "reopen_time": close + pd.Timedelta(hours=49),
         "label_time": close + pd.Timedelta(hours=50), "sigma": sigma, "weekend_crypto": crypto,
         "move": move, "label": move / sigma}
    )  # fmt: skip
    quantiles = pd.DataFrame(np.tile(Z, (n, 1)), columns=list(QUANTILE_COLUMNS))
    return pd.concat([frame, quantiles], axis=1)


def _fits(off_hours: ModuleType) -> tuple[Callable[..., Any], ...]:
    # the correction's quantiles (without its widths) and the model-free quantiles
    def correction(event: pd.DataFrame, settings: Any) -> Any:
        return off_hours.corrected_quantiles(event, settings)[0]

    return correction, off_hours.model_free_quantiles


def test_the_correction_uses_only_weekends_known_by_the_friday_close(
    off_hours: ModuleType, settings: Any
):
    event = _event_rows(60, np.random.default_rng(1))
    settings = dataclasses.replace(settings, min_fit_weekends=26)
    for arm in _fits(off_hours):
        fitted = arm(event, settings)
        assert np.isnan(fitted[:26]).all()
        assert np.isfinite(fitted[26:]).all()
        # changing the labels from weekend 40 on leaves the fits up to weekend 40 unchanged
        later = event.copy()
        later.loc[40:, "label"] = later.loc[40:, "label"] + 5.0
        np.testing.assert_array_equal(arm(later, settings)[:41], fitted[:41])
        # weekend 39's label is known before weekend 40's Friday close, so it changes 40
        earlier = event.copy()
        earlier.loc[[39], "label"] = earlier.loc[[39], "label"] + 5.0
        changed = arm(earlier, settings)
        np.testing.assert_array_equal(changed[:40], fitted[:40])
        assert not np.allclose(changed[40], fitted[40])


def test_the_fits_are_per_asset(off_hours: ModuleType, settings: Any):
    rng = np.random.default_rng(2)
    one, other = _event_rows(40, rng), _event_rows(40, rng).assign(asset="GOLD")
    for arm in _fits(off_hours):
        together = arm(pd.concat([one, other], ignore_index=True), settings)
        np.testing.assert_array_equal(together[:40], arm(one, settings))


def test_the_crypto_based_arms_centre_on_beta_times_crypto_in_label_units(
    off_hours: ModuleType, settings: Any
):
    event = _event_rows(60, np.random.default_rng(4), sigma=0.004)
    scaled = event["weekend_crypto"][30:] / 0.004
    corrected, widths = off_hours.corrected_quantiles(event, settings)
    assert np.isnan(widths[:26]).all()
    assert np.isin(widths[30:], settings.width_factors).all()
    np.testing.assert_allclose(corrected[30:, 4], 0.5 * scaled, atol=0.5)
    assert (np.diff(corrected[30:], axis=1) > 0).all()
    model_free = off_hours.model_free_quantiles(event, settings)
    np.testing.assert_allclose(model_free[30:, 4], 0.5 * scaled, atol=0.5)
    # its spread is that of the earlier residuals, 0.002 / 0.004 = 0.5 label units
    spread = (model_free[30:, -1] - model_free[30:, 0]) / (Z[-1] - Z[0])
    np.testing.assert_allclose(spread, 0.5, rtol=0.35)


# --- forecasts ----------------------------------------------------------------------------


def test_every_model_forecasts_every_arm_per_weekend_from_its_pinned_revision(
    off_hours: ModuleType, settings: Any, weekends: Any, fake_models: Any
):
    built, pinned = fake_models
    forecasts = off_hours.reopening_forecasts(weekends, ["FX"], settings)
    chosen = weekends.table[weekends.table["reopen_time"] >= settings.split_date]
    assert len(built) == len(settings.zero_shot) == 3
    assert pinned == [(repository, settings.revisions[model])
                      for model, (_, repository) in settings.zero_shot.items()]  # fmt: skip
    arms = forecasts.groupby("model")["arm"].unique().map(sorted).to_dict()
    assert all(found == sorted(off_hours.ARMS) for found in arms.values())
    assert len(forecasts) == len(chosen) * 3 * len(off_hours.ARMS)
    assert list(forecasts.columns) == [
        *off_hours.WEEKEND_COLUMNS,
        *QUANTILE_COLUMNS,
        "model",
        "arm",
        "width",
    ]
    correction = forecasts["arm"] == "correction"
    assert forecasts.loc[~correction, "width"].isna().all()
    assert forecasts.loc[correction & (forecasts["fold"] != ""), "width"].notna().all()
    folds = forecasts.drop_duplicates("reopen_time").set_index("reopen_time")["fold"]
    assert set(folds) == {"", "d1", "d2", "d3", "d4"}
    assert (folds[folds.index < "2024-11-01"] == "").all()
    assert (folds[folds.index >= "2025-08-01"] == "d4").all()
    # the event arm never sees a NaN: one call per context length, its padding cut off
    *events, (calendar,), (same_calendar, covariates) = built[0].inputs
    assert all(np.isfinite(context).all() for (context,) in events)
    assert sum(len(context) for (context,) in events) == len(chosen)
    np.testing.assert_array_equal(calendar, same_calendar)
    assert covariates.shape == (len(calendar), 1, settings.lookback)
    assert np.isnan(calendar).any()
    # the fake's event forecast is centred on Friday's close: a label median of zero
    event = forecasts[(forecasts["arm"] == "event") & (forecasts["model"] == "chronos-2")]
    np.testing.assert_allclose(event["q50"].dropna(), 0.0, atol=1e-9)


def test_the_forecasts_are_in_the_benchmarks_label_units(
    off_hours: ModuleType, settings: Any, weekends: Any, fake_models: Any
):
    forecasts = off_hours.reopening_forecasts(weekends, ["FX"], settings)
    covariate = forecasts[(forecasts["arm"] == "covariate") & forecasts["q50"].notna()]
    # the fake's covariate median is 0.8 times the crypto move, a log return over sigma
    np.testing.assert_allclose(
        covariate["q50"], 0.8 * covariate["weekend_crypto"] / covariate["sigma"], rtol=1e-6
    )
    np.testing.assert_allclose(covariate["label"], covariate["move"] / covariate["sigma"])


def test_only_the_forecast_assets_are_forecast(
    off_hours: ModuleType,
    settings: Any,
    off_hours_market: Callable[..., Any],
    off_hours_lab: Callable[..., Any],
    tmp_path: Path,
    fake_models: Any,
):
    bars, crypto_bars = off_hours_market(beta=0.8, seed=3, start="2023-06-01", end="2025-11-01")
    lab = off_hours_lab(tmp_path, {"FX": ("forex", bars), "GOLD": ("commodity", bars)}, crypto_bars)
    weekends = off_hours.read_weekends(lab, {"FX": Market.FX, "GOLD": Market.COMMODITY}, settings)
    forecasts = off_hours.reopening_forecasts(weekends, ["GOLD"], settings)
    assert set(forecasts["asset"]) == {"GOLD"}


def test_the_covariate_fake_beats_the_other_arms_and_the_fits_need_history(
    off_hours: ModuleType, settings: Any, weekends: Any, fake_models: Any
):
    forecasts = off_hours.reopening_forecasts(weekends, ["FX"], settings)
    for arm in ("correction", "model-free"):
        rows = forecasts[forecasts["arm"] == arm]
        assert rows.loc[rows["reopen_time"] < "2024-07-01", "q50"].isna().all()
        assert rows.loc[rows["fold"] != "", "q50"].notna().all()
    tables = off_hours.reopening_tables(forecasts, settings)
    cells = tables["reopening_cells"].set_index("model")
    assert (cells["status"] == "scored").all()
    assert (cells["crps_covariate"] < cells["crps_event"]).all()
    assert (cells["crps_covariate"] < cells["crps_calendar"]).all()
    tests = tables["reopening_tests"]
    against_calendar = tests[(tests["arm"] == "covariate") & (tests["reference"] == "calendar")]
    assert (against_calendar["adjusted_p"] < 0.05).all()
    assert (against_calendar["skill_low"] > 0).all()


def test_the_model_free_arm_is_the_same_for_every_model_and_a_missing_crypto_move_counts(
    off_hours: ModuleType, settings: Any, weekends: Any, fake_models: Any
):
    scored = weekends.table["reopen_time"] >= "2024-12-01"
    missing = weekends.table.index[scored][0]
    table = weekends.table.copy()
    table.loc[missing, "weekend_crypto"] = np.nan
    forecasts = off_hours.reopening_forecasts(
        dataclasses.replace(weekends, table=table), ["FX"], settings
    )
    model_free = forecasts[forecasts["arm"] == "model-free"]
    by_model = [part[list(QUANTILE_COLUMNS)].to_numpy() for _, part in model_free.groupby("model")]
    for other in by_model[1:]:
        np.testing.assert_array_equal(other, by_model[0])
    # the weekend without a crypto move is forecast, has no correction, and is excluded
    row = forecasts[forecasts["reopen_time"] == table.loc[missing, "reopen_time"]]
    assert row.loc[row["arm"] == "correction", "q50"].isna().all()
    cells = off_hours.reopening_tables(forecasts, settings)["reopening_cells"]
    assert (cells["excluded"] == 1).all()


def test_a_run_without_any_full_context_stops_with_a_clear_error(
    off_hours: ModuleType, settings: Any, weekends: Any, fake_models: Any
):
    early = dataclasses.replace(
        weekends, table=weekends.table[weekends.table["reopen_time"] < "2023-06-15"]
    )
    late = dataclasses.replace(settings, split_date=pd.Timestamp("2023-06-01", tz="UTC"))
    with pytest.raises(ValueError, match="no weekend has a full calendar context"):
        off_hours.reopening_forecasts(early, ["FX"], late)


def test_a_width_beyond_the_grid_is_flagged_at_its_edge(off_hours: ModuleType, settings: Any):
    # labels five times wider than the event forecasts reach for a width of about 5 x 1.28 / 1
    # beyond a grid that ends at 2: every fit sits at the edge, and the cell reports it
    rng = np.random.default_rng(6)
    event = _event_rows(80, rng)
    event["label"] = 8.0 * rng.standard_normal(80)
    narrow = dataclasses.replace(settings, width_factors=np.linspace(0.5, 2.0, 31))
    _, widths = off_hours.corrected_quantiles(event, narrow)
    fitted = widths[np.isfinite(widths)]
    assert len(fitted) > 0
    np.testing.assert_allclose(fitted, 2.0)
    crps = np.where(np.isfinite(widths), 1.0, np.nan)
    group = pd.DataFrame({"arm": "correction", "crps": crps, "width": widths})
    assert off_hours._comparisons._width_at_edge(group, narrow) == pytest.approx(1.0)
    assert off_hours._comparisons._width_at_edge(group, settings) == pytest.approx(0.0)
