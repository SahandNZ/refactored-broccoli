"""07-rq5-off-hours, Q1: scores, one-sided tests, skill intervals and the covariate use."""

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from src.benchmark import QUANTILE_COLUMNS
from src.models import QUANTILE_LEVELS
from src.stats import (
    BLOCK_OVERLAP_FLOOR,
    MIN_BOOTSTRAP_BLOCKS,
    diebold_mariano,
    holm,
    politis_white_block,
)

Z = ndtri(np.asarray(QUANTILE_LEVELS))
LEVELS = np.asarray(QUANTILE_LEVELS)


def _forecast_table(
    off_hours: ModuleType, rng: np.random.Generator, *, weeks: int = 40
) -> pd.DataFrame:
    # one model on two markets over `weeks` weekends of d1-d2: on "forex" the covariate arm
    # sits on the label; on "index" it is the widest
    reopen = pd.date_range("2024-11-03 22:00", periods=weeks, freq="7D", tz="UTC")
    parts: list[pd.DataFrame] = []
    for market in ("forex", "index"):
        labels = rng.standard_normal(weeks)
        centres = {"event": 0.0 * labels, "calendar": 0.0 * labels, "covariate": labels,
                   "correction": 0.5 * labels, "model-free": 0.4 * labels}  # fmt: skip
        widths = {"event": 1.0, "calendar": 1.2, "covariate": 0.2, "correction": 0.8,
                  "model-free": 0.9}  # fmt: skip
        if market == "index":
            centres["covariate"], widths["covariate"] = 0.0 * labels, 5.0
        for arm in off_hours.ARMS:
            quantiles = centres[arm][:, None] + widths[arm] * Z[None, :]
            frame = pd.DataFrame(quantiles, columns=list(QUANTILE_COLUMNS))
            parts.append(
                frame.assign(
                    market=market,
                    asset=market.upper(),
                    model="m",
                    arm=arm,
                    fold="d1",
                    week=reopen.normalize() - pd.Timedelta(days=6),
                    reopen_time=reopen,
                    label_time=reopen + pd.Timedelta(hours=1),
                    label=labels,
                    sigma=0.01,
                    weekend_crypto=0.01 * labels,
                    move=0.01 * labels,
                    width=np.nan,
                )
            )
    return pd.concat(parts, ignore_index=True)


def _losses(rows: pd.DataFrame) -> np.ndarray:
    error = rows["label"].to_numpy()[:, None] - rows[list(QUANTILE_COLUMNS)].to_numpy()
    return 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean(axis=1)


def test_every_comparison_is_a_one_sided_dm_test_holm_adjusted_over_every_cell(
    off_hours: ModuleType, settings: Any
):
    forecasts = _forecast_table(off_hours, np.random.default_rng(0))
    tables = off_hours.reopening_tables(forecasts, settings)
    tests = tables["reopening_tests"]
    assert len(tests) == 2 * len(off_hours.COMPARISONS)
    for row in tests.itertuples():
        rows: pd.DataFrame = forecasts[forecasts["market"] == str(row.market)]
        arm = _losses(rows[rows["arm"] == str(row.arm)])
        reference = _losses(rows[rows["arm"] == str(row.reference)])
        expected = diebold_mariano(arm, reference, overlap=0)
        assert row.p_value == pytest.approx(expected["p_less"])
        assert row.dm_statistic == pytest.approx(expected["statistic"])
        assert row.crps_arm == pytest.approx(arm.mean())
        assert row.skill == pytest.approx(100 * (1 - arm.mean() / reference.mean()))
        assert row.skill_low <= row.skill <= row.skill_high
        assert row.block >= 1
    np.testing.assert_allclose(tests["adjusted_p"], holm(tests["p_value"].to_numpy()))
    np.testing.assert_array_equal(tests["is_better"], tests["adjusted_p"] < settings.alpha)


def test_the_covariate_arm_is_best_only_when_it_beats_all_four_others(
    off_hours: ModuleType, settings: Any
):
    forecasts = _forecast_table(off_hours, np.random.default_rng(1))
    tables = off_hours.reopening_tables(forecasts, settings)
    cells = tables["reopening_cells"].set_index("market")
    assert cells.loc["forex", "covariate_is_best"]
    assert not cells.loc["index", "covariate_is_best"]
    assert cells.loc["forex", "coverage_covariate"] == pytest.approx(1.0)
    # a worse arm has a one-sided p near 1 and a skill interval below zero
    tests = tables["reopening_tests"].set_index(["market", "arm", "reference"])
    worse = tests.loc[("index", "covariate", "event")]
    assert worse["p_value"] > 0.99
    assert worse["skill_high"] < 0


def test_a_cell_short_of_the_covariate_tests_is_not_best(off_hours: ModuleType, settings: Any):
    forecasts = _forecast_table(off_hours, np.random.default_rng(1))
    tests = off_hours.reopening_tables(forecasts, settings)["reopening_tests"]
    cells = off_hours.reopening_tables(forecasts, settings)["reopening_cells"]
    kept = tests[~((tests["arm"] == "covariate") & (tests["reference"] == "model-free"))]
    assert not off_hours._comparisons._with_verdict(cells, kept)["covariate_is_best"].any()


def test_staggered_reopenings_of_a_market_pool_into_one_point_per_weekend(
    off_hours: ModuleType, settings: Any
):
    # a second forex asset reopens on Monday 07:00, nine hours after the first: still one
    # point per weekend, and no label overlaps the next weekend
    forecasts = _forecast_table(off_hours, np.random.default_rng(6))
    second = forecasts[forecasts["market"] == "forex"].assign(asset="OTHER")
    second["reopen_time"] += pd.Timedelta(hours=9)
    second["label_time"] += pd.Timedelta(hours=9)
    second[list(QUANTILE_COLUMNS)] += 0.1
    pooled = pd.concat([forecasts, second], ignore_index=True)
    tables = off_hours.reopening_tables(pooled, settings)
    cells = tables["reopening_cells"].set_index("market")
    assert cells.loc["forex", "times"] == 40
    assert cells.loc["forex", "weekends"] == 80
    assert cells.loc["forex", "overlap"] == 0
    forex = pooled[pooled["market"] == "forex"]
    event = forex[forex["arm"] == "event"].assign(loss=_losses)
    weekly = event.groupby("week")["loss"].mean()
    assert cells.loc["forex", "crps_event"] == pytest.approx(weekly.mean())


def test_label_overlap_counts_later_decisions_before_each_label(off_hours: ModuleType):
    times = pd.date_range("2024-01-07 22:00", periods=5, freq="7D", tz="UTC")
    assert off_hours.label_overlap(times, times + pd.Timedelta(hours=1)) == 0
    assert off_hours.label_overlap(times, times + pd.Timedelta(days=8)) == 1


def test_the_block_is_politis_white_with_a_floor_of_four_times_the_overlap(off_hours: ModuleType):
    rng = np.random.default_rng(0)
    persistent = np.convolve(rng.standard_normal(60), np.ones(12), mode="same")
    arm, reference = rng.random(80), rng.random(80) + 0.1
    test = off_hours._comparisons._test(arm, reference, 3, rng, _Settings())
    assert BLOCK_OVERLAP_FLOOR == 4
    assert test["block"] == max(politis_white_block(arm - reference), BLOCK_OVERLAP_FLOOR * 3)
    assert test["dm_lags"] >= 3
    assert test["status"] == "scored"
    noisy = rng.standard_normal(60)
    test = off_hours._comparisons._test(noisy + persistent + 5, persistent + 5, 0, rng, _Settings())
    assert test["block"] == max(politis_white_block(noisy), 1)


class _Settings:
    # the three options `_test` reads
    bootstrap_resamples = 200
    interval_level = 0.95


def test_the_one_sided_test_holds_its_size_on_weekly_points(off_hours: ModuleType, settings: Any):
    # size check: two arms with the same expected CRPS on 48 weekly points (about one year),
    # autocorrelated; the one-sided test at alpha rejects at most about alpha of the time
    rng = np.random.default_rng(11)
    p_values: list[float] = []
    for _ in range(500):
        shock = rng.standard_normal(49)
        common = np.abs(rng.standard_normal(48)) + 1
        difference = 0.2 * (shock[1:] + 0.4 * shock[:-1])
        p_values.append(diebold_mariano(common + difference, common, overlap=0)["p_less"])
    assert 0.02 <= np.mean(np.array(p_values) < settings.alpha) <= 0.08


def test_invalid_arms_and_unknown_labels_are_excluded_and_counted(
    off_hours: ModuleType, settings: Any
):
    forecasts = _forecast_table(off_hours, np.random.default_rng(1))
    forex = forecasts["market"] == "forex"
    weeks = forecasts["reopen_time"].sort_values().unique()
    forecasts.loc[
        forex & (forecasts["reopen_time"] == weeks[0]) & (forecasts["arm"] == "correction"), "q50"
    ] = np.nan
    forecasts.loc[
        forex & (forecasts["reopen_time"] == weeks[1]) & (forecasts["arm"] == "event"), "q10"
    ] = 9.0
    forecasts.loc[forex & (forecasts["reopen_time"] == weeks[2]), "label"] = np.nan
    cells = off_hours.reopening_tables(forecasts, settings)["reopening_cells"].set_index("market")
    assert cells.loc["forex", "excluded"] == 3
    assert cells.loc["forex", "weekends"] == 37
    assert cells.loc["index", "excluded"] == 0


def test_a_cell_with_too_few_weekends_is_not_tested_and_not_counted_by_holm(
    off_hours: ModuleType, settings: Any
):
    forecasts = _forecast_table(off_hours, np.random.default_rng(2))
    keep = (forecasts["market"] == "forex") | (
        forecasts["reopen_time"] <= forecasts["reopen_time"].sort_values().unique()[18]
    )
    tables = off_hours.reopening_tables(forecasts[keep], settings)
    cells = tables["reopening_cells"].set_index("market")
    assert cells.loc["index", "status"] == "not tested: 19 weekends, fewer than 20"
    assert not cells.loc["index", "covariate_is_best"]
    tests = tables["reopening_tests"]
    assert set(tests["market"]) == {"forex"}
    np.testing.assert_allclose(tests["adjusted_p"], holm(tests["p_value"].to_numpy()))


def test_only_weekends_of_the_scored_folds_are_scored(off_hours: ModuleType, settings: Any):
    forecasts = _forecast_table(off_hours, np.random.default_rng(3))
    forecasts.loc[forecasts["reopen_time"] < "2025-01-01", "fold"] = ""
    cells = off_hours.reopening_tables(forecasts, settings)["reopening_cells"]
    expected = (forecasts["reopen_time"] >= "2025-01-01").sum() // (2 * len(off_hours.ARMS))
    assert (cells["weekends"] == expected).all()


def test_no_weekend_in_the_scored_folds_raises(off_hours: ModuleType, settings: Any):
    forecasts = _forecast_table(off_hours, np.random.default_rng(5)).assign(fold="")
    with pytest.raises(ValueError, match="no forecast of a weekend in the scored folds"):
        off_hours.reopening_tables(forecasts, settings)


def test_covariate_use_recovers_the_share_of_the_crypto_move_a_model_passes_on(
    off_hours: ModuleType, settings: Any
):
    forecasts = _forecast_table(off_hours, np.random.default_rng(4))
    covariate = forecasts["arm"] == "covariate"
    calendar = forecasts.loc[forecasts["arm"] == "calendar", list(QUANTILE_COLUMNS)].to_numpy()
    crypto = forecasts.loc[covariate, "weekend_crypto"].to_numpy()
    # the covariate arm is the calendar arm moved by 0.3 times the crypto move (log units)
    forecasts.loc[covariate, list(QUANTILE_COLUMNS)] = calendar + 0.3 * crypto[:, None] / 0.01
    gaps = pd.DataFrame(
        {"market": ["forex", "forex", "index"], "asset": ["FOREX", "ELSE", "INDEX"],
         "status": "tested", "beta": [0.6, 5.0, 0.0]}
    )  # fmt: skip
    use = off_hours.covariate_use(forecasts, gaps, settings).set_index("market")
    np.testing.assert_allclose(use["use_slope"], 0.3)
    # the realised move is the crypto move itself here: slope 1, so the share is 0.3
    np.testing.assert_allclose(use["move_slope"], 1.0)
    np.testing.assert_allclose(use["use_share"], 0.3)
    assert use.loc["forex", "gap_beta"] == pytest.approx(0.6)
    assert use.loc["index", "gap_beta"] == pytest.approx(0.0)
    assert (use["weekends"] == 40).all()


def test_the_holm_family_spans_every_model_and_market(off_hours: ModuleType, settings: Any):
    first = _forecast_table(off_hours, np.random.default_rng(8))
    second = _forecast_table(off_hours, np.random.default_rng(9)).assign(model="other")
    tables = off_hours.reopening_tables(pd.concat([first, second], ignore_index=True), settings)
    tests = tables["reopening_tests"]
    assert len(tests) == 2 * 2 * len(off_hours.COMPARISONS)
    np.testing.assert_allclose(tests["adjusted_p"], holm(tests["p_value"].to_numpy()))
    alone = off_hours.reopening_tables(first, settings)["reopening_tests"]
    assert (
        tests.loc[tests["model"] == "m", "adjusted_p"].to_numpy() >= alone["adjusted_p"].to_numpy()
    ).all()


def test_covariate_use_reads_only_the_scored_weekends(off_hours: ModuleType, settings: Any):
    # one weekend without a crypto move: it has no correction or model-free forecast, so Q1
    # does not score it and the covariate use leaves it out instead of turning NaN
    forecasts = _forecast_table(off_hours, np.random.default_rng(4))
    first = forecasts["reopen_time"] == forecasts["reopen_time"].min()
    forecasts.loc[first, "weekend_crypto"] = np.nan
    crypto_arms = forecasts["arm"].isin(["correction", "model-free"])
    forecasts.loc[first & crypto_arms, "q50"] = np.nan
    use = off_hours.covariate_use(
        forecasts, pd.DataFrame(columns=["market", "asset", "status", "beta"]), settings
    )
    assert (use["weekends"] == 39).all()
    assert use[["use_slope", "move_slope", "use_share"]].notna().all().all()


def test_every_cell_reports_the_naive_reference_untested(off_hours: ModuleType, settings: Any):
    forecasts = _forecast_table(off_hours, np.random.default_rng(0))
    tables = off_hours.reopening_tables(forecasts, settings)
    cells = tables["reopening_cells"].set_index("market")
    labels = forecasts[(forecasts["market"] == "forex") & (forecasts["arm"] == "event")]["label"]
    normal = np.tile(Z, (len(labels), 1))
    error = labels.to_numpy()[:, None] - normal
    crps = 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean(axis=1)
    assert cells.loc["forex", "crps_ewma-normal"] == pytest.approx(crps.mean())
    inside = np.abs(labels.to_numpy()) <= Z[-1]
    assert cells.loc["forex", "coverage_ewma-normal"] == pytest.approx(inside.mean())
    assert "ewma-normal" not in set(tables["reopening_tests"]["reference"])


def test_a_rerun_reproduces_the_tables(off_hours: ModuleType, settings: Any):
    forecasts = _forecast_table(off_hours, np.random.default_rng(3))
    first = off_hours.reopening_tables(forecasts, settings)
    second = off_hours.reopening_tables(forecasts.copy(), settings)
    for name in ("reopening_cells", "reopening_tests"):
        pd.testing.assert_frame_equal(first[name], second[name])


def test_identical_arms_have_no_test_and_leave_the_holm_family(
    off_hours: ModuleType, settings: Any
):
    # a model that ignores its covariate gives the calendar arm's forecast: no variance of the
    # loss difference, so no p-value; Holm counts the other tests only, and the cell's
    # covariate arm cannot be the best
    forecasts = _forecast_table(off_hours, np.random.default_rng(1))
    covariate = forecasts["arm"] == "covariate"
    calendar = forecasts.loc[forecasts["arm"] == "calendar", list(QUANTILE_COLUMNS)].to_numpy()
    forecasts.loc[covariate, list(QUANTILE_COLUMNS)] = calendar
    tables = off_hours.reopening_tables(forecasts, settings)
    tests = tables["reopening_tests"]
    same = (tests["arm"] == "covariate") & (tests["reference"] == "calendar")
    assert tests.loc[same, "p_value"].isna().all()
    assert tests.loc[same, "adjusted_p"].isna().all()
    np.testing.assert_allclose(
        tests.loc[~same, "adjusted_p"], holm(tests.loc[~same, "p_value"].to_numpy())
    )
    assert not tables["reopening_cells"]["covariate_is_best"].any()


def test_a_comparison_with_fewer_than_five_blocks_is_not_tested(off_hours: ModuleType):
    # 50 weekly points with an overlap of 3 need blocks of 12: only 4 of them fit
    rng = np.random.default_rng(1)
    arm, reference = rng.random(50), rng.random(50) + 0.1
    test = off_hours._comparisons._test(arm, reference, 3, rng, _Settings())
    assert MIN_BOOTSTRAP_BLOCKS == 5
    assert test["block"] == 12
    assert test["status"] == "not tested: 50 weekly points, fewer than 5 blocks of 12"
    for name in ("p_value", "dm_statistic", "dm_lags", "skill_low", "skill_high"):
        assert np.isnan(test[name])
    assert test["skill"] == pytest.approx(100 * (1 - arm.mean() / reference.mean()))


def test_untested_comparisons_leave_the_holm_family_and_the_verdict(
    off_hours: ModuleType, settings: Any
):
    # on "index" every label is known only 15 days later: an overlap of 2 needs blocks of 8,
    # so its 22 weekly points hold fewer than 5 blocks and no comparison is tested
    forecasts = _forecast_table(off_hours, np.random.default_rng(1), weeks=22)
    index = forecasts["market"] == "index"
    forecasts.loc[index, "label_time"] = forecasts.loc[index, "reopen_time"] + pd.Timedelta(days=15)
    tables = off_hours.reopening_tables(forecasts, settings)
    tests = tables["reopening_tests"]
    untested = tests[tests["market"] == "index"]
    assert untested["status"].str.startswith("not tested: 22 weekly points").all()
    assert untested["adjusted_p"].isna().all()
    assert not untested["is_better"].any()
    tested = tests[tests["market"] == "forex"]
    assert (tested["status"] == "scored").all()
    np.testing.assert_allclose(tested["adjusted_p"], holm(tested["p_value"].to_numpy()))
    cells = tables["reopening_cells"].set_index("market")
    assert cells.loc["index", "overlap"] == 2
    assert not cells.loc["index", "covariate_is_best"]
