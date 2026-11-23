import dataclasses
import math
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import src.benchmark._scores as scores_module
from src.benchmark import SCORED, Bootstrap, ForecastCache, StudySettings, read_cache
from src.evaluate import DEVELOPMENT_FOLDS, LEAKAGE_FOLDS, TEST_FOLDS, Fold
from src.stats import bootstrap_difference, politis_white_block

MODELS = ["random-walk", "climatology", "skilled", "sampled", "leaky"]
LEAKAGE = [fold.name for fold in LEAKAGE_FOLDS]
BOOTSTRAP = Bootstrap(level=0.95, n_resamples=500, seed=0, block_hours=24, min_block=4)
RELEASE = pd.Timestamp("2025-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2024-06-30", tz="UTC")
NO_CUTOFF = pd.NaT
GRID_HOURS = 4


def _options(
    leakage: ModuleType, placebo: Sequence[str] = ("skilled",), not_built: Sequence[str] = ()
) -> object:
    return leakage.LeakageOptions(
        grid_hours=GRID_HOURS,
        placebo_models=tuple(placebo),
        placebo_split=pd.Timestamp("2024-01-01", tz="UTC"),
        not_built=tuple(not_built),
    )


def _tables(
    leakage: ModuleType,
    cache: ForecastCache,
    models: Sequence[str] = MODELS,
    *,
    before: Sequence[Fold] = LEAKAGE_FOLDS,
    after: Sequence[Fold] = TEST_FOLDS,
    options: object = None,
) -> dict[str, pd.DataFrame]:
    study = StudySettings(DEVELOPMENT_FOLDS, tuple(after), 0.05, BOOTSTRAP)
    return leakage.leakage_tables(
        cache,
        list(models),
        study,
        before_folds=tuple(before),
        options=_options(leakage) if options is None else options,
    )


@pytest.fixture(name="tables", scope="module")
def _tables_fixture(leakage: ModuleType, leakage_cache_folder: Path) -> dict[str, pd.DataFrame]:
    return _tables(leakage, read_cache(leakage_cache_folder))


def _cells(tables: dict[str, pd.DataFrame], model: str, test: str) -> pd.DataFrame:
    cells = tables["cells"]
    return cells[(cells["model"] == model) & (cells["test"] == test)]


# --- compare_periods ------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    noise = rng.standard_normal(n)
    values = np.empty(n)
    values[0] = noise[0]
    for index in range(1, n):
        values[index] = phi * values[index - 1] + noise[index]
    return values


def _moving_sum(rng: np.random.Generator, n: int, overlap: int) -> np.ndarray:
    # the skill of overlapping labels: each value shares noise with the next `overlap` ones
    noise = rng.standard_normal(n + overlap)
    return np.convolve(noise, np.ones(overlap + 1), mode="valid")


def _ar1_half(rng: np.random.Generator, n: int) -> np.ndarray:
    return _ar1(rng, n, 0.5)


def _overlap_five(rng: np.random.Generator, n: int) -> np.ndarray:
    return _moving_sum(rng, n, 5)


def _period(values: np.ndarray, overlap: int, start_time: str = "2024-01-01") -> pd.DataFrame:
    times = pd.date_range(start_time, periods=len(values), freq="4h", tz="UTC")
    return pd.DataFrame({"time": times, "skill": values, "overlap": overlap})


def _compare(
    leakage: ModuleType,
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    test: str = "leakage",
    bootstrap: Bootstrap = BOOTSTRAP,
) -> dict[str, float | int | str]:
    return leakage.compare_periods(first, second, test=test, bootstrap=bootstrap)


@pytest.mark.parametrize(
    ("make", "first_length", "second_length", "overlap", "largest_size"),
    [
        (_ar1_half, 600, 400, 0, 0.08),
        (_overlap_five, 600, 400, 5, 0.09),
    ],
    ids=["ar1", "overlapping-labels"],
)
def test_the_one_sided_test_keeps_its_size(
    leakage: ModuleType,
    make: Callable[[np.random.Generator, int], np.ndarray],
    first_length: int,
    second_length: int,
    overlap: int,
    largest_size: float,
) -> None:
    # no difference between the periods: the share of the 300 series (binomial standard
    # deviation 1.3 %) the one-sided test at 5 % rejects, with the circular-block bootstrap
    # standard error and t reference of src.stats (400 draws) and the Politis-White blocks
    # of rq4 (at least 4 overlaps). AR(1) skills: 5.3 % measured, bound 8 %. Skills of labels
    # that overlap 5 later decisions (24 bars on the 4-hour grid): 8.3 % measured with about
    # 24 blocks of 20; the bound 9 % records that known excess (the block variance of
    # overlapping labels stays biased down), not a target. A series whose block leaves fewer
    # than 5 blocks is not scored; none does here.
    rng = np.random.default_rng(2)
    p_values: list[float] = []
    for seed in range(300):
        first = _period(make(rng, first_length), overlap)
        second = _period(make(rng, second_length), overlap, "2026-01-01")
        bootstrap = dataclasses.replace(BOOTSTRAP, n_resamples=400, seed=seed)
        tested = _compare(leakage, first, second, bootstrap=bootstrap)
        if tested["status"] == SCORED:
            p_values.append(float(tested["p_value"]))
    assert len(p_values) >= 270
    # a regression pin both ways: an inflated standard error would push the size below 2 %
    assert 0.02 <= np.mean(np.array(p_values) < 0.05) <= largest_size


def test_the_one_sided_test_finds_a_real_gain(leakage: ModuleType) -> None:
    rng = np.random.default_rng(3)
    first = _period(_ar1(rng, 300, 0.5) + 1.0, 0)
    second = _period(_ar1(rng, 200, 0.5), 0, "2026-01-01")
    tested = _compare(leakage, first, second)
    assert tested["status"] == SCORED
    assert float(tested["p_value"]) < 0.01
    assert float(tested["low"]) > 0
    assert float(_compare(leakage, second, first)["p_value"]) > 0.99


def test_the_test_is_the_t_referenced_bootstrap_of_src_stats(leakage: ModuleType) -> None:
    # one-sided p_greater for leakage, the two-sided p for the placebo, and the t interval
    rng = np.random.default_rng(7)
    first = _period(_ar1(rng, 300, 0.5) + 0.1, 0)
    second = _period(_ar1(rng, 200, 0.5), 0, "2026-01-01")
    tested = _compare(leakage, first, second)
    expected = bootstrap_difference(
        first["skill"].to_numpy(),
        second["skill"].to_numpy(),
        level=BOOTSTRAP.level,
        n_resamples=BOOTSTRAP.n_resamples,
        seed=BOOTSTRAP.seed,
        blocks=(int(tested["first_block"]), int(tested["second_block"])),
    )
    assert tested["p_value"] == pytest.approx(expected["p_greater"])
    assert (tested["low"], tested["high"]) == pytest.approx((expected["low"], expected["high"]))
    assert tested["did"] == pytest.approx(expected["difference"])
    placebo = _compare(leakage, first, second, test="placebo")
    assert placebo["p_value"] == pytest.approx(expected["p_value"])


def test_the_placebo_is_two_sided(leakage: ModuleType) -> None:
    rng = np.random.default_rng(6)
    early = _period(_ar1(rng, 300, 0.5), 0)
    late = _period(_ar1(rng, 200, 0.5) + 1.0, 0, "2026-01-01")
    placebo = _compare(leakage, early, late, test="placebo")
    one_sided = _compare(leakage, early, late, test="leakage")
    assert float(placebo["did"]) < 0
    assert float(placebo["p_value"]) < 0.01
    assert float(one_sided["p_value"]) > 0.99


def test_the_block_is_politis_white_at_least_four_overlaps(leakage: ModuleType) -> None:
    rng = np.random.default_rng(1)
    noise = _period(rng.standard_normal(500), 5)
    persistent = _period(_ar1(rng, 2000, 0.9), 0, "2026-01-01")
    tested = _compare(leakage, noise, persistent)
    assert tested["first_block"] == 20
    expected = politis_white_block(persistent["skill"].to_numpy())
    assert tested["second_block"] == max(expected, 2)
    assert expected > 10
    constant = _compare(leakage, _period(np.zeros(100), 0), _period(np.zeros(100), 3, "2026"))
    assert (constant["first_block"], constant["second_block"]) == (2, 12)


def test_a_period_with_too_few_blocks_or_times_is_not_scored(leakage: ModuleType) -> None:
    rng = np.random.default_rng(4)
    long = _period(rng.standard_normal(300), 0)
    # overlap 5: the block is 4 x 5 = 20, so 5 blocks need 100 times
    short = _period(rng.standard_normal(90), 5, "2026-01-01")
    tested = _compare(leakage, long, short)
    assert tested["status"] == (
        "not scored: 90 decision times after the release on the model's grid, fewer than 5 "
        "bootstrap blocks of 20"
    )
    assert math.isnan(float(tested["p_value"]))
    enough = _compare(leakage, long, _period(rng.standard_normal(100), 5, "2026-01-01"))
    assert enough["second_block"] == 20
    assert enough["status"] == SCORED
    tiny = _period(rng.standard_normal(10), 0)
    tested = _compare(leakage, tiny, long, test="cutoff")
    assert tested["status"] == "not scored: 10 decision times before the release, fewer than 11"
    tested = _compare(leakage, long, long.iloc[:0], test="cutoff")
    assert tested["status"] == "not scored: no decision between the cutoff and the release"
    assert tested["second_times"] == 0


# --- leakage_tables ------------------------------------------------------------------------


def test_every_model_has_its_tests_with_one_cell_per_market_and_horizon(
    tables: dict[str, pd.DataFrame],
) -> None:
    cells = tables["cells"]
    tests = cells.groupby("model")["test"].agg(lambda values: sorted(set(values)))
    assert tests.to_dict() == {
        "leaky": ["cutoff", "hourly", "leakage"],
        "sampled": ["cutoff", "leakage"],
        "skilled": ["hourly", "leakage", "placebo"],
    }
    assert (cells.groupby(["model", "test"]).size() == 3 * 3).all()
    assert (cells["status"] == SCORED).all()
    assert (cells["invalid"] == 0).all()
    assert (cells["baseline"] == "climatology").all()
    np.testing.assert_allclose(cells["did"], cells["skill_first"] - cells["skill_second"])
    assert (cells["low"] <= cells["high"]).all()
    assert "times" not in cells.columns
    assert cells["first_times"].dtype == int
    assert cells["second_times"].dtype == int


def test_the_after_period_of_the_leakage_test_is_on_the_4_hour_grid(
    tables: dict[str, pd.DataFrame],
) -> None:
    # before: the leakage quarters every 4 hours; after: t1-t4 hourly, thinned to 4 hours for
    # the test leakage and kept hourly for the check hourly (a sampled model is on its grid).
    # 200 hourly decisions per quarter give 51 points: 00:00, then (0, 4] -> 4, ..., and the
    # last three hours (197-199) -> point 200, still inside the quarter
    for model in ("skilled", "leaky"):
        assert (_cells(tables, model, "leakage")["second_times"] == 4 * 51).all()
        assert (_cells(tables, model, "hourly")["second_times"] == 4 * 200).all()
    assert (_cells(tables, "sampled", "leakage")["second_times"] == 4 * 50).all()
    # the grid model's between: l11-l12 and d1, all on its own 4-hour grid (50 each)
    assert (_cells(tables, "sampled", "cutoff")["second_times"] == 3 * 50).all()
    assert _cells(tables, "sampled", "hourly").empty
    skills = tables["skills"]
    for period, spacing in (("before", 4), ("after", 4), ("after-hourly", 1)):
        rows = skills[(skills["model"] == "skilled") & (skills["period"] == period)]
        for (_, _, window), group in rows.assign(window=rows["time"].dt.year).groupby(
            ["market", "horizon", "window"]
        ):
            gaps = group["time"].sort_values().diff().dropna()
            assert gaps.min() == pd.Timedelta(hours=spacing), (period, window)


def test_each_block_is_at_least_four_times_its_periods_overlap(
    tables: dict[str, pd.DataFrame],
) -> None:
    cells = tables["cells"]
    leakage_cells = cells[cells["test"] == "leakage"]
    hourly = cells[cells["test"] == "hourly"]
    # 24-bar labels overlap 5 later decisions on the 4-hour grid and 23 hourly ones
    long = leakage_cells["horizon"] == 24
    assert (
        leakage_cells.loc[long & (leakage_cells["model"] != "sampled"), "first_block"] >= 20
    ).all()
    assert (
        leakage_cells.loc[long & (leakage_cells["model"] != "sampled"), "second_block"] >= 20
    ).all()
    hourly_long = hourly["horizon"] == 24
    assert (hourly.loc[hourly_long & (hourly["model"] != "sampled"), "second_block"] >= 92).all()
    assert (cells["first_block"] >= 1).all()


def test_a_leaky_model_is_detected_and_honest_models_are_not(
    leakage: ModuleType, tables: dict[str, pd.DataFrame]
) -> None:
    verdict = leakage.leakage_verdict(tables["cells"], alpha=0.05)
    primary = verdict[verdict["test"] == "leakage"]
    leaky = primary[primary["model"] == "leaky"]
    assert leaky["shows_leakage"].all()
    assert (leaky["low"] > 0).all()
    assert (leaky["skill_second"].abs() < 0.03).all()
    honest = primary[primary["model"] != "leaky"]
    assert not honest["shows_leakage"].any()
    # 50 decision times per leakage quarter: the noise of an honest difference is about 0.02
    assert (honest["did"].abs() < 0.06).all()
    assert not verdict.loc[verdict["test"] != "leakage", "shows_leakage"].any()


def test_the_cutoff_check_scores_the_skill_between_cutoff_and_release_per_time(
    leakage: ModuleType, tables: dict[str, pd.DataFrame]
) -> None:
    cutoff = _cells(tables, "leaky", "cutoff")
    assert (cutoff["first"] == "before").all()
    assert (cutoff["second"] == "between").all()
    assert (cutoff["first_times"] == 10 * 50).all()
    # between: l11-l12 every 4 hours (2 x 50) and the development quarter d1 before the
    # release, thinned to the same 4-hour grid (51 points of its 200 hours)
    assert (cutoff["second_times"] == 2 * 50 + 51).all()
    np.testing.assert_allclose(cutoff["skill_second"], 0.0, atol=1e-12)
    skills = tables["skills"]
    between = skills[(skills["model"] == "leaky") & (skills["period"] == "between")]
    means = between.groupby(["market", "horizon"])["skill"].mean().to_numpy()
    np.testing.assert_allclose(cutoff.sort_values(["market", "horizon"])["skill_second"], means)
    verdict = leakage.leakage_verdict(tables["cells"], alpha=0.05)
    assert verdict.loc[
        (verdict["model"] == "leaky") & (verdict["test"] == "cutoff"), "rejects"
    ].all()


def test_the_placebo_splits_the_before_period_at_the_split(
    leakage: ModuleType, tables: dict[str, pd.DataFrame]
) -> None:
    placebo = _cells(tables, "skilled", "placebo")
    assert (placebo["first_times"] == 8 * 50).all()
    assert (placebo["second_times"] == 4 * 50).all()
    skills = tables["skills"]
    split = pd.Timestamp("2024-01-01", tz="UTC")
    early = skills[skills["period"] == "early"]
    late = skills[skills["period"] == "late"]
    assert (early["time"] < split).all()
    assert (late["time"] >= split).all()
    assert set(early["model"]) | set(late["model"]) == {"skilled"}
    verdict = leakage.leakage_verdict(tables["cells"], alpha=0.05)
    assert not verdict.loc[verdict["test"] == "placebo", "rejects"].any()


def test_the_periods_table_reports_every_period(tables: dict[str, pd.DataFrame]) -> None:
    periods = tables["periods"]
    leaky = periods[periods["model"] == "leaky"]
    assert set(leaky["period"]) == {"before", "between", "after", "after-hourly"}
    by_period = leaky.groupby("period")[["model_crps", "baseline_crps"]].mean()
    before = by_period.loc["before"].to_numpy(dtype=float)
    between = by_period.loc["between"].to_numpy(dtype=float)
    assert before[0] < before[1]
    np.testing.assert_allclose(between[0], between[1])
    assert (leaky.loc[leaky["period"] == "between", "decisions"] == 2 * 50 + 51).all()
    assert "between" not in set(periods.loc[periods["model"] == "skilled", "period"])
    assert {"early", "late"} <= set(periods.loc[periods["model"] == "skilled", "period"])


def _assert_exclusive_periods(skills: pd.DataFrame) -> None:
    # before, between and after never share a time of one model and cell
    exclusive = skills[skills["period"].isin(["before", "between", "after"])]
    counts = exclusive.groupby(["model", "market", "horizon", "time"])["period"].nunique()
    assert (counts == 1).all()


def test_no_time_is_in_two_exclusive_periods(tables: dict[str, pd.DataFrame]) -> None:
    _assert_exclusive_periods(tables["skills"])


def test_a_development_run_scores_the_code_on_d1_to_d4(
    leakage: ModuleType, leakage_cache_folder: Path
) -> None:
    # the open plan: before, between and after all come from d1-d4 (synthetic d1 is before
    # the release of 2025-01-01, d2-d4 after it); each fold's baseline is chosen on the others
    tables = _tables(
        leakage, read_cache(leakage_cache_folder), before=DEVELOPMENT_FOLDS, after=DEVELOPMENT_FOLDS
    )
    _assert_exclusive_periods(tables["skills"])
    skills = tables["skills"]
    assert set(skills["time"].dt.year) <= {2024, 2025}
    primary = tables["cells"][tables["cells"]["test"] == "leakage"]
    assert (primary["status"] == SCORED).any()
    before = skills[skills["period"] == "before"]
    assert (before["time"] < RELEASE).all()


def test_the_before_period_of_a_model_without_cutoff_has_every_leakage_decision(
    tables: dict[str, pd.DataFrame],
) -> None:
    assert (_cells(tables, "skilled", "leakage")["first_times"] == 12 * 50).all()


def test_a_shift_of_every_loss_before_the_release_cancels_out(
    leakage: ModuleType, leakage_cache_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = read_cache(leakage_cache_folder)
    plain = _tables(leakage, cache)["cells"]
    original = scores_module.decision_losses

    def shifted_losses(decisions: pd.DataFrame) -> pd.DataFrame:
        losses = original(decisions)
        is_leakage = losses["window"].isin(LEAKAGE)
        return losses.assign(crps=losses["crps"] + np.where(is_leakage, 0.7, 0.0))

    monkeypatch.setattr(scores_module, "decision_losses", shifted_losses)
    monkeypatch.setattr(leakage._leakage, "decision_losses", shifted_losses)
    shifted = _tables(leakage, cache)["cells"]
    columns = ["did", "low", "high", "p_value", "skill_first", "skill_second"]
    np.testing.assert_allclose(shifted[columns], plain[columns], atol=1e-9)


def test_the_cells_are_reproducible(
    leakage: ModuleType, leakage_cache_folder: Path, tables: dict[str, pd.DataFrame]
) -> None:
    again = _tables(leakage, read_cache(leakage_cache_folder))
    pd.testing.assert_frame_equal(again["cells"], tables["cells"])


def test_leakage_tables_refuse_unknown_models_windows_and_placebo_models(
    leakage: ModuleType, leakage_cache_folder: Path
) -> None:
    cache = read_cache(leakage_cache_folder)
    with pytest.raises(ValueError, match=r"models \['absent'\], windows \[\]"):
        _tables(leakage, cache, [*MODELS, "absent"])
    later = dataclasses.replace(LEAKAGE_FOLDS[-1], name="l13")
    with pytest.raises(ValueError, match=r"models \[\], windows \['l13'\]"):
        _tables(leakage, cache, before=[*LEAKAGE_FOLDS, later])
    with pytest.raises(ValueError, match=r"placebo models \['climatology'\] are not zero-shot"):
        _tables(leakage, cache, options=_options(leakage, placebo=["climatology"]))


def test_a_model_without_decisions_after_its_release_is_not_scored(
    leakage: ModuleType, leakage_cache_folder: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "cache"
    shutil.copytree(leakage_cache_folder, folder)
    models = pd.read_csv(folder / "models.csv")
    models.loc[models["model"] == "skilled", "release_date"] = "2027-01-01"
    models.to_csv(folder / "models.csv", index=False)
    cells = _tables(leakage, read_cache(folder), ["climatology", "skilled", "leaky"])["cells"]
    skilled = cells[(cells["model"] == "skilled") & (cells["test"] == "leakage")]
    assert len(skilled) == 9
    assert (skilled["first"] == "before").all()
    assert (skilled["second"] == "after").all()
    assert (
        skilled["status"] == "not scored: no decision after the release on the model's grid"
    ).all()
    assert (skilled["second_times"] == 0).all()
    assert skilled["p_value"].isna().all()
    primary = cells[cells["test"] == "leakage"]
    assert (primary.loc[primary["model"] == "leaky", "status"] == SCORED).all()


def test_no_scored_cell_at_all_raises_with_the_reasons(
    leakage: ModuleType, leakage_cache_folder: Path
) -> None:
    cache = read_cache(leakage_cache_folder)
    with pytest.raises(
        ValueError,
        match=(
            r"no zero-shot model of rq4 could be scored on .*: 9 cells not scored: no decision "
            r"after the release on the model's grid"
        ),
    ):
        _tables(leakage, cache, ["climatology", "skilled"], after=DEVELOPMENT_FOLDS[:1])


# --- leakage_verdict -----------------------------------------------------------------------


def _verdict_cells(tests: Sequence[str], p_value: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame({"model": "m", "test": list(tests), "p_value": list(p_value)})


def test_holm_adjusts_over_the_scored_cells_of_each_test(leakage: ModuleType) -> None:
    cells = _verdict_cells(
        ["leakage", "leakage", "leakage", "placebo", "placebo"], [0.01, 0.04, 0.03, 0.01, 0.2]
    )
    verdict = leakage.leakage_verdict(cells, alpha=0.05)
    np.testing.assert_allclose(verdict["adjusted_p"], [0.03, 0.06, 0.06, 0.02, 0.2])
    assert verdict["rejects"].tolist() == [True, False, False, True, False]
    assert verdict["shows_leakage"].tolist() == [True, False, False, False, False]
    with_unscored = leakage.leakage_verdict(
        _verdict_cells(["leakage", "leakage"], [0.03, np.nan]), alpha=0.05
    )
    assert with_unscored["adjusted_p"].iloc[0] == pytest.approx(0.03)
    assert with_unscored["shows_leakage"].tolist() == [True, False]


def test_an_adjusted_p_equal_to_alpha_does_not_show_leakage(leakage: ModuleType) -> None:
    verdict = leakage.leakage_verdict(_verdict_cells(["leakage"], [0.05]), alpha=0.05)
    assert not verdict["shows_leakage"].iloc[0]


def test_a_model_not_built_on_the_leakage_quarters_is_reported_as_such(
    leakage: ModuleType, leakage_cache_folder: Path, tmp_path: Path
) -> None:
    # leaky without its l1-l12 forecasts, listed as not built: every cell says so, the
    # other models are scored as before
    folder = tmp_path / "cache"
    shutil.copytree(leakage_cache_folder, folder)
    labels = read_cache(folder).labels[["time", "asset", "horizon", "window"]]
    path = folder / "forecasts" / "leaky.parquet"
    frame = pd.read_parquet(path).merge(labels, on=["time", "asset", "horizon"])
    frame[~frame["window"].isin(LEAKAGE)].drop(columns="window").to_parquet(path)
    options = _options(leakage, not_built=["leaky"])
    cells = _tables(leakage, read_cache(folder), options=options)["cells"]
    leaky = cells[cells["model"] == "leaky"]
    assert (leaky["status"] == leakage._leakage.NOT_BUILT).all()
    assert leaky[["did", "low", "high", "p_value"]].isna().all().all()
    primary = cells[(cells["test"] == "leakage") & (cells["model"] != "leaky")]
    assert (primary["status"] == SCORED).all()


def test_not_built_models_must_be_zero_shot_and_without_leakage_forecasts(
    leakage: ModuleType, leakage_cache_folder: Path, tmp_path: Path
) -> None:
    cache = read_cache(leakage_cache_folder)
    with pytest.raises(ValueError, match=r"not built models \['absent'\] are not zero-shot"):
        _tables(leakage, cache, options=_options(leakage, not_built=["absent"]))
    with pytest.raises(ValueError, match="leaky is listed as not built but has forecasts"):
        _tables(leakage, cache, options=_options(leakage, not_built=["leaky"]))
    # also when every l1-l12 forecast of it is invalid
    folder = tmp_path / "cache"
    shutil.copytree(leakage_cache_folder, folder)
    labels = read_cache(folder).labels[["time", "asset", "horizon", "window"]]
    path = folder / "forecasts" / "leaky.parquet"
    frame = pd.read_parquet(path).merge(labels, on=["time", "asset", "horizon"])
    frame.loc[frame["window"].isin(LEAKAGE), "q50"] = np.nan
    frame.drop(columns="window").to_parquet(path)
    with pytest.raises(ValueError, match="leaky is listed as not built but has forecasts"):
        _tables(leakage, read_cache(folder), options=_options(leakage, not_built=["leaky"]))
    # with the plan open (before = d1-d4) the list does not apply: leaky is scored
    development = _tables(
        leakage,
        cache,
        before=DEVELOPMENT_FOLDS,
        after=DEVELOPMENT_FOLDS,
        options=_options(leakage, not_built=["leaky"]),
    )["cells"]
    assert not (development["status"] == leakage._leakage.NOT_BUILT).any()


def test_decisions_without_a_control_forecast_are_counted_as_unpaired(
    leakage: ModuleType, leakage_cache_folder: Path, tmp_path: Path
) -> None:
    # the control (climatology) lost one asset's l3 forecasts, as a GARCH fit that did not
    # converge would: those decisions leave the cell and are counted in unpaired
    folder = tmp_path / "cache"
    shutil.copytree(leakage_cache_folder, folder)
    labels = read_cache(folder).labels[["time", "asset", "horizon", "window"]]
    path = folder / "forecasts" / "climatology.parquet"
    frame = pd.read_parquet(path).merge(labels, on=["time", "asset", "horizon"])
    in_l3 = frame["window"] == "l3"
    middle = pd.Timestamp(frame.loc[in_l3, "time"].sort_values().iloc[int(in_l3.sum()) // 2])
    gone = in_l3 & (frame["asset"] == "BTCUSDT") & (frame["time"] < middle)
    # also a d2 decision of the control: d2 is in no period of skilled (no cutoff), so it
    # leaves no cell and is not counted
    in_d2 = (frame["window"] == "d2") & (frame["asset"] == "BTCUSDT")
    first_d2 = pd.Timestamp(frame.loc[in_d2, "time"].min())
    gone = gone | (in_d2 & (frame["time"] == first_d2))
    frame[~gone].drop(columns="window").to_parquet(path)
    cells = _tables(leakage, read_cache(folder))["cells"]
    skilled = cells[(cells["model"] == "skilled") & (cells["test"] == "leakage")]
    crypto = skilled[skilled["market"] == "crypto"]
    removed = labels[
        (labels["window"] == "l3") & (labels["asset"] == "BTCUSDT") & (labels["time"] < middle)
    ]
    expected = removed.groupby("horizon").size()
    assert crypto.set_index("horizon")["unpaired"].to_dict() == expected.to_dict()
    assert (skilled.loc[skilled["market"] != "crypto", "unpaired"] == 0).all()
    # every test of skilled reads the before period, l3 included: same count in each
    for test in ("hourly", "placebo"):
        other = cells[(cells["model"] == "skilled") & (cells["test"] == test)]
        crypto_other = other[other["market"] == "crypto"].set_index("horizon")["unpaired"]
        assert crypto_other.to_dict() == expected.to_dict()
