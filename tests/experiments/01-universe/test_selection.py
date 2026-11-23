import dataclasses
import tomllib
from statistics import NormalDist
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import LAB_DIR
from src.stats import block_indices

STUDY_TOML = LAB_DIR / "experiments" / "01-universe" / "study.toml"


def _correlation(names: list[str], pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for (a, b), value in pairs.items():
        matrix.loc[a, b] = matrix.loc[b, a] = value
    return matrix


def _pick(
    universe: ModuleType,
    correlation: pd.DataFrame,
    pool: list[str],
    *,
    anchors: tuple[str, ...] = (),
    size: int,
    cap: Any = None,
) -> tuple[list[str], dict[str, str]]:
    cap = cap or universe.SectorCap(pd.Series(dtype=str), 1)
    return universe.greedy_order(
        correlation, pool, anchors=anchors, size=size, duplicate_threshold=0.85, cap=cap
    )


def test_greedy_starts_with_anchors_and_picks_the_least_correlated(universe: ModuleType):
    names = ["BTC", "ETH", "SOL", "XMR"]
    pairs = {
        ("BTC", "ETH"): 0.8, ("BTC", "SOL"): 0.7, ("ETH", "SOL"): 0.75,
        ("BTC", "XMR"): 0.3, ("ETH", "XMR"): 0.35, ("SOL", "XMR"): 0.4,
    }  # fmt: skip
    order, dropped = _pick(universe, _correlation(names, pairs), names, anchors=("BTC",), size=4)
    assert order == ["BTC", "XMR", "SOL", "ETH"]
    assert dropped == {}


def test_greedy_stops_at_its_size(universe: ModuleType):
    names = ["BTC", "ETH", "SOL", "XMR"]
    order, _ = _pick(universe, _correlation(names, {}), names, anchors=("BTC",), size=2)
    assert order == ["BTC", "ETH"]


def test_without_anchors_the_first_most_liquid_candidate_starts(universe: ModuleType):
    names = ["MOST", "NEXT", "LAST"]
    order, _ = _pick(universe, _correlation(names, {}), names, size=1)
    assert order == ["MOST"]


def test_near_duplicates_are_dropped_with_a_reason(universe: ModuleType):
    names = ["BRENT", "WTI", "GOLD"]
    corr = _correlation(
        names, {("BRENT", "WTI"): 0.9, ("BRENT", "GOLD"): 0.1, ("WTI", "GOLD"): 0.1}
    )
    order, dropped = _pick(universe, corr, names, anchors=("BRENT",), size=3)
    assert order == ["BRENT", "GOLD"]
    assert dropped == {"WTI": "near-duplicate of BRENT (0.90)"}


def test_a_near_duplicate_of_an_asset_kept_in_another_pool_is_dropped(universe: ModuleType):
    # USA500 is kept in the index pool; a US stock that moves with it leaves the stock pool
    names = ["USA500", "GOLD", "SPY_LIKE", "XOM", "KO"]
    pairs = {("USA500", "SPY_LIKE"): 0.9, ("USA500", "XOM"): 0.5, ("GOLD", "KO"): -0.86}
    left, dropped = universe.drop_near_duplicates(
        _correlation(names, pairs), ["SPY_LIKE", "XOM", "KO"], kept=["USA500", "GOLD"],
        threshold=0.85,
    )  # fmt: skip
    assert left == ["XOM"]
    assert dropped == {
        "SPY_LIKE": "near-duplicate of USA500 (0.90)",
        "KO": "near-duplicate of GOLD (0.86)",
    }
    assert universe.drop_near_duplicates(
        _correlation(names, pairs), ["XOM", "KO"], kept=[], threshold=0.85
    ) == (["XOM", "KO"], {})
    missing = _correlation(names, pairs)
    missing.loc["XOM", "USA500"] = np.nan
    with pytest.raises(ValueError, match="missing pairs"):
        universe.drop_near_duplicates(missing, ["XOM"], kept=["USA500"], threshold=0.85)


def test_a_near_duplicate_of_a_later_pick_is_dropped_too(universe: ModuleType):
    names = ["A", "B", "B2", "C"]
    pairs = {("A", "B"): 0.1, ("A", "B2"): 0.2, ("B", "B2"): -0.95, ("A", "C"): 0.5}
    order, dropped = _pick(universe, _correlation(names, pairs), names, anchors=("A",), size=4)
    assert order == ["A", "B", "C"]
    assert dropped == {"B2": "near-duplicate of B (0.95)"}


def test_ties_go_to_the_earlier_more_liquid_candidate(universe: ModuleType):
    names = ["A", "B", "C"]
    corr = _correlation(names, {("A", "B"): 0.2, ("A", "C"): 0.2, ("B", "C"): 0.0})
    order, _ = _pick(universe, corr, ["A", "C", "B"], anchors=("A",), size=2)
    assert order == ["A", "C"]


def test_greedy_checks_its_size(universe: ModuleType):
    names = ["A", "B"]
    with pytest.raises(ValueError, match="size"):
        _pick(universe, _correlation(names, {}), names, anchors=("A", "B"), size=1)


def test_an_empty_pool_gives_an_empty_pick(universe: ModuleType):
    assert _pick(universe, _correlation(["A"], {}), [], size=4) == ([], {})


def test_a_missing_pair_stops_the_pick(universe: ModuleType):
    names = ["A", "B", "C"]
    corr = _correlation(names, {("A", "B"): np.nan})
    with pytest.raises(ValueError, match="missing"):
        _pick(universe, corr, names, anchors=("A",), size=3)


# sector cap


def _one_sector_market(universe: ModuleType) -> tuple[list[str], pd.DataFrame, Any]:
    """Six banks, the least correlated assets, and three other stocks; liquidity order."""
    banks = [f"BANK{i}" for i in range(6)]
    others = ["OIL", "SOFT", "CHIP"]
    names = banks + others
    pairs = {(a, b): 0.6 for i, a in enumerate(names) for b in names[i + 1 :]}
    pairs.update({(a, b): 0.0 for i, a in enumerate(banks) for b in banks[i + 1 :]})
    sectors = pd.Series(
        {**dict.fromkeys(banks, "Financials"), "OIL": "Energy",
         "SOFT": "Information Technology", "CHIP": "Information Technology"}
    )  # fmt: skip
    return names, _correlation(names, pairs), universe.SectorCap(sectors, 3)


def test_the_pick_takes_at_most_three_of_one_sector_and_fills_from_the_others(
    universe: ModuleType,
):
    names, corr, cap = _one_sector_market(universe)
    order, dropped = _pick(universe, corr, names, size=6, cap=cap)
    banks = [name for name in order if name.startswith("BANK")]
    assert len(order) == 6
    assert banks == ["BANK0", "BANK1", "BANK2"]
    assert set(order) - set(banks) == {"OIL", "SOFT", "CHIP"}
    assert dropped == {}


def test_a_full_sector_is_skipped_and_the_pick_stops_when_nothing_else_is_left(
    universe: ModuleType,
):
    names, corr, cap = _one_sector_market(universe)
    order, _ = _pick(universe, corr, names[:6], size=6, cap=cap)
    assert order == ["BANK0", "BANK1", "BANK2"]


def test_names_without_a_sector_are_never_capped(universe: ModuleType):
    cap = universe.SectorCap(pd.Series({"S1": "Energy", "S2": "Energy"}), 1)
    take = universe.take_with_sector_cap
    assert take(["C1", "S1", "C2", "S2", "C3"], size=5, cap=cap) == ["C1", "S1", "C2", "C3"]
    assert take(["C1", "C2", "C3"], size=2, cap=cap) == ["C1", "C2"]


def test_random_universe_respects_sizes_sector_cap_and_seed(universe: ModuleType):
    names, _, cap = _one_sector_market(universe)
    candidates = {"stock_us": names, "commodity": ["X", "Y"]}
    sizes = {"stock_us": 6, "commodity": 5}
    for seed in range(20):
        first = universe.random_universe(
            candidates, sizes, cap=cap, rng=np.random.default_rng(seed)
        )
        again = universe.random_universe(
            candidates, sizes, cap=cap, rng=np.random.default_rng(seed)
        )
        assert first == again
        assert len(first) == 8
        assert sum(name.startswith("BANK") for name in first) == 3


# liquidity


def test_most_liquid_sorts_by_score_and_keeps_the_given_order_on_ties(universe: ModuleType):
    liquidity = pd.Series({"A": 1.0, "B": 3.0, "C": 1.0, "D": 2.0})
    assert universe.most_liquid(liquidity, ["C", "A", "B", "D"]) == ["B", "D", "C", "A"]


def test_the_liquid_pool_is_the_anchors_then_the_most_liquid(universe: ModuleType):
    ranked = ["A", "B", "C", "D", "E", "F"]
    assert universe.liquid_pool(ranked, anchors=("F",), size=3) == ["F", "A", "B"]
    assert universe.liquid_pool(ranked, anchors=(), size=10) == ranked
    with pytest.raises(ValueError, match="anchors"):
        universe.liquid_pool(ranked, anchors=("Z",), size=3)


def test_the_greedy_pick_inside_the_liquid_pool_never_takes_an_asset_outside_it(
    universe: ModuleType,
):
    names = ["A", "B", "C", "D", "E", "F"]
    pairs = {(a, b): 0.5 for i, a in enumerate(names) for b in names[i + 1 :]}
    pairs[("E", "F")] = 0.0
    corr = _correlation(names, pairs)
    pool = universe.liquid_pool(names, anchors=("F",), size=3)
    order, _ = _pick(universe, corr, pool, anchors=("F",), size=3)
    # E, uncorrelated to the anchor F, is outside the pool
    assert order == ["F", "A", "B"]
    everyone, _ = _pick(universe, corr, ["F", *names[:5]], anchors=("F",), size=3)
    assert everyone[1] == "E"


# correlation and diversity


def test_returns_across_a_closed_market_are_missing_not_jumps(universe: ModuleType):
    closes = pd.DataFrame({"A": [1.0, np.nan, 2.0, 2.0]})
    returns = universe.log_returns(closes)["A"]
    assert returns.isna().tolist() == [True, True, True, False]


def _daily(values: np.ndarray, start: str = "2024-01-01") -> pd.Series:
    days = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(values)), index=days)


def _weekly_correlation(
    universe: ModuleType, closes: dict[str, pd.Series], *, start: str, end: str, min_overlap: int
) -> pd.DataFrame:
    returns = universe.weekly_returns(closes, start=start, end=end)
    return universe.correlation_matrix(returns, min_overlap=min_overlap)


def test_a_weekly_return_runs_between_the_last_closes_of_two_weeks(universe: ModuleType):
    # four ISO weeks from Monday 2024-01-01; B has no close in the second week and none on the
    # Sunday of the third, so its third-week close is the Saturday
    days = pd.date_range("2024-01-01", "2024-01-28", freq="D", tz="UTC")
    a = pd.Series(np.arange(1.0, 29.0), index=days)
    b = a[
        (days < pd.Timestamp("2024-01-08", tz="UTC"))
        | (days >= pd.Timestamp("2024-01-15", tz="UTC"))
    ]
    b = b.drop(pd.Timestamp("2024-01-21", tz="UTC"))
    returns = universe.weekly_returns({"A": a, "B": b}, start="2024-01-01", end="2024-01-29")
    weeks = pd.DatetimeIndex(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"], tz="UTC")
    pd.testing.assert_index_equal(returns.index, weeks, check_names=False)
    np.testing.assert_allclose(
        returns["A"].to_numpy(), [np.nan, np.log(14 / 7), np.log(21 / 14), np.log(28 / 21)]
    )
    # a week without a close has no return, and neither has the week after it
    np.testing.assert_allclose(returns["B"].to_numpy(), [np.nan, np.nan, np.nan, np.log(28 / 20)])


def test_weekly_returns_see_through_a_one_day_lag_of_different_time_zones(
    universe: ModuleType,
):
    # Tokyo closes before New York opens: news of day t reaches Tokyo's close on day t + 1, so
    # their daily returns barely correlate while their weekly returns move together
    rng = np.random.default_rng(4)
    news = rng.normal(0, 0.01, 3 * 364 + 1)
    new_york = _daily(news[1:])
    tokyo = _daily(news[:-1])
    daily = pd.DataFrame({"NY": new_york, "T": tokyo}).apply(np.log).diff()
    assert abs(daily.corr().loc["NY", "T"]) < 0.1
    weekly = _weekly_correlation(
        universe, {"AAPL": new_york, "7203.T": tokyo}, start="2024-01-01", end="2026-12-28",
        min_overlap=100,
    )  # fmt: skip
    assert weekly.loc["AAPL", "7203.T"] > 0.7


def test_weekly_correlation_uses_only_its_window(universe: ModuleType):
    rng = np.random.default_rng(2)
    values = rng.normal(0, 0.01, (400, 2))
    closes = {"A": _daily(values[:, 0]), "B": _daily(values[:, 1])}
    inside = _weekly_correlation(
        universe, closes, start="2024-01-01", end="2024-10-07", min_overlap=30
    )
    later = {name: series.copy() for name, series in closes.items()}
    later["B"].iloc[280:] = later["B"].iloc[280:] * rng.uniform(0.5, 2.0, 120)
    again = _weekly_correlation(
        universe, later, start="2024-01-01", end="2024-10-07", min_overlap=30
    )
    pd.testing.assert_frame_equal(inside, again)


def test_a_pair_with_too_few_common_weeks_raises_instead_of_counting_as_zero(
    universe: ModuleType,
):
    rng = np.random.default_rng(3)
    closes = {"A": _daily(rng.normal(0, 0.01, 364)), "B": _daily(rng.normal(0, 0.01, 364))}
    closes["B"] = closes["B"].iloc[250:]
    with pytest.raises(ValueError, match=r"1 pairs have fewer than 20 .*'A', 'B'"):
        _weekly_correlation(universe, closes, start="2024-01-01", end="2024-12-30", min_overlap=20)


def test_a_frozen_price_raises_zero_variance_not_too_few_weeks(universe: ModuleType):
    # a delisted coin stored with its last price: plenty of common weeks, but no variance
    rng = np.random.default_rng(5)
    closes = {"A": _daily(rng.normal(0, 0.01, 364)), "FROZEN": _daily(np.zeros(364))}
    with pytest.raises(ValueError, match=r"zero variance: .*FROZEN"):
        _weekly_correlation(universe, closes, start="2024-01-01", end="2024-12-30", min_overlap=20)


def _factor_returns(shares: dict[str, float], *, weeks: int, seed: int) -> pd.DataFrame:
    """Weekly returns of assets that each hold `shares[name]` of one common factor."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, weeks)
    columns = {
        name: share * common + (1 - share) * rng.normal(0, 0.01, weeks)
        for name, share in shares.items()
    }
    return pd.DataFrame(columns)


def _bootstrap_settings(universe: ModuleType, **changes: Any) -> Any:
    """The study's settings with other bootstrap options."""
    settings = universe.read_settings(tomllib.loads(STUDY_TOML.read_text())["options"])
    return dataclasses.replace(settings, **changes)


def _gap_case(universe: ModuleType) -> tuple[pd.DataFrame, dict[str, list[str]], float]:
    """Returns, compared sets and point gap: an independent universe against factor-heavy sets."""
    shares = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.9, "E": 0.9, "F": 0.9, "G": 0.0}
    returns = _factor_returns(shares, weeks=208, seed=6)
    chosen = ["A", "B", "C"]
    randoms = [["D", "E", "F"], ["D", "E", "G"], ["E", "F", "G"]]
    correlation = universe.correlation_matrix(returns, min_overlap=100)
    ranks = [universe.diversity(correlation.loc[m, m])["effective_rank"] for m in randoms]
    gap = universe.diversity(correlation.loc[chosen, chosen])["effective_rank"] - np.median(ranks)
    sets = {
        "Universe": chosen,
        "Liquid": ["D"],
        **{f"Random {i}": m for i, m in enumerate(randoms)},
    }
    return returns, sets, float(gap)


def test_the_gap_interval_is_centred_on_the_gap_and_follows_its_seed(universe: ModuleType):
    returns, sets, gap = _gap_case(universe)
    interval = {"bootstrap_min_overlap": 100, "bootstrap_block_weeks": 4}
    settings = _bootstrap_settings(universe, **interval, bootstrap_resamples=200)
    low, high = universe.gap_interval(returns, sets, settings, gap=gap, seed=0)
    # the normal interval holds the point gap by construction, also when the resampled gaps
    # are shifted below it (the percentile interval of the review did not)
    assert 0.0 < low < gap < high < 2.0
    assert (low + high) / 2 == pytest.approx(gap)
    assert universe.gap_interval(returns, sets, settings, gap=gap, seed=0) == (low, high)
    assert universe.gap_interval(returns, sets, settings, gap=gap, seed=1) != (low, high)
    shifted = universe.gap_interval(returns, sets, settings, gap=gap + 1.0, seed=0)
    assert shifted == pytest.approx((low + 1.0, high + 1.0))
    narrow = dataclasses.replace(settings, interval_level=0.5)
    narrow_low, narrow_high = universe.gap_interval(returns, sets, narrow, gap=gap, seed=0)
    z_ratio = NormalDist().inv_cdf(0.75) / NormalDist().inv_cdf(0.975)
    assert (narrow_high - narrow_low) / (high - low) == pytest.approx(z_ratio)
    with pytest.raises(ValueError, match="block"):
        universe.gap_interval(
            returns, sets, dataclasses.replace(settings, bootstrap_block_weeks=0), gap=gap, seed=0
        )


def test_the_gap_interval_half_width_is_z_times_the_sd_of_the_paired_gaps(
    universe: ModuleType,
):
    returns, sets, gap = _gap_case(universe)
    settings = _bootstrap_settings(
        universe, bootstrap_min_overlap=100, bootstrap_block_weeks=4, bootstrap_resamples=100
    )
    # the paired draws, rebuilt here: the same blocks of weeks for the universe and the randoms
    positions = block_indices(len(returns), np.random.default_rng(0), n_resamples=100, block=4)
    randoms = [members for name, members in sets.items() if name.startswith("Random")]
    gaps = []
    for weeks in positions:
        correlation = universe.correlation_matrix(returns.take(weeks), min_overlap=100)
        ranks = [universe.diversity(correlation.loc[m, m])["effective_rank"] for m in randoms]
        chosen = sets["Universe"]
        rank = universe.diversity(correlation.loc[chosen, chosen])["effective_rank"]
        gaps.append(rank - np.median(ranks))
    half_width = NormalDist().inv_cdf(0.975) * np.std(gaps, ddof=1)
    low, high = universe.gap_interval(returns, sets, settings, gap=gap, seed=0)
    assert (low, high) == pytest.approx((gap - half_width, gap + half_width))


def test_the_gap_interval_uses_only_the_first_random_universes(universe: ModuleType):
    returns, sets, gap = _gap_case(universe)
    settings = _bootstrap_settings(
        universe, bootstrap_min_overlap=100, bootstrap_block_weeks=4, bootstrap_resamples=20
    )
    first = universe.BOOTSTRAP_RANDOM_UNIVERSES
    padded = {name: members for name, members in sets.items() if not name.startswith("Random")}
    for draw in range(first):
        padded[f"Random {draw}"] = sets[f"Random {draw % 3}"]
    expected = universe.gap_interval(returns, padded, settings, gap=gap, seed=0)
    # a later draw with an asset that has no returns would raise if it were measured
    padded[f"Random {first}"] = ["NO_SUCH_ASSET"]
    assert universe.gap_interval(returns, padded, settings, gap=gap, seed=0) == expected


def test_the_gap_interval_is_paired_on_the_same_weeks(universe: ModuleType):
    # a random set equal to the universe has a gap of exactly 0 in every draw
    returns = _factor_returns({"A": 0.0, "B": 0.5, "C": 0.9}, weeks=104, seed=7)
    chosen = ["A", "B", "C"]
    settings = _bootstrap_settings(universe, bootstrap_resamples=50)
    sets = {"Universe": chosen, "Random 0": chosen}
    assert universe.gap_interval(returns, sets, settings, gap=0.0, seed=0) == (0.0, 0.0)


def test_the_gap_interval_raises_on_too_few_common_weeks(universe: ModuleType):
    returns = _factor_returns({"A": 0.0, "B": 0.0, "C": 0.0}, weeks=104, seed=8)
    returns.loc[30:, "C"] = np.nan
    settings = _bootstrap_settings(universe, bootstrap_min_overlap=50, bootstrap_resamples=10)
    sets = {"Universe": ["A", "B"], "Random 0": ["A", "C"]}
    with pytest.raises(ValueError, match="fewer than 50 common returns"):
        universe.gap_interval(returns, sets, settings, gap=0.0, seed=0)


def test_the_noise_level_of_independent_series_is_their_effective_rank(universe: ModuleType):
    returns = _factor_returns({f"X{i}": 0.0 for i in range(30)}, weeks=43, seed=9)
    returns.iloc[:20, :5] = np.nan
    rank = universe.diversity(universe.correlation_matrix(returns, min_overlap=20))
    noise = universe.noise_effective_rank(returns, min_overlap=20, seed=0)
    assert noise == pytest.approx(rank["effective_rank"], rel=0.05)
    # 43 weeks cap the rank of 30 assets well below 30
    assert noise < 25
    assert universe.noise_effective_rank(returns, min_overlap=20, seed=0) == noise
    assert universe.noise_effective_rank(returns, min_overlap=20, seed=1) != noise


def test_the_noise_level_depends_only_on_the_missing_data(universe: ModuleType):
    # perfectly correlated assets have an effective rank of about 1; the noise level of their
    # missing-data pattern is that of independent series all the same
    common = np.random.default_rng(10).normal(0, 0.01, 208)
    correlated = pd.DataFrame({f"X{i}": (i + 1) * common for i in range(10)})
    rank = universe.diversity(universe.correlation_matrix(correlated, min_overlap=100))
    assert rank["effective_rank"] == pytest.approx(1.0)
    independent = _factor_returns({f"X{i}": 0.0 for i in range(10)}, weeks=208, seed=11)
    noise = universe.noise_effective_rank(correlated, min_overlap=100, seed=0)
    assert noise == universe.noise_effective_rank(independent, min_overlap=100, seed=0)
    assert noise > 9.0


def test_diversity_of_independent_and_identical_sets(universe: ModuleType):
    independent = universe.diversity(pd.DataFrame(np.eye(4)))
    assert independent["effective_rank"] == pytest.approx(4.0)
    assert independent["mean_abs_correlation"] == pytest.approx(0.0)
    assert independent["min_eigenvalue"] == pytest.approx(1.0)
    identical = universe.diversity(pd.DataFrame(np.ones((4, 4))))
    assert identical["effective_rank"] == pytest.approx(1.0)
    assert identical["components_80"] == 1
    assert identical["first_component_share"] == pytest.approx(1.0)


def test_diversity_reports_a_negative_eigenvalue_and_refuses_missing_pairs(
    universe: ModuleType,
):
    # pairwise correlations that no real data set can have together
    impossible = pd.DataFrame([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    measures = universe.diversity(impossible)
    assert measures["min_eigenvalue"] < 0
    assert 1.0 <= measures["effective_rank"] <= 3.0
    with pytest.raises(ValueError, match="missing"):
        universe.diversity(pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]]))
