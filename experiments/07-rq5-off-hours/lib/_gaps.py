"""Study 07-rq5-off-hours, Q2: the reopening gap on Friday's information and weekend crypto.

`gap_tables` regresses each asset's reopening gap on Friday's information and the weekend
crypto move, with a two-sided HAC t test of the crypto coefficient (its sign is not known in
advance for every market), Holm-adjusted over the tested assets.
"""

import numpy as np
import pandas as pd
from scipy.special import stdtr, stdtrit

from src.stats import holm, newey_west_ols
from src.utils import half_years

from ._settings import OffHoursSettings

INPUTS = ("friday_target", "friday_crypto", "weekend_crypto")
"""The regressors of the reopening gap, in order: Friday's information, then the crypto move."""

STATISTICS = (
    "weekends",
    "corr",
    "rank_corr",
    "rank_corr_trimmed",
    "beta",
    "beta_low",
    "beta_high",
    "t",
    "p_value",
    "r2_friday",
    "r2_full",
    "r2_increment",
)
"""The columns of `gap_statistics`."""


def gap_statistics(weekends: pd.DataFrame, settings: OffHoursSettings) -> dict[str, float]:
    """Regress the reopening gap on Friday's information and the weekend crypto move.

    The full model is ``gap ~ 1 + friday_target + friday_crypto + weekend_crypto`` with
    Newey-West errors (`src.stats.newey_west_ols`, ``hac_lags``); the Friday model leaves the
    crypto move out, so ``r2_increment`` is what the crypto move adds. The crypto term's t is
    read against Student's t with ``n - k`` degrees of freedom: ``p_value`` is two-sided and
    ``beta_low`` .. ``beta_high`` is the interval at ``interval_level`` from the same standard
    error, so the interval leaves out zero exactly when the unadjusted p is below
    ``1 - interval_level``.

    Parameters
    ----------
    weekends : pd.DataFrame
        Weekends of one group in time order, with `INPUTS` and ``gap``, no NaN; more than
        ``trim_largest + 4`` rows.
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    dict
        ``weekends``, ``corr`` and ``rank_corr`` (Pearson and Spearman of the crypto move
        with the gap), ``rank_corr_trimmed`` (Spearman without the ``trim_largest`` weekends of
        the largest absolute crypto move), ``beta``, ``beta_low``, ``beta_high``, ``t`` and
        ``p_value`` (the crypto term), ``r2_friday``, ``r2_full`` and ``r2_increment``.
    """
    gap = weekends["gap"].to_numpy(dtype=float)
    crypto = weekends["weekend_crypto"].to_numpy(dtype=float)
    friday = np.column_stack(
        [np.ones(len(gap)), weekends[["friday_target", "friday_crypto"]].to_numpy(dtype=float)]
    )
    full = newey_west_ols(gap, np.column_stack([friday, crypto]), lags=settings.hac_lags)
    base = newey_west_ols(gap, friday, lags=settings.hac_lags)
    beta, t = float(full.coefficients[-1]), float(full.t[-1])
    degrees = len(gap) - friday.shape[1] - 1
    critical = float(stdtrit(degrees, 0.5 + settings.interval_level / 2))
    error = abs(beta / t) if t != 0 else np.nan
    kept = np.argsort(np.abs(crypto), kind="stable")[: len(crypto) - settings.trim_largest]
    return {
        "weekends": len(gap),
        "corr": float(np.corrcoef(crypto, gap)[0, 1]),
        "rank_corr": float(pd.Series(crypto).corr(pd.Series(gap), method="spearman")),
        "rank_corr_trimmed": float(
            pd.Series(crypto[kept]).corr(pd.Series(gap[kept]), method="spearman")
        ),
        "beta": beta,
        "beta_low": beta - critical * error,
        "beta_high": beta + critical * error,
        "t": t,
        "p_value": float(2 * stdtr(degrees, -abs(t))),
        "r2_friday": base.r2,
        "r2_full": full.r2,
        "r2_increment": full.r2 - base.r2,
    }


def _group_statistics(
    weekends: pd.DataFrame, keys: list[str], settings: OffHoursSettings
) -> pd.DataFrame:
    # one row per group; a group with fewer than min_weekends weekends is only counted
    rows: list[dict[str, object]] = []
    for key, group in weekends.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        if len(group) < settings.min_weekends:
            status = f"not tested: {len(group)} weekends, fewer than {settings.min_weekends}"
            row |= {"weekends": len(group), "status": status}
        else:
            row |= gap_statistics(group, settings) | {"status": "tested"}
        rows.append(row)
    return pd.DataFrame(rows, columns=[*keys, *STATISTICS, "status"])


def gap_tables(weekends: pd.DataFrame, settings: OffHoursSettings) -> dict[str, pd.DataFrame]:
    """Test per closed-market asset whether the weekend crypto move adds to Friday's (Q2).

    The two-sided p-value of each asset's crypto term is Holm-adjusted over the tested
    assets; an asset says yes when its adjusted p is below ``alpha`` and its trimmed rank
    correlation has the sign of its coefficient (``sign``: positive or negative).

    Parameters
    ----------
    weekends : pd.DataFrame
        The rows of `Weekends.table` of the Q2 assets.
    settings : OffHoursSettings
        The study's settings.

    Returns
    -------
    dict of str to pd.DataFrame
        ``gap_assets`` (one row per asset, also one without a complete weekend:
        `gap_statistics`, ``incomplete`` weekends left out for a missing price, ``status``,
        ``adjusted_p``, ``sign`` and ``is_yes``),
        ``gap_half_years`` (per asset and calendar half-year of the reopening) and
        ``gap_directions`` (per asset and sign of the crypto move, ``up`` or ``down``; a
        weekend without crypto move is in neither).
    """
    is_complete = weekends[[*INPUTS, "gap"]].notna().all(axis=1)
    complete = weekends[is_complete]
    assets = _group_statistics(complete, ["market", "asset"], settings)
    everyone = weekends[["market", "asset"]].drop_duplicates()
    assets = everyone.merge(assets, on=["market", "asset"], how="left")
    assets = assets.sort_values(["market", "asset"], ignore_index=True)
    assets["weekends"] = assets["weekends"].fillna(0).astype(int)
    assets["status"] = assets["status"].fillna("not tested: no complete weekend")
    incomplete = weekends[~is_complete].groupby("asset").size()
    assets["incomplete"] = assets["asset"].map(incomplete).fillna(0).astype(int)
    assets["adjusted_p"] = holm(assets["p_value"].to_numpy(dtype=float))
    beta = assets["beta"].to_numpy(dtype=float)
    assets["sign"] = np.where(beta > 0, "positive", np.where(beta < 0, "negative", ""))
    same_sign = np.sign(assets["rank_corr_trimmed"].to_numpy(dtype=float)) == np.sign(beta)
    assets["is_yes"] = (assets["adjusted_p"] < settings.alpha).to_numpy() & same_sign
    halves = complete.assign(half_year=half_years(pd.DatetimeIndex(complete["reopen_time"])))
    sides = complete[complete["weekend_crypto"] != 0]
    sides = sides.assign(direction=np.where(sides["weekend_crypto"] > 0, "up", "down"))
    return {
        "gap_assets": assets,
        "gap_half_years": _group_statistics(halves, ["market", "asset", "half_year"], settings),
        "gap_directions": _group_statistics(sides, ["market", "asset", "direction"], settings),
    }
