"""What each forecaster kind can read: NaN as a missing value, and past covariates.

One table for every study that shows closed hours as NaN or gives a covariate
(07-rq5-off-hours, 08-closures). Checked in the installed packages on 2026-11-16.
"""

NAN_NOT_MISSING = {
    "timesfm": (
        "TimesFM 2.5 fills NaN by linear interpolation inside forecast() "
        "(timesfm_2p5_base.linear_interpolation), so the model sees interpolated prices"
    ),
    "timesfm3": (
        "TimesFM 3.0 fills NaN by linear interpolation before forecasting "
        "(timesfm3_forecaster.linear_interpolation; a NaN tail is forward-filled), so the model "
        "sees interpolated prices"
    ),
    "toto2": (
        "the wrapper (Toto2Forecaster) marks every value observed (target_mask all ones, "
        "has_missing_values=False), so NaN would enter the model as a value"
    ),
}
"""Forecaster kinds that do not read NaN as a missing value, with the reason; every other kind
does (Chronos, TiRex, TiRex-2, t0, Moirai; Kronos reads candles and takes no NaN)."""

COVARIATE_KINDS = ("timesfm3", "tirex2", "t0")
"""The forecaster kinds whose wrapper reads past covariates (`predict_with_covariates`); of the
kind ``chronos`` only Chronos-2 does."""


def missing_value_reason(kind: str) -> str | None:
    """Return why a forecaster kind cannot read NaN as missing, or None when it can."""
    return NAN_NOT_MISSING.get(kind)


def takes_covariates(kind: str, repository: str) -> bool:
    """Return whether the model of `kind` and `repository` reads past covariates."""
    return kind in COVARIATE_KINDS or (kind == "chronos" and repository == "amazon/chronos-2")
