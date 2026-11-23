"""What each forecaster reads: NaN as missing, past covariates, and the checks of their shapes."""

import numpy as np
import pytest
import torch

from src.models import Forecast, missing_value_reason, takes_covariates
from src.models._forecaster import check_covariates
from src.models._new_models import historical_series


def test_the_kinds_that_fill_nan_have_a_reason_and_the_others_none():
    for kind in ("timesfm", "timesfm3", "toto2"):
        assert missing_value_reason(kind)
    for kind in ("chronos", "tirex", "tirex2", "t0", "moirai2"):
        assert missing_value_reason(kind) is None


def test_only_the_four_covariate_models_take_covariates():
    assert takes_covariates("chronos", "amazon/chronos-2")
    assert not takes_covariates("chronos", "amazon/chronos-bolt-small")
    for kind in ("timesfm3", "tirex2", "t0"):
        assert takes_covariates(kind, "any/repository")
    for kind in ("timesfm", "tirex", "toto2", "moirai2", "kronos"):
        assert not takes_covariates(kind, "any/repository")


def test_covariates_must_line_up_with_their_contexts():
    context = np.zeros((3, 8))
    check_covariates(context, np.zeros((3, 2, 8)))
    for wrong in (np.zeros((3, 8)), np.zeros((2, 1, 8)), np.zeros((3, 1, 7))):
        with pytest.raises(ValueError, match="covariates must have shape"):
            check_covariates(context, wrong)


def test_a_forecast_of_the_wrong_shape_is_refused():
    forecast = Forecast(point=np.zeros((2, 1)), quantiles=np.zeros((2, 1, 9)))
    assert forecast.checked_quantiles(2, 1).shape == (2, 1, 9)
    with pytest.raises(ValueError, match=r"must have shape \(3, 1, 9\)"):
        forecast.checked_quantiles(3, 1)


def test_timesfm3_fills_nan_by_interpolation_and_a_nan_tail_forward():
    # pins the library behaviour behind missing_value_reason("timesfm3")
    module = pytest.importorskip("timesfm3.torch.timesfm3_forecaster")
    x = np.array([[1.0, 1.1, np.nan, np.nan, 1.3, 1.2, np.nan, np.nan, np.nan]], dtype=np.float32)
    filled = np.asarray(module.linear_interpolation(x))
    expected = [1.0, 1.1, 1.1 + 0.2 / 3, 1.1 + 0.4 / 3, 1.3, 1.2, 1.2, 1.2, 1.2]
    np.testing.assert_allclose(filled[0], expected, rtol=1e-6)


def test_t0_reads_the_covariates_as_past_only_variates_of_the_same_series():
    t0 = pytest.importorskip("t0")
    variates = torch.randn(2, 3, 8)
    series = historical_series(variates, horizon=2)
    roles = series.variate_type[:, 0].tolist()
    assert roles == [t0.VariateType.TARGET, *[t0.VariateType.HISTORICAL] * 2] * 2
    assert series.group_ids[:, 0].tolist() == [0, 0, 0, 1, 1, 1]
    assert series.variates.shape == (6, 10)
    torch.testing.assert_close(series.variates[:3, :8], variates[0])
