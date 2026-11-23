import pandas as pd
import pytest

from src.enums import Timeframe


def test_value_round_trip():
    for timeframe in Timeframe:
        assert Timeframe(timeframe.value) is timeframe


def test_plain_enum_is_not_a_string():
    assert Timeframe.H1 != "1h"
    assert Timeframe.H1.value == "1h"


@pytest.mark.parametrize(
    ("timeframe", "minutes"),
    [
        (Timeframe.M15, 15), (Timeframe.M30, 30), (Timeframe.H1, 60), (Timeframe.H4, 240),
        (Timeframe.H12, 720), (Timeframe.D1, 1440), (Timeframe.D3, 4320), (Timeframe.W1, 10080),
    ],
)  # fmt: skip
def test_lengths(timeframe, minutes):
    assert timeframe.minutes == minutes
    assert timeframe.timedelta == pd.Timedelta(minutes=minutes)


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="2h"):
        Timeframe("2h")
