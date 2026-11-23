import tomllib

import exchange_calendars as xc
import pytest

from src import LAB_DIR
from src.data import load_data_config
from src.enums import Market, Source, Timeframe

CONFIG = """
[period]
start = "2020-01-01"
end = "2026-11-01"
[download.workers]
binance = 2
dukascopy = 2
alpaca = 1
yahoo = 1
build = 2
[quality]
min_coverage = 0.95
max_gap_days = 10
max_negative_spread_share = 0.001
outlier_sigma = 10
max_abs_return = 0.4
yahoo_return_median_bps = 30
yahoo_return_p99_bps = 300
max_flat_share = 0.25
[stock_calendars]
"" = "XNYS"
DE = "XETR"
[groups.stock]
market = "stock"
source = "dukascopy"
instruments = { AAPL = "aaplususd" }
[groups.stock_alpaca]
market = "stock"
source = "alpaca"
instruments = { META = "META" }
"""


def test_groups_become_instruments(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(CONFIG)
    config = load_data_config(path, data_dir=tmp_path)
    assert config.workers["dukascopy"] == 2
    assert config.quality.yahoo_return_p99_bps == 300
    assert [(i.name, i.source, i.market) for i in config.instruments] == [
        ("AAPL", Source.DUKASCOPY, Market.STOCK),
        ("META", Source.ALPACA, Market.STOCK),
    ]


def test_missing_value_raises(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(CONFIG.replace("min_coverage = 0.95\n", ""))
    with pytest.raises(TypeError, match="min_coverage"):
        load_data_config(path, data_dir=tmp_path)


def test_groups_filter_keeps_only_named_groups(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(CONFIG)
    names = list(tomllib.loads(CONFIG)["groups"])
    config = load_data_config(path, data_dir=tmp_path, groups=names[1:])
    assert [i.name for i in config.instruments] == ["META"]


def test_unknown_group_raises(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(CONFIG)
    with pytest.raises(KeyError, match="unknown groups"):
        load_data_config(path, data_dir=tmp_path, groups=["nope"])


def test_group_timeframe_sets_the_base_timeframe(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(
        CONFIG.replace('source = "dukascopy"\n', 'source = "dukascopy"\ntimeframe = "1h"\n')
    )
    config = load_data_config(path, data_dir=tmp_path)
    by_name = {i.name: i.timeframe for i in config.instruments}
    assert by_name == {"AAPL": Timeframe.H1, "META": Timeframe.M15}


def test_no_reference_marks_only_the_listed_stock(tmp_path):
    path = tmp_path / "data.toml"
    listed = 'instruments = { AAPL = "aaplususd", ROG = "rog" }\nno_reference = ["ROG"]'
    path.write_text(CONFIG.replace('instruments = { AAPL = "aaplususd" }', listed))
    config = load_data_config(path, data_dir=tmp_path)
    flags = {instrument.name: instrument.has_reference for instrument in config.instruments}
    assert flags == {"AAPL": True, "ROG": False, "META": True}


def test_no_reference_name_outside_its_group_raises(tmp_path):
    path = tmp_path / "data.toml"
    wrong = 'instruments = { AAPL = "aaplususd" }\nno_reference = ["META"]'
    path.write_text(CONFIG.replace('instruments = { AAPL = "aaplususd" }', wrong))
    with pytest.raises(KeyError, match="no_reference names are not in group stock"):
        load_data_config(path, data_dir=tmp_path)


INDEX_GROUP = """
[groups.index]
market = "index"
source = "dukascopy"
instruments = { USA500 = "usa500idxusd", DOLLAR = "dollaridxusd" }
"""


def test_a_group_calendars_table_sets_each_instruments_calendar(tmp_path):
    path = tmp_path / "data.toml"
    text = 'instruments = { USA500 = "usa500idxusd" }\ncalendars = { USA500 = "XNYS" }'
    path.write_text(
        CONFIG
        + INDEX_GROUP.replace(
            'instruments = { USA500 = "usa500idxusd", DOLLAR = "dollaridxusd" }', text
        )
    )
    config = load_data_config(path, data_dir=tmp_path, groups=["index"])
    assert {i.name: i.calendar for i in config.instruments} == {"USA500": "XNYS"}
    path.write_text(CONFIG + INDEX_GROUP + 'calendars = { NOPE = "XNYS" }\n')
    with pytest.raises(KeyError, match="calendars names are not in group index"):
        load_data_config(path, data_dir=tmp_path)


def test_an_instrument_without_a_calendar_has_none(tmp_path):
    path = tmp_path / "data.toml"
    path.write_text(CONFIG + INDEX_GROUP + 'calendars = { USA500 = "XNYS" }\n')
    config = load_data_config(path, data_dir=tmp_path, groups=["index"])
    assert {i.name: i.calendar for i in config.instruments} == {"USA500": "XNYS", "DOLLAR": None}


def test_a_stock_takes_its_calendar_from_its_ticker_suffix(tmp_path):
    path = tmp_path / "data.toml"
    listed = 'instruments = { AAPL = "aaplususd", "SAP.DE" = "sapdeeur" }'
    path.write_text(CONFIG.replace('instruments = { AAPL = "aaplususd" }', listed))
    config = load_data_config(path, data_dir=tmp_path)
    calendars = {i.name: i.calendar for i in config.instruments}
    assert calendars == {"AAPL": "XNYS", "SAP.DE": "XETR", "META": "XNYS"}


def test_a_stock_suffix_without_a_calendar_raises(tmp_path):
    path = tmp_path / "data.toml"
    listed = 'instruments = { AAPL = "aaplususd", "7203.T" = "7203jpjpy" }'
    path.write_text(CONFIG.replace('instruments = { AAPL = "aaplususd" }', listed))
    with pytest.raises(KeyError, match="suffix 'T' is not in \\[stock_calendars\\]"):
        load_data_config(path, data_dir=tmp_path)


def test_a_stock_group_with_its_own_calendars_raises(tmp_path):
    path = tmp_path / "data.toml"
    text = 'instruments = { AAPL = "aaplususd" }\ncalendars = { AAPL = "XLON" }'
    path.write_text(CONFIG.replace('instruments = { AAPL = "aaplususd" }', text))
    with pytest.raises(KeyError, match="takes its calendars from"):
        load_data_config(path, data_dir=tmp_path)


def test_the_lab_config_names_a_known_calendar_for_every_instrument_with_one():
    config = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=LAB_DIR / "data")
    names = {i.calendar for i in config.instruments if i.calendar is not None}
    assert names <= set(xc.get_calendar_names())
    stocks = [i for i in config.instruments if i.market is Market.STOCK]
    assert all(i.calendar is not None for i in stocks)
    commodities = {i.name: i.calendar for i in config.instruments if i.market is Market.COMMODITY}
    softs = {"COCOA": "XNYS", "COFFEE": "XNYS", "COTTON": "XNYS", "OJUICE": "XNYS",
             "SOYBEAN": "XNYS", "SUGAR": "XLON"}  # fmt: skip
    assert {name: commodities[name] for name in softs} == softs
    assert all(commodities[name] is None for name in set(commodities) - set(softs))
