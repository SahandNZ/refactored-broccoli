from src import LAB_DIR
from src.benchmark import MARKET_NAMES, MODEL_NAMES, REFERENCE_NAMES, cache_settings
from src.benchmark._settings import BASELINES
from src.experiment import load_study


def test_the_paper_names_list_the_baselines_then_the_models_in_release_order() -> None:
    assert list(MARKET_NAMES) == ["crypto", "forex", "commodity", "index", "stock"]
    names = list(MODEL_NAMES)
    assert names[:5] == list(BASELINES)
    config = load_study(LAB_DIR / "experiments" / "02-forecasts" / "study.toml")
    releases = {
        name: str(config.options["release_dates"][name]) for name in config.options["release_dates"]
    }
    zero_shot = names[5:]
    assert set(zero_shot) == set(cache_settings(config).zero_shot)
    assert [releases[name] for name in zero_shot] == sorted(releases[name] for name in zero_shot)


def test_the_trading_references_are_named_apart_from_the_models() -> None:
    assert REFERENCE_NAMES == {"buy-and-hold": "Buy and hold"}
    assert not set(REFERENCE_NAMES) & set(MODEL_NAMES)
