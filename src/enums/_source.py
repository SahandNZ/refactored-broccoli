from enum import Enum


class Source(Enum):
    """Where the raw bars come from; ``.value`` is the folder under ``data/``."""

    DUKASCOPY = "dukascopy"
    BINANCE = "binance"
    ALPACA = "alpaca"
