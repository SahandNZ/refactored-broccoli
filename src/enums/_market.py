from enum import Enum


class Market(Enum):
    """The market an instrument belongs to."""

    CRYPTO = "crypto"
    FX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    STOCK = "stock"
