"""The names of the markets and models in the paper's tables, in their order."""

MARKET_NAMES = {
    "crypto": "Crypto",
    "forex": "FX",
    "commodity": "Commodities",
    "index": "Indices",
    "stock": "Stocks",
}
"""The markets in the paper's fixed order (crypto, FX, commodities, indices, stocks), with their
display names in every paper table and figure (one spelling per market; the stock market holds US
and non-US names)."""

MODEL_NAMES = {
    "random-walk": "Random walk",
    "climatology": "Climatology",
    "ewma-normal": "EWMA-normal",
    "garch-normal": "GARCH(1,1)",
    "linear": "Ridge regression",
    "chronos-bolt-small": "Chronos-Bolt (S)",
    "chronos-bolt-base": "Chronos-Bolt (B)",
    "tirex-1.1": "TiRex",
    "kronos-small": "Kronos (S)",
    "kronos-base": "Kronos (B)",
    "moirai-2-small": "Moirai 2 (S)",
    "timesfm-2.5": "TimesFM 2.5",
    "chronos-2": "Chronos-2",
    "toto-2-22m": "Toto 2 (22M)",
    "toto-2-313m": "Toto 2 (313M)",
    "tirex-2": "TiRex-2",
    "timesfm-3": "TimesFM 3",
    "t0-gamma": "t0",
}
"""The baselines, then the zero-shot models in release order (``release_dates`` of the
02-forecasts configuration; ``tables/models.tex``), with their names in the paper."""

FINANCE_MODELS = ("kronos-small", "kronos-base")
"""The checkpoints pre-trained on financial data. The paper's tables list them last in the main
family, below a dashed rule (display only: their family, Holm correction and verdict are
unchanged)."""

REFERENCE_NAMES = {"buy-and-hold": "Buy and hold"}
"""The trading references of the paper's tables (not forecast models), with their names."""


def paper_order(families: dict[str, str]) -> list[str]:
    """Return the models in the order of the paper's tables and figures (tab:rq1).

    The models of `families` in the order of `MODEL_NAMES` (baselines first, then release
    order; a model not in it follows, in the order given), with every model of the ``late``
    Holm family moved after the others.

    Parameters
    ----------
    families : dict of str to str
        The Holm family (``main`` or ``late``) of each model to order; a baseline counts as
        ``main``.

    Returns
    -------
    list of str
        The model names, in order.

    Raises
    ------
    ValueError
        If a family is neither ``main`` nor ``late``.
    """
    unknown = sorted(set(families.values()) - {"main", "late"})
    if unknown:
        raise ValueError(f"families {unknown} are neither main nor late")
    known = list(MODEL_NAMES)
    ranked = [
        (families[name] == "late", known.index(name) if name in known else len(known), given, name)
        for given, name in enumerate(families)
    ]
    return [name for *_, name in sorted(ranked)]


def finance_last(models: list[str], families: dict[str, str]) -> list[str]:
    """Return `models` with the finance-pretrained main-family checkpoints last in that family.

    The display order of the paper's tables (`FINANCE_MODELS`); nothing else changes.

    Parameters
    ----------
    models : list of str
        The models in their order, the main family before the late family.
    families : dict of str to str
        The Holm family (``main`` or ``late``) of each model; a model not in it counts as
        ``main``.

    Returns
    -------
    list of str
        The same models: the other main models, the finance models, then the late family.
    """
    main = [model for model in models if families.get(model, "main") == "main"]
    late = [model for model in models if families.get(model, "main") != "main"]
    general = [model for model in main if model not in FINANCE_MODELS]
    finance = [model for model in main if model in FINANCE_MODELS]
    return general + finance + late


def dashed_rule(columns: int) -> str:
    r"""Return the table line of the dashed rule above the finance-pretrained checkpoints.

    Parameters
    ----------
    columns : int
        The number of columns the rule spans, from the first.

    Returns
    -------
    str
        ``\cdashline`` (arydshln) with 2 pt of space above and below, so it clears the rows.
    """
    return f"\\addlinespace[2pt]\\cdashline{{1-{columns}}}\\addlinespace[2pt]"
