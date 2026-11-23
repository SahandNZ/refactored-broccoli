"""Trading the sign of each model's median after costs (experiments/05-rq3-net).

The equal-weight sign book of one market and horizon (`book_returns`), its gross and net
result for every model, market and horizon (`net_tables`), the bootstrap of one book
(`book_bootstrap`), the verdict (`net_verdict`) and the LaTeX bodies of the paper's tables
(`break_even_body`, `appendix_body`). Modules import each other relatively, so the package
works under any module name.
"""

from ._bootstrap import book_block, book_bootstrap, is_effective
from ._net import BUY_AND_HOLD, book_returns, net_tables
from ._table import appendix_body, break_even_body, family_models
from ._verdict import LATE, MAIN, net_verdict

__all__ = [
    "BUY_AND_HOLD",
    "LATE",
    "MAIN",
    "appendix_body",
    "book_block",
    "book_bootstrap",
    "book_returns",
    "break_even_body",
    "family_models",
    "is_effective",
    "net_tables",
    "net_verdict",
]
