"""
Buy and Hold Strategy — Passive benchmark.

Always fully invested. Used as the comparison baseline against which all
active strategies are measured. A strategy that cannot consistently beat
this benchmark net of costs provides no edge.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class BuyAndHold(BaseStrategy):
    """
    Buy and Hold: always long, never trades.

    This is the passive benchmark. Transaction costs are only incurred on
    the initial entry (handled by the 1-day signal lag in the engine).
    """

    name = "Buy & Hold"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        """Return 1.0 (fully long) for every trading day."""
        return pd.Series(1.0, index=prices.index, dtype=float)
