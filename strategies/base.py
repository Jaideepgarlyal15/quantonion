"""
Abstract base class for all QuantOnion strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all QuantOnion trading strategies.

    Subclasses must implement generate_signals() which accepts a price
    series and returns a signal series (0 = flat, 1 = long).

    Contract:
        - Signals must only use information available on or before day t
          (no lookahead bias).
        - Signals are delayed by 1 day inside the backtest engine before
          positions are taken, so generate_signals() should return same-day
          signals without any additional delay.
        - NaN values in the returned signal are treated as 0 (flat) by
          the backtest engine.
        - Signals are clipped to [0, 1]; values outside this range are
          silently clamped.
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, prices: pd.Series, **kwargs: Any) -> pd.Series:
        """
        Generate trading signals from closing price data.

        Args:
            prices: Daily adjusted closing price series with DatetimeIndex.
            **kwargs: Optional context such as regime_series for RegimeFilter.

        Returns:
            pd.Series of floats (0.0 = flat, 1.0 = long) with the same
            DatetimeIndex as prices. NaN values are treated as 0.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
