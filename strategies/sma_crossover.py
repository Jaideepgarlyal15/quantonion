"""
Simple Moving Average (SMA) Crossover Strategy.

Classic trend-following strategy. Long when the fast SMA is above the slow
SMA (uptrend), flat otherwise. No shorting.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class SMACrossover(BaseStrategy):
    """
    SMA Crossover: trend-following using two simple moving averages.

    Signal logic:
        1  when fast_sma > slow_sma  (uptrend confirmed)
        0  when fast_sma <= slow_sma (downtrend or no trend)

    Warm-up period: `slow` days before any signal is generated.

    Parameters:
        fast: Fast SMA window in trading days (default 50).
        slow: Slow SMA window in trading days (default 200).

    Notes:
        - Works best in trending markets with clear directional moves.
        - Prone to whipsawing in sideways or choppy markets.
        - Longer windows reduce noise but increase lag significantly.
    """

    name = "SMA Crossover"

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError(
                f"fast window ({fast}) must be strictly less than slow ({slow})"
            )
        self.fast = fast
        self.slow = slow
        self.name = f"SMA ({fast}/{slow})"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        fast_ma = prices.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = prices.rolling(self.slow, min_periods=self.slow).mean()

        # Long only when both MAs are defined and fast is above slow
        signal = (fast_ma > slow_ma).astype(float)
        signal[slow_ma.isna()] = 0.0
        return signal
