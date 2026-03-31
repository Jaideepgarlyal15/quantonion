"""
Exponential Moving Average (EMA) Crossover Strategy.

Similar to SMA crossover but exponential weighting gives more weight to
recent prices, producing faster signals. Default parameters (12/26) are
the basis of MACD.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class EMACrossover(BaseStrategy):
    """
    EMA Crossover: trend-following using two exponential moving averages.

    Signal logic:
        1  when fast_ema > slow_ema  (upward momentum)
        0  when fast_ema <= slow_ema (downward or flat momentum)

    Warm-up period: `slow` days suppressed to avoid early noise.

    Parameters:
        fast: Fast EMA span in trading days (default 12).
        slow: Slow EMA span in trading days (default 26).

    Notes:
        - Reacts faster than SMA to price changes.
        - More responsive also means more false signals in choppy markets.
        - The 12/26 default matches the MACD signal line crossover.
    """

    name = "EMA Crossover"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        if fast >= slow:
            raise ValueError(
                f"fast span ({fast}) must be strictly less than slow ({slow})"
            )
        self.fast = fast
        self.slow = slow
        self.name = f"EMA ({fast}/{slow})"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        fast_ema = prices.ewm(span=self.fast, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow, adjust=False).mean()

        signal = (fast_ema > slow_ema).astype(float)
        # Suppress signals during warm-up to reduce spurious early trades
        signal.iloc[: self.slow] = 0.0
        return signal
