"""
RSI Mean Reversion Strategy.

Buy when RSI indicates oversold conditions, exit when overbought.
Uses a state-machine approach to hold positions between entry and exit
signals, preventing excessive trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's exponential smoothing method.

    Args:
        prices: Closing price series.
        period: Lookback period (default 14 days).

    Returns:
        RSI series, range [0, 100]. NaN for the first `period` rows.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion: buy oversold, sell overbought.

    Signal logic (state machine):
        - Enter long when RSI drops below `oversold` threshold
        - Exit to flat when RSI rises above `overbought` threshold
        - Hold position between entry and exit (no re-entry while in trade)

    Parameters:
        period:     RSI calculation period (default 14).
        oversold:   RSI level below which to enter long (default 30).
        overbought: RSI level above which to exit (default 70).

    Notes:
        - Long-only mean-reversion strategy.
        - Works well in range-bound markets; underperforms in strong trends.
        - The state-machine approach avoids excessive churn near thresholds.
        - In a sustained downtrend, strategy may stay long for long periods
          — this is a known limitation of simple RSI reversion.
    """

    name = "RSI Mean Reversion"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than overbought ({overbought})"
            )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI ({period}, {oversold:.0f}/{overbought:.0f})"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        rsi = _compute_rsi(prices, self.period)
        signal = pd.Series(0.0, index=prices.index, dtype=float)

        in_position = False
        for i in range(len(rsi)):
            r = rsi.iloc[i]
            if pd.isna(r):
                signal.iloc[i] = 0.0
                continue
            if not in_position and r < self.oversold:
                in_position = True
            elif in_position and r > self.overbought:
                in_position = False
            signal.iloc[i] = 1.0 if in_position else 0.0

        return signal
