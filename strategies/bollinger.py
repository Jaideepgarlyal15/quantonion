"""
Bollinger Band Mean Reversion Strategy.

Buy when price closes below the lower Bollinger Band (statistical extreme),
exit when price reverts back above the middle band (moving average).
"""

from __future__ import annotations

import pandas as pd

from strategies.base import BaseStrategy


class BollingerBandReversion(BaseStrategy):
    """
    Bollinger Band Mean Reversion: buy at statistical lows, exit at mean.

    Signal logic (state machine):
        - Enter long when price closes below lower band (mean - n*std)
        - Exit to flat when price closes above middle band (SMA)
        - Hold position between entry and exit

    Parameters:
        period:  Rolling window for SMA and std dev (default 20).
        num_std: Number of standard deviations for band width (default 2.0).

    Notes:
        - Lower `num_std` generates more trades; higher generates fewer.
        - Works best in low-volatility range-bound markets.
        - In trending down markets, price can stay below the lower band
          for extended periods — position will be held throughout.
        - Consider pairing with a trend filter to avoid this scenario.
    """

    name = "Bollinger Band Reversion"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period < 2:
            raise ValueError(f"period must be at least 2, got {period}")
        if num_std <= 0:
            raise ValueError(f"num_std must be positive, got {num_std}")
        self.period = period
        self.num_std = num_std
        self.name = f"Bollinger ({period}, {num_std:.1f}σ)"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        middle = prices.rolling(self.period, min_periods=self.period).mean()
        std = prices.rolling(self.period, min_periods=self.period).std()
        lower = middle - self.num_std * std

        signal = pd.Series(0.0, index=prices.index, dtype=float)
        in_position = False

        for i in range(len(prices)):
            if pd.isna(lower.iloc[i]) or pd.isna(middle.iloc[i]):
                signal.iloc[i] = 0.0
                continue
            if not in_position and prices.iloc[i] < lower.iloc[i]:
                in_position = True
            elif in_position and prices.iloc[i] > middle.iloc[i]:
                in_position = False
            signal.iloc[i] = 1.0 if in_position else 0.0

        return signal
