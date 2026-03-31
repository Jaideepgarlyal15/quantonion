"""
HMM Regime Filter Strategy.

Uses Hidden Markov Model regime labels to gate long exposure.
Only holds long during identified bull-regime periods; flat during
bear or uncertain regimes. This demonstrates the practical application
of the regime detection layer as a portfolio filter.
"""

from __future__ import annotations

from typing import Optional, Set

import pandas as pd

from strategies.base import BaseStrategy

# Default set of regime labels considered "long-friendly"
_DEFAULT_BULL_REGIMES: Set[str] = {"Calm", "Super Calm", "Bull", "Super Bull"}


class RegimeFilter(BaseStrategy):
    """
    Regime Filter: go long only during favourable market regimes.

    Uses the HMM regime labels (e.g. "Calm", "Super Calm") to determine
    when to hold a long position. This strategy has no price-level logic
    of its own — the regime model is the entire signal.

    Signal logic:
        1  when current regime label is in `bull_regimes`
        0  otherwise (flat / de-risked)

    Parameters:
        bull_regimes: Set of regime label strings to treat as long-friendly.
                      Defaults to {"Calm", "Super Calm", "Bull", "Super Bull"}.

    Usage:
        regime_series must be passed as a kwarg to generate_signals():
            signals = strategy.generate_signals(
                prices,
                regime_series=pd.Series(["Calm", "Choppy", ...], index=dates)
            )

    Notes:
        - If regime_series is not provided, the strategy defaults to always-long
          (identical to Buy & Hold). Always provide regime_series for a
          meaningful comparison.
        - Regime labels are forward-filled to handle any index gaps between
          the HMM output and the price series.
        - This strategy has zero transaction cost optimisation. It will trade
          each time the regime transitions, which can be frequent. Consider
          adding a minimum regime duration filter for live applications.
    """

    name = "Regime Filter"

    def __init__(self, bull_regimes: Optional[Set[str]] = None) -> None:
        self.bull_regimes = bull_regimes if bull_regimes is not None else _DEFAULT_BULL_REGIMES

    def generate_signals(
        self,
        prices: pd.Series,
        regime_series: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Args:
            prices: Closing price series.
            regime_series: pd.Series of regime label strings indexed by date.
                           Must be provided for meaningful results.
        """
        if regime_series is None or regime_series.empty:
            # Graceful fallback: always long, same as buy-and-hold
            return pd.Series(1.0, index=prices.index, dtype=float)

        # Align regime labels to price index, forward-fill any gaps
        regime_aligned = regime_series.reindex(prices.index).ffill().bfill()
        signal = regime_aligned.apply(
            lambda r: 1.0 if r in self.bull_regimes else 0.0
        ).astype(float)
        return signal
