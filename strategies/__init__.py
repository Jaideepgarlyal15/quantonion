"""
QuantOnion Strategy Library

All strategies implement:
    BaseStrategy.generate_signals(prices, **kwargs) -> pd.Series

Signals are 0 (flat) or 1 (long). Long-only, no leverage, no shorting.

Usage:
    from strategies import STRATEGIES

    strategy = STRATEGIES["SMA Crossover"](fast=50, slow=200)
    signals = strategy.generate_signals(prices)
"""

from strategies.base import BaseStrategy
from strategies.buy_hold import BuyAndHold
from strategies.sma_crossover import SMACrossover
from strategies.ema_crossover import EMACrossover
from strategies.rsi_reversion import RSIMeanReversion
from strategies.bollinger import BollingerBandReversion
from strategies.regime_filter import RegimeFilter

# Registry: display name → strategy class
# Order controls display ordering in the UI
STRATEGIES: dict = {
    "Buy & Hold": BuyAndHold,
    "SMA Crossover": SMACrossover,
    "EMA Crossover": EMACrossover,
    "RSI Mean Reversion": RSIMeanReversion,
    "Bollinger Band Reversion": BollingerBandReversion,
    "Regime Filter": RegimeFilter,
}

__all__ = [
    "BaseStrategy",
    "BuyAndHold",
    "SMACrossover",
    "EMACrossover",
    "RSIMeanReversion",
    "BollingerBandReversion",
    "RegimeFilter",
    "STRATEGIES",
]
