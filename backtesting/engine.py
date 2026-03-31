"""
QuantOnion Backtesting Engine

Vectorised daily backtesting with honest execution assumptions:

Execution model:
  - Signal generated on day t from close prices
  - Position entered at close of day t+1 (1-day lag, prevents lookahead)
  - Transaction costs applied on each position change (entry and exit)
  - Long-only, no leverage, no shorting (signals clipped to [0, 1])

Limitations (stated honestly):
  - Close-to-close return approximation (no intraday modelling)
  - Does not model market impact or volume constraints
  - No borrowing costs or dividend adjustments beyond Adj Close
  - Simplified slippage model (fixed bps, not volume-dependent)
  - Not suitable for high-frequency or multi-asset portfolio backtesting

This is an educational backtester. Do not use for live trading decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Run a vectorised single-asset backtest.

    Args:
        prices:
            Closing price series with DatetimeIndex. Should be adjusted
            close prices to account for splits and dividends.
        signals:
            Signal series aligned to prices. Values should be 0 (flat)
            or 1 (long). Values are clipped to [0, 1]; NaN treated as 0.
        cost_bps:
            One-way transaction cost in basis points (default 10 = 0.10%).
            Applied on each unit of position change.
        slippage_bps:
            One-way slippage assumption in basis points (default 5 = 0.05%).
            Added to cost on each trade. Both are applied together on entry
            and exit separately.

    Returns:
        pd.DataFrame with columns:
            price            — closing price
            signal           — raw signal before lag (0 or 1)
            position         — held position after 1-day lag
            daily_return     — log return of underlying price
            strategy_return  — log return of strategy after costs
            equity_curve     — cumulative strategy equity starting at 1.0
            benchmark_equity — cumulative buy-and-hold equity starting at 1.0
            trade_cost       — cost deducted on each day (per unit of change)
            drawdown         — running drawdown from equity peak (negative)

    Raises:
        Returns empty DataFrame if inputs are insufficient (< 2 aligned rows).
    """
    if prices.empty or signals.empty:
        return pd.DataFrame()

    common = prices.index.intersection(signals.index)
    if len(common) < 2:
        return pd.DataFrame()

    px = prices.loc[common].copy()
    sig = signals.reindex(common).fillna(0.0).clip(0.0, 1.0)

    # ── 1-day execution lag ──────────────────────────────────────────────────
    # Signal on day t is executed at close of day t+1.
    # This prevents any lookahead bias from same-day signal use.
    position = sig.shift(1).fillna(0.0)

    # ── Log returns ──────────────────────────────────────────────────────────
    log_ret = np.log(px / px.shift(1)).fillna(0.0)

    # ── Transaction costs ────────────────────────────────────────────────────
    # One-way cost applied on every unit of position change.
    # Both entry and exit carry cost + slippage.
    one_way_cost = (cost_bps + slippage_bps) / 10_000.0
    position_delta = position.diff().abs().fillna(0.0)
    trade_cost = position_delta * one_way_cost

    # ── Strategy return ──────────────────────────────────────────────────────
    strategy_log_ret = position * log_ret - trade_cost

    # ── Equity curves ────────────────────────────────────────────────────────
    equity = np.exp(strategy_log_ret.cumsum())
    benchmark = np.exp(log_ret.cumsum())

    # ── Drawdown ─────────────────────────────────────────────────────────────
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max

    return pd.DataFrame(
        {
            "price": px,
            "signal": sig,
            "position": position,
            "daily_return": log_ret,
            "strategy_return": strategy_log_ret,
            "equity_curve": equity,
            "benchmark_equity": benchmark,
            "trade_cost": trade_cost,
            "drawdown": drawdown,
        }
    )
