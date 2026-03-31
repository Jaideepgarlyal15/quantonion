"""
QuantOnion Backtesting Module

Provides vectorised backtesting infrastructure with no lookahead bias.

Usage:
    from backtesting.engine import run_backtest
    from backtesting.metrics import compute_metrics, format_metrics_table

    result = run_backtest(prices, signals, cost_bps=10, slippage_bps=5)
    metrics = compute_metrics(result)
"""

from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics, format_metrics_table

__all__ = ["run_backtest", "compute_metrics", "format_metrics_table"]
