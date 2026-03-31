"""
QuantOnion Backtesting Metrics

Standard quantitative performance metrics for strategy evaluation.

All metrics are computed from the output DataFrame of run_backtest().
Metrics are annualised assuming 252 trading days per year.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_metrics(
    result: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute standard backtesting performance metrics from a backtest result.

    Args:
        result:
            Output DataFrame from backtesting.engine.run_backtest().
        risk_free_rate:
            Annual risk-free rate as a decimal (e.g. 0.04 for 4%).
            Defaults to 0.0 for a conservative Sharpe estimate.

    Returns:
        Dict with keys:
            total_return    — cumulative strategy return (decimal)
            cagr            — compound annual growth rate (decimal)
            volatility      — annualised daily return std dev (decimal)
            sharpe          — annualised Sharpe ratio
            max_drawdown    — maximum peak-to-trough drawdown (negative decimal)
            calmar          — CAGR / |max_drawdown|
            win_rate        — fraction of in-position days with positive return
            profit_factor   — gross profit / gross loss (in-position days)
            n_trades        — estimated number of round-trip trades
            time_in_market  — fraction of days with non-zero position
            n_days          — total number of days in result
    """
    if result.empty or len(result) < 2:
        return _empty_metrics()

    equity = result["equity_curve"]
    strategy_ret = result["strategy_return"]
    position = result["position"]

    n_days = len(equity)
    n_years = max(n_days / 252.0, 1e-8)

    # ── Total return ─────────────────────────────────────────────────────────
    total_return = float(equity.iloc[-1] - 1.0)

    # ── CAGR ─────────────────────────────────────────────────────────────────
    final_value = max(float(equity.iloc[-1]), 1e-10)
    cagr = float(final_value ** (1.0 / n_years) - 1.0)

    # ── Annualised volatility ─────────────────────────────────────────────────
    volatility = float(strategy_ret.std() * np.sqrt(252))

    # ── Sharpe ratio ─────────────────────────────────────────────────────────
    daily_rf = risk_free_rate / 252.0
    excess = strategy_ret - daily_rf
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(252))
        if excess.std() > 1e-10
        else 0.0
    )

    # ── Max drawdown ─────────────────────────────────────────────────────────
    max_drawdown = float(result["drawdown"].min())

    # ── Calmar ratio ─────────────────────────────────────────────────────────
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-9 else 0.0

    # ── Win rate and profit factor (days with active position only) ───────────
    active = strategy_ret[position > 0.01]
    if len(active) > 0:
        wins = active[active > 0]
        losses = active[active < 0]
        win_rate = float(len(wins) / len(active))
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-12 else float("inf")
        )
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # ── Number of round-trip trades ───────────────────────────────────────────
    # Count significant position changes (> 1% threshold avoids float noise).
    pos_changes = (result["position"].diff().abs() > 0.01).sum()
    n_trades = max(int(pos_changes // 2), 0)

    # ── Time in market ────────────────────────────────────────────────────────
    time_in_market = float((position > 0.01).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "time_in_market": time_in_market,
        "n_days": n_days,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "time_in_market": 0.0,
        "n_days": 0,
    }


def format_metrics_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple strategy backtest results.

    Args:
        all_results:
            Dict mapping strategy name → {"metrics": dict, "result": DataFrame}.

    Returns:
        DataFrame with strategies as rows, key metrics as columns.
        Numbers are raw decimals; format for display in the calling layer.
    """
    rows = []
    for name, res in all_results.items():
        m = res.get("metrics", {})
        rows.append(
            {
                "Strategy": name,
                "Total Return": m.get("total_return", 0.0),
                "CAGR": m.get("cagr", 0.0),
                "Volatility": m.get("volatility", 0.0),
                "Sharpe": m.get("sharpe", 0.0),
                "Max Drawdown": m.get("max_drawdown", 0.0),
                "Calmar": m.get("calmar", 0.0),
                "Win Rate": m.get("win_rate", 0.0),
                "Profit Factor": m.get("profit_factor", 0.0),
                "Trades": m.get("n_trades", 0),
                "% In Market": m.get("time_in_market", 0.0),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Strategy")
