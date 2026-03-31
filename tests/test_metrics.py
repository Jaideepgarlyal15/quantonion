"""
Tests for backtesting/metrics.py

Uses known equity curves so expected metric values can be computed by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import compute_metrics, format_metrics_table, _empty_metrics


def _make_result(equity: list, position_val: float = 1.0) -> pd.DataFrame:
    """Build a minimal backtest result DataFrame from an equity curve list."""
    n = len(equity)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    eq = pd.Series(equity, index=idx, name="equity_curve")
    pos = pd.Series([position_val] * n, index=idx, name="position")
    # strategy_return is log of equity ratio
    strat_ret = np.log(eq / eq.shift(1)).fillna(0.0)
    drawdown = (eq - eq.cummax()) / eq.cummax()
    return pd.DataFrame({
        "equity_curve": eq,
        "position": pos,
        "strategy_return": strat_ret,
        "drawdown": drawdown,
    })


class TestComputeMetrics:
    def test_buy_and_hold_flat(self):
        """Flat equity → zero total return, zero CAGR."""
        result = _make_result([1.0] * 252)
        m = compute_metrics(result)
        assert abs(m["total_return"]) < 1e-6
        assert abs(m["cagr"]) < 1e-6

    def test_positive_return(self):
        """10% total return over 252 days → ~10% CAGR."""
        equity = list(np.linspace(1.0, 1.10, 252))
        result = _make_result(equity)
        m = compute_metrics(result)
        assert abs(m["total_return"] - 0.10) < 0.01
        assert abs(m["cagr"] - 0.10) < 0.02

    def test_max_drawdown_is_negative(self):
        result = _make_result([1.0, 1.1, 0.9, 0.95, 1.0])
        m = compute_metrics(result)
        assert m["max_drawdown"] < 0, "max_drawdown must always be non-positive"

    def test_max_drawdown_known_value(self):
        """Peak=1.1, trough=0.9 → drawdown = (0.9-1.1)/1.1 ≈ -18.2%"""
        result = _make_result([1.0, 1.1, 0.9, 1.0])
        m = compute_metrics(result)
        expected_dd = (0.9 - 1.1) / 1.1
        assert abs(m["max_drawdown"] - expected_dd) < 0.001

    def test_sharpe_zero_for_flat(self):
        result = _make_result([1.0] * 252)
        m = compute_metrics(result)
        assert m["sharpe"] == 0.0

    def test_all_keys_present(self):
        result = _make_result(list(np.linspace(1.0, 1.5, 252)))
        m = compute_metrics(result)
        expected_keys = {
            "total_return", "cagr", "volatility", "sharpe",
            "max_drawdown", "calmar", "win_rate", "profit_factor",
            "n_trades", "time_in_market", "n_days",
        }
        assert expected_keys == set(m.keys())

    def test_n_days_correct(self):
        result = _make_result([1.0] * 100)
        m = compute_metrics(result)
        assert m["n_days"] == 100

    def test_time_in_market_full(self):
        result = _make_result([1.0] * 252, position_val=1.0)
        m = compute_metrics(result)
        assert abs(m["time_in_market"] - 1.0) < 1e-6

    def test_empty_result_returns_zeros(self):
        m = compute_metrics(pd.DataFrame())
        assert m == _empty_metrics()

    def test_single_row_returns_empty_metrics(self):
        result = _make_result([1.0])
        m = compute_metrics(result)
        assert m == _empty_metrics()

    def test_calmar_is_cagr_over_abs_drawdown(self):
        equity = list(np.linspace(1.0, 1.5, 504))  # ~2 years
        result = _make_result(equity)
        m = compute_metrics(result)
        if abs(m["max_drawdown"]) > 1e-9:
            expected_calmar = m["cagr"] / abs(m["max_drawdown"])
            assert abs(m["calmar"] - expected_calmar) < 0.01


class TestFormatMetricsTable:
    def test_returns_dataframe(self):
        results = {
            "Strategy A": {"metrics": {"cagr": 0.1, "sharpe": 1.0,
                                        "total_return": 0.2, "volatility": 0.15,
                                        "max_drawdown": -0.1, "calmar": 1.0,
                                        "win_rate": 0.55, "profit_factor": 1.5,
                                        "n_trades": 10, "time_in_market": 0.8}},
        }
        df = format_metrics_table(results)
        assert isinstance(df, pd.DataFrame)
        assert "Strategy A" in df.index

    def test_empty_results_returns_empty_df(self):
        df = format_metrics_table({})
        assert df.empty
