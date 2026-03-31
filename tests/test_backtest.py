"""
Tests for backtesting/engine.py

Verifies correct execution model: 1-day lag, cost application, output shape.
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import run_backtest


def _make_prices(n: int = 252, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    px = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx, name="price")


def _always_long(prices: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=prices.index)


def _always_flat(prices: pd.Series) -> pd.Series:
    return pd.Series(0.0, index=prices.index)


class TestRunBacktest:
    def test_returns_dataframe(self):
        prices = _make_prices()
        result = run_backtest(prices, _always_long(prices))
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        prices = _make_prices()
        result = run_backtest(prices, _always_long(prices))
        expected = {
            "price", "signal", "position", "daily_return",
            "strategy_return", "equity_curve", "benchmark_equity",
            "trade_cost", "drawdown",
        }
        assert expected == set(result.columns)

    def test_output_length_matches_input(self):
        prices = _make_prices(252)
        result = run_backtest(prices, _always_long(prices))
        assert len(result) == len(prices)

    def test_equity_starts_near_one(self):
        """First equity value should be close to 1.0 (only first-day return)."""
        prices = _make_prices()
        result = run_backtest(prices, _always_long(prices))
        # Day 0 position is 0 (signal lagged 1 day) → no return on day 0
        assert abs(float(result["equity_curve"].iloc[0]) - 1.0) < 0.05

    def test_one_day_signal_lag(self):
        """Signal on day t must produce a non-zero position only from day t+1."""
        n = 50
        prices = _make_prices(n)
        # Signal = 1 only on day 0
        signals = pd.Series(0.0, index=prices.index)
        signals.iloc[0] = 1.0
        result = run_backtest(prices, signals)
        # Day 0 position should be 0 (lag not yet applied)
        assert result["position"].iloc[0] == 0.0
        # Day 1 position should be 1.0
        assert result["position"].iloc[1] == 1.0

    def test_zero_cost_buy_hold_matches_benchmark(self):
        """Buy-and-hold with zero costs should produce equity == benchmark_equity."""
        prices = _make_prices()
        signals = _always_long(prices)
        result = run_backtest(prices, signals, cost_bps=0, slippage_bps=0)
        # Equity and benchmark start aligned after the first trade entry
        # They should be very close (small rounding differences allowed)
        np.testing.assert_allclose(
            result["equity_curve"].values[2:],
            result["benchmark_equity"].values[2:],
            rtol=1e-6,
        )

    def test_costs_reduce_returns(self):
        """Strategy with costs should have lower final equity than without."""
        prices = _make_prices()
        # Alternate between long and flat to maximise trades (= max cost impact)
        signals = pd.Series(
            [float(i % 2) for i in range(len(prices))], index=prices.index
        )
        result_no_cost = run_backtest(prices, signals, cost_bps=0, slippage_bps=0)
        result_with_cost = run_backtest(prices, signals, cost_bps=20, slippage_bps=10)
        assert (
            float(result_with_cost["equity_curve"].iloc[-1])
            < float(result_no_cost["equity_curve"].iloc[-1])
        )

    def test_always_flat_stays_at_one(self):
        """A flat (never invested) strategy should have equity == 1.0 throughout."""
        prices = _make_prices()
        result = run_backtest(prices, _always_flat(prices), cost_bps=0, slippage_bps=0)
        np.testing.assert_allclose(result["equity_curve"].values, 1.0, rtol=1e-9)

    def test_signals_clipped_to_zero_one(self):
        """Signals outside [0,1] should be silently clipped."""
        prices = _make_prices(50)
        signals = pd.Series(2.0, index=prices.index)  # Over-leveraged input
        result = run_backtest(prices, signals)
        assert (result["signal"] <= 1.0).all()
        assert (result["signal"] >= 0.0).all()

    def test_drawdown_always_non_positive(self):
        prices = _make_prices()
        result = run_backtest(prices, _always_long(prices))
        assert (result["drawdown"] <= 1e-10).all()

    def test_empty_prices_returns_empty(self):
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        signals = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        result = run_backtest(empty, signals)
        assert result.empty

    def test_misaligned_signals_handled(self):
        """Signals with different (but overlapping) index should still work."""
        prices = _make_prices(100)
        # Signals only cover half the date range
        signals = pd.Series(1.0, index=prices.index[:50])
        result = run_backtest(prices, signals)
        assert not result.empty
        # After the signal index ends, position should be 0 (NaN → 0)
        assert (result["position"].iloc[51:] == 0.0).all()
