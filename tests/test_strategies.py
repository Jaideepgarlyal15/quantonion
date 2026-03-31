"""
Tests for all strategy implementations in strategies/

Verifies signal contracts: correct types, values in [0,1], no lookahead
in the signal generation logic, and correct edge-case handling.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.buy_hold import BuyAndHold
from strategies.sma_crossover import SMACrossover
from strategies.ema_crossover import EMACrossover
from strategies.rsi_reversion import RSIMeanReversion
from strategies.bollinger import BollingerBandReversion
from strategies.regime_filter import RegimeFilter


def _make_prices(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    px = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(px, index=idx, name="Adj Close")


def _assert_signal_contract(signals: pd.Series, prices: pd.Series):
    """Shared assertions for all strategy signal outputs."""
    assert isinstance(signals, pd.Series), "signals must be a pd.Series"
    assert len(signals) == len(prices), "signals must match price length"
    assert signals.index.equals(prices.index), "signals must share index with prices"
    assert signals.dtype == float or str(signals.dtype).startswith("float"), \
        "signal dtype must be float"
    non_nan = signals.dropna()
    assert ((non_nan >= 0.0) & (non_nan <= 1.0)).all(), \
        "signals must be in [0.0, 1.0]"


class TestBuyAndHold:
    def test_always_long(self):
        prices = _make_prices()
        signals = BuyAndHold().generate_signals(prices)
        assert (signals == 1.0).all()

    def test_signal_contract(self):
        prices = _make_prices()
        _assert_signal_contract(BuyAndHold().generate_signals(prices), prices)

    def test_empty_prices(self):
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        signals = BuyAndHold().generate_signals(empty)
        assert len(signals) == 0


class TestSMACrossover:
    def test_signal_contract(self):
        prices = _make_prices()
        _assert_signal_contract(SMACrossover(50, 200).generate_signals(prices), prices)

    def test_warmup_period_is_zero(self):
        prices = _make_prices()
        signals = SMACrossover(50, 200).generate_signals(prices)
        # Before slow MA is defined (< 200 bars), signal should be 0
        assert (signals.iloc[:199] == 0.0).all()

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            SMACrossover(fast=200, slow=50)

    def test_name_set_correctly(self):
        s = SMACrossover(fast=10, slow=50)
        assert "10" in s.name and "50" in s.name

    def test_uptrend_signal(self):
        """Monotonically increasing prices should eventually yield a long signal."""
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        prices = pd.Series(np.linspace(100, 200, 300), index=idx)
        signals = SMACrossover(20, 50).generate_signals(prices)
        assert (signals.iloc[50:] == 1.0).all()


class TestEMACrossover:
    def test_signal_contract(self):
        prices = _make_prices()
        _assert_signal_contract(EMACrossover(12, 26).generate_signals(prices), prices)

    def test_warmup_suppression(self):
        prices = _make_prices()
        signals = EMACrossover(12, 26).generate_signals(prices)
        assert (signals.iloc[:26] == 0.0).all()

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            EMACrossover(fast=50, slow=10)


class TestRSIMeanReversion:
    def test_signal_contract(self):
        prices = _make_prices()
        _assert_signal_contract(
            RSIMeanReversion(14, 30, 70).generate_signals(prices), prices
        )

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=70, overbought=30)

    def test_no_signal_before_warmup(self):
        prices = _make_prices()
        signals = RSIMeanReversion(period=14).generate_signals(prices)
        # First 14 rows are NaN RSI → should be 0
        assert (signals.iloc[:14] == 0.0).all()

    def test_signals_binary(self):
        """RSI strategy should only produce 0.0 or 1.0 values."""
        prices = _make_prices(n=500)
        signals = RSIMeanReversion().generate_signals(prices)
        unique_vals = set(signals.dropna().unique())
        assert unique_vals.issubset({0.0, 1.0})


class TestBollingerBandReversion:
    def test_signal_contract(self):
        prices = _make_prices()
        _assert_signal_contract(
            BollingerBandReversion(20, 2.0).generate_signals(prices), prices
        )

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            BollingerBandReversion(period=1)
        with pytest.raises(ValueError):
            BollingerBandReversion(num_std=-1.0)

    def test_signals_binary(self):
        prices = _make_prices(n=500)
        signals = BollingerBandReversion().generate_signals(prices)
        unique_vals = set(signals.dropna().unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_no_signal_during_warmup(self):
        prices = _make_prices()
        signals = BollingerBandReversion(period=20).generate_signals(prices)
        assert (signals.iloc[:20] == 0.0).all()


class TestRegimeFilter:
    def _make_regime_series(self, prices: pd.Series, pattern: list) -> pd.Series:
        labels = [pattern[i % len(pattern)] for i in range(len(prices))]
        return pd.Series(labels, index=prices.index)

    def test_signal_contract(self):
        prices = _make_prices()
        regime = self._make_regime_series(prices, ["Calm", "Choppy"])
        strat = RegimeFilter()
        _assert_signal_contract(
            strat.generate_signals(prices, regime_series=regime), prices
        )

    def test_long_in_bull_regime(self):
        prices = _make_prices(n=100)
        regime = pd.Series(["Calm"] * 100, index=prices.index)
        signals = RegimeFilter().generate_signals(prices, regime_series=regime)
        assert (signals == 1.0).all()

    def test_flat_in_bear_regime(self):
        prices = _make_prices(n=100)
        regime = pd.Series(["Stormy"] * 100, index=prices.index)
        signals = RegimeFilter().generate_signals(prices, regime_series=regime)
        assert (signals == 0.0).all()

    def test_defaults_to_always_long_without_regime(self):
        prices = _make_prices(n=50)
        signals = RegimeFilter().generate_signals(prices)
        assert (signals == 1.0).all()

    def test_custom_bull_regimes(self):
        prices = _make_prices(n=100)
        regime = pd.Series(["Choppy"] * 100, index=prices.index)
        # Treat "Choppy" as bull
        signals = RegimeFilter(bull_regimes={"Choppy"}).generate_signals(
            prices, regime_series=regime
        )
        assert (signals == 1.0).all()
