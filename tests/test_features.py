"""
Tests for core/features.py

Uses synthetic deterministic data so tests are reproducible with no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from core.features import build_features


def _make_prices(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic price DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Adj Close": prices}, index=idx)


class TestBuildFeatures:
    def test_returns_dataframe(self):
        df = _make_prices()
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_prices()
        result = build_features(df)
        assert "ret" in result.columns
        assert "vol" in result.columns
        assert "ret_x_vol" in result.columns

    def test_no_nan_in_result(self):
        df = _make_prices(n=100)
        result = build_features(df, vol_window=21)
        assert not result.isna().any().any(), "build_features should drop NaN rows"

    def test_length_less_than_input(self):
        df = _make_prices(n=100)
        result = build_features(df, vol_window=21)
        # Warm-up period + first diff row dropped → result shorter than input
        assert len(result) < len(df)

    def test_vol_window_respected(self):
        df = _make_prices(n=100)
        r20 = build_features(df, vol_window=20)
        r10 = build_features(df, vol_window=10)
        # Smaller window → longer result (less warm-up)
        assert len(r10) >= len(r20)

    def test_interaction_term_correct(self):
        df = _make_prices(n=60)
        result = build_features(df, vol_window=10)
        expected_interaction = result["ret"] * result["vol"]
        pd.testing.assert_series_equal(
            result["ret_x_vol"],
            expected_interaction,
            check_names=False,
        )

    def test_returns_are_log_returns(self):
        """Verify ret column is log(P_t / P_{t-1})."""
        df = _make_prices(n=50)
        result = build_features(df, vol_window=5)
        log_rets = np.log(df["Adj Close"]).diff().dropna()
        common = result.index.intersection(log_rets.index)
        np.testing.assert_allclose(
            result.loc[common, "ret"].values,
            log_rets.loc[common].values,
            rtol=1e-9,
        )

    def test_empty_prices_returns_empty(self):
        empty_df = pd.DataFrame({"Adj Close": []}, index=pd.DatetimeIndex([]))
        result = build_features(empty_df)
        assert result.empty

    def test_too_short_for_window_returns_empty(self):
        df = _make_prices(n=5)
        result = build_features(df, vol_window=21)
        assert result.empty
