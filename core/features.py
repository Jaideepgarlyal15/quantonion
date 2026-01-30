"""
Feature engineering for the Regime Switching Dashboard.

Builds standard features from price data:
- Log returns
- Rolling volatility
- Return x Volatility interaction term
"""

import numpy as np
import pandas as pd


def build_features(df_price: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """
    Build standard features for HMM model.
    
    Features:
    - ret: Daily log returns
    - vol: Rolling volatility (annualized)
    - ret_x_vol: Return * Volatility interaction
    
    Args:
        df_price: DataFrame with 'Adj Close' column
        vol_window: Rolling window for volatility calculation (days)
    
    Returns:
        DataFrame with feature columns, index aligned to price data
    """
    df = pd.DataFrame(index=df_price.index.copy())

    ret = np.log(df_price["Adj Close"]).diff()
    df["ret"] = ret
    df["vol"] = ret.rolling(vol_window).std() * np.sqrt(252)
    df["ret_x_vol"] = df["ret"] * df["vol"]

    return df.dropna()

