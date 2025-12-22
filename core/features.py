# core/features.py

import numpy as np
import pandas as pd


def build_features(df_price: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Build standard features: log returns, volatility, interaction."""
    df = pd.DataFrame(index=df_price.index.copy())

    ret = np.log(df_price["Adj Close"]).diff()
    df["ret"] = ret
    df["vol"] = ret.rolling(vol_window).std() * np.sqrt(252)
    df["ret_x_vol"] = df["ret"] * df["vol"]

    return df.dropna()


def prepare_ml_dataset(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    lookback: int = 20,
):
    """
    Create supervised ML dataset:
    X = rolling window of features (flattened)
    y = next-day log return
    idx = dates associated with each sample
    """
    price = df_price["Adj Close"]
    ret = np.log(price).diff()

    # Align
    f = features.loc[ret.index].dropna()
    r = ret.loc[f.index]

    X_list, y_list, idx_list = [], [], []

    for i in range(lookback, len(f) - 1):
        window = f.iloc[i - lookback : i].values.flatten()
        X_list.append(window)
        y_list.append(r.iloc[i + 1])
        idx_list.append(f.index[i])

    if not X_list:
        return np.empty((0, 0)), np.empty((0,)), pd.Index([])

    X = np.vstack(X_list)
    y = np.array(y_list)
    idx = pd.Index(idx_list)

    return X, y, idx
