# core/ml.py
"""
Lightweight ML helpers for the Regime Switching Dashboard.

We deliberately avoid heavy dependencies (TensorFlow, XGBoost, LightGBM)
so that the app starts quickly and is easy to run on any machine / free host.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# Dataset prep
# --------------------------------------------------
def prepare_ml_dataset(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    lookback: int = 20,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build supervised dataset for *next day* log-return prediction.

    X: flattened rolling windows of past `lookback` feature vectors
    y: next-day log return
    idx: index aligned to last day of each lookback window
    """
    price = df_price["Adj Close"]
    ret = np.log(price).diff()


    # Align features with returns using reindex to handle missing dates
    common_index = features.index.intersection(ret.index)
    features = features.loc[common_index]
    ret = ret.loc[common_index]

    X_list = []
    y_list = []
    idx_list = []

    for i in range(lookback, len(features) - 1):
        window_feats = features.iloc[i - lookback : i].values
        X_list.append(window_feats.flatten())
        y_list.append(ret.iloc[i + 1])  # predict next day's return
        idx_list.append(features.index[i])

    if not X_list:
        return np.empty((0, 0)), np.empty((0,)), pd.Index([])

    X = np.vstack(X_list)
    y = np.array(y_list)
    idx = pd.Index(idx_list)
    return X, y, idx


# --------------------------------------------------
# Models
# --------------------------------------------------
def train_linear_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_rf_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Random Forest as a robust non-linear model."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def ensemble_predict(
    models: Dict[str, Any],
    X_last: np.ndarray,
) -> float:
    """Weighted ensemble of available models."""
    preds = []
    weights = []

    if "linear" in models and models["linear"] is not None:
        preds.append(float(models["linear"].predict(X_last)[0]))
        weights.append(1.0)

    if "rf" in models and models["rf"] is not None:
        preds.append(float(models["rf"].predict(X_last)[0]))
        weights.append(1.3)

    if not preds:
        return 0.0

    w = np.array(weights)
    p = np.array(preds)
    return float(np.sum(w * p) / np.sum(w))


def train_all_ml_models(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    lookback: int = 20,
) -> Dict[str, Any]:
    """
    Train a small ensemble of fast models.

    Returns dict with:
      - 'linear': LinearRegression
      - 'rf': RandomForestRegressor
      - 'lookback', 'n_features'
    """
    X, y, idx = prepare_ml_dataset(df_price, features, lookback)
    if X.size == 0:
        return {}

    # Limit history for speed (last ~1500 samples is plenty)
    max_samples = 1500
    if X.shape[0] > max_samples:
        X = X[-max_samples:]
        y = y[-max_samples:]

    n_features = features.shape[1]

    models: Dict[str, Any] = {}
    models["linear"] = train_linear_model(X, y)
    models["rf"] = train_rf_model(X, y)

    models["lookback"] = lookback
    models["n_features"] = n_features
    return models



def add_ml_prediction_column(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    models: Optional[Dict[str, Any]],
) -> pd.Series:
    """
    For each day (after warm-up), predict next-day price using ensemble.

    Returns a Series aligned to df_price.index with name 'PredictedPriceNextML'.
    """
    price = df_price["Adj Close"]
    result = pd.Series(index=df_price.index, dtype=float, name="PredictedPriceNextML")

    if not models:
        return result

    lookback = int(models.get("lookback", 20))
    n_features = int(models.get("n_features", features.shape[1]))

    # Align features with price data more carefully
    common_index = features.index.intersection(price.index)
    if len(common_index) == 0:
        return result
        
    feats = features.loc[common_index].copy()
    price_aligned = price.loc[common_index]

    if len(feats) <= lookback:
        return result

    # Generate predictions for all valid windows
    prediction_count = 0
    for i in range(lookback, len(feats) - 1):
        try:
            window_feats = feats.iloc[i - lookback : i].values
            X_last = window_feats.flatten().reshape(1, -1)

            pred_ret = ensemble_predict(models, X_last)
            current_dt = feats.index[i]
            
            # Predict the following day's price from today's price and predicted return
            # Use the current price and add the predicted return
            if current_dt in price_aligned.index:
                current_price = price_aligned.loc[current_dt]
                predicted_price = current_price * np.exp(pred_ret)
                result.loc[current_dt] = float(predicted_price)
                prediction_count += 1
                
        except Exception as e:
            # Skip problematic windows but continue processing
            continue

    print(f"Generated {prediction_count} ML predictions")
    return result
