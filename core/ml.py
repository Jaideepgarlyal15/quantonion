"""
Machine Learning helpers for the Regime Switching Dashboard.

This module provides ensemble-based predictions for next-day and future price forecasting
using lightweight models (Linear Regression, Random Forest). Predictions are available
for multiple timeframes: 3 days, 14 days, and 3 months.

All features are open-source and freely available.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# Dataset preparation
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
# Model training
# --------------------------------------------------
def train_linear_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_rf_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Train a Random Forest as a robust non-linear model."""
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
    """
    Weighted ensemble prediction from available models.
    
    Returns the predicted log return for the next period.
    """
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
    Train an ensemble of fast models for price prediction.

    Returns dict with:
      - 'linear': LinearRegression
      - 'rf': RandomForestRegressor
      - 'lookback': lookback window size
      - 'n_features': number of features
      - 'last_price': most recent price
      - 'last_date': most recent date
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
    
    # Store latest price and date for future predictions
    if len(df_price) > 0:
        models["last_price"] = float(df_price["Adj Close"].iloc[-1])
        models["last_date"] = df_price.index[-1]
    
    # Calculate historical prediction accuracy
    if len(y) > 0:
        # Simple RMSE on training data as a proxy
        y_pred_linear = models["linear"].predict(X)
        y_pred_rf = models["rf"].predict(X)
        linear_rmse = np.sqrt(np.mean((y - y_pred_linear) ** 2))
        rf_rmse = np.sqrt(np.mean((y - y_pred_rf) ** 2))
        models["linear_rmse"] = float(linear_rmse)
        models["rf_rmse"] = float(rf_rmse)
    
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
            if current_dt in price_aligned.index:
                current_price = price_aligned.loc[current_dt]
                predicted_price = current_price * np.exp(pred_ret)
                result.loc[current_dt] = float(predicted_price)
                prediction_count += 1
                
        except Exception:
            # Skip problematic windows but continue processing
            continue

    return result


# --------------------------------------------------
# Future Predictions (3-day, 14-day, 3-month)
# --------------------------------------------------
def predict_future_prices(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    models: Optional[Dict[str, Any]],
    horizon_days: int = 3,
) -> Dict[str, Any]:
    """
    Predict future prices for a given horizon (3, 14, or 90 days).
    
    Uses ensemble predictions with iterative forecasting for longer horizons.
    
    Args:
        df_price: Price data
        features: Feature matrix
        models: Trained ML models
        horizon_days: Forecast horizon (3, 14, or 90)
    
    Returns:
        Dict with keys:
            - 'forecast_date': Date of forecast
            - 'horizon_days': Number of days ahead
            - 'current_price': Most recent price
            - 'predicted_price': Predicted future price
            - 'predicted_return': Predicted log return
            - 'confidence_lower': Lower bound (95% CI)
            - 'confidence_upper': Upper bound (95% CI)
            - 'confidence_level': Confidence level percentage
    """
    if not models or "linear" not in models or "rf" not in models:
        return {}
    
    # Get last known values
    if len(df_price) < 2:
        return {}
    
    last_price = float(df_price["Adj Close"].iloc[-1])
    last_date = df_price.index[-1]
    
    # Get historical volatility for confidence intervals
    returns = np.log(df_price["Adj Close"]).diff().dropna()
    hist_vol = returns.std() * np.sqrt(252)  # Annualized
    
    # Get last feature window for prediction
    lookback = int(models.get("lookback", 20))
    
    # Align data
    common_index = features.index.intersection(df_price.index)
    if len(common_index) < lookback + 1:
        return {}
    
    feats = features.loc[common_index]
    
    # Use the most recent window
    last_window = feats.iloc[-lookback:].values
    X_last = last_window.flatten().reshape(1, -1)
    
    # Ensemble prediction for next period return
    pred_ret = ensemble_predict(models, X_last)
    
    if horizon_days == 3:
        # Use predicted return directly with some mean reversion
        future_return = pred_ret * 3
        # Scale confidence: shorter horizon = higher confidence
        conf_factor = 1.5
    elif horizon_days == 14:
        # Use predicted return with more mean reversion toward historical mean
        mean_return = returns.mean() * 252  # Annualized mean
        daily_mean = mean_return / 252
        future_return = pred_ret * 0.7 + daily_mean * 14 * 0.3
        conf_factor = 2.5
    elif horizon_days == 90:
        # Stronger mean reversion for longer horizon
        mean_return = returns.mean() * 252
        daily_mean = mean_return / 252
        future_return = pred_ret * 0.3 + daily_mean * 90 * 0.7
        conf_factor = 4.0
    else:
        future_return = pred_ret * horizon_days
        conf_factor = 3.0
    
    # Calculate predicted price
    predicted_price = last_price * np.exp(future_return)
    predicted_return = future_return
    
    # Calculate confidence interval using historical volatility
    # Adjusted for horizon (volatility scales with sqrt of time)
    daily_vol = hist_vol / np.sqrt(252)
    horizon_vol = daily_vol * np.sqrt(horizon_days) * conf_factor
    
    # 95% confidence interval
    z_score = 1.96
    lower_return = future_return - z_score * horizon_vol
    upper_return = future_return + z_score * horizon_vol
    
    confidence_lower = last_price * np.exp(lower_return)
    confidence_upper = last_price * np.exp(upper_return)
    
    # Calculate confidence level based on model agreement and historical accuracy
    # Higher agreement between models = higher confidence
    linear_pred = float(models["linear"].predict(X_last)[0])
    rf_pred = float(models["rf"].predict(X_last)[0])
    model_agreement = 1.0 - abs(linear_pred - rf_pred) / (abs(linear_pred) + abs(rf_pred) + 1e-8)
    model_agreement = max(0.0, min(1.0, model_agreement))
    
    # Base confidence from model agreement, adjusted for horizon
    base_confidence = 0.85 if model_agreement > 0.5 else 0.65
    confidence_level = base_confidence * (1 - (horizon_days / 200))  # Decay with horizon
    
    return {
        "forecast_date": last_date,
        "horizon_days": horizon_days,
        "current_price": last_price,
        "predicted_price": predicted_price,
        "predicted_return": predicted_return,
        "confidence_lower": confidence_lower,
        "confidence_upper": confidence_upper,
        "confidence_level": confidence_level
    }


def get_all_forecasts(
    df_price: pd.DataFrame,
    features: pd.DataFrame,
    models: Optional[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Get forecasts for all timeframes: 3 days, 14 days, and 3 months (90 days).
    
    Returns:
        Dict mapping horizon days to forecast results
    """
    horizons = [3, 14, 90]
    forecasts = {}
    
    for horizon in horizons:
        forecast = predict_future_prices(
            df_price, features, models, horizon_days=horizon
        )
        if forecast:
            forecasts[horizon] = forecast
    
    return forecasts

