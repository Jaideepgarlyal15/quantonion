#!/usr/bin/env python3
"""
Debug script to test ML prediction functionality
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import our modules
from core.plotting import plot_price_with_regimes
from core.ml import train_all_ml_models, add_ml_prediction_column
from core.features import build_features
from core.data_loader import load_price_data

def test_ml_functionality():
    """Test if ML predictions are being generated correctly"""
    
    # Load some sample data
    print("Loading sample data...")
    df_prices, data_src = load_price_data("^GSPC", 
                                         pd.to_datetime("2023-01-01"), 
                                         pd.to_datetime("2024-01-01"))
    
    if df_prices.empty:
        print("No data loaded")
        return
    
    print(f"Loaded {len(df_prices)} price records")
    

    # Build features
    print("Building features...")
    features = build_features(df_prices, vol_window=21)
    print(f"Built {len(features)} feature records")
    
    # Train ML models
    print("Training ML models...")
    ml_models = train_all_ml_models(df_prices, features, lookback=20)
    print("ML models trained successfully")
    
    # Add ML predictions
    print("Adding ML predictions...")
    ml_predictions = add_ml_prediction_column(df_prices, features, ml_models)
    print(f"Generated {ml_predictions.notna().sum()} ML predictions")
    
    # Create test dataframe
    df_test = df_prices.copy()
    df_test['PredictedPriceNextML'] = ml_predictions
    
    # Test the plotting function
    print("Testing plot_price_with_regimes function...")
    fig = plot_price_with_regimes(df_test, [], enable_pro_ml=True)
    
    # Check if the figure has the ML trace
    print(f"Figure has {len(fig.data)} traces")
    for i, trace in enumerate(fig.data):
        print(f"Trace {i}: {trace.name}")
    
    return ml_predictions

if __name__ == "__main__":
    test_ml_functionality()
