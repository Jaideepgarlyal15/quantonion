#!/usr/bin/env python3
"""
Test script to verify ML prediction functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from core.data_loader import load_price_data
from core.features import build_features
from core.ml import train_all_ml_models, add_ml_prediction_column

def test_ml_functionality():
    print("Testing ML prediction functionality...")
    
    # Load sample data
    try:
        df_prices, data_src = load_price_data("AAPL", "2023-01-01", "2024-01-01", allow_synth=True)
        print(f"Loaded data from: {data_src}")
        print(f"Data shape: {df_prices.shape}")
        print(f"Date range: {df_prices.index[0]} to {df_prices.index[-1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Build features
    try:

        features = build_features(df_prices, vol_window=21)
        print(f"Features shape: {features.shape}")
        print(f"Features columns: {list(features.columns)}")
    except Exception as e:
        print(f"Error building features: {e}")
        return False
    
    # Train ML models
    try:
        print("Training ML models...")
        ml_models = train_all_ml_models(df_prices, features, lookback=20)
        print(f"Models trained: {list(ml_models.keys())}")
        
        if not ml_models:
            print("No models were trained successfully")
            return False
            
    except Exception as e:
        print(f"Error training models: {e}")
        return False
    
    # Generate predictions
    try:
        print("Generating ML predictions...")
        predictions = add_ml_prediction_column(df_prices, features, ml_models)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Non-null predictions: {predictions.notna().sum()}")
        
        if predictions.notna().sum() == 0:
            print("No predictions were generated")
            return False
            
        print("Sample predictions:")
        print(predictions.head(10))
        print(predictions.tail(10))
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return False
    
    print("ML functionality test completed successfully!")
    return True

if __name__ == "__main__":
    test_ml_functionality()
