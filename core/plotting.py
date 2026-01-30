"""
Plotting utilities for the Regime Switching Dashboard.

Provides interactive Plotly charts for:
- Price with regime shading
- Regime timeline
- Confidence series
- Posterior probabilities
- ML forecast comparisons with future markers
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta


# Color palette for regimes
SIMPLE_COLORS = {
    "Calm": "#2E8B57",           # Sea Green
    "Super Calm": "#228B22",      # Forest Green  
    "Choppy": "#FF8C00",          # Dark Orange
    "Stormy": "#DC143C",          # Crimson
    "Unknown": "#696969"          # Dim Gray
}

# Forecast colors and markers
FORECAST_CONFIG = {
    3: {"color": "#3498db", "symbol": "diamond", "size": 12, "name": "3-Day"},
    14: {"color": "#9b59b6", "symbol": "star", "size": 14, "name": "14-Day"},
    90: {"color": "#e74c3c", "symbol": "hexagon", "size": 16, "name": "3-Month"}
}


def price_with_regimes(price, segments, ml_series=None, forecasts=None):
    """
    Create price chart with regime shading, ML predictions, and future forecasts.
    
    Args:
        price: Historical price series
        segments: Regime shading segments
        ml_series: ML next-day predictions (optional)
        forecasts: Dict of horizon -> forecast dict (optional)
    """
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=price.index, 
        y=price, 
        name="Adj Close", 
        mode="lines",
        line=dict(color="#2c3e50", width=1.5),
        hovertemplate="$%{y:.2f}<extra>Price</extra>"
    ))

    # ML predictions (next-day predictions for historical dates)
    if ml_series is not None:
        ml_clean = ml_series.dropna()
        if len(ml_clean) > 0:
            fig.add_trace(go.Scatter(
                x=ml_clean.index,
                y=ml_clean,
                mode="lines",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                name="ML Prediction",
                opacity=0.8,
                hovertemplate="ML: $%{y:.2f}<extra>Prediction</extra>"
            ))

    # Future forecast markers
    if forecasts:
        last_date = price.index[-1]
        last_price = float(price.iloc[-1])
        
        # Create future dates for each horizon
        for horizon in sorted(forecasts.keys()):
            f = forecasts[horizon]
            config = FORECAST_CONFIG.get(horizon, {})
            
            forecast_date = last_date + timedelta(days=horizon)
            pred_price = f["predicted_price"]
            conf_lower = f["confidence_lower"]
            conf_upper = f["confidence_upper"]
            
            # Calculate returns for display
            price_return = (pred_price - last_price) / last_price * 100
            conf_lower_return = (conf_lower - last_price) / last_price * 100
            conf_upper_return = (conf_upper - last_price) / last_price * 100
            
            # Add confidence interval bar
            fig.add_trace(go.Scatter(
                x=[forecast_date, forecast_date],
                y=[conf_lower, conf_upper],
                mode="lines",
                line=dict(color=config.get("color", "#95a5a6"), width=2),
                showlegend=False,
                hoverinfo="skip"
            ))
            
            # Add forecast marker
            fig.add_trace(go.Scatter(
                x=[forecast_date],
                y=[pred_price],
                mode="markers",
                marker=dict(
                    symbol=config.get("symbol", "circle"),
                    size=config.get("size", 12),
                    color=config.get("color", "#95a5a6"),
                    line=dict(color="white", width=2)
                ),
                name=f"{config.get('name', str(horizon) + '-Day')} Forecast",
                hovertemplate=(
                    f"<b>{config.get('name', str(horizon) + '-Day')} Forecast</b><br>"
                    f"Date: {forecast_date.strftime('%Y-%m-%d')}<br>"
                    f"Price: ${pred_price:.2f}<br>"
                    f"Return: {price_return:+.1f}%<br>"
                    f"95% CI: ${conf_lower:.2f} - ${conf_upper:.2f}<extra></extra>"
                )
            ))
            
            # Add annotation line from last price
            fig.add_shape(
                type="line",
                x0=last_date,
                y0=last_price,
                x1=forecast_date,
                y1=pred_price,
                line=dict(
                    color=config.get("color", "#95a5a6"),
                    width=1,
                    dash="dot"
                ),
                opacity=0.5,
                layer="below"
            )

    # Regime shading
    for (s0, s1, label, color) in segments:
        fig.add_vrect(
            x0=s0, x1=s1, 
            fillcolor=color, 
            opacity=0.1, 
            line_width=0,
            annotation_text=label,
            annotation_position="top left"
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        height=500
    )
    return fig


def confidence_plot(conf):
    """Plot regime confidence over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=conf.index, 
        y=conf.values, 
        mode="lines",
        fill="tozeroy",
        line=dict(color="#3498db", width=1.5),
        name="Confidence",
        hovertemplate="Date: %{x}<br>Confidence: %{y:.0%}<extra></extra>"
    ))
    fig.update_yaxes(range=[0, 1], title="Confidence", tickformat=".0%")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified"
    )
    return fig


def regime_timeline(segments):
    """Create a timeline showing regime durations."""
    fig = go.Figure()

    for (s0, s1, label, color) in segments:
        fig.add_shape(
            type="rect",
            x0=s0,
            x1=s1,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.35,
            line_width=0,
        )
        
        # Add label for longer segments
        if (s1 - s0) > timedelta(days=30):
            mid_date = s0 + (s1 - s0) / 2
            fig.add_annotation(
                x=mid_date,
                y=0.5,
                text=label,
                showarrow=False,
                font=dict(size=10, color="black")
            )

    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_xaxes(type="date")
    fig.update_layout(height=100, margin=dict(l=20, r=20, t=10, b=20))
    return fig


def plot_regime_timeline(dates, states, labels):
    """Create timeline plot for regime segments."""
    segments = []
    colors = ["#DC143C", "#FF8C00", "#2E8B57", "#228B22"]  # Stormy, Choppy, Calm, Super Calm
    
    if len(states) == 0:
        return regime_timeline([])
    
    # Convert states to integers
    int_states = []
    for state in states:
        try:
            int_states.append(int(state))
        except (ValueError, TypeError):
            int_states.append(0)
    
    current_state = int_states[0]
    start_idx = 0
    
    for i, state in enumerate(int_states):
        if state != current_state or i == len(int_states) - 1:
            if i == len(int_states) - 1:
                i += 1
            
            start_date = dates[start_idx]
            end_date = dates[i-1]
            
            orig_state = states[start_idx]
            try:
                orig_state_int = int(orig_state)
                label = labels.get(orig_state_int, "State " + str(orig_state_int))
            except (ValueError, TypeError):
                label = labels.get(orig_state, "State " + str(orig_state))
            
            color_index = current_state % len(colors)
            color = colors[color_index]
            
            segments.append((start_date, end_date, label, color))
            
            current_state = state
            start_idx = i
    
    return regime_timeline(segments)


def plot_price_with_regimes(df, segments, enable_ml=False, forecasts=None):
    """
    Create price plot with regime shading.
    
    Args:
        df: DataFrame with 'Adj Close' and 'PredictedPriceNextML'
        segments: Regime segments
        enable_ml: Whether to show ML predictions
        forecasts: Dict of horizon -> forecast info
    """
    price = df['Adj Close']
    
    # ML series extraction
    ml_series = None
    if enable_ml and 'PredictedPriceNextML' in df.columns:
        ml_series = df['PredictedPriceNextML']
        if ml_series.notna().sum() == 0:
            ml_series = None
    
    # Convert segments format
    plot_segments = []
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
    for segment in segments:
        if segment is None:
            continue
            
        if len(segment) == 4:  # (start, end, label, color)
            plot_segments.append(segment)
        elif len(segment) == 3:  # (start, end, label)
            start, end, label = segment
            label_color_map = {
                "Calm": colors[0],
                "Super Calm": colors[3], 
                "Choppy": colors[1],
                "Stormy": colors[2],
                "Unknown": colors[0]
            }
            color = label_color_map.get(label, colors[0])
            plot_segments.append((start, end, label, color))
        else:  # (start, end, state_idx)
            start, end, state_idx = segment
            label = "State " + str(state_idx)
            try:
                if hasattr(state_idx, 'item'):
                    state_idx = state_idx.item()
                state_int = int(state_idx)
                color_index = state_int % len(colors)
                color = colors[color_index]
            except:
                color = colors[0]
            plot_segments.append((start, end, label, color))
    
    return price_with_regimes(price, plot_segments, ml_series, forecasts)


def plot_confidence_series(conf_series):
    """Plot confidence series."""
    return confidence_plot(conf_series)


def make_regime_scatter(df, state_clean, labels_adv):
    """Create scatter plot colored by regime."""
    fig = go.Figure()
    
    unique_states = sorted(state_clean.unique())
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
    for i, state in enumerate(unique_states):
        mask = state_clean == state
        dates = state_clean.index[mask]
        prices = df.loc[dates, 'Adj Close']
        
        try:
            if hasattr(state, 'item'):
                state_int = int(state.item())
            else:
                state_int = int(state)
        except:
            state_int = 0
        
        label = labels_adv.get(state_int)
        if label is None:
            try:
                label = labels_adv.get(state)
            except:
                label = "State " + str(state_int)
        if label is None:
            label = "State " + str(state_int)
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='markers',
            name=label,
            marker=dict(color=color, size=4),
            opacity=0.7,
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="closest"
    )
    
    return fig


def make_posterior_probs(df, labels_adv, K):
    """Create posterior probability plots."""
    fig = go.Figure()
    
    prob_cols = ['p_state_' + str(i) for i in range(K)]
    colors = ["#DC143C", "#FF8C00", "#2E8B57", "#228B22"]
    
    for i in range(K):
        if prob_cols[i] in df.columns:
            prob_data = df[prob_cols[i]]
            label = labels_adv.get(i, "State " + str(i))
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=prob_data.index,
                y=prob_data.values,
                mode='lines',
                name=label,
                line=dict(color=color, width=2),
                fill='tozeroy' if i == 0 else None,
                fillcolor=color,
                opacity=0.2,
                hovertemplate="Date: %{x}<br>Probability: %{y:.0%}<extra></extra>"
            ))
    
    fig.update_yaxes(range=[0, 1], title="Probability", tickformat=".0%")
    fig.update_layout(
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig


def plot_forecast_comparison(current_price, forecasts):
    """
    Create a bar chart comparing forecasts across time horizons.
    
    Args:
        current_price: Current price value
        forecasts: Dict of horizon -> forecast dict
    
    Returns:
        Plotly figure
    """
    horizons = sorted(forecasts.keys())
    labels = {3: "3-Day", 14: "14-Day", 90: "3-Month"}
    
    pred_prices = [forecasts[h]["predicted_price"] for h in horizons]
    conf_lower = [forecasts[h]["confidence_lower"] for h in horizons]
    conf_upper = [forecasts[h]["confidence_upper"] for h in horizons]
    
    # Calculate errors for error bars
    lower_err = [current_price - conf_lower[i] for i in range(len(horizons))]
    upper_err = [conf_upper[i] - current_price for i in range(len(horizons))]
    
    fig = go.Figure()
    
    # Current price line
    fig.add_hline(
        y=current_price, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Current",
        annotation_position="bottom right"
    )
    
    # Forecast bars
    colors = [FORECAST_CONFIG.get(h, {}).get("color", "#95a5a6") for h in horizons]
    
    fig.add_trace(go.Bar(
        x=[labels[h] for h in horizons],
        y=pred_prices,
        marker_color=colors,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[upper_err[i] - (pred_prices[i] - current_price) for i in range(len(horizons))],
            arrayminus=[(pred_prices[i] - current_price) - lower_err[i] for i in range(len(horizons))],
            color='#34495e',
            thickness=1.5,
            width=5
        ),
        text=[f"${p:.2f}" for p in pred_prices],
        textposition='outside',
        name="Predicted Price",
        hovertemplate="Horizon: %{x}<br>Price: $%{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Price Forecasts by Horizon",
        yaxis_title="Price ($)",
        xaxis_title="Forecast Horizon",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    
    return fig

