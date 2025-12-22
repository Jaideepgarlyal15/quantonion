

# core/plotting.py

import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Color palette for regimes
SIMPLE_COLORS = {
    "Calm": "#2E8B57",           # Sea Green
    "Super Calm": "#228B22",      # Forest Green  
    "Choppy": "#FF8C00",          # Dark Orange
    "Stormy": "#DC143C",          # Crimson
    "Unknown": "#696969"          # Dim Gray
}



def price_with_regimes(price, segments, ml_series=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, name="Adj Close", mode="lines"))

    if ml_series is not None:
        print(f"Adding ML prediction trace with {len(ml_series.dropna())} non-NaN values")
        print(f"ML series index range: {ml_series.index.min()} to {ml_series.index.max()}")
        print(f"ML series value range: {ml_series.min()} to {ml_series.max()}")
        
        fig.add_trace(
            go.Scatter(
                x=ml_series.index,
                y=ml_series,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="ML Prediction",
                opacity=0.8,
            )
        )
    else:
        print("ML series is None - not adding ML prediction trace")

    for (s0, s1, label, color) in segments:
        fig.add_vrect(x0=s0, x1=s1, fillcolor=color, opacity=0.1, line_width=0)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def confidence_plot(conf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conf.index, y=conf.values, mode="lines"))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title="Date")
    return fig


def regime_timeline(segments):
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

    fig.update_yaxes(visible=False)
    fig.update_xaxes(type="date")
    fig.update_layout(height=120)
    return fig


# New functions that app.py expects
def plot_regime_timeline(dates, states, labels):
    """Create timeline plot for regime segments"""
    segments = []
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
    if len(states) == 0:
        return regime_timeline([])
    
    # Convert states to integers first to avoid any type issues
    int_states = []
    for state in states:
        try:
            int_states.append(int(state))
        except (ValueError, TypeError):
            int_states.append(0)  # Default to 0 if can't convert
    
    current_state = int_states[0]
    start_idx = 0
    
    for i, state in enumerate(int_states):
        if state != current_state or i == len(int_states) - 1:
            if i == len(int_states) - 1:
                i += 1
            
            # Create segment
            start_date = dates[start_idx]
            end_date = dates[i-1]
            
            # Use original state value for label lookup, not the int version
            orig_state = states[start_idx]
            # Convert to regular Python int to avoid numpy formatting issues
            try:
                orig_state_int = int(orig_state)
                label = labels.get(orig_state_int, "State " + str(orig_state_int))
            except (ValueError, TypeError):
                label = labels.get(orig_state, "State " + str(orig_state))
            
            # Use integer state for color selection
            color_index = current_state % len(colors)
            color = colors[color_index]
            
            segments.append((start_date, end_date, label, color))
            
            current_state = state
            start_idx = i
    
    return regime_timeline(segments)




def plot_price_with_regimes(df, segments, enable_pro_ml=False):
    """Create price plot with regime shading"""
    # Extract price data and ML series if available
    price = df['Adj Close']
    
    # Enhanced ML series extraction with better debugging
    ml_series = None
    if 'PredictedPriceNextML' in df.columns:
        ml_series = df['PredictedPriceNextML']
        # Check if we have any non-NaN values
        non_nan_count = ml_series.notna().sum()
        if non_nan_count == 0:
            print("ML predictions found but all are NaN - hiding ML line")
            ml_series = None
        else:
            print(f"Found {non_nan_count} ML predictions to plot")
            # Debug: print some sample values
            print(f"ML series sample values: {ml_series.dropna().head()}")
    elif enable_pro_ml:
        print("Pro ML enabled but no 'PredictedPriceNextML' column found")
    
    # Convert segments format if needed
    plot_segments = []
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
    # Filter out None values and process segments
    for segment in segments:
        if segment is None:
            continue
            
        if len(segment) == 4:  # (start, end, label, color)
            plot_segments.append(segment)
        elif len(segment) == 3:  # (start, end, label) from regime_segments function
            start, end, label = segment
            # For 3-element tuple, we don't have state index, so use label for color
            # Map common labels to colors
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
            # Use a simple color assignment based on state index
            try:
                # Try to get state index as integer
                if hasattr(state_idx, 'item'):  # numpy scalar
                    state_idx = state_idx.item()
                state_int = int(state_idx)
                color_index = state_int % len(colors)
                color = colors[color_index]
            except (ValueError, TypeError, AttributeError):
                # Fallback - use first color
                color = colors[0]
            plot_segments.append((start, end, label, color))
    
    return price_with_regimes(price, plot_segments, ml_series)


def plot_confidence_series(conf_series):
    """Plot confidence series"""
    return confidence_plot(conf_series)






def make_regime_scatter(df, state_clean, labels_adv):
    """Create scatter plot colored by regime"""
    fig = go.Figure()
    
    # Group by regime for different colors
    unique_states = state_clean.unique()
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
    for i, state in enumerate(sorted(unique_states)):
        mask = state_clean == state
        dates = state_clean.index[mask]
        prices = df.loc[dates, 'Adj Close']
        
        # Convert state to int to avoid numpy formatting issues
        try:
            if hasattr(state, 'item'):
                state_int = int(state.item())
            else:
                state_int = int(state)
        except (ValueError, TypeError, AttributeError):
            state_int = 0
        
        # Get label safely - try multiple approaches
        label = None
        
        # First try with integer key
        try:
            label = labels_adv.get(state_int)
        except (TypeError, KeyError):
            pass
        
        # If not found, try with original state value
        if label is None:
            try:
                label = labels_adv.get(state)
            except (TypeError, KeyError):
                pass
        
        # If still not found, create fallback label
        if label is None:
            label = "State " + str(state_int)
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='markers',
            name=label,
            marker=dict(color=color, size=4),
            opacity=0.7
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig


def make_posterior_probs(df, labels_adv, K):
    """Create posterior probability plots"""
    fig = go.Figure()
    
    # Get probability columns

    prob_cols = ['p_state_' + str(i) for i in range(K)]
    colors = ["#2E8B57", "#FF8C00", "#DC143C", "#228B22"]
    
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
                line=dict(color=color, width=2)
            ))
    
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Probability",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig
