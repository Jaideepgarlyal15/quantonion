"""
QuantOnion Plotting Module

Plotly chart builders for regime analysis and backtesting visualisation.

All functions return go.Figure objects for use with st.plotly_chart().

Changes from original:
  - SIMPLE_COLORS is now a single source of truth (imported from hmm_model)
  - Fixed error bar calculation in forecast comparison chart
  - Added plot_equity_curves() and plot_drawdown_chart() for backtesting
  - Removed duplicate colour definitions
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

# Single source of truth for regime colours — defined in hmm_model.py
from core.hmm_model import SIMPLE_COLORS

# Forecast marker configuration
_FORECAST_CONFIG: Dict[int, Dict] = {
    3:  {"color": "#3498db", "symbol": "diamond",  "size": 12, "name": "3-Day"},
    14: {"color": "#9b59b6", "symbol": "star",      "size": 14, "name": "14-Day"},
    90: {"color": "#e74c3c", "symbol": "hexagon",   "size": 16, "name": "3-Month"},
}

# Colour sequence for multi-strategy charts
_STRATEGY_COLORS = [
    "#2c3e50", "#e74c3c", "#3498db", "#2ecc71",
    "#9b59b6", "#f39c12", "#1abc9c", "#e67e22",
]


# ── Regime charts ─────────────────────────────────────────────────────────────

def _build_regime_segments_with_color(segments):
    """
    Normalise segment tuples to (start, end, label, color) format.
    Input segments may be 3-tuples (start, end, label) or 4-tuples.
    """
    out = []
    for seg in segments:
        if seg is None:
            continue
        if len(seg) == 4:
            out.append(seg)
        elif len(seg) == 3:
            start, end, label = seg
            color = SIMPLE_COLORS.get(label, SIMPLE_COLORS.get("Unknown", "#696969"))
            out.append((start, end, label, color))
    return out


def plot_regime_timeline(dates, states, labels) -> go.Figure:
    """
    Compact horizontal timeline showing regime periods as coloured bands.

    Args:
        dates:  List of datetime-like dates.
        states: List of integer HMM state values.
        labels: Dict mapping integer state → regime label string.

    Returns:
        Plotly Figure with height=100.
    """
    if not states:
        fig = go.Figure()
        fig.update_layout(height=100, margin=dict(l=20, r=20, t=10, b=20))
        return fig

    segments = []
    current_state = states[0]
    start_idx = 0

    for i, state in enumerate(states):
        is_last = i == len(states) - 1
        if state != current_state or is_last:
            end_i = i if not is_last else i
            label = labels.get(current_state, f"State {current_state}")
            color = SIMPLE_COLORS.get(label, "#696969")
            segments.append((dates[start_idx], dates[end_i], label, color))
            current_state = state
            start_idx = i

    fig = go.Figure()
    for s0, s1, label, color in segments:
        fig.add_shape(
            type="rect",
            x0=s0, x1=s1, y0=0, y1=1,
            fillcolor=color, opacity=0.40, line_width=0,
        )
        if (s1 - s0) > timedelta(days=30):
            mid = s0 + (s1 - s0) / 2
            fig.add_annotation(
                x=mid, y=0.5, text=label, showarrow=False,
                font=dict(size=10, color="black"),
            )

    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_xaxes(type="date")
    fig.update_layout(height=100, margin=dict(l=20, r=20, t=10, b=20))
    return fig


def plot_price_with_regimes(
    df: pd.DataFrame,
    segments,
    enable_ml: bool = False,
    forecasts: dict | None = None,
) -> go.Figure:
    """
    Price chart with regime background shading, optional ML prediction overlay,
    and future forecast markers.

    Args:
        df:         DataFrame with 'Adj Close' and optionally 'PredictedPriceNextML'.
        segments:   Regime segments as list of (start, end, label) or (start, end, label, color).
        enable_ml:  Whether to overlay the ML next-day prediction series.
        forecasts:  Dict mapping horizon days → forecast dict (from ml.py).
    """
    price = df["Adj Close"]
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=price.index, y=price,
        name="Adj Close", mode="lines",
        line=dict(color="#2c3e50", width=1.5),
        hovertemplate="$%{y:.2f}<extra>Price</extra>",
    ))

    # ML next-day prediction overlay
    if enable_ml and "PredictedPriceNextML" in df.columns:
        ml_series = df["PredictedPriceNextML"].dropna()
        if len(ml_series) > 0:
            fig.add_trace(go.Scatter(
                x=ml_series.index, y=ml_series,
                mode="lines", name="ML Next-Day",
                line=dict(color="#e74c3c", width=1.5, dash="dash"),
                opacity=0.75,
                hovertemplate="ML: $%{y:.2f}<extra>Next-Day Prediction</extra>",
            ))

    # Future forecast markers
    if forecasts:
        last_date = price.index[-1]
        last_price = float(price.iloc[-1])

        for horizon in sorted(forecasts):
            f = forecasts[horizon]
            cfg = _FORECAST_CONFIG.get(horizon, {})
            forecast_date = last_date + timedelta(days=horizon)
            pred = f["predicted_price"]
            lo = f["confidence_lower"]
            hi = f["confidence_upper"]
            ret_pct = (pred - last_price) / last_price * 100

            # Confidence interval line
            fig.add_trace(go.Scatter(
                x=[forecast_date, forecast_date], y=[lo, hi],
                mode="lines", showlegend=False,
                line=dict(color=cfg.get("color", "#95a5a6"), width=2),
                hoverinfo="skip",
            ))
            # Forecast marker
            fig.add_trace(go.Scatter(
                x=[forecast_date], y=[pred],
                mode="markers",
                name=f"{cfg.get('name', str(horizon)+'-Day')} Forecast",
                marker=dict(
                    symbol=cfg.get("symbol", "circle"),
                    size=cfg.get("size", 12),
                    color=cfg.get("color", "#95a5a6"),
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    f"<b>{cfg.get('name', str(horizon)+'-Day')} Forecast</b><br>"
                    f"Date: {forecast_date.strftime('%Y-%m-%d')}<br>"
                    f"Price: ${pred:.2f}<br>"
                    f"Return: {ret_pct:+.1f}%<br>"
                    f"95% CI: ${lo:.2f} – ${hi:.2f}<extra></extra>"
                ),
            ))
            # Dotted projection line
            fig.add_shape(
                type="line",
                x0=last_date, y0=last_price,
                x1=forecast_date, y1=pred,
                line=dict(color=cfg.get("color", "#95a5a6"), width=1, dash="dot"),
                opacity=0.5, layer="below",
            )

    # Regime background shading
    for s0, s1, label, color in _build_regime_segments_with_color(segments):
        fig.add_vrect(
            x0=s0, x1=s1,
            fillcolor=color, opacity=0.10, line_width=0,
            annotation_text=label, annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified", height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_confidence_series(conf: pd.Series) -> go.Figure:
    """Regime model confidence over time (0–1 scale)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=conf.index, y=conf.values,
        mode="lines", fill="tozeroy",
        line=dict(color="#3498db", width=1.5),
        name="HMM Confidence",
        hovertemplate="Date: %{x}<br>Confidence: %{y:.0%}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 1], title="Confidence", tickformat=".0%")
    fig.update_layout(
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
    )
    return fig


def make_regime_scatter(df: pd.DataFrame, state_clean: pd.Series, labels_adv: dict) -> go.Figure:
    """Price coloured by regime state (scatter plot)."""
    fig = go.Figure()
    colors = list(SIMPLE_COLORS.values())

    for i, state in enumerate(sorted(state_clean.unique())):
        mask = state_clean == state
        dates = state_clean.index[mask]
        prices = df.loc[dates, "Adj Close"]
        label = labels_adv.get(int(state), f"State {state}")

        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode="markers", name=label,
            marker=dict(color=colors[i % len(colors)], size=4),
            opacity=0.7,
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price ($)",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="closest",
    )
    return fig


def make_posterior_probs(df: pd.DataFrame, labels_adv: dict, K: int) -> go.Figure:
    """Posterior regime probability series for each HMM state."""
    fig = go.Figure()
    colors = list(SIMPLE_COLORS.values())

    for i in range(K):
        col = f"p_state_{i}"
        if col not in df.columns:
            continue
        label = labels_adv.get(i, f"State {i}")
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col].values,
            mode="lines", name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate="Date: %{x}<br>P(%{fullData.name}): %{y:.0%}<extra></extra>",
        ))

    fig.update_yaxes(range=[0, 1], title="Posterior Probability", tickformat=".0%")
    fig.update_layout(
        xaxis_title="Date",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def plot_forecast_comparison(current_price: float, forecasts: dict) -> go.Figure:
    """
    Bar chart comparing predicted prices across forecast horizons.

    Error bars show the 95% confidence interval width (not relative to current price).
    """
    horizons = sorted(forecasts)
    labels = {3: "3-Day", 14: "14-Day", 90: "3-Month"}

    pred_prices = [forecasts[h]["predicted_price"] for h in horizons]
    conf_lower = [forecasts[h]["confidence_lower"] for h in horizons]
    conf_upper = [forecasts[h]["confidence_upper"] for h in horizons]

    # Symmetric error bars relative to predicted price
    err_below = [pred_prices[i] - conf_lower[i] for i in range(len(horizons))]
    err_above = [conf_upper[i] - pred_prices[i] for i in range(len(horizons))]

    bar_colors = [_FORECAST_CONFIG.get(h, {}).get("color", "#95a5a6") for h in horizons]

    fig = go.Figure()
    fig.add_hline(
        y=current_price, line_dash="dash", line_color="gray",
        annotation_text="Current Price", annotation_position="bottom right",
    )
    fig.add_trace(go.Bar(
        x=[labels.get(h, f"{h}-Day") for h in horizons],
        y=pred_prices,
        marker_color=bar_colors,
        error_y=dict(
            type="data", symmetric=False,
            array=err_above, arrayminus=err_below,
            color="#34495e", thickness=1.5, width=6,
        ),
        text=[f"${p:.2f}" for p in pred_prices],
        textposition="outside",
        hovertemplate="Horizon: %{x}<br>Forecast: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="ML Price Forecasts by Horizon (95% CI)",
        yaxis_title="Price ($)", xaxis_title="Forecast Horizon",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False, hovermode="x unified",
    )
    return fig


# ── Backtest charts ───────────────────────────────────────────────────────────

def plot_equity_curves(all_results: dict, title: str = "Strategy Equity Curves") -> go.Figure:
    """
    Multi-line equity curve chart comparing all backtested strategies.

    Args:
        all_results: Dict mapping strategy name → {result: DataFrame, metrics: dict}.
        title:       Chart title.
    """
    fig = go.Figure()

    for i, (name, res) in enumerate(all_results.items()):
        result_df = res.get("result")
        if result_df is None or result_df.empty:
            continue
        equity = result_df["equity_curve"]
        color = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]

        # Highlight Buy & Hold benchmark differently
        dash = "solid" if name != "Buy & Hold" else "dot"
        width = 2.0 if name != "Buy & Hold" else 1.5

        fig.add_trace(go.Scatter(
            x=equity.index, y=equity,
            name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Equity: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(0,0,0,0.2)")
    fig.update_layout(
        title=title,
        xaxis_title="Date", yaxis_title="Portfolio Value (start = 1.0)",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified", height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_drawdown_chart(all_results: dict) -> go.Figure:
    """
    Drawdown comparison chart for all backtested strategies.

    Shows drawdown from equity peak for each strategy over time.
    Deeper (more negative) troughs represent worse drawdowns.
    """
    fig = go.Figure()

    for i, (name, res) in enumerate(all_results.items()):
        result_df = res.get("result")
        if result_df is None or result_df.empty:
            continue
        drawdown = result_df["drawdown"]
        color = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
        dash = "solid" if name != "Buy & Hold" else "dot"

        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown * 100,
            name=name, mode="lines",
            line=dict(color=color, width=1.5, dash=dash),
            fill="tozeroy" if i == 0 else None,
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (0.05,)}",
            hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Drawdown from Equity Peak",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified", height=350,
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig
