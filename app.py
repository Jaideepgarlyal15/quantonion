"""
Regime-Switching Risk Dashboard

An open-source market regime detector using Hidden Markov Models and ML forecasting.
Educational and research purposes only - not investment advice.

Features:
- Gaussian HMM for market regime detection (Stormy, Choppy, Calm, Super Calm)
- ML ensemble predictions for 3-day, 14-day, and 3-month price forecasts
- Portfolio risk analytics (VaR, ES)
- Interactive visualizations with Plotly
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.data_loader import load_price_data, normalize_symbol
from core.features import build_features
from core.hmm_model import (
    fit_hmm,
    label_states,
    expected_durations,
    current_run_length,
    regime_segments,
    SIMPLE_COLORS,
)
from core.ml import (
    train_all_ml_models,
    add_ml_prediction_column,
    get_all_forecasts,
)
from core.plotting import (
    plot_regime_timeline,
    plot_price_with_regimes,
    plot_confidence_series,
    make_regime_scatter,
    make_posterior_probs,
    plot_forecast_comparison,
)
from core.portfolio import (
    portfolio_return_series,
    compute_var_es,
    monte_carlo_paths,
)

# --------------------------------------------------
# Page config and basic styling
# --------------------------------------------------
st.set_page_config(
    page_title="Regime Switching Dashboard", 
    layout="wide",
    page_icon="📊"
)

st.markdown(
    """
<style>
header, footer {visibility: hidden;}
.block-container {padding-top: 0.8rem; padding-bottom: 1.5rem;}
[data-testid="stSidebar"] {background: #181D25;}
</style>
""",
    unsafe_allow_html=True,
)

st.title(" Regime Switching Risk Dashboard")
st.caption(
    "Open-source HMM and ML forecasting for market regime detection. "
    "Educational use only - not investment advice."
)

# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------
with st.sidebar:
    st.header("Options")
    raw_symbol = st.text_input(
        "Ticker (e.g., ^GSPC, AAPL, ASX:BHP, ^AXJO)", value="^GSPC"
    )
    start = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End Date", value=pd.Timestamp.today().normalize())

    n_states_input = st.slider("Number of regimes", 2, 4, 3)
    window_vol = st.number_input("Volatility window (days)", 5, 120, 21)
    random_state = st.number_input("Random seed", 0, 10000, 42)
    simple_view = st.checkbox("Simple view", value=True)

    st.markdown("**Regime labelling**")
    label_metric = st.selectbox(
        "Regime labelling metric", ["mean_return", "sharpe_ratio"], index=0
    )

    auto_refit = st.checkbox("Auto refit on input change", value=False)
    fit_button = st.button("Run / refresh analysis")

    st.markdown("---")
    use_synth = st.checkbox("Allow synthetic fallback (demo)", value=False)

    st.markdown("---")
    st.subheader("ML Forecasting")
    enable_ml = st.checkbox("Enable ML price predictions", value=True)
    train_ml_button = st.button("Train / update ML models")

    st.markdown("---")
    st.subheader("Data override (optional)")
    custom_price_file = st.file_uploader(
        "Custom Price CSV (Date,Adj Close)", type=["csv"]
    )

    st.subheader("Portfolio file (optional)")
    port_file = st.file_uploader("Upload CSV (Ticker,Weight)", type=["csv"])

    st.markdown("---")
    st.markdown("""
    **About this dashboard:**
    
    - Uses Hidden Markov Models to detect market regimes
    - ML ensemble (Linear + Random Forest) for price forecasting
    - Forecasts: 3-day, 14-day, 3-month horizons
    
    Open source on GitHub.
    
    **📊 Popular Tickers to Try:**
    
    | Market | Tickers |
    |--------|---------|
    | **US Indices** | `^GSPC` (S&P 500), `^NDX` (Nasdaq), `^DJI` (Dow) |
    | **US Stocks** | `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `TSLA`, `META` |
    | **ETFs** | `SPY`, `QQQ`, `IWM`, `DIA`, `VTI` |
    | **Crypto** | `BTC-USD`, `ETH-USD`, `SOL-USD` |
    | **UK** | `^FTSE`, `BP.L`, `SHEL.L` |
    | **Europe** | `^GDAXI`, `^FCHI` |
    | **Australia** | `BHP.AX`, `CBA.AX`, `CSL.AX`, `IOZ.AX`, `^AXJO` |
    | **Japan** | `^N225`, `7203.T` |
    | **India** | `^NSEI`, `RELIANCE.NS` |
    
    *Enter any ticker above in the sidebar to analyze its regime patterns and get price forecasts.*
    """)

# --------------------------------------------------
# Symbol normalisation
# --------------------------------------------------
symbol = normalize_symbol(raw_symbol)

# --------------------------------------------------
# Require explicit click before doing heavy work
# --------------------------------------------------
if "analysis_requested" not in st.session_state:
    st.session_state["analysis_requested"] = False

if fit_button:
    st.session_state["analysis_requested"] = True

if not st.session_state["analysis_requested"]:
    st.info("Set your options in the sidebar and click **Run / refresh analysis**.")
    st.stop()

# --------------------------------------------------
# Load prices (using custom file or live data)
# --------------------------------------------------
if custom_price_file is not None:
    try:
        up = pd.read_csv(custom_price_file)
        date_col = [c for c in up.columns if c.strip().lower() == "date"]
        price_col = [
            c
            for c in up.columns
            if c.strip().lower() in {"adj close", "adj_close", "adjclose", "close"}
        ]
        if not date_col or not price_col:
            st.error("Custom CSV must have columns like Date and Adj Close or Close.")
            st.stop()
        up = up.rename(columns={date_col[0]: "Date", price_col[0]: "Adj Close"})
        up["Date"] = pd.to_datetime(up["Date"])
        up = up.set_index("Date").sort_index()[["Adj Close"]]
        up = up.loc[
            (up.index >= pd.to_datetime(start))
            & (up.index <= pd.to_datetime(end))
        ]
        df_prices, data_src = up, "upload"
    except Exception as e:
        st.error("Failed to parse custom price CSV: " + str(e))
        st.stop()
else:
    df_prices, data_src = load_price_data(symbol, start, end, allow_synth=use_synth)

if df_prices.empty:
    st.error(
        "No data available. Try symbols like SPY, AAPL, MSFT, or ASX tickers such as BHP.AX or IOZ.AX. "
        "If all live providers fail, enable synthetic fallback."
    )
    st.stop()

st.success(f"Data source: {data_src} | {len(df_prices)} trading days")

# --------------------------------------------------
# Features and HMM model fit
# --------------------------------------------------
features = build_features(df_prices, window_vol)
if features.empty:
    st.error(
        "Not enough data after feature engineering. Try a different date range or a smaller volatility window."
    )
    st.stop()

sig = (
    f"{symbol}|{pd.to_datetime(start).date()}|{pd.to_datetime(end).date()}|"
    f"{window_vol}|{n_states_input}|{random_state}|{data_src}|{len(features)}"
)


def need_refit() -> bool:
    ms = st.session_state.get("model_state")
    if ms is None:
        return True
    if ms.get("sig") != sig:
        return True
    if len(ms.get("states", [])) != len(features):
        return True
    post = ms.get("post")
    if post is None or post.shape[0] != len(features):
        return True
    return False


if auto_refit or fit_button or need_refit():
    with st.spinner("Fitting HMM model..."):
        hmm, scaler, states, post = fit_hmm(features, n_states_input, random_state)
    K = int(post.shape[1])
    st.session_state["model_state"] = dict(
        hmm=hmm, scaler=scaler, states=states, post=post, K=K, sig=sig
    )
else:
    ms = st.session_state["model_state"]
    hmm, scaler, states, post, K = (
        ms["hmm"],
        ms["scaler"],
        ms["states"],
        ms["post"],
        ms["K"],
    )

# --------------------------------------------------
# Join predictions and basic stats
# --------------------------------------------------
labels_adv, labels_simple = label_states(features, states, label_metric)

decoded = pd.DataFrame(index=features.index)
decoded["state"] = pd.Series(states, index=features.index)
for k in range(K):
    decoded["p_state_" + str(k)] = pd.Series(post[:, k], index=features.index)

df = df_prices.join(decoded, how="left")
conf_series = pd.Series(post.max(axis=1), index=features.index, name="confidence")

grouped = features.join(df["state"]).groupby("state")
stats = grouped.agg(
    mean_ret=("ret", "mean"),
    std_ret=("ret", "std"),
    mean_vol=("vol", "mean"),
    count=("ret", "count"),
)
stats["ann_mean_ret"] = stats["mean_ret"] * 252
stats["ann_vol"] = stats["std_ret"] * np.sqrt(252)
stats = stats.rename(index=labels_adv)

df["Regime"] = df["state"].map(labels_adv)
df["RegimeSimple"] = df["state"].map(labels_simple)

state_clean = df["state"].copy().ffill().dropna().astype(int)

# --------------------------------------------------
# ML predictions
# --------------------------------------------------
ml_models: Optional[Dict[str, Any]] = None
if enable_ml:
    if train_ml_button:
        with st.spinner("Training ML models..."):
            ml_models = train_all_ml_models(df_prices, features, lookback=20)
        st.session_state["ml_models"] = ml_models
    elif "ml_models" in st.session_state:
        ml_models = st.session_state["ml_models"]

if enable_ml and ml_models:
    with st.spinner("Computing ML predictions..."):
        df["PredictedPriceNextML"] = add_ml_prediction_column(
            df_prices, features, ml_models
        )
else:
    df["PredictedPriceNextML"] = np.nan

# --------------------------------------------------
# Future price forecasts
# --------------------------------------------------
forecasts = {}
if enable_ml and ml_models and not state_clean.empty:
    with st.spinner("Computing future price forecasts..."):
        forecasts = get_all_forecasts(df_prices, features, ml_models)

# --------------------------------------------------
# Simple view
# --------------------------------------------------
if state_clean.empty:
    st.warning("Not enough decoded state points yet.")
else:
    latest_state = int(state_clean.iloc[-1])
    latest_label_simple = labels_simple.get(latest_state, "Unknown")
    conf_clean = conf_series.reindex(state_clean.index).dropna()
    latest_conf = float(conf_clean.iloc[-1]) if not conf_clean.empty else 0.0
    typical_days = float(
        1.0
        / (1.0 - np.clip(hmm.transmat_[latest_state, latest_state], 1e-9, 0.999999))
    )
    days_in_regime = current_run_length(state_clean)

    if simple_view:
        st.markdown("### Quick View")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Regime", latest_label_simple)
        c2.metric("Confidence", f"{latest_conf:.0%}")
        c3.metric(
            "Days in Regime",
            f"{days_in_regime}",
            help=f"Typical duration: ~{typical_days:.1f} days",
        )

        advanced_name = labels_adv.get(latest_state, list(labels_adv.values())[0])
        expl = stats.loc[advanced_name]
        ann_r = expl["ann_mean_ret"]
        ann_v = expl["ann_vol"]
        blurb = {
            "Calm": "Trend up with relatively lower daily swings.",
            "Super Calm": "Strong, steady trend with low noise.",
            "Choppy": "Mixed direction with frequent whipsaws.",
            "Stormy": "Downtrends or sharp swings with elevated risk.",
        }.get(
            latest_label_simple,
            "Returns and volatility are typical for this regime.",
        )
        st.info(
            f"**Typical behavior:** {blurb}\n\n"
            f"Historical annual return: **{ann_r:.1%}** | "
            f"Volatility: **{ann_v:.1%}**"
        )

        # Regime timeline
        idx_clean = state_clean.index.to_list()
        segs = regime_segments(idx_clean, state_clean.to_list(), labels_simple)
        tl = plot_regime_timeline(idx_clean, state_clean.to_list(), labels_simple)
        st.plotly_chart(tl, use_container_width=True)

        # Price with regime shading and future forecasts
        pfig = plot_price_with_regimes(df, segs, enable_ml, forecasts)
        st.plotly_chart(pfig, use_container_width=True)

        # Confidence
        conf_fig = plot_confidence_series(conf_series)
        st.plotly_chart(conf_fig, use_container_width=True)

# --------------------------------------------------
# Future Price Forecasts Section
# --------------------------------------------------
if enable_ml and forecasts:
    st.markdown("---")
    st.markdown("### 🔮 ML Price Forecasts")
    
    col1, col2, col3 = st.columns(3)
    
    current_price = float(df_prices["Adj Close"].iloc[-1])
    forecast_dates = {
        3: col1,
        14: col2,
        90: col3
    }
    
    for horizon, col in forecast_dates.items():
        if horizon in forecasts:
            f = forecasts[horizon]
            pred_price = f["predicted_price"]
            pred_return = f["predicted_return"]
            conf_lower = f["confidence_lower"]
            conf_upper = f["confidence_upper"]
            conf_level = f["confidence_level"]
            
            with col:
                st.markdown(f"**{horizon}-Day Forecast**")
                st.metric(
                    "Predicted Price",
                    f"${pred_price:.2f}",
                    delta=f"{pred_return:.2%}",
                    help=f"Current: ${current_price:.2f}\n95% CI: ${conf_lower:.2f} - ${conf_upper:.2f}"
                )
                st.caption(f"Confidence: {conf_level:.0%}")
        else:
            with col:
                st.markdown(f"**{horizon}-Day Forecast**")
                st.info("Forecast unavailable")

    # Forecast comparison chart
    if forecasts:
        fig_forecast = plot_forecast_comparison(current_price, forecasts)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.caption(
            "Forecasts use ensemble ML (Linear + Random Forest) with confidence intervals. "
            "Longer horizons include mean reversion toward historical averages."
        )

# --------------------------------------------------
# Advanced diagnostics and plots
# --------------------------------------------------
if not simple_view and not state_clean.empty:
    st.subheader(f"{symbol} Regime Summary")
    st.dataframe(
        stats.style.format(
            {
                "mean_ret": "{:.5f}",
                "std_ret": "{:.5f}",
                "mean_vol": "{:.4f}",
                "ann_mean_ret": "{:.2%}",
                "ann_vol": "{:.2%}",
                "count": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    trans = pd.DataFrame(
        hmm.transmat_,
        index=[labels_adv.get(i, "State " + str(i)) for i in range(hmm.n_components)],
        columns=[labels_adv.get(i, "State " + str(i)) for i in range(hmm.n_components)],
    )
    dur = pd.Series(
        expected_durations(hmm.transmat_),
        index=trans.index,
        name="Expected duration (days)",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Transition Matrix**")
        st.dataframe(
            trans.style.format("{:.3f}"),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Expected Duration**")
        st.dataframe(
            dur.to_frame().style.format("{:.1f}"),
            use_container_width=True,
        )

    # Price coloured by regime
    fig = make_regime_scatter(df, state_clean, labels_adv)
    st.plotly_chart(fig, use_container_width=True)

    # Posterior regime probabilities
    prob_fig = make_posterior_probs(
        df, labels_adv, int(st.session_state["model_state"]["K"])
    )
    st.plotly_chart(prob_fig, use_container_width=True)

# --------------------------------------------------
# Portfolio analytics (if uploaded)
# --------------------------------------------------
if port_file is not None:
    try:
        port_df = pd.read_csv(port_file)
        if not {"Ticker", "Weight"}.issubset({c.strip().title() for c in port_df.columns}):
            st.warning("Portfolio CSV must have columns `Ticker` and `Weight`.")
        else:
            port_df.columns = [c.strip().title() for c in port_df.columns]
            weights = dict(zip(port_df["Ticker"], port_df["Weight"]))

            price_dict = {symbol: df_prices}
            port_rets = portfolio_return_series(price_dict, {symbol: 1.0})
            var95, es95 = compute_var_es(port_rets, alpha=0.95)

            st.subheader("Portfolio Risk Snapshot")
            st.write(f"**95% one-day VaR:** {var95:.2%}")
            st.write(f"**95% one-day Expected Shortfall:** {es95:.2%}")

    except Exception as e:
        st.warning(f"Failed to parse portfolio file: {e}")

# --------------------------------------------------
# Signal and CSV download
# --------------------------------------------------
if not state_clean.empty and not stats.empty:
    latest_state_for_signal = int(state_clean.iloc[-1])
    latest_regime_adv = labels_adv.get(
        latest_state_for_signal, "State " + str(latest_state_for_signal)
    )
    ann_means = stats["ann_mean_ret"].sort_values(ascending=False)
    best_regime = ann_means.index[0]
    st.markdown(
        f"**Latest regime:** `{latest_regime_adv}`  |  "
        f"**Best historical regime:** `{best_regime}`"
    )

out = df[
    ["Adj Close", "PredictedPriceNextML", "state", "Regime", "RegimeSimple"]
].copy()
out["Return"] = np.log(out["Adj Close"]).diff()
out = out.dropna()

st.download_button(
    "📥 Download Labelled Time Series (CSV)",
    data=out.to_csv().encode("utf-8"),
    file_name=f"{symbol.replace('^', '')}_labelled_timeseries.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Data providers: Yahoo Finance (primary), Twelve Data, Alpha Vantage. "
    "ML engine: Ensemble of Linear Regression and Random Forest. "
    "Open source - contributions welcome!"
)

