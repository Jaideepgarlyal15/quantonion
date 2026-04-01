"""
QuantOnion — Open-Source Agentic Quant Research & Backtesting Platform

Streamlit portal providing:
  - Vectorised backtesting with transaction cost modelling
  - Market regime detection via Gaussian HMM
  - Multi-strategy comparison (SMA, EMA, RSI, Bollinger, Regime Filter)
  - ML ensemble price forecasting
  - ConnectOnion-powered research agent for plain-English analysis

Educational and research purposes only. Not investment advice.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.data_loader import load_price_data, normalize_symbol
from core.features import build_features
from core.hmm_model import (
    current_run_length,
    expected_durations,
    fit_hmm,
    label_states,
    regime_segments,
)
from core.ml import (
    add_ml_prediction_column,
    get_all_forecasts,
    train_all_ml_models,
)
from core.plotting import (
    make_posterior_probs,
    make_regime_scatter,
    plot_confidence_series,
    plot_drawdown_chart,
    plot_equity_curves,
    plot_forecast_comparison,
    plot_price_with_regimes,
    plot_regime_timeline,
)
import plotly.graph_objects as go

from core.portfolio import compute_var_es, portfolio_return_series
from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics, format_metrics_table
from strategies import STRATEGIES
from agents.research_agent import create_research_agent, run_agent_analysis

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantOnion — Agentic Quant Research",
    page_icon="🧅",
    layout="wide",
)

st.markdown(
    """
<style>
header, footer {visibility: hidden;}
.block-container {padding-top: 0.8rem; padding-bottom: 1.5rem;}
[data-testid="stSidebar"] {background: #181D25;}
.stTabs [data-baseweb="tab"] {font-size: 0.95rem; font-weight: 500;}
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧅 QuantOnion")
    st.caption("Agentic Quant Research & Backtesting")

    st.divider()
    st.subheader("Asset & Period")
    raw_symbol = st.text_input(
        "Ticker (e.g. ^GSPC, AAPL, BTC-USD, BHP.AX)",
        value="^GSPC",
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start = st.date_input("From", value=pd.to_datetime("2015-01-01"))
    with col_d2:
        end = st.date_input("To", value=pd.Timestamp.today().normalize())

    st.divider()
    st.subheader("Backtest Settings")

    selected_strategy_names = st.multiselect(
        "Strategies to test",
        options=list(STRATEGIES.keys()),
        default=["Buy & Hold", "SMA Crossover", "Regime Filter"],
        help="Buy & Hold is always included as the benchmark.",
    )

    cost_bps = st.slider(
        "Transaction Cost (bps)",
        0, 50, 10,
        help="One-way cost: 10 bps = 0.10% per trade leg",
    )
    slippage_bps = st.slider(
        "Slippage (bps)",
        0, 30, 5,
        help="Execution slippage per trade leg",
    )

    with st.expander("Strategy Parameters"):
        sma_fast = st.number_input("SMA Fast window", 5, 200, 50)
        sma_slow = st.number_input("SMA Slow window", 20, 500, 200)
        ema_fast = st.number_input("EMA Fast span", 5, 100, 12)
        ema_slow = st.number_input("EMA Slow span", 10, 200, 26)
        rsi_period = st.number_input("RSI Period", 5, 50, 14)
        rsi_oversold = st.slider("RSI Oversold", 10, 45, 30)
        rsi_overbought = st.slider("RSI Overbought", 55, 90, 70)
        bb_period = st.number_input("Bollinger Period", 5, 50, 20)
        bb_std = st.slider("Bollinger Std Dev", 1.0, 3.5, 2.0, step=0.1)

    with st.expander("Regime Detection"):
        n_states = st.slider("Number of regimes", 2, 4, 3)
        vol_window = st.number_input("Volatility window (days)", 5, 120, 21)
        random_seed = st.number_input("Random seed", 0, 10000, 42)
        label_metric = st.selectbox(
            "Label regimes by",
            ["mean_return", "sharpe_ratio"],
            index=0,
        )
        simple_view = st.checkbox("Simple view (regimes tab)", value=True)

    with st.expander("ML Forecasting (optional)"):
        enable_ml = st.checkbox("Enable ML price forecasts", value=False)
        train_ml_btn = st.button("Train / update ML models")

    st.divider()
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    use_synth = st.checkbox("Synthetic fallback (demo)", value=False)

    st.divider()
    st.subheader("Custom Data (optional)")
    custom_csv = st.file_uploader("Price CSV (Date, Close/Adj Close)", type=["csv"])
    port_csv = st.file_uploader("Portfolio CSV (Ticker, Weight)", type=["csv"])

    st.divider()
    st.markdown(
        """
**Popular tickers:**

| Region | Examples |
|--------|---------|
| US Indices | `^GSPC` `^NDX` `^DJI` |
| US Stocks | `AAPL` `MSFT` `NVDA` `TSLA` |
| ETFs | `SPY` `QQQ` `IWM` `VTI` |
| Crypto | `BTC-USD` `ETH-USD` |
| UK / EU | `^FTSE` `^GDAXI` |
| Australia | `BHP.AX` `CBA.AX` `^AXJO` |
| India | `^NSEI` `RELIANCE.NS` |
""",
        unsafe_allow_html=False,
    )

# ── Symbol normalisation ───────────────────────────────────────────────────────
symbol = normalize_symbol(raw_symbol)

# ── Landing page (before first run) ───────────────────────────────────────────
if "results" not in st.session_state and not run_button:
    st.markdown("# 🧅 QuantOnion")
    st.markdown("### Open-Source Agentic Quant Research & Backtesting Platform")
    st.markdown(
        """
        > Select a ticker and strategies in the sidebar, then click **🚀 Run Analysis**.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        "**📈 Backtest Strategies**\n\n"
        "Run reproducible backtests with transaction cost "
        "and slippage modelling across 6 built-in strategies."
    )
    col2.markdown(
        "**🔮 Regime Detection**\n\n"
        "Gaussian HMM detects market regimes (Bull, Bear, Choppy, Calm) "
        "with confidence scores and expected durations."
    )
    col3.markdown(
        "**🤖 AI Research Agent**\n\n"
        "ConnectOnion-powered agent explains results in plain English, "
        "compares strategies, and highlights regime sensitivity."
    )
    st.divider()
    st.caption(
        "Data: Yahoo Finance (free, no key required). "
        "Optional: Alpha Vantage, Twelve Data keys in `.env`. "
        "Not investment advice."
    )
    st.stop()


# ── Load / parse price data ────────────────────────────────────────────────────
def _load_prices() -> tuple[pd.DataFrame, str]:
    if custom_csv is not None:
        try:
            up = pd.read_csv(custom_csv)
            date_col = next(
                (c for c in up.columns if c.strip().lower() == "date"), None
            )
            price_col = next(
                (
                    c for c in up.columns
                    if c.strip().lower() in {"adj close", "adj_close", "adjclose", "close"}
                ),
                None,
            )
            if not date_col or not price_col:
                st.error("CSV must have a 'Date' column and a 'Close' or 'Adj Close' column.")
                st.stop()
            up = up.rename(columns={date_col: "Date", price_col: "Adj Close"})
            up["Date"] = pd.to_datetime(up["Date"])
            up = up.set_index("Date").sort_index()[["Adj Close"]]
            up = up.loc[
                (up.index >= pd.to_datetime(start))
                & (up.index <= pd.to_datetime(end))
            ]
            return up, "upload"
        except Exception as exc:
            st.error(f"Failed to parse custom CSV: {exc}")
            st.stop()
    return load_price_data(symbol, start, end, allow_synth=use_synth)


# ── Run analysis ───────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Loading market data…"):
        df_prices, data_src = _load_prices()

    if df_prices.empty:
        st.error(
            f"No data returned for **{symbol}**. "
            "Try a different ticker or enable the synthetic fallback."
        )
        st.stop()

    # Feature engineering
    features = build_features(df_prices, vol_window)
    if features.empty:
        st.error(
            "Not enough data after feature engineering. "
            "Reduce the volatility window or expand the date range."
        )
        st.stop()

    # HMM regime model
    with st.spinner("Fitting HMM regime model…"):
        hmm, scaler, states, post = fit_hmm(features, n_states, random_seed)

    K = int(post.shape[1])
    labels_adv, labels_simple = label_states(features, states, label_metric)

    # Build DataFrame with state columns
    decoded = pd.DataFrame(index=features.index)
    decoded["state"] = pd.Series(states, index=features.index)
    for k in range(K):
        decoded[f"p_state_{k}"] = pd.Series(post[:, k], index=features.index)

    df = df_prices.join(decoded, how="left")
    df["Regime"] = df["state"].map(labels_adv)
    df["RegimeSimple"] = df["state"].map(labels_simple)

    state_clean = df["state"].copy().ffill().dropna().astype(int)
    latest_state = int(state_clean.iloc[-1])
    current_regime = labels_simple.get(latest_state, "Unknown")

    conf_series = pd.Series(post.max(axis=1), index=features.index, name="confidence")

    # Regime stats
    grouped = features.join(decoded["state"]).groupby("state")
    regime_stats = grouped.agg(
        mean_ret=("ret", "mean"),
        std_ret=("ret", "std"),
        mean_vol=("vol", "mean"),
        count=("ret", "count"),
    )
    regime_stats["ann_mean_ret"] = regime_stats["mean_ret"] * 252
    regime_stats["ann_vol"] = regime_stats["std_ret"] * np.sqrt(252)
    regime_stats = regime_stats.rename(index=labels_adv)

    # Regime label series for regime filter strategy
    regime_series = pd.Series(
        [labels_simple.get(int(s), "Unknown") for s in states],
        index=features.index,
    )

    # Instantiate strategies
    strategy_params: Dict[str, Dict[str, Any]] = {
        "SMA Crossover": {"fast": int(sma_fast), "slow": int(sma_slow)},
        "EMA Crossover": {"fast": int(ema_fast), "slow": int(ema_slow)},
        "RSI Mean Reversion": {
            "period": int(rsi_period),
            "oversold": float(rsi_oversold),
            "overbought": float(rsi_overbought),
        },
        "Bollinger Band Reversion": {
            "period": int(bb_period),
            "num_std": float(bb_std),
        },
    }

    # Always include Buy & Hold benchmark
    strategy_names_to_run = list(dict.fromkeys(["Buy & Hold"] + selected_strategy_names))

    strategy_instances: Dict[str, Any] = {}
    for name in strategy_names_to_run:
        cls = STRATEGIES[name]
        params = strategy_params.get(name, {})
        try:
            strategy_instances[name] = cls(**params)
        except ValueError as exc:
            st.warning(f"Strategy '{name}' parameter error: {exc}. Using defaults.")
            strategy_instances[name] = cls()

    prices = df_prices["Adj Close"]

    # Run backtests
    backtest_results: Dict[str, Dict] = {}
    with st.spinner("Running backtests…"):
        for name, strat in strategy_instances.items():
            try:
                if name == "Regime Filter":
                    signals = strat.generate_signals(
                        prices, regime_series=regime_series
                    )
                else:
                    signals = strat.generate_signals(prices)

                result_df = run_backtest(
                    prices,
                    signals,
                    cost_bps=float(cost_bps),
                    slippage_bps=float(slippage_bps),
                )
                if not result_df.empty:
                    metrics = compute_metrics(result_df)
                    backtest_results[name] = {
                        "result": result_df,
                        "metrics": metrics,
                        "signals": signals,
                    }
            except Exception as exc:
                st.warning(f"Backtest failed for '{name}': {exc}")

    # ML models (optional)
    ml_models: Optional[Dict] = st.session_state.get("ml_models")
    df["PredictedPriceNextML"] = np.nan  # always present; filled below if ML trained
    if enable_ml:
        if train_ml_btn:
            with st.spinner("Training ML models…"):
                ml_models = train_all_ml_models(df_prices, features, lookback=20)
            st.session_state["ml_models"] = ml_models
        if ml_models:
            with st.spinner("Computing ML predictions…"):
                df["PredictedPriceNextML"] = add_ml_prediction_column(
                    df_prices, features, ml_models
                )

    # ML forecasts
    forecasts: Dict = {}
    if enable_ml and ml_models and not state_clean.empty:
        with st.spinner("Computing ML forecasts…"):
            forecasts = get_all_forecasts(df_prices, features, ml_models)

    # Store everything in session state
    st.session_state["results"] = {
        "symbol": symbol,
        "data_src": data_src,
        "df_prices": df_prices,
        "df": df,
        "features": features,
        "hmm": hmm,
        "states": states,
        "post": post,
        "K": K,
        "labels_adv": labels_adv,
        "labels_simple": labels_simple,
        "regime_stats": regime_stats,
        "regime_series": regime_series,
        "current_regime": current_regime,
        "conf_series": conf_series,
        "state_clean": state_clean,
        "backtest_results": backtest_results,
        "forecasts": forecasts,
        "enable_ml": enable_ml,
        "period": f"{start} to {end}",
    }
    # Reset chat on new run
    st.session_state.pop("chat_history", None)
    st.session_state.pop("initial_analysis", None)
    st.session_state.pop("agent_instance", None)

# ── Display results ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

r = st.session_state["results"]

# Data info banner
st.success(
    f"**{r['symbol']}** | Source: {r['data_src']} | "
    f"{len(r['df_prices'])} trading days | "
    f"Current regime: **{r['current_regime']}**"
)

tab_bt, tab_regimes, tab_agent, tab_about = st.tabs(
    ["📈 Backtest", "🔮 Regimes", "🤖 Agent", "ℹ️ About"]
)

# ── TAB: BACKTEST ──────────────────────────────────────────────────────────────
with tab_bt:
    bt_results = r["backtest_results"]

    if not bt_results:
        st.info("No backtest results available. Click **🚀 Run Analysis**.")
    else:
        # ── Metrics table ──────────────────────────────────────────────────────
        st.markdown("### Performance Metrics")
        metrics_df = format_metrics_table(bt_results)

        # Format columns for display
        pct_cols = ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate", "% In Market"]
        float2_cols = ["Sharpe", "Calmar", "Profit Factor"]

        def _fmt(val, col):
            if col in pct_cols:
                return f"{val:.1%}"
            if col in float2_cols:
                return f"{val:.2f}" if val < 1e6 else "∞"
            if col == "Trades":
                return str(int(val))
            return f"{val:.2f}"

        display_df = metrics_df.copy()
        for col in display_df.columns:
            if col in pct_cols + float2_cols or col == "Trades":
                display_df[col] = display_df[col].apply(lambda v, c=col: _fmt(v, c))

        # Highlight best Sharpe row
        def _highlight_best(s):
            styles = [""] * len(s)
            try:
                best_idx = metrics_df["Sharpe"].idxmax()
                if s.name == best_idx:
                    styles = ["background-color: rgba(46,204,113,0.12)"] * len(s)
            except Exception:
                pass
            return styles

        st.dataframe(
            display_df.style.apply(_highlight_best, axis=1),
            use_container_width=True,
        )
        st.caption(
            "Metrics computed in-sample on the full date range selected. "
            "Green row = highest Sharpe ratio. "
            "Costs: "
            f"{cost_bps} bps transaction + {slippage_bps} bps slippage (one-way per trade)."
        )

        # ── Equity curves ──────────────────────────────────────────────────────
        st.markdown("### Equity Curves")
        fig_eq = plot_equity_curves(bt_results, title=f"{r['symbol']} — Strategy Equity Curves")
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown ───────────────────────────────────────────────────────────
        st.markdown("### Drawdown from Peak")
        fig_dd = plot_drawdown_chart(bt_results)
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Strategy signals inspector ─────────────────────────────────────────
        with st.expander("Signal Inspector", expanded=False):
            selected_inspect = st.selectbox(
                "Inspect signals for:", list(bt_results.keys()), key="inspect_strat"
            )
            if selected_inspect and selected_inspect in bt_results:
                res_df = bt_results[selected_inspect]["result"]
                sig_fig = go.Figure()
                sig_fig.add_trace(go.Scatter(
                    x=res_df.index, y=res_df["price"],
                    name="Price", mode="lines",
                    line=dict(color="#2c3e50", width=1.2),
                    yaxis="y1",
                ))
                sig_fig.add_trace(go.Scatter(
                    x=res_df.index, y=res_df["position"],
                    name="Position", mode="lines",
                    line=dict(color="#3498db", width=1.5),
                    fill="tozeroy", yaxis="y2",
                ))
                sig_fig.update_layout(
                    title=f"{selected_inspect} — Price & Position",
                    yaxis=dict(title="Price ($)", side="left"),
                    yaxis2=dict(
                        title="Position (0/1)", side="right",
                        overlaying="y", range=[-0.1, 1.5],
                    ),
                    hovermode="x unified", height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                st.plotly_chart(sig_fig, use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────────
        st.markdown("### Export")
        export_rows = []
        for name, res in bt_results.items():
            m = res["metrics"]
            export_rows.append({"Strategy": name, **m})
        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "📥 Download Metrics CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{r['symbol'].replace('^','')}_metrics.csv",
            mime="text/csv",
        )

        # Also offer equity curves CSV
        equity_dict = {
            name: res["result"]["equity_curve"]
            for name, res in bt_results.items()
            if not res["result"].empty
        }
        if equity_dict:
            equity_export = pd.DataFrame(equity_dict)
            st.download_button(
                "📥 Download Equity Curves CSV",
                data=equity_export.to_csv().encode("utf-8"),
                file_name=f"{r['symbol'].replace('^','')}_equity_curves.csv",
                mime="text/csv",
            )

# ── TAB: REGIMES ──────────────────────────────────────────────────────────────
with tab_regimes:
    state_clean = r["state_clean"]
    labels_adv = r["labels_adv"]
    labels_simple = r["labels_simple"]
    df = r["df"]
    conf_series = r["conf_series"]
    hmm = r["hmm"]
    regime_stats = r["regime_stats"]
    K = r["K"]

    if state_clean.empty:
        st.warning("No regime data available.")
    else:
        # Quick metrics
        latest_state_int = int(state_clean.iloc[-1])
        latest_label = labels_simple.get(latest_state_int, "Unknown")
        latest_label_adv = labels_adv.get(latest_state_int, "Unknown")
        conf_clean = conf_series.reindex(state_clean.index).dropna()
        latest_conf = float(conf_clean.iloc[-1]) if not conf_clean.empty else 0.0
        typical_days = float(
            1.0 / (1.0 - np.clip(hmm.transmat_[latest_state_int, latest_state_int], 1e-9, 0.999999))
        )
        days_in_regime = current_run_length(state_clean)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Regime", latest_label)
        c2.metric("HMM Confidence", f"{latest_conf:.0%}")
        c3.metric(
            "Days in Regime",
            f"{days_in_regime}",
            help=f"Typical duration for this regime: ~{typical_days:.1f} days",
        )

        if latest_label_adv in regime_stats.index:
            expl_row = regime_stats.loc[latest_label_adv]
            ann_r = expl_row["ann_mean_ret"]
            ann_v = expl_row["ann_vol"]
            descriptions = {
                "Calm": "Steady upward trend with moderate daily swings.",
                "Super Calm": "Strong, persistent trend with unusually low noise.",
                "Choppy": "Mixed direction with frequent reversals and no clear trend.",
                "Stormy": "Downtrend or high-volatility environment with elevated risk.",
            }
            blurb = descriptions.get(latest_label, "Regime characteristics vary.")
            st.info(
                f"**{latest_label}:** {blurb}\n\n"
                f"Historical in-sample: Ann. return **{ann_r:.1%}** | "
                f"Ann. volatility **{ann_v:.1%}**"
            )

        # Regime timeline
        idx_list = state_clean.index.to_list()
        tl_fig = plot_regime_timeline(idx_list, state_clean.to_list(), labels_simple)
        st.plotly_chart(tl_fig, use_container_width=True)

        # Price with regime shading
        segs = regime_segments(idx_list, state_clean.to_list(), labels_simple)
        price_fig = plot_price_with_regimes(
            df, segs, r.get("enable_ml", False), r.get("forecasts", {})
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # ML forecasts section
        forecasts = r.get("forecasts", {})
        if forecasts:
            st.markdown("### ML Price Forecasts")
            current_price = float(r["df_prices"]["Adj Close"].iloc[-1])
            fc_cols = st.columns(len(forecasts))
            horizon_labels = {3: "3-Day", 14: "14-Day", 90: "3-Month"}
            for col, (h, f) in zip(fc_cols, sorted(forecasts.items())):
                with col:
                    st.markdown(f"**{horizon_labels.get(h, f'{h}-Day')}**")
                    st.metric(
                        "Forecast",
                        f"${f['predicted_price']:.2f}",
                        delta=f"{f['predicted_return']:.2%}",
                        help=f"95% CI: ${f['confidence_lower']:.2f} – ${f['confidence_upper']:.2f}",
                    )
                    st.caption(f"Model confidence: {f['confidence_level']:.0%}")

            fc_fig = plot_forecast_comparison(current_price, forecasts)
            st.plotly_chart(fc_fig, use_container_width=True)
            st.caption(
                "Forecasts use ensemble ML (Linear + Random Forest) with mean reversion "
                "applied at longer horizons. Longer-horizon forecasts carry much higher "
                "uncertainty. These are not trading signals."
            )

        # Confidence series
        conf_fig = plot_confidence_series(conf_series)
        st.plotly_chart(conf_fig, use_container_width=True)

        # Advanced diagnostics
        if not simple_view:
            st.markdown("### Advanced Diagnostics")
            st.markdown(f"**Regime Statistics — {r['symbol']}**")
            st.dataframe(
                regime_stats.style.format(
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
                index=[labels_adv.get(i, f"State {i}") for i in range(hmm.n_components)],
                columns=[labels_adv.get(i, f"State {i}") for i in range(hmm.n_components)],
            )
            dur = pd.Series(
                expected_durations(hmm.transmat_),
                index=trans.index,
                name="Expected duration (days)",
            )

            col_tm, col_dur = st.columns([2, 1])
            with col_tm:
                st.markdown("**Transition Matrix**")
                st.dataframe(trans.style.format("{:.3f}"), use_container_width=True)
            with col_dur:
                st.markdown("**Expected Duration**")
                st.dataframe(dur.to_frame().style.format("{:.1f}"), use_container_width=True)

            scatter_fig = make_regime_scatter(df, state_clean, labels_adv)
            st.plotly_chart(scatter_fig, use_container_width=True)

            prob_fig = make_posterior_probs(df, labels_adv, K)
            st.plotly_chart(prob_fig, use_container_width=True)

    # Portfolio risk (if CSV uploaded)
    if port_csv is not None:
        try:
            port_df = pd.read_csv(port_csv)
            cols_title = {c.strip().title() for c in port_df.columns}
            if not {"Ticker", "Weight"}.issubset(cols_title):
                st.warning("Portfolio CSV must have 'Ticker' and 'Weight' columns.")
            else:
                port_df.columns = [c.strip().title() for c in port_df.columns]
                price_dict = {r["symbol"]: r["df_prices"]}
                port_rets = portfolio_return_series(price_dict, {r["symbol"]: 1.0})
                var95, es95 = compute_var_es(port_rets, alpha=0.95)
                st.subheader("Portfolio Risk Snapshot")
                rc1, rc2 = st.columns(2)
                rc1.metric("95% 1-Day VaR", f"{var95:.2%}")
                rc2.metric("95% 1-Day Expected Shortfall", f"{es95:.2%}")
        except Exception as exc:
            st.warning(f"Could not parse portfolio CSV: {exc}")

    # Labelled series download
    out = df[["Adj Close", "PredictedPriceNextML", "state", "Regime", "RegimeSimple"]].copy()
    out["LogReturn"] = np.log(out["Adj Close"]).diff()
    out = out.dropna()
    st.download_button(
        "📥 Download Labelled Time Series (CSV)",
        data=out.to_csv().encode("utf-8"),
        file_name=f"{r['symbol'].replace('^','')}_labelled.csv",
        mime="text/csv",
    )

# ── TAB: AGENT ─────────────────────────────────────────────────────────────────
with tab_agent:
    st.markdown("### 🤖 QuantOnion Research Agent")
    st.caption("Powered by ConnectOnion · co/gemini-2.5-pro")

    try:
        from connectonion import Agent as _CoAgent  # noqa: F401
        _co_available = True
    except ImportError:
        _co_available = False

    if not _co_available:
        st.info(
            "**ConnectOnion not installed.**\n\n"
            "Run `pip install connectonion` then `co auth` to enable full AI analysis. "
            "A deterministic rule-based summary is shown in the meantime.",
            icon="ℹ️",
        )

    # Agent context
    agent_context = {
        "backtest_results": r.get("backtest_results", {}),
        "regime_stats": r.get("regime_stats"),
        "current_regime": r.get("current_regime", "Unknown"),
        "ticker": r.get("symbol", symbol),
        "period": r.get("period", ""),
    }

    # Pre-run summary button
    if st.button("🔍 Generate Research Summary", key="gen_summary"):
        with st.spinner("Agent analysing results…"):
            agent = create_research_agent(agent_context)
            analysis = run_agent_analysis(
                agent_context,
                (
                    "Provide a comprehensive research summary. "
                    "Use your tools to retrieve strategy metrics and regime context. "
                    "Compare the strategies against Buy & Hold and highlight the most "
                    "interesting findings. Include appropriate caveats."
                ),
                agent=agent,
            )
        st.session_state["initial_analysis"] = analysis
        st.session_state["agent_instance"] = agent

    if "initial_analysis" in st.session_state:
        with st.chat_message("assistant", avatar="🧅"):
            st.markdown(st.session_state["initial_analysis"])

    st.divider()
    st.markdown("**Ask the agent a follow-up question:**")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        role_avatar = "🧅" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=role_avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input(
        "Ask about strategies, regimes, drawdowns, comparisons…"
    ):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Agent thinking…"):
            agent = st.session_state.get("agent_instance") or create_research_agent(
                agent_context
            )
            response = run_agent_analysis(agent_context, prompt, agent=agent)
            st.session_state["agent_instance"] = agent

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response}
        )
        with st.chat_message("assistant", avatar="🧅"):
            st.markdown(response)

# ── TAB: ABOUT ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("## About QuantOnion")
    st.markdown(
        """
QuantOnion is an open-source agentic quant research and backtesting platform
built on [ConnectOnion](https://connectonion.com), Streamlit, and Python.

### Architecture
```
Streamlit UI
└── Sidebar (asset, dates, strategy params)
    ├── Backtest Tab
    │   ├── backtesting/engine.py  — vectorised backtest, no lookahead
    │   ├── backtesting/metrics.py — CAGR, Sharpe, drawdown, win rate
    │   └── strategies/            — 6 strategy implementations
    ├── Regimes Tab
    │   ├── core/hmm_model.py      — Gaussian HMM (hmmlearn)
    │   ├── core/features.py       — log returns, volatility
    │   └── core/ml.py             — ensemble ML forecasting
    └── Agent Tab
        └── agents/research_agent.py — ConnectOnion Agent (re_act plugin)
```

### Why ConnectOnion?
ConnectOnion's `re_act` plugin allows the agent to reason step-by-step,
calling tools iteratively to retrieve strategy metrics and regime data before
composing an analysis. This is more reliable than one-shot prompting for
multi-strategy comparisons where structured data retrieval is required.

### Data Sources
| Source | Free? | Notes |
|--------|-------|-------|
| Yahoo Finance (yfinance) | Yes | Primary; no key required |
| Alpha Vantage | Free tier | 25 requests/day; set ALPHAVANTAGE_API_KEY |
| Twelve Data | Free tier | 800 requests/day; set TWELVE_DATA_API_KEY |

### Limitations
- Backtests are in-sample and likely overstate real-world performance
- HMM regime labels are assigned retrospectively; live detection lags
- ML forecasts use simple ensemble models; do not use for trading decisions
- Transaction cost model is simplified (fixed bps, no market impact)
- yfinance data has ~15-minute delay; not suitable for intraday strategies
- No multi-asset portfolio backtesting in the current version

### Disclaimer
This software is provided for **educational and research purposes only**.
It does not constitute financial advice. Past performance does not predict
future results. Use at your own risk.

---

**Open source — MIT License**

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) and
[ROADMAP.md](ROADMAP.md) for details.
"""
    )
    st.divider()
    st.caption("QuantOnion · Built with ConnectOnion, Streamlit, hmmlearn, scikit-learn, and yfinance")
