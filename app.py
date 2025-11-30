
import warnings
warnings.filterwarnings("ignore")

import os, time, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import requests
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# Optional paywall
# ------------------------------
PAID_DEFAULT = False
user = {"paid": PAID_DEFAULT}
try:
    # Only runs if you installed st-paywall
    from st_paywall import add_auth
    add_auth(
        google_oauth_client_id=st.secrets.get("GOOGLE_OAUTH_CLIENT_ID", os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")),
        stripe_publishable_key=st.secrets.get("STRIPE_PUBLISHABLE_KEY", os.environ.get("STRIPE_PUBLISHABLE_KEY", "")),
        stripe_price_id=st.secrets.get("STRIPE_PRICE_ID", os.environ.get("STRIPE_PRICE_ID", "")),
        redirect_uri=st.secrets.get("PAYWALL_REDIRECT_URI", os.environ.get("PAYWALL_REDIRECT_URI", "")),
        testing_mode=st.secrets.get("PAYWALL_TESTING", os.environ.get("PAYWALL_TESTING", "true")).lower() == "true",
    )
    user = st.session_state.get("user", user)
except Exception:
    # No paywall package or no secrets set. App still runs; premium panes are gated off.
    pass

# ------------------------------
# Page + minimal dark styling
# ------------------------------
st.set_page_config(page_title="Regime Switching Risk Dashboard", layout="wide")
st.markdown("""
<style>
header, footer {visibility: hidden;}
.block-container {padding-top: 0.8rem; padding-bottom: 1.5rem;}
[data-testid="stSidebar"] {background: #181D25;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Regime Switching Risk Dashboard")
st.caption("Hidden Markov Model (Gaussian) for market regime inference — educational use only.")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Options")
    raw_symbol = st.text_input("Ticker (e.g. ^GSPC • AAPL • ASX:BHP • ^AXJO)", value="^GSPC")
    start = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End Date", value=pd.Timestamp.today().normalize())

    n_states_input = st.slider("Number of regimes (target)", 2, 4, 3)
    window_vol = st.number_input("Volatility window (days)", 5, 120, 21)
    random_state = st.number_input("Random seed", 0, 10000, 42)
    simple_view = st.checkbox("Simple view", value=True)

    st.markdown("**Quant settings**")
    label_metric = st.selectbox("Regime labelling metric", ["mean_return", "sharpe_ratio"], index=0)

    auto_refit = st.checkbox("Auto-refit on any input change", value=True)
    fit_button = st.button("Run / Refit model")

    st.markdown("---")
    use_synth = st.checkbox("Allow synthetic fallback (demo)", value=False)
    st.caption("Set ALPHAVANTAGE_API_KEY in environment or .streamlit/secrets.toml for another live source.")

    st.subheader("Data override (optional)")
    custom_price_file = st.file_uploader("Custom Price CSV (Date,Adj Close)", type=["csv"])

    st.subheader("Portfolio (premium)")
    port_file = st.file_uploader("Upload CSV (Ticker,Weight)", type=["csv"])

# ------------------------------
# Symbol normalisation / ASX helpers
# ------------------------------
ASX_SUFFIX = ".AX"

def normalize_symbol(sym: str) -> str:
    """Accept raw inputs like ASX:BHP, bhp, bhp.ax, ^AXJO, etc., return a best-effort canonical symbol."""
    if not sym:
        return sym
    s = sym.strip()
    if s.upper().startswith("ASX:"):
        core = s.split(":", 1)[1].strip().upper()
        return core if core.endswith(ASX_SUFFIX) else f"{core}{ASX_SUFFIX}"
    # Treat bare three-letter Australian codes as ASX if user typed in uppercase and no suffix
    if re.fullmatch(r"[A-Z]{3,4}", s) and not s.startswith("^"):
        # Prefer ASX convention when ambiguous
        return s + ASX_SUFFIX
    return s

def alias_candidates(sym: str) -> list[str]:
    """Possible alternates for fragile tickers (indices and ASX). Ordered by preference."""
    s = (sym or "").upper().strip()
    cands = [sym]
    # S&P500 index → ETF proxies if needed
    if s in {"^GSPC", "^SPX", "GSPC", "SPX"}:
        cands += ["^GSPC", "SPY", "IVV"]
    # Nasdaq-100
    if s in {"^NDX", "NDX"}:
        cands += ["^NDX", "QQQ"]
    # ASX 200 index
    if s in {"^AXJO", "AXJO"}:
        cands += ["^AXJO", "IOZ.AX", "STW.AX"]
    # If already X.AX, try the US dual list as ultimate fallback
    if s.endswith(".AX"):
        cands += [s.replace(".AX", ""), s]
    # De-duplicate while preserving order
    seen = set(); out = []
    for x in cands:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

symbol = normalize_symbol(raw_symbol)

# ------------------------------
# Provider helpers with retries
# ------------------------------
def _standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.rename(columns=str.title)
    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            return pd.DataFrame()
    return df[["Adj Close"]]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf_once(sym, start_date, end_date):
    try:
        df = yf.download(
            tickers=sym,
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date) + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
            threads=False,  # more deterministic under rate-limits
        )
        return _standardize_price_df(df)
    except Exception:
        return pd.DataFrame()

def fetch_yfinance(sym, start_date, end_date, retries=2, pause=1.0):
    for attempt in range(retries + 1):
        df = fetch_yf_once(sym, start_date, end_date)
        if not df.empty:
            return df
        time.sleep(pause)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq(sym: str, start_date, end_date) -> pd.DataFrame:
    # Map indices to ETF proxies for Stooq
    sym2 = alias_candidates(sym)[-1]  # last entry tends to be proxy like IOZ.AX
    try:
        df = pdr.DataReader(sym2, "stooq", pd.to_datetime(start_date), pd.to_datetime(end_date))
        return _standardize_price_df(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alpha_vantage(sym: str, start_date, end_date, api_key: str | None) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    # Alpha Vantage cannot serve ^ indices; use proxy if needed
    for trial in alias_candidates(sym):
        if trial.startswith("^"):
            continue
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": trial,
            "outputsize": "full",
            "apikey": api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=25)
            js = r.json()
            ts = js.get("Time Series (Daily)", {})
            if not ts:
                continue
            df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            if "5. adjusted close" in df.columns:
                out = df[["5. adjusted close"]].rename(columns={"5. adjusted close": "Adj Close"})
            elif "4. close" in df.columns:
                out = df[["4. close"]].rename(columns={"4. close": "Adj Close"})
            else:
                continue
            mask = (out.index >= pd.to_datetime(start_date)) & (out.index <= pd.to_datetime(end_date))
            clip = out.loc[mask]
            if not clip.empty:
                return clip
        except Exception:
            continue
    return pd.DataFrame()

# Main multi-source loader
@st.cache_data(show_spinner=True)
def load_data_multi(sym: str, start_date, end_date, allow_synth=False):
    sym = (sym or "").strip()
    if not sym:
        return pd.DataFrame(), "none"
    start_date = pd.to_datetime(start_date); end_date = pd.to_datetime(end_date)
    if end_date <= start_date:
        return pd.DataFrame(), "none"

    # Try yfinance with aliases first
    for trial in alias_candidates(sym):
        df = fetch_yfinance(trial, start_date, end_date)
        if not df.empty:
            return df, f"yfinance:{trial}"

    # Then stooq
    for trial in alias_candidates(sym):
        df = fetch_stooq(trial, start_date, end_date)
        if not df.empty:
            return df, f"stooq:{trial}"

    # Then Alpha Vantage
    api_key = st.secrets.get("ALPHAVANTAGE_API_KEY", os.environ.get("ALPHAVANTAGE_API_KEY"))
    if api_key:
        df = fetch_alpha_vantage(sym, start_date, end_date, api_key)
        if not df.empty:
            return df, "alphavantage"

    # Synthetic
    if allow_synth:
        idx = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(idx) > 3:
            rng = np.random.default_rng(abs(hash(sym)) % (2**32))
            rets = rng.normal(loc=0.0002, scale=0.01, size=len(idx))
            price = 100 * np.exp(np.cumsum(rets))
            return pd.DataFrame({"Adj Close": price}, index=idx), "synthetic"
    return pd.DataFrame(), "none"

# ------------------------------
# Features & HMM
# ------------------------------
def build_features(df_price: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    f = pd.DataFrame(index=df_price.index.copy())
    ret = np.log(df_price["Adj Close"]).diff()
    f["ret"] = ret
    f["vol"] = ret.rolling(vol_window).std() * np.sqrt(252.0)
    f["ret_x_vol"] = f["ret"] * f["vol"]
    return f.dropna()

def fit_hmm(features: pd.DataFrame, n_states: int, seed: int):
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=seed,
        verbose=False,
    )
    hmm.fit(X)
    states = hmm.predict(X)
    post = hmm.predict_proba(X)
    return hmm, scaler, states, post

# Labels
SIMPLE_NAMES = ["Stormy", "Choppy", "Calm", "Super Calm"]
SIMPLE_COLORS = {"Stormy": "#e74c3c", "Choppy": "#f39c12", "Calm": "#2ecc71", "Super Calm": "#1abc9c"}

def label_states(features: pd.DataFrame, states: np.ndarray, metric: str = "mean_return"):
    uniq = np.unique(states)
    ret_means = {s: features.loc[states == s, "ret"].mean() for s in uniq}
    stds = {s: features.loc[states == s, "ret"].std() for s in uniq}
    ann_ret = {s: ret_means[s] * 252 for s in uniq}
    ann_vol = {s: (stds[s] * np.sqrt(252)) if pd.notna(stds[s]) else np.nan for s in uniq}
    sharpe = {s: (ann_ret[s] / ann_vol[s]) if ann_vol[s] and ann_vol[s] > 0 else -np.inf for s in uniq}
    order = sorted(uniq, key=(lambda s: sharpe[s]) if metric == "sharpe_ratio" else (lambda s: ret_means[s]))
    adv_names_master = ["Crisis/Bear", "Neutral", "Bull", "Super Bull"]
    adv_map = {s: adv_names_master[i] for i, s in enumerate(order)}
    simple_map = {s: SIMPLE_NAMES[i] for i, s in enumerate(order)}
    return adv_map, simple_map

def expected_durations(transmat: np.ndarray) -> np.ndarray:
    pii = np.clip(np.diag(transmat), 1e-9, 0.999999)
    return 1.0 / (1.0 - pii)

def current_run_length(series_states: pd.Series) -> int:
    if series_states.empty: return 0
    s = series_states.values; run = 1
    for i in range(len(s) - 2, -1, -1):
        if s[i] == s[-1]: run += 1
        else: break
    return run

def regime_segments(index, states, labels_dict):
    out, start_idx = [], 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            out.append((index[start_idx], index[i - 1], labels_dict.get(states[i - 1], "Unknown")))
            start_idx = i
    out.append((index[start_idx], index[-1], labels_dict.get(states[-1], "Unknown")))
    return out

# ------------------------------
# Load prices (respect override)
# ------------------------------
if custom_price_file is not None:
    try:
        up = pd.read_csv(custom_price_file)
        date_col = [c for c in up.columns if c.strip().lower() == "date"]
        price_col = [c for c in up.columns if c.strip().lower() in {"adj close", "adj_close", "adjclose", "close"}]
        if not date_col or not price_col:
            st.error("Custom CSV must have columns like 'Date' and 'Adj Close' (or 'Close').")
            st.stop()
        up = up.rename(columns={date_col[0]: "Date", price_col[0]: "Adj Close"})
        up["Date"] = pd.to_datetime(up["Date"])
        up = up.set_index("Date").sort_index()[["Adj Close"]]
        up = up.loc[(up.index >= pd.to_datetime(start)) & (up.index <= pd.to_datetime(end))]
        df_prices, data_src = up, "upload"
    except Exception as e:
        st.error(f"Failed to parse custom price CSV: {e}")
        st.stop()
else:
    df_prices, data_src = load_data_multi(symbol, start, end, allow_synth=use_synth)

if df_prices.empty:
    st.error("No data available. Enable synthetic fallback or configure Alpha Vantage. Try SPY/AAPL/MSFT or ASX tickers like BHP.AX / IOZ.AX if indices are blocked.")
    st.stop()

st.success(f"Data source: {data_src}", icon="✅" if data_src != "synthetic" else "⚠️")

# ------------------------------
# Features and robust refit policy
# ------------------------------
features = build_features(df_prices, window_vol)
if features.empty:
    st.error("Not enough data after feature engineering. Try a different date range or smaller vol window.")
    st.stop()

sig = (
    f"{symbol}|{pd.to_datetime(start).date()}|{pd.to_datetime(end).date()}|"
    f"{window_vol}|{n_states_input}|{random_state}|{data_src}|{len(features)}"
)

def need_refit():
    ms = st.session_state.get("model_state")
    if ms is None: return True
    if ms.get("sig") != sig: return True
    if len(ms.get("states", [])) != len(features): return True
    post = ms.get("post", np.empty((0, 0)))
    if post.shape[0] != len(features): return True
    return False

if auto_refit or fit_button or need_refit():
    hmm, scaler, states, post = fit_hmm(features, n_states_input, random_state)
    K = int(post.shape[1])
    st.session_state["model_state"] = dict(hmm=hmm, scaler=scaler, states=states, post=post, K=K, sig=sig)
else:
    ms = st.session_state["model_state"]
    hmm, scaler, states, post, K = ms["hmm"], ms["scaler"], ms["states"], ms["post"], ms["K"]

# ------------------------------
# Join predictions (defensive)
# ------------------------------
labels_adv, labels_simple = label_states(features, states, label_metric)

decoded = pd.DataFrame(index=features.index)
decoded["state"] = pd.Series(states, index=features.index)
for k in range(K):
    decoded[f"p_state_{k}"] = pd.Series(post[:, k], index=features.index)

df = df_prices.join(decoded, how="left")
conf_series = pd.Series(post.max(axis=1), index=features.index, name="confidence")

grouped = features.join(df["state"]).groupby("state")
stats = grouped.agg(mean_ret=("ret", "mean"), std_ret=("ret", "std"), mean_vol=("vol", "mean"), count=("ret", "count"))
stats["ann_mean_ret"] = stats["mean_ret"] * 252
stats["ann_vol"] = stats["std_ret"] * np.sqrt(252)
stats = stats.rename(index=labels_adv)

df["Regime"] = df["state"].map(labels_adv)
df["RegimeSimple"] = df["state"].map(labels_simple)

state_clean = df["state"].copy().ffill().dropna().astype(int)

# ------------------------------
# Simple view
# ------------------------------
if state_clean.empty:
    st.warning("Not enough decoded state points yet.")
else:
    latest_state = int(state_clean.iloc[-1])
    latest_label_simple = labels_simple.get(latest_state, "Unknown")
    conf_clean = conf_series.reindex(state_clean.index).dropna()
    latest_conf = float(conf_clean.iloc[-1]) if not conf_clean.empty else 0.0
    typical_days = float(1.0 / (1.0 - np.clip(hmm.transmat_[latest_state, latest_state], 1e-9, 0.999999)))
    days_in_regime = current_run_length(state_clean)

    if simple_view:
        st.markdown("### Quick view")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current regime", latest_label_simple)
        c2.metric("Confidence", f"{latest_conf:.0%}")
        c3.metric("Days in regime", f"{days_in_regime}", help=f"Typical duration ≈ {typical_days:.1f} days")

        advanced_name = labels_adv.get(latest_state, list(labels_adv.values())[0])
        expl = stats.loc[advanced_name]
        ann_r = expl["ann_mean_ret"]; ann_v = expl["ann_vol"]
        blurb = {
            "Calm": "Trend up with relatively lower daily swings.",
            "Super Calm": "Strong, steady trend with low noise.",
            "Choppy": "Mixed direction; whipsaws and ranges.",
            "Stormy": "Downtrends or sharp swings; elevated risk.",
        }.get(latest_label_simple, "Returns and volatility typical for this regime.")
        st.success(f"**What this usually means:** {blurb} Historically here: annualised return ≈ **{ann_r:.1%}**, volatility ≈ **{ann_v:.1%}**.")

        # Regime timeline
        st.subheader("Regime Timeline")
        idx_clean = state_clean.index.to_list()
        segs = regime_segments(idx_clean, state_clean.to_list(), labels_simple)
        tl = go.Figure()
        for (s0, s1, lab) in segs:
            tl.add_shape(type="rect", x0=s0, x1=s1, y0=0, y1=1, line=dict(width=0),
                         fillcolor=SIMPLE_COLORS.get(lab, "#95a5a6"), opacity=0.35)
        tl.update_yaxes(visible=False)
        tl.update_xaxes(type="date", title="Date")
        tl.update_layout(height=120, margin=dict(l=20, r=20, t=10, b=10), showlegend=False)
        st.plotly_chart(tl, use_container_width=True)

        # Price with shading
        st.subheader("Price (with regime shading)")
        price = df["Adj Close"].dropna()
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name="Adj Close"))
        for (s0, s1, lab) in segs:
            pfig.add_vrect(x0=s0, x1=s1, fillcolor=SIMPLE_COLORS.get(lab, "#95a5a6"), opacity=0.08, line_width=0)
        pfig.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                           xaxis_title="Date", yaxis_title="Price",
                           legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
        pfig.update_xaxes(type="date", title="Date")
        st.plotly_chart(pfig, use_container_width=True)

        # Confidence
        st.subheader("Model confidence (most-likely regime)")
        conf_fig = go.Figure()
        conf_fig.add_trace(go.Scatter(x=conf_series.index, y=conf_series.values, mode="lines", name="Confidence"))
        conf_fig.update_yaxes(range=[0, 1])
        conf_fig.update_xaxes(type="date", title="Date")
        conf_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Confidence (0–1)",
                               legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
        st.plotly_chart(conf_fig, use_container_width=True)

# ------------------------------
# Advanced diagnostics (paywalled)
# ------------------------------
if not simple_view:
    if not user.get("paid", False):
        st.info("Subscribe to unlock advanced diagnostics, transition stress tests, and portfolio analytics.")
    else:
        st.subheader(f"{symbol} — Regime Summary")
        st.dataframe(
            stats.style.format({
                "mean_ret": "{:.5f}", "std_ret": "{:.5f}", "mean_vol": "{:.4f}",
                "ann_mean_ret": "{:.2%}", "ann_vol": "{:.2%}", "count": "{:,.0f}",
            }),
            use_container_width=True,
        )

        trans = pd.DataFrame(
            hmm.transmat_,
            index=[labels_adv.get(i, f"State {i}") for i in range(hmm.n_components)],
            columns=[labels_adv.get(i, f"State {i}") for i in range(hmm.n_components)],
        )
        dur = pd.Series(expected_durations(hmm.transmat_), index=trans.index, name="Expected Duration (days)")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Transition Matrix**")
            st.dataframe(trans.style.format("{:.3f}"), use_container_width=True)
        with col2:
            st.markdown("**Expected Duration per Regime**")
            st.dataframe(dur.to_frame().style.format("{:.1f}"), use_container_width=True)

        st.subheader("Price with regime colouring")
        price = df["Adj Close"].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price.values, mode="lines", name="Adj Close", opacity=0.35))
        for s in sorted(np.unique(state_clean)):
            mask = (df["state"] == s)
            fig.add_trace(go.Scatter(
                x=df.index[mask], y=df.loc[mask, "Adj Close"],
                mode="markers", name=labels_adv.get(s, f"State {s}"), marker=dict(size=5)
            ))
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                          legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
                          xaxis_title="Date", yaxis_title="Price")
        fig.update_xaxes(type="date", title="Date")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Posterior regime probabilities")
        prob_fig = go.Figure()
        for s in range(int(st.session_state["model_state"]["K"])):
            name = labels_adv.get(s, f"State {s}")
            prob_fig.add_trace(go.Scatter(x=df.index, y=df.get(f"p_state_{s}"), mode="lines", name=f"P({name})"))
        prob_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                               legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
                               xaxis_title="Date", yaxis_title="Probability", yaxis=dict(range=[0, 1]))
        prob_fig.update_xaxes(type="date", title="Date")
        st.plotly_chart(prob_fig, use_container_width=True)

# ------------------------------
# Signal + download
# ------------------------------
if not state_clean.empty and not stats.empty:
    latest_state_for_signal = int(state_clean.iloc[-1])
    latest_regime_adv = labels_adv.get(latest_state_for_signal, f"State {latest_state_for_signal}")
    ann_means = stats["ann_mean_ret"].sort_values(ascending=False)
    best_regime = ann_means.index[0]
    st.markdown(f"**Latest regime (advanced)**: `{latest_regime_adv}`  |  **Historically best regime**: `{best_regime}`")

out = df[["Adj Close", "state", "Regime", "RegimeSimple"]].dropna()
out["Return"] = np.log(out["Adj Close"]).diff()
out = out.dropna()
st.download_button(
    "Download labeled time series (CSV)",
    data=out.to_csv().encode("utf-8"),
    file_name=f"{symbol.replace('^','')}_labeled_timeseries.csv",
    mime="text/csv",
)

# ------------------------------
# Portfolio analytics (premium)
# ------------------------------
def clean_portfolio(df: pd.DataFrame, weight_cap=0.30):
    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].str.replace(r"^ASX:", "", regex=True).str.upper()
    df["Ticker"] = df["Ticker"].str.replace(r"\s+", "", regex=True)
    df["Ticker"] = df["Ticker"].str.replace("^BRK[.-]B$", "BRK-B", regex=True)
    df = df.groupby("Ticker", as_index=False)["Weight"].sum()
    df["Weight"] = df["Weight"].astype(float).clip(lower=-0.10, upper=weight_cap)
    df = df[np.abs(df["Weight"]) > 1e-6]
    s = df["Weight"].abs().sum()
    if s > 0:
        df["Weight"] = df["Weight"] / s
    df["Ticker"] = df["Ticker"].apply(normalize_symbol)
    return df

def fetch_one(ticker, start_dt, end_dt):
    return ticker, load_data_multi(ticker, start_dt, end_dt, allow_synth=False)[0]

def fetch_many(tickers, start_dt, end_dt, max_workers=8, sleep_every=20, sleep_secs=12):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, t in enumerate(tickers, 1):
            futs.append(ex.submit(fetch_one, t, start_dt, end_dt))
            if i % sleep_every == 0:
                time.sleep(sleep_secs)
        prog = st.progress(0.0, text="Downloading portfolio prices…")
        done = 0
        for fut in as_completed(futs):
            t, dfp = fut.result()
            if dfp is not None and not dfp.empty:
                results[t] = dfp["Adj Close"]
            else:
                st.warning(f"No live data for {t}. Skipping.")
            done += 1
            prog.progress(done / len(futs))
        prog.empty()
    return pd.DataFrame(results)

st.subheader("📊 Portfolio Regime Analytics")
if not user.get("paid", False):
    st.info("Subscribe to analyse portfolio performance by regime with parallel data fetching and Sharpe/vol.")
else:
    if port_file is None:
        st.caption("Upload CSV: Ticker,Weight")
    else:
        try:
            port_df = clean_portfolio(pd.read_csv(port_file))
            if port_df.empty:
                st.error("No valid tickers or weights.")
            else:
                start_dt = pd.to_datetime(start); end_dt = pd.to_datetime(end)
                prices = fetch_many(port_df["Ticker"].tolist(), start_dt, end_dt)
                prices = prices.reindex(df.index).ffill().dropna(how="all")
                if prices.empty:
                    st.error("No aligned price series found.")
                else:
                    rets = np.log(prices / prices.shift(1)).dropna()
                    w = port_df.set_index("Ticker")["Weight"]
                    w = w[w.index.isin(rets.columns)]
                    if w.empty:
                        st.error("None of the portfolio tickers matched downloaded price data.")
                    else:
                        port_ret = (rets[w.index] * w).sum(axis=1)
                        port_ret.name = "PortfolioReturn"
                        port = pd.concat([port_ret, df["Regime"], df["RegimeSimple"]], axis=1).dropna()
                        reg_stats = port.groupby("RegimeSimple")["PortfolioReturn"].agg(
                            mean_ret="mean", std_ret="std", count="count"
                        )
                        reg_stats["ann_mean_ret"] = reg_stats["mean_ret"] * 252
                        reg_stats["ann_vol"] = reg_stats["std_ret"] * np.sqrt(252)
                        reg_stats["sharpe"] = reg_stats["ann_mean_ret"] / reg_stats["ann_vol"]
                        st.dataframe(
                            reg_stats.style.format({
                                "mean_ret": "{:.5f}",
                                "std_ret": "{:.5f}",
                                "ann_mean_ret": "{:.2%}",
                                "ann_vol": "{:.2%}",
                                "sharpe": "{:.2f}",
                                "count": "{:,.0f}",
                            }), use_container_width=True,
                        )
                        cum_growth = np.exp(port["PortfolioReturn"].cumsum())
                        fig_port = go.Figure()
                        fig_port.add_trace(go.Scatter(x=port.index, y=cum_growth, mode="lines",
                                                      name="Portfolio (Cumulative Growth)"))
                        regime_changes = port["RegimeSimple"].ne(port["RegimeSimple"].shift(1))
                        for dt, _reg in port.loc[regime_changes, "RegimeSimple"].items():
                            fig_port.add_vline(x=dt, line_width=1, opacity=0.15)
                        fig_port.update_layout(
                            title="Portfolio Growth with Regime Changes Marked",
                            xaxis_title="Date", yaxis_title="Growth (normalised, start=1)",
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        fig_port.update_xaxes(type="date", title="Date")
                        st.plotly_chart(fig_port, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing portfolio: {e}")

st.info("Providers are tried in order with retries and proxies: yfinance → Stooq → Alpha Vantage (if configured). For ^AXJO we proxy to IOZ.AX/STW.AX when needed; for ^GSPC to SPY/IVV.")
