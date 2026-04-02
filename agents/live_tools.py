"""
QuantOnion live tools for the ConnectOnion research agent.

Eight tools the agent calls to fetch live market data and compute results.
Each returns a plain string. Each accepts only primitive argument types.
Exceptions are caught internally; errors are returned as strings.
"""

import re
import time
import warnings
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from core.data_loader import load_price_data, normalize_symbol
from core.features import build_features
from core.hmm_model import fit_hmm, label_states, current_run_length, expected_durations
from core.ml import train_all_ml_models, get_all_forecasts
from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics
from strategies import STRATEGIES


_DEFAULT_START = "2022-01-01"

# ── In-process data cache (TTL: 5 minutes) ───────────────────────────────────
_cache: dict = {}
_CACHE_TTL = 300  # seconds


def _cache_get(key: str):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    return None


def _cache_set(key: str, val):
    _cache[key] = {"val": val, "ts": time.time()}


def _load_cached(ticker: str, start: str, end: str):
    """Load price data + features with 5-minute cache."""
    key = f"{ticker}|{start}|{end}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    df, src = load_price_data(ticker, start, end, allow_synth=False)
    features = build_features(df, vol_window=21) if not df.empty else pd.DataFrame()
    result = (df, src, features)
    _cache_set(key, result)
    return result


def _today():
    return datetime.today().strftime("%Y-%m-%d")


def list_available_strategies() -> str:
    """
    Return the names of all available backtesting strategies.

    Call this before run_backtest_analysis() if the user has not named a strategy.

    Returns:
        Newline-separated list of strategy names.
    """
    names = list(STRATEGIES.keys())
    return "Available strategies:\n" + "\n".join(f"  {n}" for n in names)


def detect_current_regime(
    ticker: str,
    n_states: int = 2,
    start: str = _DEFAULT_START,
) -> str:
    """
    Fetch live price data and detect the current market regime via Gaussian HMM.

    Fits a Hidden Markov Model to historical log returns and rolling volatility.
    Labels each regime from worst to best performing (Stormy / Choppy / Calm /
    Super Calm). Reports the most recently active regime, HMM confidence, days
    spent in it, and per-regime historical statistics.

    Args:
        ticker: Yahoo Finance ticker. Examples: "AAPL", "^GSPC", "BTC-USD", "BHP.AX".
        n_states: Number of HMM regime states (2, 3, or 4). Default 3.
        start: Historical data start date YYYY-MM-DD. Default "2018-01-01".

    Returns:
        Formatted string: current regime, confidence, days in regime,
        per-regime ann. return and vol, and the HMM transition matrix.
    """
    end = _today()
    ticker = normalize_symbol(ticker.strip())

    df, src, features = _load_cached(ticker, start, end)
    if df.empty:
        return f"No data for '{ticker}'. Check the symbol (Yahoo Finance format, e.g. '^GSPC')."

    if features.empty or len(features) < 60:
        return (
            f"Insufficient data for '{ticker}' ({len(features)} days after feature engineering). "
            "Need at least 60. Try an earlier start date."
        )

    try:
        hmm, _, states, post = fit_hmm(features, n_states=n_states, seed=42)
    except Exception as exc:
        return f"HMM fitting failed for '{ticker}': {exc}"

    _, labels = label_states(features, states, metric="mean_return")

    state_series = pd.Series(states, index=features.index).ffill().dropna().astype(int)
    latest = int(state_series.iloc[-1])
    label = labels.get(latest, "Unknown")
    conf = float(post[-1].max())
    days_in = current_run_length(state_series)
    diag = float(np.clip(hmm.transmat_[latest, latest], 1e-9, 0.9999))
    typical = float(1.0 / (1.0 - diag))
    durations = expected_durations(hmm.transmat_)

    lines = [
        f"Regime Analysis: {ticker}",
        f"Source: {src}  |  {start} to {end} ({len(df)} trading days)",
        "",
        f"Current Regime : {label}",
        f"HMM Confidence : {conf:.0%}",
        f"Days in regime : {days_in}  (typical ~{typical:.0f} days)",
        "",
        "Historical Regime Statistics (in-sample, annualised):",
    ]

    for sid in sorted(labels):
        lbl = labels[sid]
        mask = states == sid
        if mask.sum() == 0:
            continue
        ann_ret = float(features.loc[mask, "ret"].mean() * 252)
        ann_vol = float(features.loc[mask, "ret"].std() * np.sqrt(252))
        count = int(mask.sum())
        dur = float(durations[sid]) if sid < len(durations) else 0.0
        current = "  <- CURRENT" if sid == latest else ""
        lines.append(
            f"  {lbl:<12}: Ann.Return={ann_ret:+.1%}  Ann.Vol={ann_vol:.1%}"
            f"  Days={count:>4}  TypDur~{dur:.0f}d{current}"
        )

    lines += ["", "Transition matrix (row=from, col=to):"]
    col_headers = "  ".join(f"{labels.get(j, f'S{j}'):>10}" for j in range(n_states))
    lines.append(f"  {'':10}  {col_headers}")
    for i in range(n_states):
        row_label = labels.get(i, f"S{i}")
        row_vals = "  ".join(f"{hmm.transmat_[i, j]:>10.3f}" for j in range(n_states))
        lines.append(f"  {row_label:10}  {row_vals}")

    lines += [
        "",
        "Note: HMM regime labels are assigned in-sample. Live detection lags by ~1 day.",
    ]
    return "\n".join(lines)


def run_backtest_analysis(
    ticker: str,
    strategy: str = "Regime Filter",
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> str:
    """
    Fetch live price data and run a single-strategy backtest vs Buy & Hold.

    Executes a vectorised backtest with a 1-day execution lag (no lookahead),
    transaction cost and slippage modelling. Returns full performance metrics
    for the chosen strategy alongside the Buy & Hold benchmark.

    Args:
        ticker: Asset ticker. Examples: "AAPL", "^GSPC", "BTC-USD".
        strategy: Strategy name. Call list_available_strategies() to see options.
        start: Start date YYYY-MM-DD. Default "2018-01-01".
        end: End date YYYY-MM-DD. Defaults to today.
        cost_bps: One-way transaction cost in basis points. Default 10 (= 0.10%).
        slippage_bps: One-way slippage in basis points. Default 5.

    Returns:
        Formatted string with CAGR, Sharpe, max drawdown, win rate, and
        delta vs Buy & Hold.
    """
    if end is None:
        end = _today()
    ticker = normalize_symbol(ticker.strip())

    if strategy not in STRATEGIES:
        return f"Unknown strategy '{strategy}'. Available: {', '.join(STRATEGIES)}"

    df, src, features = _load_cached(ticker, start, end)
    if df.empty:
        return f"No data for '{ticker}'."

    prices = df["Adj Close"]

    regime_series = None
    if strategy == "Regime Filter":
        if len(features) < 60:
            return "Regime Filter needs 60+ days of feature data. Try an earlier start date."
        try:
            _, _, states, _ = fit_hmm(features, n_states=3, seed=42)
            _, labels = label_states(features, states, metric="mean_return")
            regime_series = pd.Series(
                [labels.get(int(s), "Unknown") for s in states],
                index=features.index,
            )
        except Exception as exc:
            return f"HMM fitting failed: {exc}"

    results = {}
    for name in ["Buy & Hold", strategy]:
        if name not in STRATEGIES:
            continue
        try:
            strat = STRATEGIES[name]()
            sigs = (
                strat.generate_signals(prices, regime_series=regime_series)
                if name == "Regime Filter" and regime_series is not None
                else strat.generate_signals(prices)
            )
            res = run_backtest(prices, sigs, cost_bps=cost_bps, slippage_bps=slippage_bps)
            if not res.empty:
                results[name] = compute_metrics(res)
        except Exception as exc:
            return f"Backtest failed for '{name}': {exc}"

    if not results:
        return "Backtest produced no results. Need at least 60 trading days."

    def _block(name, m):
        pf = m.get("profit_factor", 0)
        return [
            f"--- {name} ---",
            f"Total Return  {m.get('total_return', 0):>10.1%}",
            f"CAGR          {m.get('cagr', 0):>10.1%}",
            f"Volatility    {m.get('volatility', 0):>10.1%}",
            f"Sharpe        {m.get('sharpe', 0):>10.2f}",
            f"Max Drawdown  {m.get('max_drawdown', 0):>10.1%}",
            f"Calmar        {m.get('calmar', 0):>10.2f}",
            f"Win Rate      {m.get('win_rate', 0):>10.1%}",
            f"Profit Factor {(f'{pf:.2f}' if pf < 1000 else 'inf'):>10}",
            f"In Market     {m.get('time_in_market', 0):>10.1%}",
            f"Trades        {m.get('n_trades', 0):>10}",
        ]

    lines = [
        f"Backtest: {strategy} on {ticker}",
        f"Period: {start} to {end}  |  Source: {src}",
        f"Costs: {cost_bps} bps + {slippage_bps} bps slippage (one-way per leg)",
        "",
    ]

    for name, m in results.items():
        lines.extend(_block(name, m))
        lines.append("")

    if strategy != "Buy & Hold" and "Buy & Hold" in results and strategy in results:
        bh = results["Buy & Hold"]
        st = results[strategy]
        lines += [
            f"--- {strategy} vs Buy & Hold ---",
            f"CAGR delta    {st.get('cagr', 0) - bh.get('cagr', 0):>+10.1%}",
            f"Sharpe delta  {st.get('sharpe', 0) - bh.get('sharpe', 0):>+10.2f}",
            f"Max DD delta  {st.get('max_drawdown', 0) - bh.get('max_drawdown', 0):>+10.1%}",
            "",
        ]

    lines.append("In-sample historical backtest. Past performance does not predict future results.")
    return "\n".join(lines)


def compare_all_strategies(
    ticker: str,
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> str:
    """
    Fetch live data and run all 6 built-in strategies, returning a ranked table.

    Runs Buy & Hold, SMA Crossover, EMA Crossover, RSI Mean Reversion,
    Bollinger Band Reversion, and Regime Filter on the same ticker and date
    range. Results are sorted by Sharpe ratio (best risk-adjusted return first).

    Args:
        ticker: Asset ticker. Examples: "AAPL", "^GSPC", "BTC-USD".
        start: Start date YYYY-MM-DD. Default "2018-01-01".
        end: End date YYYY-MM-DD. Defaults to today.
        cost_bps: One-way transaction cost in basis points. Default 10.
        slippage_bps: One-way slippage in basis points. Default 5.

    Returns:
        Formatted comparison table sorted by Sharpe, with CAGR delta vs benchmark.
    """
    if end is None:
        end = _today()
    ticker = normalize_symbol(ticker.strip())

    df, src, features = _load_cached(ticker, start, end)
    if df.empty:
        return f"No data for '{ticker}'."

    prices = df["Adj Close"]

    regime_series = None
    if len(features) >= 60:
        try:
            _, _, states, _ = fit_hmm(features, n_states=3, seed=42)
            _, labels = label_states(features, states, metric="mean_return")
            regime_series = pd.Series(
                [labels.get(int(s), "Unknown") for s in states],
                index=features.index,
            )
        except Exception:
            pass

    results = {}
    skipped = []
    for name, cls in STRATEGIES.items():
        try:
            strat = cls()
            if name == "Regime Filter":
                if regime_series is None:
                    skipped.append(name)
                    continue
                sigs = strat.generate_signals(prices, regime_series=regime_series)
            else:
                sigs = strat.generate_signals(prices)
            res = run_backtest(prices, sigs, cost_bps=cost_bps, slippage_bps=slippage_bps)
            if not res.empty:
                results[name] = compute_metrics(res)
        except Exception:
            skipped.append(name)

    if not results:
        return "No backtest results. Need at least 60 trading days."

    bh_cagr = results.get("Buy & Hold", {}).get("cagr", 0.0)
    best_sharpe = max(m.get("sharpe", -999) for m in results.values())

    sorted_results = sorted(results.items(), key=lambda x: x[1].get("sharpe", -999), reverse=True)

    c1, c2, c3, c4, c5, c6, c7 = 26, 8, 8, 8, 9, 8, 7
    header = (
        f"{'Strategy':<{c1}} {'CAGR':>{c2}} {'vs B&H':>{c3}} "
        f"{'Sharpe':>{c4}} {'Max DD':>{c5}} {'WinRate':>{c6}} {'Trades':>{c7}}"
    )
    sep = "-" * (c1 + c2 + c3 + c4 + c5 + c6 + c7 + 6)

    lines = [
        f"Strategy Comparison: {ticker}",
        f"Period: {start} to {end}  |  Source: {src}",
        f"Costs: {cost_bps} bps + {slippage_bps} bps slippage  |  Sorted by Sharpe",
        "",
        header,
        sep,
    ]

    for name, m in sorted_results:
        cagr = m.get("cagr", 0.0)
        delta = cagr - bh_cagr if name != "Buy & Hold" else 0.0
        delta_str = f"{delta:+.1%}" if name != "Buy & Hold" else " bench"
        star = " *" if abs(m.get("sharpe", -999) - best_sharpe) < 1e-9 else ""
        lines.append(
            f"{name:<{c1}} {cagr:>{c2}.1%} {delta_str:>{c3}} "
            f"{m.get('sharpe', 0):>{c4}.2f} {m.get('max_drawdown', 0):>{c5}.1%} "
            f"{m.get('win_rate', 0):>{c6}.1%} {m.get('n_trades', 0):>{c7}}{star}"
        )

    lines += ["", "* = Best Sharpe ratio"]
    if skipped:
        lines.append(f"Skipped (insufficient data): {', '.join(skipped)}")
    lines.append("In-sample historical backtest. Not representative of future performance.")
    return "\n".join(lines)


def get_risk_metrics(
    ticker: str,
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
) -> str:
    """
    Fetch live price data and compute downside risk metrics.

    Returns historical Value-at-Risk (VaR), Expected Shortfall (CVaR),
    annualised and recent volatility, and daily return extremes.

    Args:
        ticker: Asset ticker. Examples: "AAPL", "^GSPC", "BTC-USD".
        start: Start date YYYY-MM-DD. Default "2018-01-01".
        end: End date YYYY-MM-DD. Defaults to today.

    Returns:
        Formatted string with VaR at 95% and 99%, CVaR, volatility
        (annualised and recent 21-day), and best/worst single days.
    """
    if end is None:
        end = _today()
    ticker = normalize_symbol(ticker.strip())

    df, src, _ = _load_cached(ticker, start, end)
    if df.empty:
        return f"No data for '{ticker}'."

    log_rets = np.log(df["Adj Close"] / df["Adj Close"].shift(1)).dropna()
    if len(log_rets) < 30:
        return f"Insufficient data for '{ticker}' ({len(log_rets)} days; need 30+)."

    ann_vol = float(log_rets.std() * np.sqrt(252))
    recent_vol = float(log_rets.tail(21).std() * np.sqrt(252))
    vol_flag = (
        "elevated vs history" if recent_vol > ann_vol * 1.1 else
        "below historical average" if recent_vol < ann_vol * 0.9 else
        "near historical average"
    )

    var_95 = float(np.percentile(log_rets, 5))
    var_99 = float(np.percentile(log_rets, 1))
    tail_95 = log_rets[log_rets <= var_95]
    tail_99 = log_rets[log_rets <= var_99]
    es_95 = float(tail_95.mean()) if not tail_95.empty else var_95
    es_99 = float(tail_99.mean()) if not tail_99.empty else var_99

    n_years = max(len(log_rets) / 252.0, 1e-8)
    cagr = float((np.exp(log_rets.sum())) ** (1.0 / n_years) - 1.0)

    worst = float(log_rets.min())
    best = float(log_rets.max())
    worst_date = str(log_rets.idxmin().date())
    best_date = str(log_rets.idxmax().date())

    lines = [
        f"Risk Metrics: {ticker}",
        f"Source: {src}  |  {start} to {end} ({len(log_rets)} trading days)",
        "",
        "Return profile:",
        f"  CAGR                    : {cagr:>8.1%}",
        f"  Annualised volatility   : {ann_vol:>8.1%}",
        f"  Recent vol (21-day)     : {recent_vol:>8.1%}  ({vol_flag})",
        "",
        "1-Day Value-at-Risk (historical simulation):",
        f"  95% VaR  : {var_95:>8.2%}  (loss worse than this ~1 in 20 days)",
        f"  99% VaR  : {var_99:>8.2%}  (loss worse than this ~1 in 100 days)",
        "",
        "Expected Shortfall / CVaR (average loss beyond VaR):",
        f"  95% ES   : {es_95:>8.2%}",
        f"  99% ES   : {es_99:>8.2%}",
        "",
        "Return extremes:",
        f"  Worst day : {worst:>8.2%}  ({worst_date})",
        f"  Best day  : {best:>8.2%}  ({best_date})",
        "",
        "Historical VaR assumes past return distribution repeats.",
        "Tail events in structural breaks will exceed these estimates.",
    ]
    return "\n".join(lines)


def get_ml_forecast(
    ticker: str,
    start: str = "2020-01-01",
) -> str:
    """
    Train an ensemble ML model on live price data and return price forecasts.

    Trains a Linear Regression + Random Forest ensemble on rolling 20-day
    feature windows (log returns, rolling volatility, momentum, RSI-proxy).
    Returns 3-day, 14-day, and 90-day price forecasts with 95% confidence
    intervals and model confidence scores.

    Args:
        ticker: Asset ticker. Examples: "AAPL", "^GSPC", "BTC-USD".
        start: Start date for training data YYYY-MM-DD. Default "2020-01-01".

    Returns:
        Formatted string with predicted prices, directional bias (UP/DOWN),
        95% CI, model agreement score, and in-sample RMSE.
    """
    end = _today()
    ticker = normalize_symbol(ticker.strip())

    df, src, features = _load_cached(ticker, start, end)
    if df.empty:
        return f"No data for '{ticker}'."

    if len(features) < 80:
        return f"Insufficient data for ML training ({len(features)} days; need 80+)."

    try:
        models = train_all_ml_models(df, features, lookback=20)
    except Exception as exc:
        return f"ML training failed: {exc}"

    if not models:
        return "ML training returned no models. Check that there is sufficient data."

    try:
        forecasts = get_all_forecasts(df, features, models)
    except Exception as exc:
        return f"Forecast generation failed: {exc}"

    if not forecasts:
        return "No forecasts generated."

    current_price = float(df["Adj Close"].iloc[-1])
    last_date = str(df.index[-1].date())
    linear_rmse = models.get("linear_rmse", float("nan"))
    rf_rmse = models.get("rf_rmse", float("nan"))

    lines = [
        f"ML Price Forecast: {ticker}",
        f"Source: {src}  |  Training: {start} to {end} ({len(df)} trading days)",
        f"Current price: ${current_price:.2f}  (as of {last_date})",
        "Model: Linear Regression + Random Forest ensemble (60/40 weighted RF-heavy)",
        "",
        "In-sample RMSE (log-return scale):",
        f"  Linear Regression : {linear_rmse:.6f}",
        f"  Random Forest     : {rf_rmse:.6f}",
        "",
        "Forecasts:",
    ]

    horizon_labels = {3: "3-Day", 14: "2-Week", 90: "3-Month"}
    for h in sorted(forecasts):
        f = forecasts[h]
        pred = f.get("predicted_price", current_price)
        ret = f.get("predicted_return", 0.0)
        lo = f.get("confidence_lower", pred * 0.9)
        hi = f.get("confidence_upper", pred * 1.1)
        conf = f.get("confidence_level", 0.5)
        direction = "UP" if ret > 0 else "DOWN"
        lbl = horizon_labels.get(h, f"{h}-Day")
        lines += [
            f"  {lbl} ({h}d):",
            f"    Predicted price : ${pred:.2f}  ({ret:+.1%}  {direction})",
            f"    95% CI          : ${lo:.2f} to ${hi:.2f}",
            f"    Model confidence: {conf:.0%}",
        ]

    lines += [
        "",
        "Notes:",
        "  - Predictions are ensemble averages on rolling 20-day windows.",
        "  - Confidence intervals widen with horizon (vol scales with sqrt of time).",
        "  - 3-month forecasts: CI too wide for price targets, treat as directional only.",
        "  - In-sample RMSE measures training fit, not out-of-sample accuracy.",
        "  - These forecasts are for research illustration. Do not trade on them.",
    ]
    return "\n".join(lines)


def get_market_sentiment(ticker: str) -> str:
    """
    Return current market sentiment context for a ticker.

    Sources used (all free, no auth required):
      - VIX (CBOE Volatility Index) via yfinance: equity market fear proxy
      - alternative.me Fear & Greed Index: crypto sentiment (7-day trend)
      - Yahoo Finance news headlines: keyword-scored for bullish/bearish tone

    Sentiment is directional context, not a trading signal. High VIX alone
    is not a sell signal; extreme fear has historically preceded recoveries.

    Args:
        ticker: Asset ticker. Examples: "AAPL", "^GSPC", "BTC-USD".

    Returns:
        Formatted string with VIX level and interpretation, crypto Fear & Greed
        score and trend, and news headline sentiment summary.
    """
    ticker_clean = ticker.strip().upper()
    lines = [f"Market Sentiment: {ticker_clean}", f"As of {_today()}", ""]

    # VIX
    lines.append("Equity Fear (VIX):")
    try:
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if not vix_hist.empty:
            vix = float(vix_hist["Close"].iloc[-1])
            vix_prev = float(vix_hist["Close"].iloc[-2]) if len(vix_hist) >= 2 else vix
            change = vix - vix_prev
            if vix < 15:
                reading = "Extreme complacency — markets pricing in very low risk"
            elif vix < 20:
                reading = "Below average — calm conditions"
            elif vix < 25:
                reading = "Near historical average — normal uncertainty"
            elif vix < 35:
                reading = "Elevated — markets pricing in meaningful risk"
            else:
                reading = "Extreme fear — crisis-level volatility"
            lines += [
                f"  VIX: {vix:.2f}  ({'+' if change >= 0 else ''}{change:.2f} vs prev)",
                f"  Reading: {reading}",
            ]
        else:
            lines.append("  VIX: unavailable")
    except Exception:
        lines.append("  VIX: fetch failed")

    lines.append("")

    # Crypto Fear & Greed (relevant for all assets as macro context)
    lines.append("Crypto Fear & Greed (alternative.me):")
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=7&format=json",
            timeout=6,
            headers={"User-Agent": "QuantOnion/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                score = int(data[0].get("value", 50))
                classification = data[0].get("value_classification", "Neutral")
                values = [int(d.get("value", 50)) for d in data if d.get("value")]
                trend = ""
                if len(values) >= 4:
                    recent = sum(values[:3]) / 3
                    older = sum(values[3:]) / max(len(values[3:]), 1)
                    diff = recent - older
                    trend = " (rising)" if diff > 5 else " (falling)" if diff < -5 else " (stable)"
                lines += [
                    f"  Score: {score}/100  ->  {classification}{trend}",
                    "  Scale: 0=Extreme Fear, 50=Neutral, 100=Extreme Greed",
                ]
        else:
            lines.append("  Fear & Greed: unavailable")
    except Exception:
        lines.append("  Fear & Greed: fetch failed")

    lines.append("")

    # News headlines (Yahoo Finance, keyword scored)
    lines.append(f"News Sentiment ({ticker_clean}):")
    bull_words = {
        "bullish", "rally", "surge", "gain", "rise", "beat", "record",
        "growth", "upgrade", "buy", "recover", "positive", "strong",
    }
    bear_words = {
        "bearish", "crash", "drop", "fall", "plunge", "decline", "miss",
        "downgrade", "sell", "recession", "warn", "weak", "negative", "loss",
    }

    def _score(text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        bull = len(words & bull_words)
        bear = len(words & bear_words)
        total = bull + bear
        return (bull - bear) / total if total > 0 else 0.0

    try:
        yf_ticker = yf.Ticker(ticker_clean)
        news = yf_ticker.news or []
        articles = []
        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title", "") or item.get("title", "")
            if title:
                pub = content.get("pubDate") or item.get("providerPublishTime")
                date_str = ""
                if pub:
                    try:
                        if isinstance(pub, int):
                            date_str = datetime.fromtimestamp(pub).strftime("%Y-%m-%d")
                        else:
                            date_str = str(pub)[:10]
                    except Exception:
                        pass
                articles.append({"title": title, "date": date_str, "score": _score(title)})

        if articles:
            scores = [a["score"] for a in articles]
            avg = sum(scores) / len(scores)
            bull_count = sum(1 for s in scores if s > 0.1)
            bear_count = sum(1 for s in scores if s < -0.1)
            neutral_count = len(scores) - bull_count - bear_count
            label_str = (
                "Bullish" if avg > 0.15 else
                "Bearish" if avg < -0.15 else
                "Neutral"
            )
            lines += [
                f"  Source: Yahoo Finance  |  Articles: {len(articles)}",
                f"  Overall: {label_str}  (score {avg:+.2f})",
                f"  Distribution: Bullish={bull_count}  Neutral={neutral_count}  Bearish={bear_count}",
                "  Recent headlines:",
            ]
            for a in articles[:5]:
                date_pfx = f"[{a['date']}] " if a["date"] else ""
                tone = "+" if a["score"] > 0.1 else ("-" if a["score"] < -0.1 else "~")
                lines.append(f"    {tone} {date_pfx}{a['title'][:80]}")
        else:
            lines.append("  No news articles found via Yahoo Finance.")
    except Exception:
        lines.append("  News fetch failed.")

    lines += [
        "",
        "Sentiment is context, not a signal. High fear has historically preceded recoveries.",
        "Do not trade on sentiment data alone.",
    ]
    return "\n".join(lines)


def get_macro_context() -> str:
    """
    Fetch macro regime context: US yield curve, US Dollar Index, and Fed rate proxy.

    Data sources (all free, no API key required):
      - US Treasury daily yield curve XML: 2-year and 10-year nominal yields
      - DX-Y.NYB (US Dollar Index) and ^IRX (13-week T-bill) via yfinance

    Returns:
        Formatted string with 2Y/10Y yields, spread and curve interpretation,
        DXY level and 5-day trend, Fed proxy rate and outlook, and a one-line
        macro regime label.
    """
    cached = _cache_get("macro_context")
    if cached is not None:
        return cached

    lines = ["Macro Context", f"As of {_today()}", ""]

    # ── 1. US Treasury Yield Curve ────────────────────────────────────────────
    lines.append("US Treasury Yield Curve:")
    y2, y10, spread = None, None, None
    try:
        now = datetime.today()
        atom_ns = "http://www.w3.org/2005/Atom"
        meta_ns = "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
        # Try current month, then prior two months in case data not yet published
        for months_back in range(3):
            d = now - timedelta(days=30 * months_back)
            resp = requests.get(
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml",
                params={
                    "data": "daily_treasury_yield_curve",
                    "field_tdate_month": f"{d.month:02d}",
                    "field_tdate_year": str(d.year),
                },
                timeout=12,
                headers={"User-Agent": "QuantOnion/1.0"},
            )
            if resp.status_code != 200:
                break  # non-200 is a server error — do not retry with older months

            root = ET.fromstring(resp.text)
            entries = root.findall(f"{{{atom_ns}}}entry")
            if not entries:
                continue  # empty month — try the previous one

            props = entries[-1].find(f"{{{atom_ns}}}content/{{{meta_ns}}}properties")
            if props is None:
                continue

            # Treasury tag names vary: BC_2YEAR or d:BC_2YEAR — strip namespace prefix
            for elem in props.iter():
                tag = elem.tag.split("}")[-1].upper()
                if "2YEAR" in tag and elem.text and elem.text.strip():
                    try:
                        y2 = float(elem.text)
                    except ValueError:
                        pass
                if "10YEAR" in tag and elem.text and elem.text.strip():
                    try:
                        y10 = float(elem.text)
                    except ValueError:
                        pass

            if y2 is not None and y10 is not None:
                break

        if y2 is not None and y10 is not None:
            spread = y10 - y2
            if spread > 1.0:
                curve_label = "Steep — markets pricing in strong growth"
            elif spread > 0.25:
                curve_label = "Normal — modest growth expectations"
            elif spread > 0:
                curve_label = "Flat — late cycle, growth slowing"
            elif spread > -0.5:
                curve_label = "Mildly inverted — recession risk elevated, monitor"
            else:
                curve_label = "Deeply inverted — strong recession warning (historically reliable)"

            lines += [
                f"  2-Year yield  : {y2:.2f}%",
                f"  10-Year yield : {y10:.2f}%",
                f"  Spread (10-2) : {spread:+.2f}%  ({curve_label})",
            ]
        else:
            lines.append("  Yield data unavailable — Treasury XML returned no entries.")
    except Exception as exc:
        lines.append(f"  Yield curve fetch failed: {exc}")

    lines.append("")

    # ── 2. US Dollar Index (DXY) and Fed Rate Proxy (^IRX) — fetched together ─
    dxy_hist = None
    try:
        bulk = yf.download(["DX-Y.NYB", "^IRX"], period="7d", progress=False)
        # yfinance returns MultiIndex columns when downloading multiple tickers
        close = bulk["Close"] if "Close" in bulk.columns else bulk.xs("Close", axis=1, level=0)

        lines.append("US Dollar Index (DXY):")
        dxy_series = close["DX-Y.NYB"].dropna() if "DX-Y.NYB" in close else pd.Series(dtype=float)
        if len(dxy_series) >= 2:
            dxy_hist = dxy_series  # kept for regime section below
            dxy = float(dxy_series.iloc[-1])
            chg_1d = dxy - float(dxy_series.iloc[-2])
            chg_5d = dxy - float(dxy_series.iloc[max(0, len(dxy_series) - 6)])

            if dxy > 105:
                dxy_label = "Strong — headwind for commodities, EM equities, and non-USD assets"
            elif dxy > 100:
                dxy_label = "Elevated — mildly negative for risk assets"
            elif dxy > 95:
                dxy_label = "Moderate — broadly neutral"
            else:
                dxy_label = "Weak — tailwind for commodities and international equities"

            trend = (
                "strengthening" if chg_5d > 0.5 else
                "weakening" if chg_5d < -0.5 else
                "stable"
            )
            lines += [
                f"  DXY  : {dxy:.2f}  "
                f"({'+' if chg_1d >= 0 else ''}{chg_1d:.2f} today, "
                f"{'+' if chg_5d >= 0 else ''}{chg_5d:.2f} over 5 days — {trend})",
                f"  Reading: {dxy_label}",
            ]
        else:
            lines.append("  DXY: unavailable")

        lines.append("")
        lines.append("Fed Rate Proxy (3-month T-bill, ^IRX):")
        irx_series = close["^IRX"].dropna() if "^IRX" in close else pd.Series(dtype=float)
        if len(irx_series) >= 1:
            irx = float(irx_series.iloc[-1])
            irx_5d = float(irx_series.iloc[max(0, len(irx_series) - 6)])
            irx_chg = irx - irx_5d

            if irx_chg > 0.15:
                fed_outlook = "Rising — market pricing in rate hike(s) or delay in cuts"
            elif irx_chg < -0.15:
                fed_outlook = "Falling — market pricing in rate cut(s)"
            else:
                fed_outlook = "Stable — no near-term move priced in"

            lines += [
                f"  3M T-bill : {irx:.2f}%  "
                f"({'+' if irx_chg >= 0 else ''}{irx_chg:.2f}% vs 5 days ago)",
                f"  Fed outlook: {fed_outlook}",
                "  Note: 3M T-bill closely tracks the effective Fed funds rate.",
            ]
        else:
            lines.append("  T-bill data unavailable.")

    except Exception as exc:
        lines.append(f"  Market data fetch failed: {exc}")

    lines.append("")

    # ── 3. Macro Regime One-liner ─────────────────────────────────────────────
    lines.append("Macro Regime:")
    signals = []
    if spread is not None:
        signals.append(f"inverted curve ({spread:+.2f}%)" if spread < 0 else f"normal curve ({spread:+.2f}%)")
    if dxy_hist is not None and len(dxy_hist) >= 1:
        dxy_last = float(dxy_hist.iloc[-1])
        signals.append("strong USD" if dxy_last > 102 else "weak USD")
    lines.append(f"  {' | '.join(signals)}" if signals else "  Insufficient data for macro regime summary.")

    lines += [
        "",
        "Yield curve inversion precedes recessions by 6-24 months on average.",
        "It is a directional indicator, not a timing signal — do not use alone.",
    ]

    result = "\n".join(lines)
    _cache_set("macro_context", result)
    return result
