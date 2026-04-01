"""
QuantOnion live tools for the ConnectOnion research agent.

Six functions the agent calls to fetch live market data and compute results.
Each returns a plain string. Each accepts only primitive argument types.
Exceptions are caught internally; errors are returned as strings.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
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


_DEFAULT_START = "2018-01-01"


def _today():
    return datetime.today().strftime("%Y-%m-%d")


def _fmt_pct(v):
    return f"{v:.1%}"


def _fmt2(v):
    return f"{v:.2f}"


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
    n_states: int = 3,
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

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return f"No data for '{ticker}'. Check the symbol (Yahoo Finance format, e.g. '^GSPC')."

    features = build_features(df, vol_window=21)
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

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return f"No data for '{ticker}'."

    prices = df["Adj Close"]
    features = build_features(df, vol_window=21)

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

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return f"No data for '{ticker}'."

    prices = df["Adj Close"]
    features = build_features(df, vol_window=21)

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

    df, src = load_price_data(ticker, start, end, allow_synth=False)
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

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return f"No data for '{ticker}'."

    features = build_features(df, vol_window=21)
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
