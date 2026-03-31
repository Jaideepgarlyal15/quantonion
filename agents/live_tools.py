"""
QuantOnion Live Agent Tools

Five tool functions for the standalone ConnectOnion agent (agent.py).
Each tool fetches live market data and computes results on demand —
no pre-loaded Streamlit session required.

Design contract:
  - All args are plain Python scalars (str, float, int) for LLM compatibility
  - All return values are plain strings
  - Failures return a descriptive error string, never raise
  - No Streamlit imports — these run outside Streamlit context
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.data_loader import load_price_data, normalize_symbol
from core.features import build_features
from core.hmm_model import (
    fit_hmm,
    label_states,
    current_run_length,
    expected_durations,
)
from backtesting.engine import run_backtest
from backtesting.metrics import compute_metrics
from strategies import STRATEGIES

_DEFAULT_START = "2018-01-01"


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: list_available_strategies
# ─────────────────────────────────────────────────────────────────────────────

def list_available_strategies() -> str:
    """
    Return the list of all available backtesting strategies.

    Call this first if unsure which strategy names to pass to
    run_backtest_analysis() or compare_all_strategies().

    Returns:
        Formatted string listing all strategy names.
    """
    names = list(STRATEGIES.keys())
    lines = ["Available strategies in QuantOnion:"]
    for n in names:
        lines.append(f"  - {n}")
    lines += [
        "",
        "Pass any of these names to run_backtest_analysis() or compare_all_strategies().",
        "The 'Regime Filter' strategy uses HMM regime detection as its signal.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: detect_current_regime
# ─────────────────────────────────────────────────────────────────────────────

def detect_current_regime(
    ticker: str,
    n_states: int = 3,
    start: str = _DEFAULT_START,
) -> str:
    """
    Fetch live price data and detect the current market regime using a Gaussian HMM.

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD', 'BHP.AX').
        n_states: Number of HMM regime states to detect (2, 3, or 4). Default 3.
        start: Start date for historical data in YYYY-MM-DD format. Default '2018-01-01'.

    Returns:
        Formatted string with current regime label, HMM confidence score,
        days in current regime, transition matrix, and per-regime statistics.
    """
    end = _today()
    ticker = normalize_symbol(ticker.strip())

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return (
            f"Could not fetch data for '{ticker}'. "
            "Check the ticker symbol. Examples: 'AAPL', '^GSPC', 'BTC-USD', 'BHP.AX'."
        )

    features = build_features(df, vol_window=21)
    if features.empty or len(features) < 60:
        return (
            f"Insufficient data for '{ticker}' to fit HMM "
            f"(got {len(features)} days, need at least 60). "
            "Try a longer date range: e.g. start='2015-01-01'."
        )

    try:
        hmm, _, states, post = fit_hmm(features, n_states=n_states, seed=42)
    except Exception as exc:
        return f"HMM fitting failed for '{ticker}': {exc}"

    _, labels_simple = label_states(features, states, metric="mean_return")

    # Derive advanced labels (worst to best by mean return)
    uniq = np.unique(states)
    ret_means = {s: features.loc[states == s, "ret"].mean() for s in uniq}
    order = sorted(uniq, key=lambda s: ret_means[s])
    adv_names = ["Crisis/Bear", "Neutral", "Bull", "Super Bull"]
    labels_adv = {s: adv_names[i] for i, s in enumerate(order)}

    state_series = pd.Series(states, index=features.index).ffill().dropna().astype(int)
    latest_state = int(state_series.iloc[-1])
    current_label = labels_simple.get(latest_state, "Unknown")
    current_conf = float(post[-1].max())
    days_in = current_run_length(state_series)
    diag_p = float(hmm.transmat_[latest_state, latest_state])
    typical_dur = float(1.0 / (1.0 - max(diag_p, 1e-9)))
    durations = expected_durations(hmm.transmat_)

    lines = [
        f"=== Regime Analysis: {ticker} ===",
        f"Data source : {src}",
        f"History     : {start} to {end} ({len(df)} trading days)",
        "",
        f"Current Regime : {current_label}",
        f"HMM Confidence : {current_conf:.0%}",
        f"Days in regime : {days_in}  (typical duration ~{typical_dur:.0f} days)",
        "",
        "Historical Regime Statistics (in-sample, annualised):",
    ]

    for state_id in sorted(labels_simple.keys()):
        label = labels_simple[state_id]
        adv = labels_adv.get(state_id, "")
        mask = states == state_id
        if mask.sum() == 0:
            continue
        ann_ret = float(features.loc[mask, "ret"].mean() * 252)
        ann_vol = float(features.loc[mask, "ret"].std() * np.sqrt(252))
        count = int(mask.sum())
        dur = float(durations[state_id]) if state_id < len(durations) else 0.0
        marker = "  <- CURRENT" if state_id == latest_state else ""
        lines.append(
            f"  {label:<12} ({adv:<12}): "
            f"Ann.Return={ann_ret:>7.1%}  Ann.Vol={ann_vol:>7.1%}  "
            f"Days={count:>4}  TypDur~{dur:.0f}d{marker}"
        )

    lines += [
        "",
        "Transition Matrix (rows = from-state, cols = to-state):",
    ]
    header_labels = [labels_simple.get(i, f"S{i}") for i in range(n_states)]
    lines.append("  " + "  ".join(f"{h:<10}" for h in header_labels))
    for i in range(n_states):
        row_label = labels_simple.get(i, f"S{i}")
        row = "  ".join(f"{hmm.transmat_[i, j]:.4f}    " for j in range(n_states))
        lines.append(f"  {row_label:<10}  {row}")

    lines += [
        "",
        "Note: HMM labels are assigned in-sample on historical data.",
        "Live regime detection lags reality by at least 1 trading day.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: run_backtest_analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_analysis(
    ticker: str,
    strategy: str = "Regime Filter",
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> str:
    """
    Fetch live price data and run a backtest for one strategy vs Buy & Hold.

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD').
        strategy: Strategy name. Call list_available_strategies() to see options.
                  Default 'Regime Filter'.
        start: Start date YYYY-MM-DD. Default '2018-01-01'.
        end: End date YYYY-MM-DD. Defaults to today.
        cost_bps: One-way transaction cost in basis points. Default 10 (= 0.10%).
        slippage_bps: One-way slippage in basis points. Default 5 (= 0.05%).

    Returns:
        Formatted string with full performance metrics for the strategy and
        Buy & Hold benchmark, plus CAGR/Sharpe/drawdown delta.
    """
    if end is None:
        end = _today()

    ticker = normalize_symbol(ticker.strip())

    if strategy not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        return f"Unknown strategy '{strategy}'. Available: {available}"

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return (
            f"Could not fetch data for '{ticker}'. "
            "Check the ticker symbol. Examples: 'AAPL', '^GSPC', 'BTC-USD'."
        )

    prices = df["Adj Close"]
    features = build_features(df, vol_window=21)

    # Build regime series for Regime Filter strategy
    regime_series = None
    if strategy == "Regime Filter":
        if features.empty or len(features) < 60:
            return (
                "Not enough data to fit HMM for Regime Filter "
                f"(got {len(features)} feature days, need 60+). "
                "Try a longer start date."
            )
        try:
            _, _, states, _ = fit_hmm(features, n_states=3, seed=42)
            _, labels_simple = label_states(features, states, metric="mean_return")
            regime_series = pd.Series(
                [labels_simple.get(int(s), "Unknown") for s in states],
                index=features.index,
            )
        except Exception as exc:
            return f"HMM fitting failed while preparing Regime Filter: {exc}"

    results: dict = {}
    for name in ["Buy & Hold", strategy]:
        if name not in STRATEGIES:
            continue
        cls = STRATEGIES[name]
        try:
            strat = cls()
            if name == "Regime Filter" and regime_series is not None:
                sigs = strat.generate_signals(prices, regime_series=regime_series)
            else:
                sigs = strat.generate_signals(prices)
            res = run_backtest(prices, sigs, cost_bps=cost_bps, slippage_bps=slippage_bps)
            if not res.empty:
                results[name] = compute_metrics(res)
        except Exception as exc:
            return f"Backtest failed for '{name}': {exc}"

    if not results:
        return (
            "Backtest produced no results. "
            "Ensure sufficient data (minimum 60 trading days)."
        )

    def _fmt(name: str, m: dict) -> list:
        pf = m.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf < 1_000 else "inf"
        return [
            f"--- {name} ---",
            f"Total Return    {m.get('total_return', 0):>10.1%}",
            f"CAGR            {m.get('cagr', 0):>10.1%}",
            f"Volatility      {m.get('volatility', 0):>10.1%}",
            f"Sharpe Ratio    {m.get('sharpe', 0):>10.2f}",
            f"Max Drawdown    {m.get('max_drawdown', 0):>10.1%}",
            f"Calmar Ratio    {m.get('calmar', 0):>10.2f}",
            f"Win Rate        {m.get('win_rate', 0):>10.1%}",
            f"Profit Factor   {pf_str:>10}",
            f"Time in Market  {m.get('time_in_market', 0):>10.1%}",
            f"# Trades        {m.get('n_trades', 0):>10}",
            f"# Days          {m.get('n_days', 0):>10}",
        ]

    lines = [
        f"=== Backtest: {strategy} on {ticker} ===",
        f"Period  : {start} to {end}  |  Source: {src}",
        f"Costs   : {cost_bps} bps transaction + {slippage_bps} bps slippage (one-way per leg)",
        "",
    ]

    for name, m in results.items():
        lines.extend(_fmt(name, m))
        lines.append("")

    if strategy != "Buy & Hold" and "Buy & Hold" in results and strategy in results:
        bh = results["Buy & Hold"]
        st = results[strategy]
        cagr_d = st.get("cagr", 0) - bh.get("cagr", 0)
        sharpe_d = st.get("sharpe", 0) - bh.get("sharpe", 0)
        dd_d = st.get("max_drawdown", 0) - bh.get("max_drawdown", 0)
        lines += [
            f"--- {strategy} vs Buy & Hold ---",
            f"CAGR delta      {cagr_d:>+10.1%}",
            f"Sharpe delta    {sharpe_d:>+10.2f}",
            f"Max DD delta    {dd_d:>+10.1%}  (negative = less drawdown = better protection)",
            "",
        ]

    lines.append(
        "WARNING: In-sample backtest on historical data. "
        "Past performance does not predict future results. Not financial advice."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: compare_all_strategies
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_strategies(
    ticker: str,
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> str:
    """
    Fetch live data and run all 6 built-in strategies, returning a comparison table.

    Results are ranked by Sharpe ratio (best risk-adjusted return first).

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD', '^NSEI').
        start: Start date YYYY-MM-DD. Default '2018-01-01'.
        end: End date YYYY-MM-DD. Defaults to today.
        cost_bps: One-way transaction cost in basis points. Default 10.
        slippage_bps: One-way slippage in basis points. Default 5.

    Returns:
        Formatted comparison table with CAGR, CAGR-vs-BuyHold, Sharpe,
        Max Drawdown, Win Rate, and Trade Count for all strategies.
    """
    if end is None:
        end = _today()

    ticker = normalize_symbol(ticker.strip())

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return (
            f"Could not fetch data for '{ticker}'. "
            "Check the ticker symbol."
        )

    prices = df["Adj Close"]
    features = build_features(df, vol_window=21)

    # Fit HMM once for Regime Filter
    regime_series = None
    if not features.empty and len(features) >= 60:
        try:
            _, _, states, _ = fit_hmm(features, n_states=3, seed=42)
            _, labels_simple = label_states(features, states, metric="mean_return")
            regime_series = pd.Series(
                [labels_simple.get(int(s), "Unknown") for s in states],
                index=features.index,
            )
        except Exception:
            pass  # Regime Filter will be skipped below

    results: dict = {}
    skipped: list = []
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
            continue

    if not results:
        return (
            "No backtest results produced. "
            "Ensure sufficient data (minimum 60 trading days after feature engineering)."
        )

    bh_cagr = results.get("Buy & Hold", {}).get("cagr", 0.0)
    best_sharpe = max(m.get("sharpe", -999) for m in results.values())

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("sharpe", -999),
        reverse=True,
    )

    col1, col2, col3, col4, col5, col6, col7 = 26, 8, 8, 8, 9, 8, 7
    header = (
        f"{'Strategy':<{col1}} "
        f"{'CAGR':>{col2}} "
        f"{'vs B&H':>{col3}} "
        f"{'Sharpe':>{col4}} "
        f"{'Max DD':>{col5}} "
        f"{'WinRate':>{col6}} "
        f"{'Trades':>{col7}}"
    )
    sep = "-" * (col1 + col2 + col3 + col4 + col5 + col6 + col7 + 6)

    lines = [
        f"=== Strategy Comparison: {ticker} ===",
        f"Period : {start} to {end}  |  Source: {src}",
        f"Costs  : {cost_bps} bps transaction + {slippage_bps} bps slippage  |  Sorted by Sharpe",
        "",
        header,
        sep,
    ]

    for name, m in sorted_results:
        cagr = m.get("cagr", 0.0)
        sharpe = m.get("sharpe", 0.0)
        maxdd = m.get("max_drawdown", 0.0)
        win_rate = m.get("win_rate", 0.0)
        trades = m.get("n_trades", 0)
        delta = cagr - bh_cagr if name != "Buy & Hold" else 0.0
        delta_str = f"{delta:+.1%}" if name != "Buy & Hold" else "  bench"
        star = " *" if sharpe == best_sharpe else ""
        lines.append(
            f"{name:<{col1}} "
            f"{cagr:>{col2}.1%} "
            f"{delta_str:>{col3}} "
            f"{sharpe:>{col4}.2f} "
            f"{maxdd:>{col5}.1%} "
            f"{win_rate:>{col6}.1%} "
            f"{trades:>{col7}}{star}"
        )

    lines += ["", "* = Best risk-adjusted return (Sharpe ratio)"]

    if skipped:
        lines.append(f"Skipped (insufficient data): {', '.join(skipped)}")

    lines += [
        "",
        "WARNING: In-sample backtest. Results do not represent live or future performance.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: get_risk_metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_metrics(
    ticker: str,
    start: str = _DEFAULT_START,
    end: Optional[str] = None,
) -> str:
    """
    Fetch live price data and return risk metrics: VaR, Expected Shortfall,
    annualised volatility, recent volatility, and return extremes.

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD').
        start: Start date YYYY-MM-DD. Default '2018-01-01'.
        end: End date YYYY-MM-DD. Defaults to today.

    Returns:
        Formatted string with 95%/99% historical VaR and CVaR,
        annualised and recent volatility, and best/worst single-day returns.
    """
    if end is None:
        end = _today()

    ticker = normalize_symbol(ticker.strip())

    df, src = load_price_data(ticker, start, end, allow_synth=False)
    if df.empty:
        return (
            f"Could not fetch data for '{ticker}'. "
            "Check the ticker symbol."
        )

    prices = df["Adj Close"]
    log_rets = np.log(prices / prices.shift(1)).dropna()

    if len(log_rets) < 30:
        return (
            f"Insufficient data for '{ticker}' "
            f"(got {len(log_rets)} days, need at least 30)."
        )

    ann_vol = float(log_rets.std() * np.sqrt(252))
    recent_vol = float(log_rets.tail(21).std() * np.sqrt(252))
    if recent_vol > ann_vol * 1.1:
        vol_flag = "elevated vs history"
    elif recent_vol < ann_vol * 0.9:
        vol_flag = "below historical average"
    else:
        vol_flag = "near historical average"

    var_95 = float(np.percentile(log_rets, 5))
    var_99 = float(np.percentile(log_rets, 1))
    tail_95 = log_rets[log_rets <= var_95]
    tail_99 = log_rets[log_rets <= var_99]
    es_95 = float(tail_95.mean()) if not tail_95.empty else var_95
    es_99 = float(tail_99.mean()) if not tail_99.empty else var_99

    worst_day = float(log_rets.min())
    best_day = float(log_rets.max())
    worst_date = str(log_rets.idxmin().date())
    best_date = str(log_rets.idxmax().date())

    n_years = max(len(log_rets) / 252.0, 1e-8)
    total_ret = float(np.exp(log_rets.sum()) - 1.0)
    cagr = float((1.0 + total_ret) ** (1.0 / n_years) - 1.0)

    lines = [
        f"=== Risk Metrics: {ticker} ===",
        f"Data source   : {src}",
        f"Period        : {start} to {end}  ({len(log_rets)} trading days)",
        "",
        "Return Profile:",
        f"  Annualised Return (CAGR)   : {cagr:>8.1%}",
        f"  Annualised Volatility      : {ann_vol:>8.1%}",
        f"  Recent Volatility (21-day) : {recent_vol:>8.1%}  ({vol_flag})",
        "",
        "1-Day Value-at-Risk (Historical Simulation):",
        f"  95% VaR  : {var_95:>8.2%}  (1-in-20 days loss threshold)",
        f"  99% VaR  : {var_99:>8.2%}  (1-in-100 days loss threshold)",
        "",
        "Expected Shortfall / CVaR (avg loss beyond VaR):",
        f"  95% ES   : {es_95:>8.2%}",
        f"  99% ES   : {es_99:>8.2%}",
        "",
        "Return Extremes:",
        f"  Worst single day : {worst_day:>8.2%}  ({worst_date})",
        f"  Best single day  : {best_day:>8.2%}  ({best_date})",
        "",
        "Note: Historical VaR/ES assumes the past return distribution repeats.",
        "Fat-tailed events (crashes) can exceed 99% VaR estimates.",
        "These metrics are for educational analysis only, not risk management.",
    ]
    return "\n".join(lines)
