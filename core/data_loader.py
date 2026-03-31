"""
QuantOnion Data Loader

Multi-source market data retrieval with caching and graceful fallback.

Priority order (free sources first):
  1. Yahoo Finance via yfinance  — free, no key, primary source
  2. Alpha Vantage               — free tier (25 requests/day), requires key
  3. Twelve Data                 — free tier (800 requests/day), requires key
  4. Synthetic                   — deterministic GBM, demo/testing only

API keys are read from environment variables (or st.secrets in Streamlit).
The app functions fully without any paid keys.

Data freshness note:
  Yahoo Finance data has a ~15-minute delay for most assets and is suitable
  for daily close-based analysis. It is not suitable for intraday trading.
"""

import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

ASX_SUFFIX = ".AX"


# ── Symbol normalisation ───────────────────────────────────────────────────────

def normalize_symbol(sym: str) -> str:
    """
    Normalise user-provided ticker symbols.

    Only converts "ASX:XXX" prefix notation to "XXX.AX" format.
    All other symbols (including US tickers like AAPL, MSFT) are
    returned unchanged — they must be passed exactly as yfinance expects.

    Args:
        sym: Raw ticker string from user input.

    Returns:
        Normalised ticker string.

    Examples:
        "ASX:BHP"  → "BHP.AX"
        "^GSPC"    → "^GSPC"
        "AAPL"     → "AAPL"
        "BHP.AX"   → "BHP.AX"
    """
    if not sym:
        return sym
    s = sym.strip()
    if s.upper().startswith("ASX:"):
        core = s.split(":", 1)[1].strip().upper()
        return core if core.endswith(ASX_SUFFIX) else f"{core}{ASX_SUFFIX}"
    return s


def alias_candidates(sym: str) -> list:
    """
    Return a list of symbol aliases to try when the primary symbol fails.

    Provides fallbacks for common index aliases and ASX dual-listings.
    """
    s = (sym or "").upper().strip()
    cands = [sym]

    if s in {"^GSPC", "^SPX", "GSPC", "SPX"}:
        cands += ["^GSPC", "SPY", "IVV"]
    elif s in {"^NDX", "NDX"}:
        cands += ["^NDX", "QQQ"]
    elif s in {"^AXJO", "AXJO"}:
        cands += ["^AXJO", "IOZ.AX", "STW.AX"]
    elif s.endswith(".AX"):
        # Also try without suffix in case the exchange listing is elsewhere
        cands.append(s.replace(".AX", ""))

    # Deduplicate while preserving order
    seen: set = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a price DataFrame has a DatetimeIndex and an 'Adj Close' column."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # yfinance ≥0.2 returns MultiIndex columns like (Price, Ticker) for single tickers.
    # Flatten to just the price-type level before any column operations.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.rename(columns=str.title)

    if "Adj Close" not in df.columns:
        if "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        else:
            return pd.DataFrame()

    return df[["Adj Close"]]


def _get_secret(key: str) -> str:
    """
    Read a secret from st.secrets (Streamlit) or environment variables.
    Returns empty string if not found anywhere.
    """
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, "")


# ── Individual loaders ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_yf(sym: str, start, end) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Free, no API key required. Suitable for daily historical data.
    Data has approximately 15-minute delay and is not tick-accurate.
    """
    try:
        df = yf.download(
            sym,
            start=pd.to_datetime(start),
            end=pd.to_datetime(end) + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        # yfinance ≥0.2 returns MultiIndex columns (Price, Ticker) for single tickers.
        # Flatten to single level before any further processing.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _standardize_price_df(df)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_alpha_vantage(sym: str, start, end) -> pd.DataFrame:
    """
    Fetch daily adjusted prices from Alpha Vantage.

    Requires ALPHAVANTAGE_API_KEY. Free tier: 25 requests/day.
    Falls back gracefully if key is absent or rate limited.
    """
    api_key = _get_secret("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return pd.DataFrame()

    for trial in alias_candidates(sym):
        if trial.startswith("^"):
            continue  # Alpha Vantage does not support index tickers
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": trial,
                "outputsize": "full",
                "apikey": api_key,
            }
            r = requests.get(
                "https://www.alphavantage.co/query", params=params, timeout=20
            )
            ts = r.json().get("Time Series (Daily)", {})
            if not ts:
                continue

            df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            if "5. adjusted close" in df.columns:
                df = df.rename(columns={"5. adjusted close": "Adj Close"})
            elif "4. close" in df.columns:
                df = df.rename(columns={"4. close": "Adj Close"})
            else:
                continue

            df = df[["Adj Close"]]
            df = df.loc[
                (df.index >= pd.to_datetime(start))
                & (df.index <= pd.to_datetime(end))
            ]
            if not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_twelve_data(sym: str, start, end) -> pd.DataFrame:
    """
    Fetch daily close prices from Twelve Data.

    Requires TWELVE_DATA_API_KEY. Free tier: 800 requests/day.
    Falls back gracefully if key is absent.
    """
    api_key = _get_secret("TWELVE_DATA_API_KEY")
    if not api_key:
        return pd.DataFrame()

    for trial in alias_candidates(sym):
        if trial.startswith("^"):
            continue

        variations = [trial]
        if trial.endswith(".AX"):
            variations.append(trial.replace(".AX", ""))

        for cand in variations:
            try:
                params = {
                    "symbol": cand,
                    "interval": "1day",
                    "outputsize": 5000,
                    "apikey": api_key,
                }
                r = requests.get(
                    "https://api.twelvedata.com/time_series",
                    params=params,
                    timeout=20,
                )
                values = r.json().get("values")
                if not values:
                    continue

                df = pd.DataFrame(values)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime").sort_index()
                df["close"] = df["close"].astype(float)
                df = df.rename(columns={"close": "Adj Close"})[["Adj Close"]]
                df = df.loc[
                    (df.index >= pd.to_datetime(start))
                    & (df.index <= pd.to_datetime(end))
                ]
                if not df.empty:
                    return df
            except Exception:
                continue

    return pd.DataFrame()


# ── Unified loader ────────────────────────────────────────────────────────────

def load_price_data(
    symbol: str,
    start,
    end,
    allow_synth: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Load adjusted daily close prices with automatic source fallback.

    Sources tried in order:
      1. Yahoo Finance (free, no key required)
      2. Alpha Vantage (if ALPHAVANTAGE_API_KEY is set)
      3. Twelve Data (if TWELVE_DATA_API_KEY is set)
      4. Synthetic GBM (only if allow_synth=True, for demos/testing)

    Args:
        symbol:     Normalised ticker string.
        start:      Start date (date, datetime, or string).
        end:        End date (date, datetime, or string).
        allow_synth: If True and all sources fail, return synthetic data.

    Returns:
        Tuple of (DataFrame with 'Adj Close' column, source label string).
        Returns (empty DataFrame, "none") if all sources fail.
    """
    symbol = normalize_symbol(symbol)

    # yfinance first — free, no key, handles most global markets
    for trial in alias_candidates(symbol):
        df = fetch_yf(trial, start, end)
        if not df.empty:
            return df, f"yfinance:{trial}"

    # Alpha Vantage — if key available
    df = fetch_alpha_vantage(symbol, start, end)
    if not df.empty:
        return df, "alphavantage"

    # Twelve Data — if key available
    df = fetch_twelve_data(symbol, start, end)
    if not df.empty:
        return df, "twelvedata"

    # Synthetic fallback — deterministic, for demo purposes only
    if allow_synth:
        idx = pd.date_range(start=start, end=end, freq="B")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0003, 0.012, len(idx))
        px = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame({"Adj Close": px}, index=idx), "synthetic"

    return pd.DataFrame(), "none"
