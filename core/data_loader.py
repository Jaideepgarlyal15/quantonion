import os
import time
import re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st


# -------------------------
# Normalise symbols
# -------------------------
ASX_SUFFIX = ".AX"

def normalize_symbol(sym: str) -> str:
    if not sym:
        return sym
    s = sym.strip()
    if s.upper().startswith("ASX:"):
        core = s.split(":", 1)[1].strip().upper()
        return core if core.endswith(ASX_SUFFIX) else f"{core}{ASX_SUFFIX}"
    if re.fullmatch(r"[A-Z]{3,4}", s) and not s.startswith("^"):
        return s + ASX_SUFFIX
    return s


def alias_candidates(sym: str):
    s = (sym or "").upper().strip()
    cands = [sym]

    if s in {"^GSPC", "^SPX", "GSPC", "SPX"}:
        cands += ["^GSPC", "SPY", "IVV"]

    if s in {"^NDX", "NDX"}:
        cands += ["^NDX", "QQQ"]

    if s in {"^AXJO", "AXJO"}:
        cands += ["^AXJO", "IOZ.AX", "STW.AX"]

    if s.endswith(".AX"):
        cands += [s.replace(".AX", ""), s]

    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -------------------------
# Helper standardisation
# -------------------------
def _standardize_price_df(df):
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


# -------------------------
# TwelveData loader
# -------------------------
@st.cache_data(ttl=3600)
def fetch_twelve_data(sym, start, end):
    api_key = st.secrets.get("TWELVE_DATA_API_KEY")
    if not api_key:
        return pd.DataFrame()

    url = "https://api.twelvedata.com/time_series"

    for trial in alias_candidates(sym):
        if trial.startswith("^"):
            continue

        variations = [trial]
        if trial.endswith(".AX"):
            variations.append(trial.replace(".AX", ""))
        if trial.isalpha() and not trial.endswith(".AX"):
            variations.append(f"{trial}.AX")

        for cand in variations:
            try:
                params = {
                    "symbol": cand,
                    "interval": "1day",
                    "outputsize": 5000,
                    "apikey": api_key,
                }
                r = requests.get(url, params=params, timeout=20)
                js = r.json()
                values = js.get("values")
                if not values:
                    continue

                df = pd.DataFrame(values)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime").sort_index()
                df["close"] = df["close"].astype(float)

                df = df.rename(columns={"close": "Adj Close"})
                df = df[["Adj Close"]]

                df = df.loc[(df.index >= pd.to_datetime(start)) &
                            (df.index <= pd.to_datetime(end))]

                if not df.empty:
                    return df
            except Exception:
                continue
    return pd.DataFrame()


# -------------------------
# Yahoo Finance loader
# -------------------------
@st.cache_data(ttl=3600)
def fetch_yf(sym, start, end):
    try:
        df = yf.download(
            sym,
            start=pd.to_datetime(start),
            end=pd.to_datetime(end) + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        return _standardize_price_df(df)
    except:
        return pd.DataFrame()


# -------------------------
# Alpha Vantage loader
# -------------------------
@st.cache_data(ttl=3600)
def fetch_alpha_vantage(sym, start, end):
    api_key = st.secrets.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return pd.DataFrame()

    for trial in alias_candidates(sym):
        if trial.startswith("^"):
            continue

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": trial,
                "outputsize": "full",
                "apikey": api_key,
            }
            r = requests.get(url, params=params, timeout=20)
            js = r.json()
            ts = js.get("Time Series (Daily)", {})
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

            df = df.loc[(df.index >= pd.to_datetime(start)) &
                        (df.index <= pd.to_datetime(end))]

            if not df.empty:
                return df
        except:
            continue

    return pd.DataFrame()


# -------------------------
# Unified loader
# -------------------------
def load_price_data(symbol, start, end, allow_synth=False):
    symbol = normalize_symbol(symbol)

    df = fetch_twelve_data(symbol, start, end)
    if not df.empty:
        return df, "twelvedata"

    for trial in alias_candidates(symbol):
        df = fetch_yf(trial, start, end)
        if not df.empty:
            return df, f"yfinance:{trial}"

    df = fetch_alpha_vantage(symbol, start, end)
    if not df.empty:
        return df, "alphavantage"

    if allow_synth:
        idx = pd.date_range(start=start, end=end, freq="B")
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0003, 0.012, len(idx))
        px = 100 * np.exp(np.cumsum(rets))
        return pd.DataFrame({"Adj Close": px}, index=idx), "synthetic"

    return pd.DataFrame(), "none"
