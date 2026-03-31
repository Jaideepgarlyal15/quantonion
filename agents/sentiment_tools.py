"""
QuantOnion Sentiment Tools

Four tool functions providing sentiment data from multiple free sources.
Designed for the standalone ConnectOnion agent — no Streamlit required.

Sources used (all free / no auth required unless noted):
  - Alpha Vantage NEWS_SENTIMENT  (requires ALPHAVANTAGE_API_KEY in env)
  - Yahoo Finance news headlines  (via yfinance, no key needed)
  - StockTwits public stream      (no auth, public API)
  - Reddit JSON API               (no auth, User-Agent header only)
  - alternative.me Fear & Greed   (crypto, no auth)
  - VIX via yfinance              (equity fear proxy, no key)
  - Finnhub news                  (optional FINNHUB_API_KEY for richer data)

Design contract:
  - All args are plain Python scalars (str) for LLM compatibility
  - All return values are plain strings
  - Failures return a descriptive error string, never raise
  - No Streamlit imports
  - No heavy NLP dependencies — uses lightweight keyword scoring
"""

from __future__ import annotations

import os
import re
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from typing import Optional

import requests
import yfinance as yf


# ── Helpers ──────────────────────────────────────────────────────────────────

_BULL_WORDS = {
    "bullish", "bull", "rally", "surge", "soar", "jump", "gain", "up",
    "rise", "rising", "outperform", "beat", "record", "high", "strong",
    "growth", "profit", "upgrade", "buy", "long", "boom", "recover",
    "breakthrough", "positive", "optimistic", "opportunity", "upside",
}

_BEAR_WORDS = {
    "bearish", "bear", "crash", "drop", "fall", "plunge", "decline",
    "down", "loss", "miss", "downgrade", "sell", "short", "recession",
    "inflation", "risk", "concern", "warn", "fear", "weak", "negative",
    "pessimistic", "correction", "selloff", "volatile", "uncertainty",
}

_HEADERS = {"User-Agent": "QuantOnion/1.0 Research-Tool (educational use)"}
_TIMEOUT = 8  # seconds per HTTP request


def _get_secret(key: str) -> Optional[str]:
    """Read from environment (works for both local .env and deployed contexts)."""
    val = os.environ.get(key, "").strip()
    return val if val else None


def _score_text(text: str) -> float:
    """
    Lightweight keyword-based sentiment scorer.
    Returns a score in [-1.0, +1.0]:
      +1.0 = strongly bullish, -1.0 = strongly bearish, 0.0 = neutral.
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    bull = len(words & _BULL_WORDS)
    bear = len(words & _BEAR_WORDS)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def _sentiment_label(score: float) -> str:
    if score >= 0.5:
        return "Strongly Bullish"
    if score >= 0.15:
        return "Bullish"
    if score <= -0.5:
        return "Strongly Bearish"
    if score <= -0.15:
        return "Bearish"
    return "Neutral"


def _safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Optional[dict]:
    """HTTP GET with timeout and error suppression."""
    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers or _HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _today() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def _clean_ticker_for_social(ticker: str) -> str:
    """Strip exchange suffixes for social APIs (BTC-USD -> BTC, BHP.AX -> BHP)."""
    ticker = ticker.upper()
    for sep in ["-", "."]:
        if sep in ticker:
            ticker = ticker.split(sep)[0]
    return ticker


# ── Tool 1: get_news_sentiment ────────────────────────────────────────────────

def get_news_sentiment(ticker: str) -> str:
    """
    Fetch recent news headlines for a ticker and return sentiment analysis.

    Pulls from two sources (whichever are available):
      1. Alpha Vantage NEWS_SENTIMENT API (requires ALPHAVANTAGE_API_KEY in env)
      2. Yahoo Finance news headlines (no key required)

    Scores each article using pre-computed scores (Alpha Vantage) or a
    lightweight keyword scorer (Yahoo Finance fallback).

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD').

    Returns:
        Formatted string with per-article headline + sentiment,
        overall score distribution, and source quality note.
    """
    ticker = ticker.strip().upper()
    lines = [f"=== News Sentiment: {ticker} ===", f"As of {_today()}", ""]

    av_key = _get_secret("ALPHAVANTAGE_API_KEY")
    articles_used: list[dict] = []
    source_note = ""

    # ── Source 1: Alpha Vantage NEWS_SENTIMENT ────────────────────────────────
    if av_key:
        av_ticker = ticker.replace("^", "").replace("-USD", "")
        data = _safe_get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": av_ticker,
                "limit": 15,
                "apikey": av_key,
            },
        )
        if data and "feed" in data:
            for item in data["feed"][:10]:
                headline = item.get("title", "")
                ts = item.get("time_published", "")[:10]
                source = item.get("source", "")
                # Use Alpha Vantage's own overall_sentiment_score if present
                av_score = None
                for ts_data in item.get("ticker_sentiment", []):
                    if ts_data.get("ticker", "").upper() == av_ticker.upper():
                        try:
                            av_score = float(ts_data.get("ticker_sentiment_score", 0))
                        except (ValueError, TypeError):
                            pass
                        break
                if av_score is None:
                    try:
                        av_score = float(item.get("overall_sentiment_score", 0))
                    except (ValueError, TypeError):
                        av_score = _score_text(headline)
                articles_used.append({
                    "headline": headline,
                    "date": ts,
                    "source": source,
                    "score": av_score,
                })
            source_note = "Alpha Vantage NEWS_SENTIMENT (pre-computed ticker-level scores)"

    # ── Source 2: Yahoo Finance headlines (fallback or supplement) ────────────
    if len(articles_used) < 5:
        try:
            yf_ticker = yf.Ticker(ticker)
            yf_news = yf_ticker.news or []
            for item in yf_news[:10]:
                content = item.get("content", {})
                headline = (
                    content.get("title", "")
                    or item.get("title", "")
                )
                ts = ""
                pub = content.get("pubDate") or item.get("providerPublishTime")
                if pub:
                    try:
                        if isinstance(pub, int):
                            ts = datetime.fromtimestamp(pub).strftime("%Y-%m-%d")
                        else:
                            ts = str(pub)[:10]
                    except Exception:
                        pass
                provider = content.get("provider", {}).get("displayName", "") or "Yahoo Finance"
                score = _score_text(headline)
                articles_used.append({
                    "headline": headline,
                    "date": ts,
                    "source": provider,
                    "score": score,
                })
            if not source_note:
                source_note = "Yahoo Finance headlines (keyword-scored)"
            else:
                source_note += " + Yahoo Finance headlines"
        except Exception:
            pass

    if not articles_used:
        lines.append(
            "No news data available. Possible reasons:\n"
            "  - ALPHAVANTAGE_API_KEY not set in environment\n"
            "  - Yahoo Finance returned no articles for this ticker\n"
            "  - Network timeout"
        )
        return "\n".join(lines)

    # ── Score distribution ────────────────────────────────────────────────────
    scores = [a["score"] for a in articles_used]
    avg_score = sum(scores) / len(scores)
    bull_count = sum(1 for s in scores if s > 0.15)
    bear_count = sum(1 for s in scores if s < -0.15)
    neutral_count = len(scores) - bull_count - bear_count

    lines += [
        f"Source         : {source_note}",
        f"Articles found : {len(articles_used)}",
        f"Overall Score  : {avg_score:+.3f}  →  {_sentiment_label(avg_score)}",
        f"Distribution   : Bullish={bull_count}  Neutral={neutral_count}  Bearish={bear_count}",
        "",
        "Recent Headlines:",
    ]

    for a in articles_used[:8]:
        label = _sentiment_label(a["score"])
        date_str = f"[{a['date']}] " if a["date"] else ""
        lines.append(f"  {date_str}{label:>18}  {a['headline'][:90]}")

    lines += [
        "",
        "Note: Sentiment scores measure tone, not accuracy. Headlines can be misleading.",
        "Do not trade based on sentiment signals alone.",
    ]
    return "\n".join(lines)


# ── Tool 2: get_social_buzz ───────────────────────────────────────────────────

def get_social_buzz(ticker: str) -> str:
    """
    Fetch social media sentiment for a ticker from StockTwits and Reddit.

    Sources:
      - StockTwits public stream (bull/bear counts from tagged messages)
      - Reddit: r/wallstreetbets, r/stocks, r/investing (keyword scored)

    No authentication required for either source.

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD').

    Returns:
        Formatted string with StockTwits bull/bear ratio, top Reddit posts,
        and overall social sentiment score.
    """
    ticker = ticker.strip().upper()
    social_ticker = _clean_ticker_for_social(ticker)
    lines = [f"=== Social Sentiment: {ticker} ===", f"As of {_today()}", ""]

    overall_scores: list[float] = []

    # ── StockTwits ────────────────────────────────────────────────────────────
    lines.append("[ StockTwits ]")
    st_data = _safe_get(
        f"https://api.stocktwits.com/api/2/streams/symbol/{social_ticker}.json",
        headers=_HEADERS,
    )
    if st_data and "messages" in st_data:
        messages = st_data["messages"]
        def _st_sentiment(m: dict) -> str:
            ent = m.get("entities") or {}
            sent = ent.get("sentiment") or {}
            return sent.get("basic", "") if isinstance(sent, dict) else ""

        bull = sum(1 for m in messages if _st_sentiment(m) == "Bullish")
        bear = sum(1 for m in messages if _st_sentiment(m) == "Bearish")
        tagged = bull + bear
        untagged = len(messages) - tagged

        lines += [
            f"  Messages fetched : {len(messages)}",
            f"  Bullish tagged   : {bull}",
            f"  Bearish tagged   : {bear}",
            f"  Untagged/Neutral : {untagged}",
        ]
        if tagged > 0:
            st_score = (bull - bear) / tagged
            overall_scores.append(st_score)
            lines.append(f"  Sentiment ratio  : {_sentiment_label(st_score)}  (score {st_score:+.2f})")

        # Show a few recent bodies
        lines.append("  Recent posts:")
        for m in messages[:4]:
            body = m.get("body", "").replace("\n", " ")[:80]
            ent = m.get("entities") or {}
            sent = ent.get("sentiment") or {}
            sentiment_tag = sent.get("basic", "") if isinstance(sent, dict) else ""
            tag_str = f"[{sentiment_tag}] " if sentiment_tag else ""
            lines.append(f"    {tag_str}{body}")
    else:
        lines.append("  StockTwits: no data (ticker may not be listed, or rate limited)")

    lines.append("")

    # ── Reddit ────────────────────────────────────────────────────────────────
    lines.append("[ Reddit ]")
    subreddits = ["wallstreetbets", "stocks", "investing"]
    reddit_posts: list[dict] = []

    for sub in subreddits:
        data = _safe_get(
            f"https://www.reddit.com/r/{sub}/search.json",
            params={
                "q": social_ticker,
                "sort": "top",
                "t": "week",
                "limit": 5,
                "type": "link",
            },
            headers=_HEADERS,
        )
        if data and "data" in data:
            for post in data["data"].get("children", []):
                pd_ = post.get("data", {})
                title = pd_.get("title", "")
                score = pd_.get("score", 0)
                comments = pd_.get("num_comments", 0)
                if title:
                    sentiment_score = _score_text(title)
                    reddit_posts.append({
                        "sub": sub,
                        "title": title,
                        "score": score,
                        "comments": comments,
                        "sentiment": sentiment_score,
                    })

    if reddit_posts:
        reddit_scores = [p["sentiment"] for p in reddit_posts]
        avg_reddit = sum(reddit_scores) / len(reddit_scores)
        overall_scores.append(avg_reddit)
        lines += [
            f"  Posts found (last 7 days) : {len(reddit_posts)} across {', '.join('r/'+s for s in subreddits)}",
            f"  Reddit sentiment          : {_sentiment_label(avg_reddit)}  (score {avg_reddit:+.2f})",
            "  Top posts:",
        ]
        sorted_posts = sorted(reddit_posts, key=lambda x: x["score"], reverse=True)
        for p in sorted_posts[:5]:
            lines.append(
                f"    r/{p['sub']} | ↑{p['score']:>5} | {_sentiment_label(p['sentiment']):>18} | {p['title'][:70]}"
            )
    else:
        lines.append("  Reddit: no posts found (rate limited or no recent discussion)")

    lines.append("")

    # ── Overall social summary ────────────────────────────────────────────────
    if overall_scores:
        combined = sum(overall_scores) / len(overall_scores)
        lines += [
            f"Overall Social Sentiment : {_sentiment_label(combined)}  (score {combined:+.2f})",
            "",
            "Note: Social sentiment is highly noisy and often reflects short-term speculation.",
            "High Reddit volume ≠ informed opinion. StockTwits tags are self-reported.",
        ]
    else:
        lines.append("No social data collected. Both sources returned empty results.")

    return "\n".join(lines)


# ── Tool 3: get_fear_and_greed_index ─────────────────────────────────────────

def get_fear_and_greed_index() -> str:
    """
    Return current fear and greed readings for both equity and crypto markets.

    Sources:
      - VIX (CBOE Volatility Index) via yfinance — equity market fear proxy
      - alternative.me Fear & Greed Index — crypto market sentiment (7-day trend)

    No API key required.

    Returns:
        Formatted string with current VIX level, VIX interpretation,
        crypto F&G index value and classification, and 7-day trend.
    """
    lines = [f"=== Fear & Greed Indicators ===", f"As of {_today()}", ""]

    # ── VIX (equity fear) ─────────────────────────────────────────────────────
    lines.append("[ Equity Fear: VIX (CBOE Volatility Index) ]")
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            vix_val = float(vix_hist["Close"].iloc[-1])
            vix_prev = float(vix_hist["Close"].iloc[-2]) if len(vix_hist) >= 2 else vix_val
            vix_change = vix_val - vix_prev

            if vix_val < 15:
                vix_label = "Extreme Greed / Very Low Fear"
                vix_context = "Markets are complacent. Historically precedes volatility spikes."
            elif vix_val < 20:
                vix_label = "Low Fear (Calm)"
                vix_context = "Below historical average (~20). Low near-term uncertainty priced in."
            elif vix_val < 25:
                vix_label = "Moderate Fear"
                vix_context = "Near historical average. Normal market uncertainty."
            elif vix_val < 35:
                vix_label = "Elevated Fear"
                vix_context = "Above average. Markets pricing in material uncertainty or risk events."
            else:
                vix_label = "Extreme Fear / Crisis"
                vix_context = "Crisis-level volatility. Historical spikes above 35 mark major sell-offs."

            change_str = f"{'+'if vix_change>=0 else ''}{vix_change:.2f} vs prev close"
            lines += [
                f"  Current VIX : {vix_val:.2f}  ({change_str})",
                f"  Reading     : {vix_label}",
                f"  Context     : {vix_context}",
            ]
        else:
            lines.append("  VIX: data unavailable from yfinance")
    except Exception as e:
        lines.append(f"  VIX: error fetching data — {e}")

    lines.append("")

    # ── alternative.me Crypto Fear & Greed ───────────────────────────────────
    lines.append("[ Crypto Fear & Greed Index (alternative.me) ]")
    fg_data = _safe_get("https://api.alternative.me/fng/?limit=7&format=json")
    if fg_data and "data" in fg_data and len(fg_data["data"]) > 0:
        today_entry = fg_data["data"][0]
        today_score = int(today_entry.get("value", 50))
        today_class = today_entry.get("value_classification", "Neutral")

        # 7-day trend
        values = []
        for entry in fg_data["data"]:
            try:
                values.append(int(entry["value"]))
            except (KeyError, ValueError):
                pass

        trend = ""
        if len(values) >= 3:
            recent_avg = sum(values[:3]) / 3
            older_avg = sum(values[4:]) / max(len(values[4:]), 1)
            delta = recent_avg - older_avg
            if delta > 5:
                trend = "↑ Rising (shifting toward greed)"
            elif delta < -5:
                trend = "↓ Falling (shifting toward fear)"
            else:
                trend = "→ Stable"

        lines += [
            f"  Today's Score  : {today_score}/100  →  {today_class}",
            f"  7-Day Trend    : {trend}",
            "  Scale          : 0=Extreme Fear, 25=Fear, 50=Neutral, 75=Greed, 100=Extreme Greed",
            "",
            "  7-Day History:",
        ]
        for i, entry in enumerate(fg_data["data"][:7]):
            val = entry.get("value", "?")
            cls = entry.get("value_classification", "")
            ts = entry.get("timestamp", "")
            try:
                date_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
            except Exception:
                date_str = "?"
            marker = "  <- TODAY" if i == 0 else ""
            lines.append(f"    {date_str}  {val:>3}/100  {cls}{marker}")
    else:
        lines.append("  Crypto F&G: data unavailable (alternative.me may be down)")

    lines += [
        "",
        "Interpretation Guide:",
        "  VIX <15           → equity complacency (contrarian bearish signal)",
        "  VIX 15–25         → normal conditions",
        "  VIX >30           → elevated stress; often buying opportunity historically",
        "  Crypto F&G <25    → fear: historically good entry for long-term BTC holders",
        "  Crypto F&G >75    → greed: historically elevated drawdown risk",
        "",
        "Note: Fear and greed indices are sentiment proxies, not timing signals.",
        "They are most useful as context, not as standalone trade triggers.",
    ]
    return "\n".join(lines)


# ── Tool 4: get_research_brief ────────────────────────────────────────────────

def get_research_brief(
    ticker: str,
    start: str = "2020-01-01",
) -> str:
    """
    Generate a comprehensive research brief for a ticker.

    Combines quantitative analysis (regime detection + backtesting) with
    sentiment analysis (news, social, fear & greed) into a structured report
    that equips the user to make their own informed decision.

    The brief explicitly presents bull and bear evidence side-by-side and
    does NOT recommend buying, selling, or any specific position.

    Args:
        ticker: Asset ticker symbol (e.g. 'AAPL', '^GSPC', 'BTC-USD').
        start: Start date for quant analysis in YYYY-MM-DD format. Default '2020-01-01'.

    Returns:
        Formatted multi-section research brief with:
          1. Quantitative Regime Analysis
          2. Recent Strategy Performance
          3. News Sentiment Summary
          4. Social Sentiment Summary
          5. Market Fear & Greed Context
          6. Signal Scorecard (bull / bear / neutral)
          7. Decision Framework (what to consider, what to verify)
          8. Disclaimers
    """
    ticker = ticker.strip().upper()
    lines = [
        f"{'='*60}",
        f"  QUANTONION RESEARCH BRIEF: {ticker}",
        f"  Generated: {_today()}",
        f"{'='*60}",
        "",
        "This brief presents data and evidence for your own analysis.",
        "It does NOT recommend any position. All figures are historical.",
        "",
    ]

    bull_signals: list[str] = []
    bear_signals: list[str] = []
    neutral_signals: list[str] = []

    # ── Section 1: Regime ─────────────────────────────────────────────────────
    lines.append("━━━ 1. MARKET REGIME (HMM) ━━━")
    try:
        from agents.live_tools import detect_current_regime
        regime_text = detect_current_regime(ticker, n_states=3, start=start)
        for line in regime_text.split("\n"):
            if "Current Regime" in line or "HMM Confidence" in line or "Days in regime" in line:
                lines.append(f"  {line.strip()}")
                # Classify signal
                regime_lower = line.lower()
                if any(w in regime_lower for w in ["bull", "calm", "super"]):
                    bull_signals.append(f"Regime: {line.strip()}")
                elif any(w in regime_lower for w in ["bear", "crisis", "stormy", "choppy"]):
                    bear_signals.append(f"Regime: {line.strip()}")
                else:
                    neutral_signals.append(f"Regime: {line.strip()}")
    except Exception as exc:
        lines.append(f"  Regime detection unavailable: {exc}")

    lines.append("")

    # ── Section 2: Strategy Performance (top strategy vs BH) ─────────────────
    lines.append("━━━ 2. RECENT STRATEGY PERFORMANCE ━━━")
    try:
        from agents.live_tools import compare_all_strategies
        comp_text = compare_all_strategies(ticker, start=start)
        # Pull the header and top 3 rows from the comparison table
        in_table = False
        row_count = 0
        for line in comp_text.split("\n"):
            if line.startswith("===") or line.startswith("Period") or line.startswith("Costs"):
                lines.append(f"  {line.strip()}")
            if "Strategy" in line and "CAGR" in line:
                in_table = True
                lines.append(f"  {line}")
                continue
            if in_table:
                if line.startswith("---") or not line.strip():
                    break
                lines.append(f"  {line}")
                row_count += 1
                # Classify based on best strategy Sharpe vs Buy & Hold
                if row_count == 1 and "Buy & Hold" not in line:
                    # Best Sharpe strategy is not BH → active strategy adds value
                    bull_signals.append(f"Strategy edge: active strategy outperforms Buy & Hold on Sharpe")
    except Exception as exc:
        lines.append(f"  Strategy comparison unavailable: {exc}")

    lines.append("")

    # ── Section 3: News Sentiment ─────────────────────────────────────────────
    lines.append("━━━ 3. NEWS SENTIMENT ━━━")
    try:
        news_text = get_news_sentiment(ticker)
        for line in news_text.split("\n"):
            if any(k in line for k in ["Overall Score", "Distribution", "Source", "Articles found"]):
                lines.append(f"  {line.strip()}")
                if "Overall Score" in line:
                    if "Bullish" in line:
                        bull_signals.append(f"News: {line.strip()}")
                    elif "Bearish" in line:
                        bear_signals.append(f"News: {line.strip()}")
                    else:
                        neutral_signals.append(f"News: Neutral sentiment")
    except Exception as exc:
        lines.append(f"  News sentiment unavailable: {exc}")

    lines.append("")

    # ── Section 4: Social Buzz ────────────────────────────────────────────────
    lines.append("━━━ 4. SOCIAL SENTIMENT ━━━")
    try:
        social_text = get_social_buzz(ticker)
        for line in social_text.split("\n"):
            if any(k in line for k in ["Overall Social", "StockTwits:", "Reddit sentiment", "Posts found"]):
                lines.append(f"  {line.strip()}")
                if "Overall Social" in line:
                    if "Bullish" in line:
                        bull_signals.append(f"Social: {line.strip()}")
                    elif "Bearish" in line:
                        bear_signals.append(f"Social: {line.strip()}")
                    else:
                        neutral_signals.append(f"Social: {line.strip()}")
    except Exception as exc:
        lines.append(f"  Social sentiment unavailable: {exc}")

    lines.append("")

    # ── Section 5: Fear & Greed ───────────────────────────────────────────────
    lines.append("━━━ 5. MARKET FEAR & GREED ━━━")
    try:
        fg_text = get_fear_and_greed_index()
        for line in fg_text.split("\n"):
            if any(k in line for k in ["Current VIX", "Reading", "Today's Score", "7-Day Trend"]):
                lines.append(f"  {line.strip()}")
                if "Reading" in line:
                    if "Greed" in line or "Low Fear" in line:
                        neutral_signals.append(f"VIX: {line.strip()} (complacency risk)")
                    elif "Extreme Fear" in line or "Crisis" in line:
                        bear_signals.append(f"VIX: {line.strip()}")
                    else:
                        neutral_signals.append(f"VIX: {line.strip()}")
    except Exception as exc:
        lines.append(f"  Fear & Greed unavailable: {exc}")

    lines.append("")

    # ── Section 6: Signal Scorecard ───────────────────────────────────────────
    total = len(bull_signals) + len(bear_signals) + len(neutral_signals)
    if total > 0:
        bull_pct = len(bull_signals) / total
        bear_pct = len(bear_signals) / total
        if bull_pct > 0.6:
            overall_bias = "CAUTIOUSLY CONSTRUCTIVE (more bullish signals than bearish)"
        elif bear_pct > 0.6:
            overall_bias = "CAUTIOUSLY CAUTIOUS (more bearish signals than bullish)"
        else:
            overall_bias = "MIXED / NEUTRAL (signals do not strongly favour either direction)"
    else:
        overall_bias = "INSUFFICIENT DATA"

    lines += [
        "━━━ 6. SIGNAL SCORECARD ━━━",
        f"  Overall Bias : {overall_bias}",
        f"  Bull signals : {len(bull_signals)}",
        f"  Bear signals : {len(bear_signals)}",
        f"  Neutral      : {len(neutral_signals)}",
        "",
        "  Bullish Evidence:",
    ]
    for s in bull_signals:
        lines.append(f"    + {s}")
    if not bull_signals:
        lines.append("    (none detected)")

    lines.append("  Bearish Evidence:")
    for s in bear_signals:
        lines.append(f"    - {s}")
    if not bear_signals:
        lines.append("    (none detected)")

    lines.append("  Neutral / Context:")
    for s in neutral_signals:
        lines.append(f"    ~ {s}")
    if not neutral_signals:
        lines.append("    (none detected)")

    lines += [
        "",
        "━━━ 7. DECISION FRAMEWORK ━━━",
        "  Before making any decision, consider verifying the following:",
        "",
        "  Fundamental questions:",
        "    □ What is the company/asset's current earnings trend or macro driver?",
        "    □ Does the current price reflect fair value by standard valuation metrics?",
        "    □ Are there catalysts (earnings, central bank decisions, macro events) in the near term?",
        "",
        "  Risk management questions:",
        "    □ What is your maximum acceptable loss on this position?",
        "    □ Does this fit your overall portfolio allocation?",
        "    □ What is your exit strategy if the thesis is wrong?",
        "",
        "  Questions this tool CANNOT answer:",
        "    □ Whether current price is a good entry point",
        "    □ How long the current regime will persist",
        "    □ Whether sentiment is leading or lagging price",
        "",
        "━━━ 8. DISCLAIMERS ━━━",
        "  ⚠  This research brief is for EDUCATIONAL PURPOSES ONLY.",
        "  ⚠  It does not constitute investment advice or a recommendation.",
        "  ⚠  All backtest figures are IN-SAMPLE and subject to overfitting.",
        "  ⚠  Sentiment indicators are lagging or coincident, not predictive.",
        "  ⚠  Past performance does not predict future results.",
        "  ⚠  Consult a qualified financial advisor before making investment decisions.",
        "",
        "  QuantOnion Research Agent — quantitative and sentiment analysis platform",
        f"  Report generated: {_today()}",
    ]

    return "\n".join(lines)
