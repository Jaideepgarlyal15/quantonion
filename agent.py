"""
QuantOnion — Standalone ConnectOnion Research Agent

This is the agent entrypoint. It runs independently of the Streamlit app.

Usage:
    python agent.py                     # local: opens browser with built-in UI
    co deploy                           # deploy to ConnectOnion Cloud

What this agent can do (via natural language):
    "What regime is AAPL in right now?"
    "Run a backtest for TSLA with RSI Mean Reversion since 2020"
    "Compare all strategies on BTC-USD for the last 3 years"
    "What are the risk metrics for ^GSPC?"
    "Which strategy had the best Sharpe on NVDA in 2022-2023?"

The agent uses the re_act plugin to reason step-by-step and call tools
iteratively before composing its final answer. No Streamlit required.

Authentication:
    Run `co auth` once in your terminal before using AI model features.
    The agent works without auth for local testing (uses fallback responses).
"""

from __future__ import annotations

from pathlib import Path

from connectonion import Agent, host
from connectonion.useful_plugins import re_act

from agents.live_tools import (
    list_available_strategies,
    detect_current_regime,
    run_backtest_analysis,
    compare_all_strategies,
    get_risk_metrics,
)
from agents.sentiment_tools import (
    get_news_sentiment,
    get_social_buzz,
    get_fear_and_greed_index,
    get_research_brief,
)

# ── System prompt ─────────────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.txt"


def _load_system_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return (
            "You are QuantOnion Research Agent, an expert quantitative analyst. "
            "Use your tools to fetch live market data, run backtests on demand, "
            "and detect market regimes. Never give financial advice. "
            "Always caveat that past performance does not predict future results."
        )


# ── Agent factory (host() requires a callable, not an instance) ───────────────
_TOOLS = [
    list_available_strategies,
    detect_current_regime,
    run_backtest_analysis,
    compare_all_strategies,
    get_risk_metrics,
    get_news_sentiment,
    get_social_buzz,
    get_fear_and_greed_index,
    get_research_brief,
]


def create_agent() -> Agent:
    return Agent(
        name="quantonion_research_agent",
        model="co/gemini-2.5-pro",
        plugins=[re_act],
        tools=_TOOLS,
        system_prompt=_load_system_prompt(),
        max_iterations=10,
    )


# Module-level instance (used by tests / imports)
agent = create_agent()

# ── Local run (python agent.py) or co deploy ─────────────────────────────────
if __name__ == "__main__":
    host(create_agent)
