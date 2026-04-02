"""
QuantOnion Research Agent

ConnectOnion-powered agent that analyses backtest results and market regimes,
providing plain-English explanations and answering researcher questions.

ConnectOnion usage (v0.8.x):
    from connectonion import Agent
    from connectonion.useful_plugins import re_act

    agent = Agent(
        name="...",
        model="co/gemini-2.5-pro",   # ConnectOnion-hosted model
        plugins=[re_act],            # imported objects, not strings
        tools=[fn1, fn2, ...],
        system_prompt="prompts/system.txt",
    )
    response = agent.input("your query")   # .input(), not .run()

Why re_act:
    The re_act plugin implements the ReAct (Reason + Act) pattern, letting the
    agent reason step-by-step and call tools iteratively before composing its
    final answer. For multi-strategy comparison this is more reliable than
    one-shot prompting because the agent retrieves actual computed data for
    each strategy before drawing conclusions.

Authentication:
    ConnectOnion authenticates via the CLI — run `co auth` once in your terminal.
    No API key is required in code or environment variables.
    Falls back to a deterministic summary when ConnectOnion is not installed or
    authentication fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agents.tools import make_tools

# ── ConnectOnion availability check ──────────────────────────────────────────
try:
    from connectonion import Agent  # type: ignore[import]
    _CONNECTONION_AVAILABLE = True
except ImportError:
    _CONNECTONION_AVAILABLE = False
    Agent = None      # type: ignore[assignment, misc]

# ── System prompt path ────────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system.txt"


def _load_system_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return (
            "You are QuantOnion Research Agent, an expert quantitative analyst. "
            "Analyse backtest results and market regimes honestly and clearly. "
            "Never give financial advice. Always caveat past performance."
        )


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_research_agent(context: Dict[str, Any]) -> Optional[Any]:
    """
    Create a ConnectOnion Research Agent with tools bound to the backtest context.

    Uses:
      - 5 tools               — strategy metrics, regime stats, benchmark comparison, risk summary, ML forecasts
      - co/gpt-5-nano         — ConnectOnion-hosted GPT-5 Nano model (fast)
      - CLI auth              — authenticate once via `co auth` (no API key in code)

    Args:
        context: Dict with backtest_results, regime_stats, current_regime,
                 ticker, period.

    Returns:
        Agent instance, or None if ConnectOnion is not installed or not authenticated.
    """
    if not _CONNECTONION_AVAILABLE:
        return None

    tools = make_tools(context)
    system_prompt = _load_system_prompt()

    try:
        agent = Agent(
            name="quantonion_research_agent",
            model="co/gemini-2.5-pro",
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=8,
        )
        return agent
    except Exception:
        return None


# ── Run helper ────────────────────────────────────────────────────────────────

def run_agent_analysis(
    context: Dict[str, Any],
    query: str,
    agent: Optional[Any] = None,
) -> str:
    """
    Run the research agent and return its analysis as a string.

    Falls back to a deterministic rule-based summary when ConnectOnion
    or the OpenAI API key is not available — the Agent tab is always useful.

    Args:
        context: Backtest and regime context dict.
        query:   User question or analysis request.
        agent:   Pre-created agent (avoids recreating on every call). Pass
                 None to create one from context.

    Returns:
        Analysis string (always non-empty).
    """
    if agent is None:
        agent = create_research_agent(context)

    if agent is not None:
        try:
            response = agent.input(query)   # ConnectOnion API: .input(), not .run()
            return str(response)
        except Exception as exc:
            fallback = _deterministic_summary(context)
            return (
                f"*Agent error: {exc}*\n\n"
                f"**Deterministic summary (no AI required):**\n\n{fallback}"
            )

    return _deterministic_summary(context)


# ── Deterministic fallback ────────────────────────────────────────────────────

def _deterministic_summary(context: Dict[str, Any]) -> str:
    """
    Rule-based analysis summary for when ConnectOnion / OpenAI is unavailable.

    Produces a useful, factual summary from backtest context without any LLM call.
    """
    results = context.get("backtest_results", {})
    ticker = context.get("ticker", "Unknown")
    current_regime = context.get("current_regime", "Unknown")
    period = context.get("period", "Unknown")

    if not results:
        return (
            "No backtest results available yet.\n\n"
            "Run a backtest using the sidebar controls, then return here for analysis."
        )

    lines = [
        f"## QuantOnion Research Summary — {ticker}",
        f"**Period:** {period}  |  **Current Regime:** {current_regime}",
        "",
        "### Strategy Performance",
    ]

    sorted_by_sharpe = sorted(
        results.items(),
        key=lambda x: x[1].get("metrics", {}).get("sharpe", -999),
        reverse=True,
    )
    bh_cagr = results.get("Buy & Hold", {}).get("metrics", {}).get("cagr", 0.0)

    for name, res in sorted_by_sharpe:
        m = res.get("metrics", {})
        cagr = m.get("cagr", 0.0)
        sharpe = m.get("sharpe", 0.0)
        maxdd = m.get("max_drawdown", 0.0)
        trades = m.get("n_trades", 0)

        vs_bh = ""
        if name != "Buy & Hold":
            delta = cagr - bh_cagr
            vs_bh = f" *(vs B&H: {delta:+.1%})*"

        lines.append(
            f"- **{name}**: CAGR {cagr:.1%}{vs_bh}, "
            f"Sharpe {sharpe:.2f}, Max DD {maxdd:.1%}, {trades} trades"
        )

    stats = context.get("regime_stats")
    if stats is not None and not stats.empty:
        lines.extend(["", "### Regime Analysis"])
        for regime_name, row in stats.iterrows():
            ann_ret = row.get("ann_mean_ret", 0)
            ann_vol = row.get("ann_vol", 0)
            days = int(row.get("count", 0))
            marker = " ← current" if regime_name == current_regime else ""
            lines.append(
                f"- **{regime_name}**: {ann_ret:.1%} ann. return, "
                f"{ann_vol:.1%} volatility, {days} days{marker}"
            )

    lines.extend(
        [
            "",
            "---",
            "### To unlock full AI analysis",
            "Install ConnectOnion (`pip install connectonion`) and run `co auth` "
            "in your terminal to authenticate. No API key required.",
            "",
            "*Past performance does not predict future results. "
            "Educational purposes only — not financial advice.*",
        ]
    )

    return "\n".join(lines)
