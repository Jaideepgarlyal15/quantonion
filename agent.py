"""QuantOnion Research Agent — ConnectOnion entrypoint."""

from pathlib import Path

from connectonion import Agent, host

from agents.live_tools import (
    list_available_strategies,
    detect_current_regime,
    run_backtest_analysis,
    compare_all_strategies,
    get_risk_metrics,
    get_ml_forecast,
    get_market_sentiment,
    get_macro_context,
)

_TOOLS = [
    list_available_strategies,
    detect_current_regime,
    run_backtest_analysis,
    compare_all_strategies,
    get_risk_metrics,
    get_ml_forecast,
    get_market_sentiment,
    get_macro_context,
]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.txt"


def _system_prompt():
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return (
            "You are QuantOnion Research Agent. Use your tools to fetch live market data, "
            "run backtests, detect regimes, and assess sentiment. Never give financial advice."
        )


def create_agent():
    return Agent(
        name="quantonion_research_agent",
        model="co/gemini-2.5-pro",
        tools=_TOOLS,
        system_prompt=_system_prompt(),
        max_iterations=8,
    )


agent = create_agent()

if __name__ == "__main__":
    host(create_agent)
