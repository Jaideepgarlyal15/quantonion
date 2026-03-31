"""
QuantOnion Agent Layer

ConnectOnion-powered research agent for backtest analysis and regime explanation.

Usage:
    from agents.research_agent import create_research_agent, run_agent_analysis

    context = {
        "backtest_results": {...},
        "regime_stats": df,
        "current_regime": "Calm",
        "ticker": "^GSPC",
        "period": "2015-01-01 to 2024-12-31",
    }
    agent = create_research_agent(context)
    response = run_agent_analysis(context, "Why did RSI underperform?", agent=agent)
"""

from agents.research_agent import create_research_agent, run_agent_analysis

__all__ = ["create_research_agent", "run_agent_analysis"]
