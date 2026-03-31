# QuantOnion — Architecture

## Overview

QuantOnion is structured as a layered Python application with a Streamlit UI. Each layer has a clear responsibility with no circular dependencies.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Streamlit UI                              │
│                            (app.py)                                 │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Backtest Tab │  │ Regimes Tab  │  │Agent Tab │  │About Tab │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  └──────────┘  │
└─────────┼────────────────┼───────────────┼──────────────────────────┘
          │                │               │
          ▼                ▼               ▼
┌─────────────────┐  ┌─────────────┐  ┌─────────────────────────────┐
│  Backtesting    │  │    Core     │  │     Agent Layer             │
│  Module         │  │   Module    │  │     (agents/)               │
│                 │  │             │  │                             │
│  engine.py      │  │ hmm_model   │  │  research_agent.py          │
│  metrics.py     │  │ features    │  │  ├── create_research_agent()│
│                 │  │ ml          │  │  └── run_agent_analysis()   │
└────────┬────────┘  │ plotting    │  │                             │
         │           │ data_loader │  │  tools.py                   │
         │           │ portfolio   │  │  ├── get_strategy_summary() │
         ▼           └──────┬──────┘  │  ├── get_regime_context()  │
┌─────────────────┐         │         │  ├── compare_vs_benchmark() │
│  Strategies     │         │         │  └── get_risk_analysis()    │
│  (strategies/)  │         │         │                             │
│                 │         │         │  ConnectOnion Agent         │
│  BaseStrategy   │         │         │  (re_act plugin)            │
│  BuyAndHold     │         │         └─────────────────────────────┘
│  SMACrossover   │         │
│  EMACrossover   │         │
│  RSIReversion   │         ▼
│  BollingerBand  │  ┌─────────────┐
│  RegimeFilter   │  │  Data Layer │
└─────────────────┘  │             │
                     │ Yahoo Finance│
                     │ Alpha Vantage│
                     │ Twelve Data │
                     └─────────────┘
```

---

## Data Flow

### Backtest Path

```
User selects ticker + strategies
         │
         ▼
core/data_loader.py          # Fetch adjusted close prices (yfinance first)
         │
         ▼
core/features.py             # log returns, rolling volatility, interaction
         │
         ├──────────────────►  core/hmm_model.py
         │                      - fit GaussianHMM
         │                      - label states (Stormy/Choppy/Calm/Super Calm)
         │                      - compute posterior probabilities
         │
         ▼
strategies/                  # Generate 0/1 signals per strategy
         │
         ▼
backtesting/engine.py        # run_backtest(prices, signals, cost_bps, slippage_bps)
  - 1-day execution lag
  - transaction cost + slippage on position changes
  - equity curve, drawdown series
         │
         ▼
backtesting/metrics.py       # compute_metrics(result)
  - CAGR, Sharpe, MaxDD, Calmar, WinRate, ProfitFactor, TimeInMarket
         │
         ▼
core/plotting.py             # plot_equity_curves(), plot_drawdown_chart()
         │
         ▼
Agent context built from results → agents/research_agent.py
```

### Regime Path

```
features → hmm_model.fit_hmm() → states, posterior
         → label_states()       → Stormy/Choppy/Calm/Super Calm
         → regime_segments()    → timeline chart
         → plot_price_with_regimes()
         → plot_confidence_series()
```

---

## Key Design Decisions

### No Lookahead Bias
Signals on day `t` are executed at the close of day `t+1` (1-day lag in `engine.py`). This models the realistic scenario where a researcher processes end-of-day data and enters the next day.

### Graceful Data Fallback
Data loading tries sources in order: yfinance → Alpha Vantage → Twelve Data → synthetic. Each source returns an empty DataFrame on failure. The UI shows the source used so users know data quality.

### Deterministic Agent Fallback
The ConnectOnion agent is optional. When ConnectOnion is not installed or not authenticated (`co auth`), `run_agent_analysis()` returns a deterministic rule-based summary computed from the same tool functions, so the Agent tab is always useful. No API key is required — ConnectOnion authenticates via its CLI.

### Strategy as Object
All strategies implement `BaseStrategy.generate_signals(prices, **kwargs) -> pd.Series`. This makes the backtest engine strategy-agnostic and makes adding new strategies a self-contained task.

### Tool Closures in Agent
Agent tools are created via `make_tools(context)` which captures the backtest context in a closure. This means the agent tools have access to live computed data without any global state or re-computation.

---

## Module Responsibilities

| Module | Responsibility | External Dependencies |
|--------|---------------|----------------------|
| `core/data_loader.py` | Fetch market data | yfinance, requests |
| `core/features.py` | Feature engineering | pandas, numpy |
| `core/hmm_model.py` | Gaussian HMM | hmmlearn, sklearn |
| `core/ml.py` | ML forecasting | sklearn |
| `core/plotting.py` | Chart builders | plotly |
| `core/portfolio.py` | VaR/ES, Monte Carlo | numpy |
| `backtesting/engine.py` | Vectorised backtest | pandas, numpy |
| `backtesting/metrics.py` | Performance metrics | numpy, pandas |
| `strategies/*.py` | Signal generation | pandas, numpy |
| `agents/research_agent.py` | ConnectOnion agent | connectonion |
| `agents/tools.py` | Agent tool functions | (pure Python) |
| `app.py` | Streamlit UI | all modules above |
