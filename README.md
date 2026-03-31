# 🧅 QuantOnion

**Open-source agentic quant research and backtesting platform.**

> Run reproducible backtests · Detect market regimes · Get AI-powered analysis · Export everything

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![ConnectOnion](https://img.shields.io/badge/agent%20framework-ConnectOnion-6C63FF.svg)](https://connectonion.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What is QuantOnion?

QuantOnion is a free, open-source web portal for quantitative traders and researchers. It combines:

- **Vectorised backtesting** with honest cost modelling (no lookahead bias)
- **Gaussian HMM regime detection** that identifies bull, bear, choppy, and calm market states
- **Six built-in strategies** with side-by-side performance comparison
- **ML ensemble price forecasting** using Linear Regression + Random Forest
- **ConnectOnion-powered research agent** that explains results in plain English

No paid subscriptions required. Free market data via Yahoo Finance. Deploy to Render or Streamlit Cloud in minutes.

---

## Screenshots

> _Screenshots are from a local run on S&P 500 (^GSPC) 2015–2024_

| Backtest Tab | Regimes Tab | Agent Tab |
|---|---|---|
| ![Backtest](assets/screenshot_backtest.png) | ![Regimes](assets/screenshot_regimes.png) | ![Agent](assets/screenshot_agent.png) |

*(Add your own screenshots after first run)*

---

## Features

### Backtesting Engine
- Vectorised daily backtesting with 1-day signal execution lag (no lookahead)
- Transaction cost and slippage modelling (configurable basis points)
- Equity curve, drawdown, and trade count tracking
- Side-by-side multi-strategy comparison

### Performance Metrics
| Metric | Description |
|--------|-------------|
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Risk-adjusted return (annualised) |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar Ratio | CAGR / Max Drawdown |
| Win Rate | % of in-position days with positive return |
| Profit Factor | Gross profit / Gross loss |
| Time in Market | % of days with active long position |

### Strategies
| Strategy | Type | Parameters |
|----------|------|------------|
| Buy & Hold | Passive benchmark | — |
| SMA Crossover | Trend following | Fast/slow window |
| EMA Crossover | Trend following | Fast/slow span |
| RSI Mean Reversion | Mean reversion | Period, oversold/overbought |
| Bollinger Band Reversion | Mean reversion | Period, std dev |
| Regime Filter | Regime-conditional | Bull regime labels |

### Regime Detection
- Gaussian Hidden Markov Model (2–4 states)
- Regime labelling: Stormy / Choppy / Calm / Super Calm
- Posterior probability charts and transition matrices
- Expected regime duration estimates

### ConnectOnion Research Agent
The `re_act` plugin enables multi-step reasoning:
1. Retrieves strategy metrics via tool calls
2. Retrieves regime statistics and current state
3. Compares strategies vs Buy & Hold benchmark
4. Composes a plain-English analysis with appropriate caveats

Graceful fallback to deterministic summaries when API key is absent.

### Data Sources
| Source | Free? | Key Required |
|--------|-------|-------------|
| Yahoo Finance | ✅ Yes | No |
| Alpha Vantage | ✅ Free tier | Optional |
| Twelve Data | ✅ Free tier | Optional |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/quantonion.git
cd quantonion

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env to add optional API keys
```

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), select a ticker, choose strategies, and click **🚀 Run Analysis**.

---

## Project Structure

```
quantonion/
├── app.py                    # Streamlit portal (Backtest | Regimes | Agent | About)
├── requirements.txt
├── pyproject.toml
├── render.yaml               # Render.com deployment
├── .env.example              # Environment variable template
├── core/
│   ├── data_loader.py        # Multi-source market data (yfinance primary)
│   ├── features.py           # Feature engineering (log returns, volatility)
│   ├── hmm_model.py          # Gaussian HMM regime detection
│   ├── ml.py                 # Ensemble ML price forecasting
│   ├── plotting.py           # Plotly chart builders
│   └── portfolio.py          # VaR/ES, Monte Carlo
├── backtesting/
│   ├── engine.py             # Vectorised backtest runner
│   └── metrics.py            # CAGR, Sharpe, drawdown, win rate…
├── strategies/
│   ├── base.py               # Abstract BaseStrategy
│   ├── buy_hold.py           # Buy & Hold benchmark
│   ├── sma_crossover.py      # SMA Crossover
│   ├── ema_crossover.py      # EMA Crossover
│   ├── rsi_reversion.py      # RSI Mean Reversion
│   ├── bollinger.py          # Bollinger Band Reversion
│   └── regime_filter.py      # HMM Regime Filter
├── agents/
│   ├── research_agent.py     # ConnectOnion Agent + fallback
│   └── tools.py              # Agent tool functions
├── prompts/
│   └── system.txt            # Agent system prompt
└── tests/
    ├── test_features.py
    ├── test_strategies.py
    ├── test_metrics.py
    └── test_backtest.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│  Sidebar → Backtest Tab │ Regimes Tab │ Agent Tab │ About   │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐
  │ Data     │  │ Regime   │  │ Backtesting          │
  │ Layer    │  │ Engine   │  │ Engine               │
  │          │  │          │  │                      │
  │yfinance  │  │HMM fit   │  │run_backtest()        │
  │AV        │  │label_    │  │compute_metrics()     │
  │TwelveData│  │states()  │  │format_metrics_table()│
  └──────────┘  └──────────┘  └──────────────────────┘
        │             │                │
        └─────────────┼────────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
       ┌─────────────┐  ┌─────────────────────────┐
       │  Strategy   │  │  ConnectOnion Agent      │
       │  Library    │  │                          │
       │  (6 strats) │  │  Agent(re_act plugin)    │
       │             │  │  Tools: strategy_summary │
       │             │  │        regime_context    │
       │             │  │        compare_benchmark │
       │             │  │        risk_analysis     │
       └─────────────┘  └─────────────────────────┘
```

---

## Why ConnectOnion?

ConnectOnion was chosen as the agent framework because:

1. **Tool integration**: The `re_act` plugin enables the agent to call Python tool functions iteratively, retrieving actual computed backtest data rather than reasoning from memory.
2. **Multi-step reasoning**: Complex analysis (compare 5 strategies across regime environments) benefits from step-by-step tool calling rather than one-shot prompting.
3. **Deployment**: The `.co/` directory integrates with ConnectOnion's deploy flow for agent hosting.
4. **Python-native**: Tools are plain Python functions with docstrings — no special wrapper format required.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Deployment

### Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to your fork, `app.py` as entry point
4. Add secrets in the Streamlit dashboard (optional data sources):
   ```
   ALPHAVANTAGE_API_KEY = "..."
   TWELVE_DATA_API_KEY = "..."
   ```
   For AI agent features on Streamlit Cloud, authenticate via ConnectOnion CLI locally first.

### Render.com (free tier)

```bash
# render.yaml is already configured
# Connect your GitHub repo at render.com → New Web Service
```

### Local with Docker (optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Environment Variables

See [`.env.example`](.env.example) for a full list. None are required for basic use.

| Variable | Required | Description |
|----------|----------|-------------|
| ConnectOnion CLI auth | No | Run `co auth` once to enable full AI agent analysis |
| `ALPHAVANTAGE_API_KEY` | No | Alpha Vantage data fallback |
| `TWELVE_DATA_API_KEY` | No | Twelve Data fallback |

---

## Limitations

Be honest about what this tool can and cannot do:

- **In-sample metrics only**: All performance figures are computed on historical data seen during model fitting. They do not represent live trading performance.
- **HMM regime labels are retrospective**: The model assigns labels after seeing all data. Live regime detection would lag reality.
- **ML forecasts are educational**: The ensemble model captures rough trends but cannot reliably forecast prices. Do not use for trading.
- **Transaction cost model is simplified**: Fixed-bps costs ignore market impact, spread widening, and partial fills.
- **No multi-asset portfolios**: The current version is single-asset only.
- **Yahoo Finance data accuracy**: Data may contain adjusted-close inaccuracies for delisted or merged securities.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full roadmap.

**Near-term:**
- Walk-forward / out-of-sample backtesting
- Multi-asset portfolio backtesting
- Streamlit session persistence
- Additional strategies (Momentum, MACD, mean-reversion pairs)

**Medium-term:**
- Browser-based research automation via ConnectOnion browser plugin
- Export to PDF research reports
- Live data integration with WebSocket support

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs and issues welcome.

---

## Disclaimer

This software is provided for **educational and research purposes only**. It does not constitute financial advice. Past performance does not predict future results. Use at your own risk. The authors accept no liability for financial decisions made based on this tool.

---

## Acknowledgements

Built with:
- [ConnectOnion](https://connectonion.com) — agent framework
- [Streamlit](https://streamlit.io) — web portal
- [hmmlearn](https://hmmlearn.readthedocs.io) — Gaussian HMM
- [scikit-learn](https://scikit-learn.org) — ML ensemble
- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- [Plotly](https://plotly.com) — interactive charts

---

**MIT License · Open source · Contributions welcome**
