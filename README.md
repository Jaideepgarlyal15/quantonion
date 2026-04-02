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

> Run `streamlit run app.py` locally, interact with each tab, then save screenshots as
> `assets/screenshot_backtest.png`, `assets/screenshot_regimes.png`, `assets/screenshot_agent.png`.

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
4. Fetches news sentiment, social buzz, and fear & greed data
5. Synthesises everything into a structured research brief
6. Composes a plain-English analysis with appropriate caveats

Graceful fallback to deterministic summaries when API key is absent.

### Standalone Agent Tools (`agents/live_tools.py`)

Seven tools the ConnectOnion agent calls to fetch live data:

| Tool | What it does | Auth |
|------|-------------|------|
| `list_available_strategies()` | Lists all 6 built-in strategies | None |
| `detect_current_regime(ticker, ...)` | Fetches live prices, runs Gaussian HMM, returns regime + confidence | None |
| `run_backtest_analysis(ticker, strategy, ...)` | Fetches live data, runs full backtest, returns CAGR/Sharpe/MaxDD/win rate | None |
| `compare_all_strategies(ticker, ...)` | Runs all 6 strategies, returns table sorted by Sharpe | None |
| `get_risk_metrics(ticker, ...)` | Returns VaR 95/99, Expected Shortfall, annualised vol | None |
| `get_ml_forecast(ticker, ...)` | Trains LR + RF ensemble, returns 3-day/2-week/3-month forecasts with 95% CI | None |
| `get_market_sentiment(ticker)` | VIX via yfinance, crypto Fear & Greed (alternative.me), Yahoo Finance headline sentiment | None |

All tools use free public data sources — no API key required for any of them.

### Data Sources
| Source | Free? | Key Required |
|--------|-------|-------------|
| Yahoo Finance | ✅ Yes | No |
| Alpha Vantage | ✅ Free tier | Optional |
| Twelve Data | ✅ Free tier | Optional |
| StockTwits | ✅ Yes | No |
| Reddit (WSB/stocks/investing) | ✅ Yes | No |
| alternative.me Fear & Greed | ✅ Yes | No |
| VIX (^VIX via yfinance) | ✅ Yes | No |
| Finnhub | ✅ Free tier | Optional |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/quantonion.git
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

**Option A — Streamlit dashboard (full interactive UI)**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501), select a ticker, choose strategies, and click **🚀 Run Analysis**.

**Option B — Standalone ConnectOnion agent (chat interface)**
```bash
co auth            # authenticate once — enables AI model
python agent.py    # launches built-in chat UI in your browser
```
Ask in plain English: *"What regime is AAPL in?"*, *"Compare all strategies on BTC-USD since 2020"*, *"Run a Regime Filter backtest on ^GSPC"*.

**Option C — Deploy to ConnectOnion Cloud**
```bash
co deploy          # deploys agent.py and gives you a public URL
```

---

## Project Structure

```
quantonion/
├── app.py                    # Streamlit portal (Backtest | Regimes | Agent | About)
├── agent.py                  # Standalone ConnectOnion agent (co deploy / python agent.py)
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
│   ├── research_agent.py     # ConnectOnion Agent for Streamlit tab + fallback
│   ├── tools.py              # Streamlit agent tools (reads pre-loaded session data)
│   └── live_tools.py         # Standalone agent tools (fetch + compute on demand)
├── prompts/
│   ├── system.txt            # System prompt for Streamlit agent tab
│   └── agent_system.txt      # System prompt for standalone ConnectOnion agent
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
       │  (6 strats) │  │  Agent(re_act plugin)     │
       │             │  │  Tools: strategy_summary  │
       │             │  │        regime_context     │
       │             │  │        compare_benchmark  │
       │             │  │        risk_analysis      │
       │             │  │        ml_forecast_summary│
       └─────────────┘  └──────────────────────────┘
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
   For full AI agent features, use the standalone agent (`python agent.py` or `co deploy`).
   The Streamlit app's Agent tab shows a deterministic rule-based summary when ConnectOnion is unavailable.

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
| `ALPHAVANTAGE_API_KEY` | No | Alpha Vantage data + NEWS_SENTIMENT for `get_news_sentiment()` |
| `TWELVE_DATA_API_KEY` | No | Twelve Data price data fallback |
| `FINNHUB_API_KEY` | No | Finnhub news (optional supplement to AV + Yahoo Finance) |

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
