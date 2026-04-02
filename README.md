# QuantOnion

**AI-powered quantitative research agent for traders and researchers.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ConnectOnion](https://img.shields.io/badge/agent%20framework-ConnectOnion-6C63FF.svg)](https://connectonion.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Try it now — no setup required

**[→ Chat with QuantOnion](https://chat.openonion.ai/0x2c52555603cf676e39ed12cceea5f9c3d3580a1b7a2e9e16b348fcb23d81d8e9)**

The agent is live. Type any of these to get started:

```
What is the current market regime for AAPL?
Compare all strategies for SPY
Show ML price forecasts for NVDA
What are the risk metrics for BTC-USD?
Analyse TSLA — full research brief
```

No account. No API key. Just click and ask.

---

## What it does

QuantOnion is a ConnectOnion agent that answers quantitative research questions about any Yahoo Finance ticker. Ask it anything:

```
"What is the current market regime for AAPL?"
→ calls detect_current_regime() → returns HMM state, confidence, days in regime

"Compare all strategies for SPY"
→ calls compare_all_strategies() → ranked table of 6 strategies by Sharpe ratio

"Analyse TSLA — full research brief"
→ calls all 5 tools in sequence → structured brief with regime, edge, ML signal, risk, sentiment
```

No paid APIs. No authentication required. Free market data via Yahoo Finance.

---

## Tools

The agent has 7 tools, each callable independently or as part of a full brief:

| Tool | What it does |
|---|---|
| `list_available_strategies` | Returns all strategy names. Called first if user hasn't named one. |
| `detect_current_regime` | Fits a Gaussian HMM to live price data. Returns current regime label, HMM confidence, days in regime, per-regime historical return and vol. |
| `run_backtest_analysis` | Runs a single strategy vs Buy & Hold with 1-day lag, cost + slippage modelling. Returns CAGR, Sharpe, max drawdown, win rate. |
| `compare_all_strategies` | Runs all 6 strategies on the same ticker and period. Returns a table sorted by Sharpe ratio. |
| `get_risk_metrics` | Historical VaR (95%/99%), Expected Shortfall, annualised and 21-day volatility, best/worst single-day returns. |
| `get_ml_forecast` | Linear Regression + Random Forest ensemble. Returns 3-day, 2-week, 3-month price forecasts with 95% CI and model confidence. |
| `get_market_sentiment` | VIX level and reading, Crypto Fear & Greed index (alternative.me), Yahoo Finance news headline sentiment (keyword-scored). |

---

## Quick start

### Run locally

```bash
git clone https://github.com/Jaideepgarlyal15/quantonion
cd quantonion
pip install -r requirements.txt
co auth          # one-time ConnectOnion login
python agent.py
```

The agent starts on `http://localhost:8000`, connects to the relay, and is immediately accessible at your chat.openonion.ai agent link. No other keys required.

### Deploy to Render (always-on)

So the demo link works 24/7 without your laptop running:

1. Fork this repo and connect it to [render.com](https://render.com) → New Web Service
2. Render will use `render.yaml` automatically (`bash start.sh` → `python agent.py`)
3. In the Render dashboard, add one environment variable:
   ```
   CO_AGENT_KEY_B64  =  <output of: base64 -i .co/keys/agent.key | tr -d '\n'>
   ```
   This restores your agent's cryptographic identity (same address, same chat link) on the Render server.
4. Deploy. Your agent URL stays the same — `chat.openonion.ai/0x2c52555...`

---

## Design decisions

### Why 7 separate tools instead of one big `analyse()` function?

Routing. A user asking "what's the regime for AAPL?" should not trigger ML training (the slowest operation, ~8s) or a full backtest run. With separate tools, the system prompt routes each query type to exactly one tool. A single-tool query costs ~$0.002 and returns in 1–2s. A full brief uses all tools sequentially and costs ~$0.02.

This also means each tool is independently testable and cacheable.

### Why Gaussian HMM for regime detection instead of a simpler indicator?

Indicators like RSI or SMA tell you where price has been, not what state the market is in. HMM is a generative latent-variable model: it assumes the market is always in one of N hidden states, each with its own return/volatility distribution, and transitions between them follow a Markov chain. The transition matrix gives you persistence — how long a regime typically lasts — which no moving average can give you. The tradeoff is that HMM labels are in-sample (the model sees all history when fitting), so live regime detection always lags by ~1 day.

### Why gemini-2.5-pro over a smaller/faster model?

The system prompt has strict routing rules: "for regime queries, call only this tool; for full briefs, call tools in this exact order." Smaller models (`gpt-5-nano` was tried) ignore these rules — they either call every tool regardless of the query or hallucinate numbers without calling any tool. Gemini 2.5 Pro follows the routing rules reliably. A single-tool query is still fast because the model routes correctly and stops.

### Why max_iterations=6?

A full research brief calls 5 tools in a fixed order. If `max_iterations` is 3, the agent runs out mid-brief and returns partial output. 6 gives 5 tool calls plus 1 final synthesis step, with 0 wasted iterations.

### Why a 5-minute in-process cache?

A full brief calls `detect_current_regime`, `compare_all_strategies`, `get_risk_metrics`, and `get_ml_forecast` — all of which need the same yfinance price data. Without a cache, that's 4 separate network calls for the same ticker and date range. The cache key is `ticker|start|end`, so it correctly misses on different parameters.

### Why 1-day execution lag in the backtester?

You cannot execute on the same bar you generate a signal from. A signal generated from day T's close can only be executed at day T+1's open (modelled here as T+1's close with cost+slippage). Without this lag, backtests are unrealistically profitable — the most common source of backtest inflation.

---

## Architecture

```
agent.py                    ← ConnectOnion entrypoint (Agent + host)
prompts/agent_system.txt    ← System prompt with routing rules
agents/live_tools.py        ← 7 tool functions (the agent's hands)
core/
  data_loader.py            ← yfinance price fetch + symbol normalisation
  features.py               ← log returns, rolling vol, momentum, RSI-proxy
  hmm_model.py              ← Gaussian HMM fit, label_states, run length
  ml.py                     ← Linear Regression + Random Forest ensemble
backtesting/
  engine.py                 ← Vectorised backtest with lag + cost modelling
  metrics.py                ← CAGR, Sharpe, max drawdown, Calmar, win rate
strategies/                 ← Buy & Hold, SMA, EMA, RSI, Bollinger, Regime Filter
.co/host.yaml               ← ConnectOnion host config (port, relay, permissions)
```

---

## Strategies

| Strategy | Type |
|---|---|
| Buy & Hold | Passive benchmark |
| SMA Crossover | Trend following (fast/slow window) |
| EMA Crossover | Trend following (fast/slow span) |
| RSI Mean Reversion | Mean reversion (period, oversold/overbought) |
| Bollinger Band Reversion | Mean reversion (period, std dev) |
| Regime Filter | Regime-conditional (long only in Calm/Super Calm regimes) |

---

## What I learned

**On agent design:** The hardest part isn't the tools — it's the system prompt. The routing logic (which tool to call for which query type) is what makes the agent feel fast and accurate. A vague system prompt produces a slow agent that calls everything for every query.

**On ConnectOnion:** The `host()` function handles the full relay registration, WebSocket keepalive, and HTTP server setup. What looks like a lot of infrastructure is actually 4 lines of code. The `host.yaml` permissions system is useful for pre-approving read-only tools so the agent never pauses mid-brief to ask for confirmation.

**On backtesting honesty:** It's easy to build a backtester that looks great. The 1-day lag, symmetric cost/slippage modelling, and explicit "in-sample only" disclaimers are the minimum bar for results that aren't misleading.

---

## Limitations

- **In-sample metrics only.** All performance figures are computed on historical data. They are not forward-looking.
- **HMM labels are retrospective.** The model fits all history before labelling regimes. Live regime detection lags by ~1 day.
- **ML forecasts are for research only.** The ensemble captures rough trends on training data. Do not use for trading.
- **Single-asset only.** No multi-asset or portfolio-level analysis.
- **Yahoo Finance data.** May contain adjusted-close inaccuracies for delisted or corporate-action-affected securities.

---

## Environment variables

See [`.env.example`](.env.example). No variables are required for basic use.

| Variable | Required | Purpose |
|---|---|---|
| ConnectOnion auth (`co auth`) | Yes | Connects agent to relay |
| `ALPHAVANTAGE_API_KEY` | No | Alternative price data source |
| `TWELVE_DATA_API_KEY` | No | Second price data fallback |

---

## Disclaimer

Educational and research purposes only. Not financial advice. Past performance does not predict future results.

---

Built with [ConnectOnion](https://connectonion.com) · [hmmlearn](https://hmmlearn.readthedocs.io) · [scikit-learn](https://scikit-learn.org) · [yfinance](https://github.com/ranaroussi/yfinance) · MIT License
