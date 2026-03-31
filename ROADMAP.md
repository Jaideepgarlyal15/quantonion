# QuantOnion Roadmap

This document outlines planned and potential future work. Items are grouped by effort and impact.
Community contributions on any item are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## v1.1 — Robustness (near-term)

- [ ] **Walk-forward backtesting** — out-of-sample testing with rolling train/test windows
- [ ] **Parameter sensitivity analysis** — heatmap of Sharpe vs strategy params
- [ ] **Benchmark comparison table** — annotate which strategies beat Buy & Hold net of costs
- [ ] **Improved mobile layout** — responsive Streamlit config for smaller screens
- [ ] **Session persistence** — save/load last-run settings via URL params

---

## v1.2 — More Strategies

- [ ] **MACD crossover** — classic momentum signal
- [ ] **Momentum strategy** — 12-1 month lookback momentum (Fama-French style)
- [ ] **Trend + regime filter combo** — SMA trend filtered by HMM regime
- [ ] **Volatility targeting** — scale position size to target a fixed portfolio vol
- [ ] **Dual momentum** (absolute + relative) — Gary Antonacci approach

---

## v1.3 — Portfolio Backtesting

- [ ] **Multi-asset portfolio** — weight multiple tickers in a single backtest
- [ ] **Correlation matrix** — regime-conditional correlation viewer
- [ ] **Portfolio-level drawdown** — combined drawdown across strategies

---

## v1.4 — Reporting

- [ ] **PDF export** — one-page research report from backtest results
- [ ] **Agent summary export** — download agent analysis as markdown
- [ ] **Chart export** — PNG/SVG download for individual charts
- [ ] **Shareable links** — URL-encoded backtest configuration

---

## v2.0 — Agent & Automation

- [ ] **Browser research automation** — ConnectOnion `browser` plugin for live data scraping
- [ ] **Scheduled agent runs** — `co deploy` to run nightly summaries
- [ ] **Multi-agent setup** — separate Strategy Agent + Regime Agent with subagent plugin
- [ ] **Agent memory** — remember user preferences across sessions
- [ ] **Shell approval workflow** — `shell_approval` plugin for any file mutations

---

## Ideas / Community Suggestions

- [ ] Live paper trading simulation mode
- [ ] Factor exposure analysis (beta, momentum, quality)
- [ ] Crypto-specific features (on-chain data, funding rates)
- [ ] Options strategy backtesting
- [ ] Integration with Polygon.io for extended history

---

*Submit ideas by opening a GitHub issue with the `enhancement` label.*
