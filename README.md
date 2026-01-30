# Regime-Switching Risk Dashboard

An open-source web application that detects market regimes and forecasts prices using Hidden Markov Models and Machine Learning.

---

## What This Dashboard Does

### 1. Market Regime Detection
- Uses **Gaussian Hidden Markov Models (HMM)** to identify market states
- Classifies regimes as: **Stormy**, **Choppy**, **Calm**, or **Super Calm**
- Shows regime confidence levels and transition probabilities
- Displays expected duration of current regime

### 2. Price Forecasting
- Uses **ML Ensemble** (Linear Regression + Random Forest)
- Predicts prices for 3 time horizons:
  - **3-Day** forecast
  - **14-Day** forecast
  - **3-Month** forecast
- Each forecast includes 95% confidence intervals

### 3. Interactive Visualizations
- Price chart with regime-colored background shading
- Future forecast markers on the price chart
- Regime timeline showing historical regime changes
- Posterior probability charts for each regime
- Confidence series over time

### 4. Portfolio Analytics
- Value-at-Risk (VaR) calculation
- Expected Shortfall (ES)
- Export labelled time series to CSV

---

## Supported Markets

| Market | Examples |
|--------|----------|
| US Indices | `^GSPC` (S&P 500), `^NDX` (Nasdaq), `^DJI` (Dow) |
| US Stocks | `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `TSLA`, `META` |
| ETFs | `SPY`, `QQQ`, `IWM`, `DIA`, `VTI` |
| Crypto | `BTC-USD`, `ETH-USD`, `SOL-USD` |
| UK | `^FTSE`, `BP.L`, `SHEL.L` |
| Europe | `^GDAXI`, `^FCHI` |
| Australia | `BHP.AX`, `CBA.AX`, `IOZ.AX`, `^AXJO` |
| Japan | `^N225`, `7203.T` |
| India | `RELIANCE.NS` |

---

## How It Works

### Regime Detection (HMM)
1. Calculate daily log returns and rolling volatility
2. Fit Gaussian HMM with 2-4 regime states
3. Label states by mean return (Stormy → Calm)
4. Plot regime periods on price chart

### Price Forecasting (ML)
1. Create supervised dataset with rolling feature windows
2. Train Linear Regression + Random Forest ensemble
3. Predict next-day returns iteratively for each horizon
4. Apply mean-reversion for longer horizons (14-day, 3-month)
5. Show predictions with confidence intervals

---

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- yfinance
- hmmlearn
- scikit-learn
- requests

---

## Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Past performance does not guarantee future results.

---

## GitHub

[Regime-Switching-Risk-Dashboard](https://github.com/Jaideepgarlyal15/Regime-Switching-Risk-Dashboard-Streamlit-HMM-)

Contributions and feedback welcome!

---

## License

MIT License

