# Regime-Switching Risk Dashboard

An open-source web application that detects market regimes and forecasts prices using Hidden Markov Models and Machine Learning techniques.

---

## Live Demo

- **Streamlit Cloud**: [https://yourusername-regime-switching-risk-dashboard.streamlit.app](https://share.streamlit.io)
- **Hugging Face**: [https://huggingface.co/spaces/yourusername/regime-switching-risk-dashboard](https://huggingface.co/spaces)

---

## What This Project Does

- Detects market regimes using Gaussian Hidden Markov Models
- Classifies market states as Stormy, Choppy, Calm, or Super Calm
- Shows regime confidence levels and transition probabilities
- Displays expected duration of the current regime
- Provides price forecasts for multiple time horizons
- Calculates Value-at-Risk and Expected Shortfall for portfolio analytics
- Exports labelled time series data to CSV format

---

## Key Features

- Interactive price charts with regime-coloured background shading
- Future forecast markers with 95% confidence intervals
- Regime timeline showing historical regime changes
- Posterior probability charts for each regime state
- Support for global markets including US, UK, Europe, Australia, Japan, India, and Crypto
- Real-time data retrieval from Yahoo Finance

---

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualisation**: plotly
- **Machine Learning**: hmmlearn, scikit-learn
- **Data Source**: yfinance
- **HTTP Requests**: requests

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Jaideepgarlyal15/Regime-Switching-Risk-Dashboard-Streamlit-HMM-.git
cd Regime-Switching-Risk-Dashboard-Streamlit-HMM-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## Project Structure

```
regime-switching-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT Licence
├── README.md                 # This file
├── GITHUB_SETUP.md           # GitHub setup and deployment guide
├── core/
│   ├── data_loader.py        # Data retrieval from Yahoo Finance
│   ├── features.py           # Feature engineering for ML models
│   ├── hmm_model.py          # Hidden Markov Model implementation
│   ├── ml.py                 # ML ensemble for price forecasting
│   ├── plotting.py           # Plotly visualisation functions
│   └── portfolio.py          # Portfolio risk analytics
└── .streamlit/
    ├── config.toml           # Streamlit configuration
    ├── secrets.toml          # Secrets management
    └── architecture.md       # Architecture documentation
```

---

## Licence

This project is licensed under the MIT Licence. See the LICENSE file for details.

---

## Disclaimer

This software is provided for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Use this software at your own risk.

---

## Contributing

Contributions are welcome. Please open issues for bug reports or feature requests, or submit pull requests for improvements.

