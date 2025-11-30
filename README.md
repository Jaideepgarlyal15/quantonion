# Regime-Switching Risk Dashboard · Streamlit + HMM

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen.svg)](https://streamlit.io/)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-0.3.x-informational.svg)](https://hmmlearn.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An educational dashboard that infers **market regimes** using a **Gaussian Hidden Markov Model** on daily log-returns and rolling volatility. Works with global tickers and ASX helpers.

> Educational use only. This is not investment advice.

---

## Quick start

```bash
# 1) Create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
