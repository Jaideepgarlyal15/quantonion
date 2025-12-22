# core/portfolio.py

import numpy as np
import pandas as pd


def portfolio_return_series(price_df_dict: dict, weights: dict):
    """Combine multiple price series (dict of {ticker: price_df}) into weighted portfolio returns."""
    rets = []
    for t, df in price_df_dict.items():
        r = np.log(df["Adj Close"]).diff().rename(t)
        rets.append(r)

    R = pd.concat(rets, axis=1).dropna()
    w = np.array([weights[t] for t in R.columns])
    R["portfolio"] = R.values @ w
    return R["portfolio"]


def compute_var_es(returns, alpha=0.95):
    """Compute VaR & ES using historical simulation."""
    sorted_rets = np.sort(returns.dropna())
    index = int((1 - alpha) * len(sorted_rets))

    var = sorted_rets[index]
    es = sorted_rets[:index].mean() if index > 0 else var
    return float(var), float(es)


def monte_carlo_paths(
    last_price: float, vol: float, days=30, n_sims=5000, seed=42
):
    """Generate Monte Carlo price paths using lognormal model."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    paths = []

    for _ in range(n_sims):
        rand = rng.normal(0, vol * np.sqrt(dt), size=days)
        price_path = last_price * np.exp(np.cumsum(rand))
        paths.append(price_path)

    return np.array(paths)
