# core/hmm_model.py

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

# --------------------------------------------------------------------
# Simple regime names & colours (used by app + plotting)
# --------------------------------------------------------------------
SIMPLE_NAMES: List[str] = ["Stormy", "Choppy", "Calm", "Super Calm"]

SIMPLE_COLORS: Dict[str, str] = {
    "Stormy": "#e74c3c",
    "Choppy": "#f39c12",
    "Calm": "#2ecc71",
    "Super Calm": "#1abc9c",
}


# --------------------------------------------------------------------
# HMM fit
# --------------------------------------------------------------------
def fit_hmm(
    features: pd.DataFrame,
    n_states: int,
    seed: int,
) -> Tuple[GaussianHMM, StandardScaler, np.ndarray, np.ndarray]:
    """
    Fit a Gaussian HMM on the feature matrix.

    Returns:
        hmm: fitted GaussianHMM
        scaler: fitted StandardScaler
        states: most likely state sequence
        post: posterior probabilities (T x K)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=seed,
        verbose=False,
    )
    hmm.fit(X)

    states = hmm.predict(X)
    post = hmm.predict_proba(X)
    return hmm, scaler, states, post


# --------------------------------------------------------------------
# State labelling
# --------------------------------------------------------------------
def label_states(
    features: pd.DataFrame,
    states: np.ndarray,
    metric: str = "mean_return",
):
    """
    Label numeric HMM states as Bull / Bear etc using mean returns or Sharpe.
    """
    uniq = np.unique(states)

    ret_means = {s: features.loc[states == s, "ret"].mean() for s in uniq}
    stds = {s: features.loc[states == s, "ret"].std() for s in uniq}

    ann_ret = {s: ret_means[s] * 252 for s in uniq}
    ann_vol = {
        s: (stds[s] * np.sqrt(252)) if pd.notna(stds[s]) else np.nan for s in uniq
    }

    sharpe = {
        s: (ann_ret[s] / ann_vol[s]) if ann_vol[s] and ann_vol[s] > 0 else -np.inf
        for s in uniq
    }

    # Order states from worst to best
    if metric == "sharpe_ratio":
        order = sorted(uniq, key=lambda s: sharpe[s])
    else:
        order = sorted(uniq, key=lambda s: ret_means[s])

    adv_names_master = ["Crisis/Bear", "Neutral", "Bull", "Super Bull"]
    adv_map = {s: adv_names_master[i] for i, s in enumerate(order)}
    simple_map = {s: SIMPLE_NAMES[i] for i, s in enumerate(order)}
    return adv_map, simple_map


# --------------------------------------------------------------------
# Durations and segments
# --------------------------------------------------------------------
def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Expected duration of each regime in days given transition matrix."""
    pii = np.clip(np.diag(transmat), 1e-9, 0.999999)
    return 1.0 / (1.0 - pii)


def current_run_length(series_states: pd.Series) -> int:
    """Number of consecutive days spent in the latest state."""
    if series_states.empty:
        return 0
    s = series_states.values
    run = 1
    for i in range(len(s) - 2, -1, -1):
        if s[i] == s[-1]:
            run += 1
        else:
            break
    return run


def regime_segments(index, states, labels_dict):
    """
    Convert a state series into (start, end, label) segments for plotting.
    """
    out = []
    start_idx = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            out.append(
                (index[start_idx], index[i - 1], labels_dict.get(states[i - 1], "Unknown"))
            )
            start_idx = i
    out.append((index[start_idx], index[-1], labels_dict.get(states[-1], "Unknown")))
    return out
