# Contributing to QuantOnion

Thank you for your interest in contributing. QuantOnion is an open-source project and all contributions are welcome — bug fixes, new strategies, documentation improvements, and feature additions.

---

## Getting Started

```bash
git clone https://github.com/yourusername/quantonion.git
cd quantonion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Run the test suite to verify everything works:

```bash
pytest tests/ -v
```

---

## Contribution Areas

### Bug Reports
Open a GitHub issue with:
- Python version and OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant error output

### New Strategies
To add a new strategy:

1. Create `strategies/your_strategy.py` implementing `BaseStrategy`:
   ```python
   from strategies.base import BaseStrategy
   import pandas as pd

   class YourStrategy(BaseStrategy):
       name = "Your Strategy"

       def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
           # Return pd.Series of 0.0 (flat) or 1.0 (long)
           ...
   ```
2. Register it in `strategies/__init__.py` in the `STRATEGIES` dict
3. Add parameters to the sidebar in `app.py` under "Strategy Parameters"
4. Write tests in `tests/test_strategies.py`

**Signal contract requirements:**
- Values must be in `[0.0, 1.0]` (long-only, no leverage)
- NaN values are treated as 0.0 by the engine
- No lookahead: signal on day `t` must use only data up to and including day `t`

### New Metrics
Add metric functions to `backtesting/metrics.py` and update `compute_metrics()`.

### Data Sources
New data sources should be added to `core/data_loader.py` following the existing pattern:
- Use `@st.cache_data(ttl=3600)` for caching
- Return empty DataFrame on failure (never raise)
- Try the source last if it requires a paid key

### UI Improvements
The Streamlit app is in `app.py`. Keep the tab structure intact.
Avoid adding UI elements that have no corresponding logic.

---

## Code Standards

- **Type annotations** on all public functions
- **Docstrings** on all public functions and classes
- **No debug prints** — use `st.warning()` for user-visible issues
- **No lookahead bias** in strategy or backtest code
- **No fake precision** — do not claim metrics that aren't computed
- Tests must pass: `pytest tests/ -v`

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Open a PR with a clear description of what changed and why
5. Reference any relevant issues

---

## Licence

By contributing, you agree your contributions will be licensed under the MIT Licence.
