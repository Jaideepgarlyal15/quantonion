# Contributing to QuantOnion

All contributions are welcome — new strategies, bug fixes, new tools for the agent, and documentation improvements.

---

## Getting Started

```bash
git clone https://github.com/Jaideepgarlyal15/quantonion.git
cd quantonion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## Contribution Areas

### New Strategies

Add a file to `strategies/` implementing `BaseStrategy`:

```python
from strategies.base import BaseStrategy
import pandas as pd

class YourStrategy(BaseStrategy):
    name = "Your Strategy"

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        # Return pd.Series of 0.0 (flat) or 1.0 (long)
        ...
```

Register it in `strategies/__init__.py` in the `STRATEGIES` dict, then write tests in `tests/test_strategies.py`.

**Signal contract:**
- Values in `[0.0, 1.0]` — long-only, no leverage
- NaN treated as 0.0 by the engine
- No lookahead: signal on day `t` must use only data up to and including day `t`

### New Agent Tools

Add a function to `agents/live_tools.py` following the existing pattern:
- Accepts only primitive argument types (str, int, float)
- Returns a plain string
- Catches all exceptions internally — never raise
- Register it in the `_TOOLS` list in `agent.py`
- Add a docstring block and routing keyword to `prompts/agent_system.txt`

### New Metrics

Add to `backtesting/metrics.py` and update `compute_metrics()`.

### Data Sources

New sources go in `core/data_loader.py`. Return an empty DataFrame on failure — never raise.

---

## Code Standards

- Type annotations on all public functions
- No lookahead bias in strategy or backtest code
- No fake precision — only report metrics that are actually computed
- Tests must pass: `pytest tests/ -v`

---

## Pull Request Process

1. Fork and branch: `git checkout -b feature/your-feature`
2. Make changes and run `pytest tests/ -v`
3. Open a PR with a clear description of what changed and why

---

## Licence

By contributing, you agree your contributions will be licensed under the MIT Licence.
