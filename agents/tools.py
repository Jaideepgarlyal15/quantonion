"""
QuantOnion Agent Tool Functions

These are the callable tools provided to the ConnectOnion Research Agent.
Each tool is a pure Python function that reads from the backtest context
dictionary. Tools are created via a factory function so that the context
is captured in a closure — the agent calls the functions without needing
to pass context explicitly.

Tool contract:
  - Each function takes simple scalar or string arguments (for LLM compatibility)
  - Each function returns a string (the agent reads string tool outputs)
  - Failures return a descriptive error string rather than raising exceptions
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def make_tools(context: Dict[str, Any]) -> List[Callable]:
    """
    Create agent tool functions with closure over the backtest context.

    Args:
        context: Dict containing:
            backtest_results  — Dict[strategy_name, {metrics, result}]
            regime_stats      — pd.DataFrame of regime statistics (or None)
            current_regime    — string label of the most recent regime
            ticker            — asset ticker string
            period            — human-readable date range string

    Returns:
        List of callable tool functions ready for the ConnectOnion agent.
    """

    def get_strategy_summary(strategy_name: str = "all") -> str:
        """
        Return performance metrics for a specific strategy or all strategies.

        Args:
            strategy_name: Strategy name (e.g. "SMA Crossover") or "all".

        Returns:
            Formatted string with CAGR, Sharpe, max drawdown, win rate,
            profit factor, time in market, and number of trades.
        """
        results = context.get("backtest_results", {})
        if not results:
            return "No backtest results available. Ask the user to run a backtest first."

        if strategy_name == "all":
            lines = ["Strategy Performance Summary:", ""]
            for name, res in results.items():
                m = res.get("metrics", {})
                lines.append(
                    f"{name}: CAGR={m.get('cagr', 0):.1%} | "
                    f"Sharpe={m.get('sharpe', 0):.2f} | "
                    f"MaxDD={m.get('max_drawdown', 0):.1%} | "
                    f"WinRate={m.get('win_rate', 0):.1%} | "
                    f"Trades={m.get('n_trades', 0)}"
                )
            return "\n".join(lines)

        res = results.get(strategy_name)
        if res is None:
            available = ", ".join(results.keys())
            return f"Strategy '{strategy_name}' not found. Available: {available}"

        m = res.get("metrics", {})
        pf = m.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf < 1000 else "∞"

        return (
            f"Strategy: {strategy_name}\n"
            f"Total Return:   {m.get('total_return', 0):.1%}\n"
            f"CAGR:           {m.get('cagr', 0):.1%}\n"
            f"Volatility:     {m.get('volatility', 0):.1%}\n"
            f"Sharpe Ratio:   {m.get('sharpe', 0):.2f}\n"
            f"Max Drawdown:   {m.get('max_drawdown', 0):.1%}\n"
            f"Calmar Ratio:   {m.get('calmar', 0):.2f}\n"
            f"Win Rate:       {m.get('win_rate', 0):.1%}\n"
            f"Profit Factor:  {pf_str}\n"
            f"Time in Market: {m.get('time_in_market', 0):.1%}\n"
            f"# Trades:       {m.get('n_trades', 0)}\n"
            f"# Days:         {m.get('n_days', 0)}"
        )

    def get_regime_context() -> str:
        """
        Return HMM regime statistics and the current market regime.

        Returns:
            Formatted string with per-regime annualised return, volatility,
            and day count, plus the most recently detected regime.
        """
        stats = context.get("regime_stats")
        current_regime = context.get("current_regime", "Unknown")
        ticker = context.get("ticker", "Unknown")
        period = context.get("period", "Unknown")

        lines = [
            f"Asset: {ticker}",
            f"Period: {period}",
            f"Current Detected Regime: {current_regime}",
            "",
            "Regime Historical Statistics (annualised, in-sample):",
        ]

        if stats is not None and not stats.empty:
            for regime, row in stats.iterrows():
                lines.append(
                    f"  {regime}: "
                    f"Ann.Return={row.get('ann_mean_ret', 0):.1%}, "
                    f"Ann.Vol={row.get('ann_vol', 0):.1%}, "
                    f"Days={int(row.get('count', 0))}"
                )
        else:
            lines.append("  Regime statistics not available.")

        lines.extend(
            [
                "",
                "Note: HMM regime labels are assigned in-sample and may not "
                "generalise to out-of-sample periods.",
            ]
        )
        return "\n".join(lines)

    def compare_vs_benchmark() -> str:
        """
        Compare all strategies against the Buy & Hold benchmark.

        Returns:
            Formatted comparison table showing CAGR delta, Sharpe delta,
            and max drawdown for each strategy relative to Buy & Hold.
        """
        results = context.get("backtest_results", {})
        if not results:
            return "No backtest results available."

        bh = results.get("Buy & Hold", {}).get("metrics", {})
        bh_cagr = bh.get("cagr", 0.0)
        bh_sharpe = bh.get("sharpe", 0.0)

        lines = [
            "Strategy Comparison vs Buy & Hold Benchmark:",
            f"  Buy & Hold: CAGR={bh_cagr:.1%}, Sharpe={bh_sharpe:.2f}",
            "",
        ]

        for name, res in results.items():
            if name == "Buy & Hold":
                continue
            m = res.get("metrics", {})
            cagr = m.get("cagr", 0.0)
            sharpe = m.get("sharpe", 0.0)
            maxdd = m.get("max_drawdown", 0.0)
            bh_maxdd = bh.get("max_drawdown", 0.0)

            cagr_delta = cagr - bh_cagr
            sharpe_delta = sharpe - bh_sharpe
            dd_delta = maxdd - bh_maxdd  # negative means worse drawdown

            lines.append(
                f"  {name}:\n"
                f"    CAGR: {cagr:.1%} (delta vs B&H: {cagr_delta:+.1%})\n"
                f"    Sharpe: {sharpe:.2f} (delta: {sharpe_delta:+.2f})\n"
                f"    Max DD: {maxdd:.1%} (vs B&H {bh_maxdd:.1%}, delta: {dd_delta:+.1%})\n"
            )

        return "\n".join(lines)

    def get_risk_analysis() -> str:
        """
        Return a risk summary including best and worst strategies by drawdown and Sharpe.

        Returns:
            Formatted risk analysis string.
        """
        results = context.get("backtest_results", {})
        if not results:
            return "No backtest results available."

        by_sharpe = sorted(
            results.items(),
            key=lambda x: x[1].get("metrics", {}).get("sharpe", -999),
            reverse=True,
        )
        by_dd = sorted(
            results.items(),
            key=lambda x: x[1].get("metrics", {}).get("max_drawdown", -999),
            reverse=True,
        )

        best_sharpe_name, best_sharpe_res = by_sharpe[0]
        worst_sharpe_name, worst_sharpe_res = by_sharpe[-1]
        shallowest_dd_name, _ = by_dd[0]
        deepest_dd_name, deepest_dd_res = by_dd[-1]

        lines = [
            "Risk Analysis Summary:",
            f"  Best Sharpe:      {best_sharpe_name} "
            f"({best_sharpe_res['metrics'].get('sharpe', 0):.2f})",
            f"  Worst Sharpe:     {worst_sharpe_name} "
            f"({worst_sharpe_res['metrics'].get('sharpe', 0):.2f})",
            f"  Shallowest DD:    {shallowest_dd_name} "
            f"({by_dd[0][1]['metrics'].get('max_drawdown', 0):.1%})",
            f"  Deepest DD:       {deepest_dd_name} "
            f"({deepest_dd_res['metrics'].get('max_drawdown', 0):.1%})",
            "",
            "Important: Sharpe and drawdown are computed in-sample on the full "
            "history. They likely overstate real-world performance due to "
            "survivorship bias, parameter fitting, and regime stationarity assumptions.",
        ]
        return "\n".join(lines)

    def get_ml_forecast_summary() -> str:
        """
        Return ML price forecasts for 3-day, 2-week, and 3-month horizons.

        Returns:
            Formatted string with predicted price, direction, expected return,
            and 95% confidence interval for each horizon.
        """
        forecasts = context.get("forecasts", {})
        current_price = context.get("current_price")
        ticker = context.get("ticker", "Unknown")

        if not forecasts:
            return (
                "ML forecasts not available. "
                "Enable ML in the sidebar and click Run Analysis to generate forecasts."
            )

        horizon_labels = {3: "3-Day", 14: "2-Week", 90: "3-Month"}
        lines = [f"ML Price Forecasts for {ticker}:"]
        if current_price:
            lines.append(f"Current Price: ${current_price:.2f}")
        lines.append("")

        for h in sorted(forecasts.keys()):
            f = forecasts[h]
            direction = "UP" if f["predicted_return"] > 0 else "DOWN"
            label = horizon_labels.get(h, f"{h}-Day")
            lines.append(
                f"{label}: ${f['predicted_price']:.2f} ({direction} {f['predicted_return']:+.2%})"
                f" | 95% CI: ${f['confidence_lower']:.2f}–${f['confidence_upper']:.2f}"
                f" | Confidence: {f['confidence_level']:.0%}"
            )

        return "\n".join(lines)

    return [
        get_strategy_summary,
        get_regime_context,
        compare_vs_benchmark,
        get_risk_analysis,
        get_ml_forecast_summary,
    ]
