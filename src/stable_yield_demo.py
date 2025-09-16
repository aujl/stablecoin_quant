from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stable_yield_lab import (
    CSVSource,
    HistoricalCSVSource,
    Metrics,
    Pipeline,
    Visualizer,
    performance,
    risk_metrics,
)
from stable_yield_lab.reporting import cross_section_report


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base``."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(cast(dict[str, Any], base[key]), value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load configuration from a TOML file and merge with defaults.

    Parameters
    ----------
    path:
        Optional path to a configuration file. When ``None`` or missing, the
        built-in defaults are used.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with any file overrides applied.
    """

    default = {
        "csv": {"path": str(Path(__file__).with_name("sample_pools.csv"))},
        "yields_csv": str(Path(__file__).with_name("sample_yields.csv")),
        "initial_investment": 1_000.0,
        "filters": {
            "min_tvl": 100_000,
            "min_base_apy": 0.06,
            "auto_only": True,
            # "chains": ["Ethereum"],
            # "stablecoins": ["USDC"]
        },
        "output": {"outdir": None, "show": True, "charts": ["bar", "scatter", "chain"]},
        "reporting": {"top_n": 10, "perf_fee_bps": 0.0, "mgmt_fee_bps": 0.0},
        "benchmarks": {
            "tickers": [],
            "weights": {},
            "cash_rate": 0.0,
            "cash_series": None,
            "title": "Rebalanced NAV vs Benchmarks",
            "labels": {
                "rebalance": "Rebalanced NAV",
                "buy_and_hold": "Buy & Hold",
                "cash": "Cash",
            },
        },
    }

    cfg_path = Path(path) if path else None

    if cfg_path and cfg_path.is_file():
        with open(cfg_path, "rb") as f:

            file_cfg = tomllib.load(f)

        for k, v in file_cfg.items():

            if isinstance(v, dict) and k in default and isinstance(default[k], dict):
                _deep_update(cast(dict[str, Any], default[k]), cast(dict[str, Any], v))

            else:

                default[k] = v

    else:

        if cfg_path:

            print(f"[WARN] Config file not found at {cfg_path}. Using defaults.")

    return default


def main() -> None:
    """Run the demo using configuration from file or environment variables."""
    cfg_file = os.getenv("STABLE_YIELD_CONFIG") or (sys.argv[1] if len(sys.argv) > 1 else None)
    cfg = load_config(cfg_file)

    if csv_env := os.getenv("STABLE_YIELD_CSV"):
        cfg.setdefault("csv", {})["path"] = csv_env
    if outdir_env := os.getenv("STABLE_YIELD_OUTDIR"):
        cfg.setdefault("output", {})["outdir"] = outdir_env
    if yields_env := os.getenv("STABLE_YIELD_YIELDS_CSV"):
        cfg["yields_csv"] = yields_env
    if init_env := os.getenv("STABLE_YIELD_INITIAL_INVESTMENT"):
        try:
            cfg["initial_investment"] = float(init_env)
        except ValueError:
            pass

    # Load data
    csv_path = cfg["csv"]["path"]
    src = CSVSource(path=csv_path)
    repo = Pipeline([src]).run()

    # Apply filters
    f = cfg.get("filters", {})
    filtered = repo.filter(
        min_tvl=float(f.get("min_tvl", 0.0)),
        min_base_apy=float(f.get("min_base_apy", 0.0)),
        auto_only=bool(f.get("auto_only", False)),
        chains=f.get("chains"),
        stablecoins=f.get("stablecoins"),
    )

    df = filtered.to_dataframe().sort_values("base_apy", ascending=False)
    print(f"Pools after filter: {len(df)}")

    # Summaries
    by_chain = Metrics.groupby_chain(filtered)
    top_n = int(cfg.get("reporting", {}).get("top_n", 10))
    Metrics.top_n(filtered, n=top_n, key="base_apy")

    # Outputs
    out = cfg.get("output", {})
    outdir = Path(out.get("outdir") or "") if out.get("outdir") else None
    show = bool(out.get("show", True)) if not outdir else False
    charts = out.get("charts", [])

    # Risk metrics derived from time-series returns (base APY as placeholder)
    returns = df.pivot_table(index="timestamp", columns="name", values="base_apy")
    stats = frontier = None
    try:
        stats = risk_metrics.summary_statistics(returns)
        frontier = risk_metrics.efficient_frontier(returns)
    except Exception as exc:
        print(f"Skipping risk metrics: {exc}")

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        cross_section_report(
            filtered,
            outdir,
            perf_fee_bps=float(cfg.get("reporting", {}).get("perf_fee_bps", 0.0)),
            mgmt_fee_bps=float(cfg.get("reporting", {}).get("mgmt_fee_bps", 0.0)),
            top_n=top_n,
        )
        if stats is not None:
            stats.to_csv(outdir / "risk_stats.csv")
        if frontier is not None:
            frontier.to_csv(outdir / "efficient_frontier.csv", index=False)

    if "scatter" in charts:
        if "volatility" in df.columns:
            Visualizer.scatter_risk_return(
                df,
                title="Volatility vs Base APY (bubble=TVL)",
                save_path=str(outdir / "scatter_risk_return.png") if outdir else None,
                show=show,
            )
        else:
            Visualizer.scatter_tvl_apy(
                df,
                title="TVL vs Base APY (bubble=risk)",
                save_path=str(outdir / "scatter_tvl_apy.png") if outdir else None,
                show=show,
            )
    if "chain" in charts:
        Visualizer.bar_group_chain(
            by_chain,
            title="TVL‑gewichteter Base‑APY je Chain",
            save_path=str(outdir / "bar_group_chain.png") if outdir else None,
            show=show,
        )

    # Performance trajectories from historical yields
    if cfg.get("yields_csv"):
        hist_src = HistoricalCSVSource(str(cfg["yields_csv"]))
        returns_ts = Pipeline([hist_src]).run_history()
        initial = float(cfg.get("initial_investment", 1.0))
        nav_ts = performance.nav_trajectories(returns_ts, initial_investment=initial)
        yield_ts = performance.yield_trajectories(returns_ts) * 100.0
        Visualizer.line_chart(
            yield_ts,
            title="Yield over time",
            ylabel="Yield (%)",
            save_path=str(outdir / "yield_vs_time.png") if outdir else None,
            show=show,
        )
        Visualizer.line_chart(
            nav_ts,
            title="NAV over time",
            ylabel="NAV (USD)",
            save_path=str(outdir / "nav_vs_time.png") if outdir else None,
            show=show,
        )

        benchmark_cfg = cfg.get("benchmarks", {})
        weights_cfg = benchmark_cfg.get("weights") or {}
        weights = pd.Series(weights_cfg, dtype=float) if weights_cfg else None

        tickers_cfg = benchmark_cfg.get("tickers") or []
        if not tickers_cfg and weights is not None:
            tickers_cfg = [t for t in weights.index if t in returns_ts.columns]
        selected = [t for t in tickers_cfg if t in returns_ts.columns]
        missing = sorted({t for t in tickers_cfg if t not in returns_ts.columns})
        if missing:
            print(f"[WARN] Missing benchmark tickers: {', '.join(missing)}")

        portfolio_returns = returns_ts[selected] if selected else returns_ts
        if weights is not None and not portfolio_returns.empty:
            weights = weights.reindex(portfolio_returns.columns).fillna(0.0)

        cash_returns: pd.Series | float | None
        cash_series_name = benchmark_cfg.get("cash_series")
        cash_returns = None
        if cash_series_name:
            if cash_series_name in returns_ts.columns:
                cash_returns = returns_ts[cash_series_name]
            else:
                print(f"[WARN] Cash benchmark series '{cash_series_name}' not found; falling back to cash_rate.")

        if cash_returns is None:
            cash_returns = float(benchmark_cfg.get("cash_rate", 0.0))

        labels_cfg = benchmark_cfg.get("labels") or {}
        title = str(benchmark_cfg.get("title", "Rebalanced NAV vs Benchmarks"))

        if not portfolio_returns.empty:
            nav_overlay = Visualizer.nav_with_benchmarks(
                portfolio_returns,
                weights=weights,
                cash_returns=cash_returns,
                initial_investment=initial,
                labels=labels_cfg,
                title=title,
                save_path=str(outdir / "nav_vs_benchmarks.png") if outdir else None,
                show=show,
            )
            if outdir and nav_overlay is not None and not nav_overlay.empty:
                nav_overlay.to_csv(outdir / "nav_vs_benchmarks.csv")


if __name__ == "__main__":
    main()
