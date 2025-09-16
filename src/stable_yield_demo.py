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
from stable_yield_lab.reporting import cross_section_report, scenario_comparison_report


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
        "rebalance": {"benchmark": None, "selected": [], "scenarios": {}},
    }

    cfg_path = Path(path) if path else None

    if cfg_path and cfg_path.is_file():
        with open(cfg_path, "rb") as f:

            file_cfg = tomllib.load(f)

        for k, v in file_cfg.items():

            if isinstance(v, dict) and k in default and isinstance(default[k], dict):

                cast(dict, default[k]).update(v)

            else:

                default[k] = v

    else:

        if cfg_path:

            print(f"[WARN] Config file not found at {cfg_path}. Using defaults.")

    return default


def _build_rebalance_calendar(index: pd.DatetimeIndex, params: dict[str, Any]) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([])

    calendar = params.get("calendar")
    if calendar:
        cal_idx = pd.DatetimeIndex(pd.to_datetime(calendar))
    else:
        cadence = params.get("cadence")
        if cadence:
            try:
                grouped = index.to_series().groupby(index.to_period(str(cadence))).last()
                cal_idx = pd.DatetimeIndex(grouped.values)
            except Exception:
                cal_idx = pd.DatetimeIndex(index)
        else:
            cal_idx = pd.DatetimeIndex(index)

    cal_idx = pd.DatetimeIndex(cal_idx).unique().sort_values()
    return cal_idx.intersection(index)


def _infer_portfolio_weights(returns: pd.DataFrame, pools: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    if not pools.empty and {"name", "tvl_usd"}.issubset(pools.columns):
        tvl = pools.set_index("name")["tvl_usd"].reindex(returns.columns).fillna(0.0)
        if tvl.sum() > 0:
            return tvl / tvl.sum()

    return pd.Series(1.0 / returns.shape[1], index=returns.columns)


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

        rebalance_cfg = cfg.get("rebalance", {})
        scenario_defs = rebalance_cfg.get("scenarios", {})
        selected = rebalance_cfg.get("selected") or list(scenario_defs.keys())
        if isinstance(selected, str):
            selected = [selected]

        scenario_map: dict[str, performance.RebalanceScenario] = {}
        for name in selected:
            params = scenario_defs.get(name)
            if not isinstance(params, dict):
                continue
            calendar = _build_rebalance_calendar(returns_ts.index, params)
            cost_bps = float(params.get("cost_bps", 0.0))
            scenario_map[name] = performance.RebalanceScenario(calendar=calendar, cost_bps=cost_bps)

        if scenario_map:
            benchmark_name = rebalance_cfg.get("benchmark")
            if benchmark_name and benchmark_name not in scenario_map:
                print(
                    f"[WARN] Benchmark scenario '{benchmark_name}' not found; "
                    "using frictionless benchmark."
                )
                benchmark_name = None

            weights = _infer_portfolio_weights(returns_ts, df)
            summary = performance.run_rebalance_scenarios(
                returns_ts,
                weights,
                scenario_map,
                benchmark=benchmark_name,
                initial_nav=initial,
            )
            print("Scenario comparison:")
            print(summary.metrics)

            if outdir:
                scenario_comparison_report(summary, outdir)

            apy_chart_df = summary.metrics.reset_index().rename(
                columns={"scenario": "name", "realized_apy": "base_apy"}
            )
            apy_chart_path = outdir / "rebalance_realized_apy.png" if outdir else None
            Visualizer.bar_apr(
                apy_chart_df,
                title="Realised APY by scenario",
                save_path=str(apy_chart_path) if apy_chart_path else None,
                show=show,
            )
            nav_chart_path = outdir / "rebalance_nav.png" if outdir else None
            Visualizer.line_chart(
                summary.navs,
                title="Scenario NAV comparison",
                ylabel="NAV (USD)",
                save_path=str(nav_chart_path) if nav_chart_path else None,
                show=show,
            )


if __name__ == "__main__":
    main()
