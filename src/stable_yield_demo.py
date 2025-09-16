from __future__ import annotations

import argparse
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for the demo CLI."""

    parser = argparse.ArgumentParser(description="Stable yield analytics demo")
    parser.add_argument("config", nargs="?", help="Path to TOML configuration file")
    parser.add_argument(
        "--lookback-days",
        type=int,
        dest="lookback_days",
        help="Lookback horizon in days for realised APY estimation.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        dest="min_observations",
        help="Minimum history observations required for realised APY.",
    )
    return parser.parse_args(argv)


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
        "reporting": {
            "top_n": 10,
            "perf_fee_bps": 0.0,
            "mgmt_fee_bps": 0.0,
            "realised_apy_lookback_days": 90,
            "realised_apy_min_observations": 4,
        },
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


def main() -> None:
    """Run the demo using configuration from file or environment variables."""
    args = parse_args(sys.argv[1:])
    cfg_file = os.getenv("STABLE_YIELD_CONFIG") or args.config
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

    report_cfg = cfg.setdefault("reporting", {})
    if lookback_env := os.getenv("STABLE_YIELD_LOOKBACK_DAYS"):
        try:
            report_cfg["realised_apy_lookback_days"] = int(lookback_env)
        except ValueError:
            pass
    if min_obs_env := os.getenv("STABLE_YIELD_MIN_OBSERVATIONS"):
        try:
            report_cfg["realised_apy_min_observations"] = int(min_obs_env)
        except ValueError:
            pass

    if args.lookback_days is not None:
        report_cfg["realised_apy_lookback_days"] = args.lookback_days
    if args.min_observations is not None:
        report_cfg["realised_apy_min_observations"] = args.min_observations

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

    lookback_cfg = report_cfg.get("realised_apy_lookback_days")
    lookback_days = int(lookback_cfg) if lookback_cfg is not None else None
    min_obs_cfg = report_cfg.get("realised_apy_min_observations", 1)
    try:
        min_obs = int(min_obs_cfg)
    except (TypeError, ValueError):
        min_obs = 1

    returns_history = None
    if cfg.get("yields_csv"):
        hist_src = HistoricalCSVSource(str(cfg["yields_csv"]))
        returns_history = Pipeline([hist_src]).run_history()

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
        report_paths = cross_section_report(
            filtered,
            outdir,
            perf_fee_bps=float(cfg.get("reporting", {}).get("perf_fee_bps", 0.0)),
            mgmt_fee_bps=float(cfg.get("reporting", {}).get("mgmt_fee_bps", 0.0)),
            top_n=top_n,
            returns=returns_history,
            realised_apy_lookback_days=lookback_days,
            realised_apy_min_observations=min_obs,
        )
        warnings_path = report_paths.get("warnings")
        if warnings_path and warnings_path.is_file():
            warnings_df = pd.read_csv(warnings_path)
            if not warnings_df.empty:
                print(
                    f"[WARN] {len(warnings_df)} pools lack sufficient history. Details: {warnings_path}"
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
    if returns_history is not None and not returns_history.empty:
        initial = float(cfg.get("initial_investment", 1.0))
        nav_ts = performance.nav_trajectories(returns_history, initial_investment=initial)
        yield_ts = performance.yield_trajectories(returns_history) * 100.0
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


if __name__ == "__main__":
    main()
