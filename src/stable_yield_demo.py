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
        "output": {
            "outdir": None,
            "show": True,
            "charts": ["bar", "scatter", "chain"],
            "history_charts": ["rolling_apy", "drawdowns"],
        },
        "reporting": {
            "top_n": 10,
            "perf_fee_bps": 0.0,
            "mgmt_fee_bps": 0.0,
            "history": {
                "enabled": True,
                "rolling_windows": [4, 12],
                "periods_per_year": 52,
                "target_field": "net_apy",
            },
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
    reporting_cfg = cfg.get("reporting", {})
    top_n = int(reporting_cfg.get("top_n", 10))
    Metrics.top_n(filtered, n=top_n, key="base_apy")

    # Outputs
    out = cfg.get("output", {})
    outdir = Path(out.get("outdir") or "") if out.get("outdir") else None
    show = bool(out.get("show", True)) if not outdir else False
    charts = out.get("charts", [])
    history_charts = out.get("history_charts", [])

    history_cfg = reporting_cfg.get("history", {})
    history_enabled = bool(history_cfg.get("enabled", False))
    rolling_windows_cfg = history_cfg.get("rolling_windows", [])
    rolling_windows = tuple(int(w) for w in rolling_windows_cfg) if rolling_windows_cfg else (4, 12)
    periods_per_year = int(history_cfg.get("periods_per_year", 52))
    target_field = str(history_cfg.get("target_field", "net_apy"))

    # Load historical returns once for reuse
    returns_ts = pd.DataFrame()
    if history_enabled and cfg.get("yields_csv"):
        hist_src = HistoricalCSVSource(str(cfg["yields_csv"]))
        returns_ts = Pipeline([hist_src]).run_history()

    # Risk metrics derived from time-series returns (base APY as placeholder)
    returns = df.pivot_table(index="timestamp", columns="name", values="base_apy")
    stats = frontier = None
    try:
        stats = risk_metrics.summary_statistics(returns)
        frontier = risk_metrics.efficient_frontier(returns)
    except Exception as exc:
        print(f"Skipping risk metrics: {exc}")

    report_paths: dict[str, Path] = {}

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        report_paths = cross_section_report(
            filtered,
            outdir,
            perf_fee_bps=float(cfg.get("reporting", {}).get("perf_fee_bps", 0.0)),
            mgmt_fee_bps=float(cfg.get("reporting", {}).get("mgmt_fee_bps", 0.0)),
            top_n=top_n,
            returns=returns_ts if history_enabled and not returns_ts.empty else None,
            rolling_windows=rolling_windows,
            periods_per_year=periods_per_year,
            target_field=target_field,
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
    if history_enabled and not returns_ts.empty:
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

        if outdir and history_charts:
            if "rolling_apy" in history_charts and "rolling_apy" in report_paths:
                rolling_df = pd.read_csv(report_paths["rolling_apy"], parse_dates=["timestamp"])
                for window in sorted(rolling_df["window"].unique()):
                    pivot = (
                        rolling_df[rolling_df["window"] == window]
                        .pivot(index="timestamp", columns="name", values="rolling_apy")
                        .sort_index()
                    )
                    if pivot.empty:
                        continue
                    Visualizer.line_chart(
                        pivot * 100.0,
                        title=f"Rolling APY ({window}-period)",
                        ylabel="Rolling APY (%)",
                        save_path=str(outdir / f"rolling_apy_{window}.png"),
                        show=show,
                    )
            if "drawdowns" in history_charts and "drawdowns" in report_paths:
                drawdown_df = pd.read_csv(report_paths["drawdowns"], parse_dates=["timestamp"])
                pivot = drawdown_df.pivot(index="timestamp", columns="name", values="drawdown").sort_index()
                if not pivot.empty:
                    Visualizer.line_chart(
                        pivot * 100.0,
                        title="Drawdown trajectories",
                        ylabel="Drawdown (%)",
                        save_path=str(outdir / "drawdowns.png"),
                        show=show,
                    )
            if "realised_vs_target" in history_charts and "realised_vs_target" in report_paths:
                realised_df = pd.read_csv(report_paths["realised_vs_target"])
                Visualizer.bar_realised_vs_target(
                    realised_df,
                    title="Realised vs target APY",
                    save_path=str(outdir / "realised_vs_target.png"),
                    show=show,
                )


if __name__ == "__main__":
    main()
