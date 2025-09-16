from __future__ import annotations

import logging
import os
import sys
import tomllib
import urllib.request
import warnings
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stable_yield_lab import (
    HistoricalCSVSource,
    Metrics,
    Pipeline,
    SchemaAwareCSVSource,
    Visualizer,
    performance,
    risk_metrics,
)
from stable_yield_lab.reporting import cross_section_report


logger = logging.getLogger(__name__)

def _normalise_lookbacks(raw: Any) -> dict[str, Any]:
    """Coerce configuration lookback definitions into a labelled mapping."""

    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(v): v for v in raw}
    return {}


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
        "csv": {
            "path": str(Path(__file__).with_name("sample_pools.csv")),
            "validation": "warn",
            "expected_frequency": None,
            "auto_refresh": False,
            "refresh_url": None,
        },
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
            "realised_apy_lookbacks": {"last 2 weeks": "14D", "last 52 weeks": "52W"},
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

    lookbacks_cfg = _normalise_lookbacks(cfg.get("reporting", {}).get("realised_apy_lookbacks", {}))
    realised_apys: pd.DataFrame | None = None
    nav_ts: pd.DataFrame | None = None
    yield_ts_pct: pd.DataFrame | None = None

    # Load data
    csv_cfg = cfg.get("csv", {})
    csv_path = str(csv_cfg.get("path", cfg["csv"]["path"]))
    validation_level = str(csv_cfg.get("validation", "warn"))
    expected_frequency = csv_cfg.get("expected_frequency")
    if expected_frequency:
        expected_frequency = str(expected_frequency)
    else:
        expected_frequency = None
    frequency_column = str(csv_cfg.get("frequency_column", "timestamp"))
    auto_refresh = bool(csv_cfg.get("auto_refresh", False))
    refresh_url = csv_cfg.get("refresh_url") or None

    refresh_callback = None
    if refresh_url:
        url = str(refresh_url)

        def refresh_callback(target: Path, *, _url: str = url) -> None:
            with urllib.request.urlopen(_url) as response:
                target.write_bytes(response.read())

    elif auto_refresh:
        logger.warning("Auto refresh requested but no refresh_url provided; skipping refresh.")
        auto_refresh = False

    src = SchemaAwareCSVSource(
        path=csv_path,
        validation=validation_level,
        expected_frequency=expected_frequency,
        frequency_column=frequency_column,
        auto_refresh=auto_refresh,
        refresh_callback=refresh_callback,
    )
    repo = Pipeline([src]).run()

    if src.detected_frequency:
        print(f"Detected CSV frequency: {src.detected_frequency}")

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

    pool_names = {pool.name for pool in filtered}
    history_path = cfg.get("yields_csv")
    historical_returns = pd.DataFrame()
    if history_path:
        hist_src = HistoricalCSVSource(str(history_path))
        historical_returns = Pipeline([hist_src]).run_history()
        if historical_returns.empty:
            warnings.warn(
                f"Historical returns from {history_path} produced no rows; skipping risk metrics.",
                stacklevel=2,
            )
    else:
        warnings.warn("No historical returns configured; skipping risk metrics.", stacklevel=2)

    returns_for_metrics = pd.DataFrame()
    if pool_names and not historical_returns.empty:
        matched_cols = [col for col in historical_returns.columns if col in pool_names]
        if matched_cols:
            matched_returns = historical_returns.loc[:, matched_cols]
            matched_returns = matched_returns.dropna(axis=1, how="all").dropna(how="all")
            if matched_returns.empty or matched_returns.columns.empty:
                warnings.warn(
                    "Historical returns contain only missing values after filtering; skipping risk metrics.",
                    stacklevel=2,
                )
            else:
                returns_for_metrics = matched_returns
        else:
            warnings.warn(
                "No historical returns matched the filtered pools; skipping risk metrics.",
                stacklevel=2,
            )
    elif not pool_names:
        warnings.warn("No pools available after filtering; skipping risk metrics.", stacklevel=2)

    stats = frontier = None
    if not returns_for_metrics.empty:
        try:
            stats = risk_metrics.summary_statistics(returns_for_metrics)
            frontier = risk_metrics.efficient_frontier(returns_for_metrics)
        except Exception as exc:
            print(f"Skipping risk metrics: {exc}")

    if cfg.get("yields_csv"):
        hist_src = HistoricalCSVSource(str(cfg["yields_csv"]))
        returns_ts = Pipeline([hist_src]).run_history()
        if not returns_ts.empty:
            initial = float(cfg.get("initial_investment", 1.0))
            nav_ts = performance.nav_trajectories(returns_ts, initial_investment=initial)
            yield_ts_pct = performance.yield_trajectories(returns_ts) * 100.0
            if lookbacks_cfg:
                apy_table = performance.horizon_apys(nav_ts, lookbacks=lookbacks_cfg, value_type="nav")
                if not apy_table.empty:
                    apy_table = apy_table.rename(
                        columns={label: f"Realised APY ({label})" for label in apy_table.columns}
                    )
                    realised_apys = apy_table
                    if not df.empty:
                        df = df.merge(apy_table, left_on="name", right_index=True, how="left")
                    pretty = apy_table.mul(100.0).round(2)
                    print("Realised APY horizons (annualised, %):")
                    print(pretty.to_string())

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        cross_section_report(
            filtered,
            outdir,
            perf_fee_bps=float(cfg.get("reporting", {}).get("perf_fee_bps", 0.0)),
            mgmt_fee_bps=float(cfg.get("reporting", {}).get("mgmt_fee_bps", 0.0)),
            top_n=top_n,
            horizon_apys=realised_apys,
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
    returns_for_nav = returns_for_metrics if not returns_for_metrics.empty else historical_returns
    if not returns_for_nav.empty:
        initial = float(cfg.get("initial_investment", 1.0))
        nav_ts = performance.nav_trajectories(returns_for_nav, initial_investment=initial)
        yield_ts = performance.yield_trajectories(returns_for_nav) * 100.0

        Visualizer.line_chart(
            yield_ts_pct,
            title="Yield over time",
            ylabel="Yield (%)",
            save_path=str(outdir / "yield_vs_time.png") if outdir else None,
            show=show,
        )
    if nav_ts is not None:
        Visualizer.line_chart(
            nav_ts,
            title="NAV over time",
            ylabel="NAV (USD)",
            save_path=str(outdir / "nav_vs_time.png") if outdir else None,
            show=show,
        )


if __name__ == "__main__":
    main()
