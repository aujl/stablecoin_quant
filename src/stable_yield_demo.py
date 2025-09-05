from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

from stable_yield_lab import CSVSource, Metrics, Pipeline, Visualizer
from stable_yield_lab.reporting import cross_section_report


def load_config(path: str | Path | None) -> dict[str, Any]:
    default = {
        "csv": {"path": str(Path(__file__).with_name("sample_pools.csv"))},
        "filters": {
            "min_tvl": 100_000,
            "min_base_apy": 0.06,
            "auto_only": True,
            # "chains": ["Ethereum"],
            # "stablecoins": ["USDC"],
        },
        "output": {
            "outdir": None,
            "show": True,
            "charts": ["bar", "scatter", "chain"],
        },
        "reporting": {"top_n": 10, "perf_fee_bps": 0.0, "mgmt_fee_bps": 0.0},
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
    cfg_file = os.getenv("STABLE_YIELD_CONFIG") or (sys.argv[1] if len(sys.argv) > 1 else None)
    cfg = load_config(cfg_file)

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
    df.head(20)

    # Summaries
    by_chain = Metrics.groupby_chain(filtered)
    top_n = int(cfg.get("reporting", {}).get("top_n", 10))
    top = Metrics.top_n(filtered, n=top_n, key="base_apy")

    # Outputs
    out = cfg.get("output", {})
    outdir = Path(out.get("outdir") or "") if out.get("outdir") else None
    show = bool(out.get("show", True)) if not outdir else False

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        cross_section_report(
            filtered,
            outdir,
            perf_fee_bps=float(cfg.get("reporting", {}).get("perf_fee_bps", 0.0)),
            mgmt_fee_bps=float(cfg.get("reporting", {}).get("mgmt_fee_bps", 0.0)),
            top_n=top_n,
        )

    charts = out.get("charts", ["bar", "scatter", "chain"]) or []
    if "bar" in charts:
        Visualizer.bar_apr(
            top,
            title="Top‑Stablecoin Pools – Base APY",
            save_path=str(outdir / "bar_apr.png") if outdir else None,
            show=show,
        )
    if "scatter" in args.charts:
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
    if "chain" in args.charts:
        Visualizer.bar_group_chain(
            by_chain,
            title="TVL‑gewichteter Base‑APY je Chain",
            save_path=str(outdir / "bar_group_chain.png") if outdir else None,
            show=show,
        )


if __name__ == "__main__":
    main()
