from __future__ import annotations

import argparse
import os
from pathlib import Path

from stable_yield_lab import CSVSource, Metrics, Pipeline, Visualizer
from stable_yield_lab.reporting import cross_section_report


def parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def get_args() -> argparse.Namespace:
    d_csv = os.getenv("STABLE_YIELD_CSV") or str(Path(__file__).with_name("sample_pools.csv"))
    d_min_tvl = float(os.getenv("STABLE_YIELD_MIN_TVL", "100000"))
    d_min_base_apy = float(os.getenv("STABLE_YIELD_MIN_BASE_APY", "0.06"))
    d_auto_only = os.getenv("STABLE_YIELD_AUTO_ONLY", "true").lower() in {"1", "true", "yes"}
    d_chains = parse_list(os.getenv("STABLE_YIELD_CHAINS"))
    d_stablecoins = parse_list(os.getenv("STABLE_YIELD_STABLECOINS"))
    d_outdir = os.getenv("STABLE_YIELD_OUTDIR")

    p = argparse.ArgumentParser(
        description="StableYield demo with configurable filters and charts."
    )
    p.add_argument("--csv", default=d_csv, help="Path to CSV dataset (default: sample in repo)")
    p.add_argument("--min-tvl", type=float, default=d_min_tvl, help="Minimum TVL filter")
    p.add_argument(
        "--min-base-apy", type=float, default=d_min_base_apy, help="Minimum base APY (fraction)"
    )
    p.add_argument(
        "--auto-only",
        action=argparse.BooleanOptionalAction,
        default=d_auto_only,
        help="Filter to auto-only pools",
    )
    p.add_argument(
        "--chains", default=",".join(d_chains or []), help="Comma-separated list of chains"
    )
    p.add_argument(
        "--stablecoins",
        default=",".join(d_stablecoins or []),
        help="Comma-separated list of stablecoins",
    )
    p.add_argument(
        "--charts",
        nargs="*",
        default=["bar", "scatter", "chain"],
        choices=["bar", "scatter", "chain"],
        help="Which charts to render",
    )
    p.add_argument("--outdir", default=d_outdir, help="Directory to save charts and CSVs")
    p.add_argument(
        "--fee-bps", type=float, default=0.0, help="Performance+management fees in bps for net APY"
    )
    p.add_argument(
        "--no-show", action="store_true", help="Do not display charts (use with --outdir)"
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    chains = parse_list(args.chains)
    stablecoins = parse_list(args.stablecoins)

    # Load data
    src = CSVSource(path=args.csv)
    repo = Pipeline([src]).run()

    # Apply filters
    filtered = repo.filter(
        min_tvl=args.min_tvl,
        min_base_apy=args.min_base_apy,
        auto_only=args.auto_only,
        chains=chains,
        stablecoins=stablecoins,
    )

    df = filtered.to_dataframe().sort_values("base_apy", ascending=False)
    print(f"Pools after filter: {len(df)}")
    df.head(20)

    # Summaries
    by_chain = Metrics.groupby_chain(filtered)
    top10 = Metrics.top_n(filtered, n=10, key="base_apy")

    # Prepare outputs
    show = not args.no_show and not args.outdir
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        # Write standardized cross-section report (includes pools, rollups, topN, concentration)
        cross_section_report(
            filtered, outdir, perf_fee_bps=args.fee_bps, mgmt_fee_bps=0.0, top_n=10
        )

    # Render charts (file-first if outdir provided)
    if "bar" in args.charts:
        Visualizer.bar_apr(
            top10,
            title="Top‑10 Stablecoin Pools – Base APY",
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
