from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

from stable_yield_lab import CSVSource, Metrics, Pipeline, Visualizer
from stable_yield_lab.reporting import cross_section_report


def parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def load_defaults(config_path: str | None) -> dict[str, object]:
    """Load default values, optionally from a TOML config file."""

    defaults: dict[str, object] = {
        "csv": str(Path(__file__).with_name("sample_pools.csv")),
        "min_tvl": 100000.0,
        "min_base_apy": 0.06,
        "auto_only": True,
        "chains": "",
        "stablecoins": "",
        "charts": ["bar", "scatter", "chain"],
        "outdir": None,
        "fee_bps": 0.0,
        "no_show": False,
    }
    if config_path:
        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
        defaults.update(cfg)
    return defaults


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = None
    if argv and not argv[0].startswith("-"):
        config_path = argv.pop(0)
    d = load_defaults(config_path)

    p = argparse.ArgumentParser(
        description="StableYield demo with configurable filters and charts."
    )
    p.add_argument("--csv", default=d["csv"], help="Path to CSV dataset")
    p.add_argument("--min-tvl", type=float, default=d["min_tvl"], help="Minimum TVL filter")
    p.add_argument(
        "--min-base-apy", type=float, default=d["min_base_apy"], help="Minimum base APY (fraction)"
    )
    p.add_argument(
        "--auto-only",
        action=argparse.BooleanOptionalAction,
        default=bool(d["auto_only"]),
        help="Filter to auto-only pools",
    )
    p.add_argument(
        "--chains", default=d["chains"], help="Comma-separated list of chains"
    )
    p.add_argument(
        "--stablecoins", default=d["stablecoins"], help="Comma-separated list of stablecoins"
    )
    p.add_argument(
        "--charts",
        nargs="*",
        default=d["charts"],
        choices=["bar", "scatter", "chain"],
        help="Which charts to render",
    )
    p.add_argument("--outdir", default=d["outdir"], help="Directory to save charts and CSVs")
    p.add_argument(
        "--fee-bps", type=float, default=d["fee_bps"], help="Performance+management fees in bps for net APY"
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        default=bool(d["no_show"]),
        help="Do not display charts (use with --outdir)",
    )
    return p.parse_args(argv)


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
