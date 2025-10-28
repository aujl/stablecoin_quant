#!/usr/bin/env python3
"""Run a live end-to-end StableYieldLab workflow.

The script orchestrates data collection across DefiLlama, Morpho and Beefy,
derives risk-adjusted pool metrics, generates reports/charts, and attempts to
download historical yield data from DefiLlama for the highest-yielding pools.
It degrades gracefully when network access is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from stable_yield_lab import (
    BeefySource,
    DefiLlamaHistorySource,
    DefiLlamaSource,
    MorphoSource,
    Pipeline,
    Visualizer,
    performance,
    portfolio,
)
from stable_yield_lab.analytics.metrics import Metrics
from stable_yield_lab.core import PoolRepository
from stable_yield_lab.reporting import cross_section_report
from stable_yield_lab.risk_scoring import score_pool


logger = logging.getLogger(__name__)


def _build_repo(cache_root: Path) -> PoolRepository:
    """Fetch pools from all live adapters and apply risk scoring overrides."""

    defillama_source = DefiLlamaSource(
        stable_only=True,
        cache_path=str(cache_root / "defillama_pools.json"),
    )
    morpho_source = MorphoSource(cache_path=str(cache_root / "morpho_markets.json"))
    beefy_source = BeefySource(cache_dir=str(cache_root / "beefy"))

    snapshot_pipeline = Pipeline([defillama_source, morpho_source, beefy_source])
    repo = snapshot_pipeline.run()

    snapshot_df = repo.to_dataframe()
    has_source_col = not snapshot_df.empty and "source" in snapshot_df.columns
    audits_override = (
        {row["name"]: 4 for _, row in snapshot_df[snapshot_df["source"] == "morpho"].iterrows()}
        if has_source_col
        else {}
    )
    volatility_override = (
        {row["name"]: 0.25 for _, row in snapshot_df[snapshot_df["source"] == "beefy"].iterrows()}
        if has_source_col
        else {}
    )

    scored = [
        score_pool(
            pool,
            protocol_audits=audits_override,
            yield_volatility=volatility_override,
        )
        for pool in repo
    ]
    return PoolRepository(scored)


def _download_history(
    defillama_source: DefiLlamaSource,
    targets: list[str],
    cache_root: Path,
    resample: str = "W-FRI",
) -> pd.DataFrame:
    """Fetch DefiLlama history for selected pools."""

    try:
        pool_id_lookup = defillama_source.pool_id_map()
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("DefiLlama pool ID lookup failed: %s", exc)
        return pd.DataFrame()

    missing_ids = [name for name in targets if name not in pool_id_lookup]
    if missing_ids:
        logger.warning("No chart IDs for pools: %s", ", ".join(missing_ids))
        targets = [name for name in targets if name in pool_id_lookup]

    if not targets:
        return pd.DataFrame()

    history_source = DefiLlamaHistorySource(
        pool_ids=pool_id_lookup,
        targets=targets,
        cache_dir=cache_root / "defillama_history",
        resample=resample,
    )

    history_pipeline = Pipeline([history_source])
    return history_pipeline.run_history()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cache_root = Path("outputs/cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    repo = _build_repo(cache_root)
    if len(repo) == 0:
        print(
            "No pools collected from live adapters. Check network access or adapter configuration."
        )
        return

    filtered = repo.filter(
        min_tvl=1_000_000,
        min_base_apy=0.04,
        auto_only=False,
    )
    if len(filtered) == 0:
        print("Filters removed all pools. Relax thresholds and try again.")
        return

    seen_names: set[str] = set()
    unique_pools = []
    for pool in filtered:
        if pool.name in seen_names:
            continue
        seen_names.add(pool.name)
        unique_pools.append(pool)
    filtered_unique = PoolRepository(unique_pools)
    if len(filtered_unique) == 0:
        print("No unique pools available after deduplication.")
        return

    pools_df = Metrics.add_net_apy_column(
        filtered_unique.to_dataframe().sort_values("base_apy", ascending=False),
        perf_fee_bps=15,
        mgmt_fee_bps=50,
    )
    by_chain = Metrics.groupby_chain(filtered_unique)

    defillama_mask = pools_df["source"] == "defillama"
    top_targets: list[str] = []
    for name in pools_df.loc[defillama_mask, "name"]:
        if name not in top_targets:
            top_targets.append(name)
        if len(top_targets) >= 8:
            break

    returns = pd.DataFrame()
    if top_targets:
        returns = _download_history(
            DefiLlamaSource(
                stable_only=True,
                cache_path=str(cache_root / "defillama_pools.json"),
            ),
            top_targets,
            cache_root,
        )

    if returns.empty:
        logger.warning(
            "DefiLlama history unavailable. Continuing without performance trajectories."
        )
        nav_paths = pd.DataFrame()
        yield_paths = pd.DataFrame()
        portfolio_metrics = pd.Series(dtype=float)
        portfolio_nav = pd.Series(dtype=float)
    else:
        returns = returns.apply(pd.to_numeric, errors="coerce")
        nav_paths = performance.nav_trajectories(returns, initial_investment=10_000.0)
        yield_paths = performance.yield_trajectories(returns)
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        portfolio_metrics, portfolio_nav = portfolio.apy_performance_summary(
            returns,
            weights,
            freq=52,
            initial_nav=10_000.0,
        )

    output_dir = Path("outputs/investor_run_live")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_paths = cross_section_report(
        filtered_unique,
        output_dir,
        perf_fee_bps=15,
        mgmt_fee_bps=50,
        top_n=10,
        returns=returns if not returns.empty else None,
    )

    Visualizer.bar_apr(
        pools_df,
        title="Net APY per Pool",
        y_col="net_apy",
        max_bars=60,
        label_max_chars=28,
        save_path=str(output_dir / "bar_apr.png"),
        show=False,
    )
    Visualizer.scatter_tvl_apy(
        pools_df,
        title="TVL vs Net APY",
        y_col="net_apy",
        annotate_top_n=25,
        label_max_chars=24,
        save_path=str(output_dir / "scatter_tvl_apy.png"),
        show=False,
    )
    if not nav_paths.empty:
        Visualizer.line_chart(
            nav_paths.join(portfolio_nav.rename("Portfolio NAV")),
            title="NAV trajectories",
            ylabel="NAV (USD)",
            save_path=str(output_dir / "nav_paths.png"),
            show=False,
        )
    if not yield_paths.empty:
        Visualizer.line_chart(
            yield_paths * 100.0,
            title="Yield trajectories (%)",
            ylabel="Yield (%)",
            save_path=str(output_dir / "yield_paths.png"),
            show=False,
        )

    print("TVL by chain:")
    print(by_chain)
    print("\nTop pools (base APY):")
    print(pools_df.head(10)[["name", "base_apy", "net_apy", "tvl_usd", "source"]])
    if not returns.empty:
        print("\nPortfolio metrics:")
        print(portfolio_metrics.to_frame(name="value"))
    else:
        print("\nPortfolio metrics: historical returns unavailable.")

    print("\nReport files:")
    for label, path in report_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
