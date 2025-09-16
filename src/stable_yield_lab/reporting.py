from __future__ import annotations

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _prepare_returns_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise the historical returns frame."""

    if returns.empty:
        return returns

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must be indexed by timestamps")

    if not returns.index.is_monotonic_increasing:
        returns = returns.sort_index()

    numeric_returns = returns.apply(pd.to_numeric, errors="coerce")
    # Drop columns that remain entirely NaN after coercion.
    numeric_returns = numeric_returns.dropna(axis=1, how="all")
    return numeric_returns


def _ordered_pool_names(df: pd.DataFrame) -> list[str]:
    """Return pool names preserving their first occurrence order."""

    if "name" not in df.columns:
        return []
    names = df["name"].dropna().astype(str)
    return list(dict.fromkeys(names))


def _filter_returns_for_pools(returns: pd.DataFrame, pool_names: list[str]) -> pd.DataFrame:
    """Select the subset of returns aligned with the provided pool names."""

    if returns.empty or not pool_names:
        return pd.DataFrame(index=returns.index)
    available = [name for name in pool_names if name in returns.columns]
    if not available:
        return pd.DataFrame(index=returns.index)
    return returns[available]


def _rolling_apy_tidy(
    returns: pd.DataFrame, rolling_windows: Sequence[int], periods_per_year: int
) -> pd.DataFrame:
    """Compute annualised rolling APYs and return a tidy DataFrame."""

    frames: list[pd.DataFrame] = []
    gross_returns = 1.0 + returns

    seen_windows: list[int] = []
    for raw_window in rolling_windows:
        window = int(raw_window)
        if window <= 0 or window in seen_windows:
            continue
        seen_windows.append(window)
        if len(gross_returns) < window:
            continue
        compounded = gross_returns.rolling(window=window, min_periods=window).apply(
            lambda arr: float(pd.Series(arr).prod()),
            raw=True,
        )
        apy = compounded.pow(periods_per_year / window) - 1.0
        tidy = (
            apy.reset_index()
            .rename(columns={apy.index.name or "index": "timestamp"})
            .melt(id_vars="timestamp", var_name="name", value_name="rolling_apy")
        )
        tidy = tidy.dropna(subset=["rolling_apy"])
        if tidy.empty:
            continue
        tidy["window"] = window
        frames.append(tidy)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "name", "rolling_apy", "window"])

    return pd.concat(frames, ignore_index=True)


def _nav_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Construct net asset value trajectories from periodic returns."""

    if returns.empty:
        return returns
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    return nav


def _drawdown_long_and_summary(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute drawdown time-series and per-pool summary statistics."""

    nav = _nav_from_returns(returns)
    if nav.empty:
        columns = ["timestamp", "name", "drawdown"]
        drawdown_long = pd.DataFrame(columns=columns)
        summary = pd.DataFrame(columns=["name", "max_drawdown", "current_drawdown"])
        return drawdown_long, summary

    running_max = nav.cummax()
    drawdowns = nav.divide(running_max).subtract(1.0)
    drawdown_long = (
        drawdowns.reset_index()
        .rename(columns={drawdowns.index.name or "index": "timestamp"})
        .melt(id_vars="timestamp", var_name="name", value_name="drawdown")
        .dropna(subset=["drawdown"])
    )
    current = drawdowns.ffill().iloc[-1]
    summary = pd.DataFrame(
        {
            "name": drawdowns.columns,
            "max_drawdown": drawdowns.min(skipna=True).values,
            "current_drawdown": current.values,
        }
    )
    return drawdown_long, summary


def _target_series(df: pd.DataFrame, target_field: str) -> pd.Series:
    """Extract the target metric for each pool keyed by name."""

    if "name" not in df.columns or target_field not in df.columns:
        return pd.Series(dtype=float)
    grouped = df.dropna(subset=["name"]).groupby("name", sort=False)[target_field].first()
    target = pd.to_numeric(grouped, errors="coerce")
    return target


def _realised_vs_target(
    returns: pd.DataFrame, targets: pd.Series, periods_per_year: int
) -> pd.DataFrame:
    """Compare realised APYs against configured targets on a per-pool basis."""

    if returns.empty:
        return pd.DataFrame(columns=["name", "realised_apy", "target_apy", "realised_minus_target"])

    gross = (1.0 + returns).prod(skipna=True, min_count=1)
    counts = returns.notna().sum()
    valid_counts = counts.where(counts > 0)
    annualised = gross.pow(periods_per_year / valid_counts) - 1.0
    annualised = annualised.reindex(returns.columns)
    comparison = pd.DataFrame(
        {
            "name": returns.columns,
            "realised_apy": annualised.values,
            "target_apy": targets.reindex(returns.columns).values,
        }
    )
    valid_names = counts[counts > 0].index
    comparison = comparison[comparison["name"].isin(valid_names)].reset_index(drop=True)
    if comparison.empty:
        return comparison
    comparison["realised_minus_target"] = (
        comparison["realised_apy"] - comparison["target_apy"]
    )
    return comparison


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    horizon_apys: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    rolling_windows: Sequence[int] = (4, 12),
    periods_per_year: int = 52,
    target_field: str = "net_apy",
) -> dict[str, Path]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin

    When ``horizon_apys`` is provided the corresponding columns are merged into
    the pool-level CSVs, enabling persistence of realised performance metrics
    such as ``Realised APY (last 52 weeks)``.

    When ``returns`` are supplied, additional historical analytics are
    generated:
      - rolling_apy.csv: annualised rolling yields derived from the returns
      - drawdowns.csv / drawdown_summary.csv: pathwise and summary drawdowns
      - realised_vs_target.csv: realised APY versus the configured target

    Returns a dict of file label -> path for convenience.
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(df, perf_fee_bps=perf_fee_bps, mgmt_fee_bps=mgmt_fee_bps)
    if horizon_apys is not None and not horizon_apys.empty:
        df = df.merge(horizon_apys, left_on="name", right_index=True, how="left")
    paths["pools"] = out / "pools.csv"
    df.to_csv(paths["pools"], index=False)

    # Aggregations
    by_chain = Metrics.groupby_chain(repo)
    paths["by_chain"] = out / "by_chain.csv"
    by_chain.to_csv(paths["by_chain"], index=False)

    def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            return df
        g = df.groupby(key).agg(
            pools=("name", "count"),
            tvl=("tvl_usd", "sum"),
            apr_avg=("base_apy", "mean"),
            apr_wavg=(
                "base_apy",
                lambda x: (x * df.loc[x.index, "tvl_usd"]).sum() / df.loc[x.index, "tvl_usd"].sum(),
            ),
        )
        return g.reset_index()

    by_source = _agg(df, "source")
    by_stable = _agg(df, "stablecoin")
    paths["by_source"] = out / "by_source.csv"
    paths["by_stablecoin"] = out / "by_stablecoin.csv"
    by_source.to_csv(paths["by_source"], index=False)
    by_stable.to_csv(paths["by_stablecoin"], index=False)

    # Top N
    top = df.sort_values("base_apy", ascending=False).head(top_n)
    paths["topN"] = out / "topN.csv"
    top.to_csv(paths["topN"], index=False)

    # Concentration metrics
    hhi_total = Metrics.hhi(df, value_col="tvl_usd")
    hhi_chain = Metrics.hhi(df, value_col="tvl_usd", group_col="chain")
    hhi_stable = Metrics.hhi(df, value_col="tvl_usd", group_col="stablecoin")
    conc = {
        "scope": ["total"],
        "hhi": list(hhi_total["hhi"]) if not hhi_total.empty else [float("nan")],
    }
    conc_df = pd.DataFrame(conc)
    hhi_chain["scope"] = "chain:" + hhi_chain["chain"].astype(str)
    hhi_stable["scope"] = "stablecoin:" + hhi_stable["stablecoin"].astype(str)
    conc_all = pd.concat(
        [conc_df, hhi_chain[["scope", "hhi"]], hhi_stable[["scope", "hhi"]]], ignore_index=True
    )
    paths["concentration"] = out / "concentration.csv"
    conc_all.to_csv(paths["concentration"], index=False)

    if returns is not None and not returns.empty:
        prepared_returns = _prepare_returns_frame(returns)
        pool_names = _ordered_pool_names(df)
        returns_subset = _filter_returns_for_pools(prepared_returns, pool_names)

        if not returns_subset.empty:
            rolling_df = _rolling_apy_tidy(returns_subset, rolling_windows, periods_per_year)
            if not rolling_df.empty:
                rolling_path = out / "rolling_apy.csv"
                rolling_df.to_csv(rolling_path, index=False)
                paths["rolling_apy"] = rolling_path

            drawdown_long, drawdown_summary = _drawdown_long_and_summary(returns_subset)
            if not drawdown_long.empty:
                drawdown_path = out / "drawdowns.csv"
                drawdown_long.to_csv(drawdown_path, index=False)
                paths["drawdowns"] = drawdown_path
            if not drawdown_summary.empty:
                summary_path = out / "drawdown_summary.csv"
                drawdown_summary.to_csv(summary_path, index=False)
                paths["drawdown_summary"] = summary_path

            targets = _target_series(df, target_field)
            comparison = _realised_vs_target(returns_subset, targets, periods_per_year)
            if not comparison.empty:
                comparison_path = out / "realised_vs_target.csv"
                comparison.to_csv(comparison_path, index=False)
                paths["realised_vs_target"] = comparison_path

    return paths
