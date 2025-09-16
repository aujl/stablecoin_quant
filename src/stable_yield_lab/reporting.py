from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository, performance


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    returns: pd.DataFrame | None = None,
    realised_apy_lookback_days: int | None = None,
    realised_apy_min_observations: int = 1,
) -> dict[str, Path]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with realised APY diagnostics
      - by_chain.csv: aggregated by chain with realised APY averages
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin
      - warnings.csv: data-quality flags for realised APY estimation
    Returns a dict of file label -> path for convenience.
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(df, perf_fee_bps=perf_fee_bps, mgmt_fee_bps=mgmt_fee_bps)

    realised_df = pd.DataFrame()
    if returns is not None and not returns.empty:
        realised_df = performance.estimate_realised_apy(
            returns,
            lookback_days=realised_apy_lookback_days,
            min_observations=realised_apy_min_observations,
        )

    if not realised_df.empty:
        df = df.merge(realised_df, how="left", left_on="name", right_index=True)

    defaults = {
        "realised_apy": float("nan"),
        "realised_apy_observations": 0,
        "realised_apy_window_start": pd.NaT,
        "realised_apy_window_end": pd.NaT,
        "realised_apy_coverage_days": float("nan"),
        "realised_apy_warning": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            if col == "realised_apy_warning":
                df[col] = df[col].fillna("")
            else:
                df[col] = df[col].fillna(default)

    if returns is not None:
        missing_mask = (df["realised_apy_observations"] == 0) & (df["realised_apy_warning"] == "")
        df.loc[missing_mask, "realised_apy_warning"] = "No observations within lookback window."

    warnings_records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        message = str(row.get("realised_apy_warning", "")).strip()
        if message:
            warnings_records.append({"pool": str(row.get("name", "")), "message": message})

    paths["pools"] = out / "pools.csv"
    df.to_csv(paths["pools"], index=False)

    warnings_path = out / "warnings.csv"
    warnings_df = pd.DataFrame(warnings_records, columns=["pool", "message"])
    warnings_df.to_csv(warnings_path, index=False)
    paths["warnings"] = warnings_path

    def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            return df
        grouped = df.groupby(key)
        g = grouped.agg(
            pools=("name", "count"),
            tvl=("tvl_usd", "sum"),
            apr_avg=("base_apy", "mean"),
        )

        def _apr_wavg(grp: pd.DataFrame) -> float:
            total = float(grp["tvl_usd"].sum())
            return float((grp["base_apy"] * grp["tvl_usd"]).sum() / total) if total else float("nan")

        g["apr_wavg"] = grouped.apply(_apr_wavg, include_groups=False)

        if "realised_apy" in df.columns:
            g["realised_apy_avg"] = grouped["realised_apy"].mean()

            def _realised_wavg(grp: pd.DataFrame) -> float:
                valid = grp["realised_apy"].notna()
                if not valid.any():
                    return float("nan")
                weights = grp.loc[valid, "tvl_usd"]
                total = float(weights.sum())
                return (
                    float((grp.loc[valid, "realised_apy"] * weights).sum() / total)
                    if total
                    else float("nan")
                )

            g["realised_apy_wavg"] = grouped.apply(_realised_wavg, include_groups=False)

        return g.reset_index()

    by_chain = _agg(df, "chain")
    paths["by_chain"] = out / "by_chain.csv"
    by_chain.to_csv(paths["by_chain"], index=False)

    by_source = _agg(df, "source")
    by_stable = _agg(df, "stablecoin")
    paths["by_source"] = out / "by_source.csv"
    paths["by_stablecoin"] = out / "by_stablecoin.csv"
    by_source.to_csv(paths["by_source"], index=False)
    by_stable.to_csv(paths["by_stablecoin"], index=False)

    top = df.sort_values("base_apy", ascending=False).head(top_n)
    paths["topN"] = out / "topN.csv"
    top.to_csv(paths["topN"], index=False)

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

    return paths
