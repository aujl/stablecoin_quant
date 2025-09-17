from pathlib import Path

import pandas as pd

from stable_yield_lab import HistoricalCSVSource, Pipeline
from stable_yield_lab.core.models import Pool
from stable_yield_lab.core.repositories import PoolRepository
from stable_yield_lab.reporting import cross_section_report


def test_cross_section_report_includes_realised_apy(tmp_path: Path) -> None:
    yields_csv = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"

    repo = PoolRepository(
        [
            Pool(name="PoolA", chain="Ethereum", stablecoin="USDC", tvl_usd=1_000_000, base_apy=0.08),
            Pool(name="PoolB", chain="Ethereum", stablecoin="USDT", tvl_usd=500_000, base_apy=0.07),
        ]
    )
    returns = Pipeline([HistoricalCSVSource(str(yields_csv))]).run_history()

    outdir = tmp_path / "report"
    paths = cross_section_report(
        repo,
        outdir,
        returns=returns,
        realised_apy_lookback_days=30,
        realised_apy_min_observations=3,
    )

    pools_df = pd.read_csv(paths["pools"])
    assert {"realised_apy", "realised_apy_observations", "realised_apy_warning"}.issubset(pools_df.columns)
    assert pools_df["realised_apy"].isna().all()
    warning_text = pools_df["realised_apy_warning"].fillna("")
    assert (warning_text.str.contains("Only 2 observations")).all()

    warnings_df = pd.read_csv(paths["warnings"])
    assert set(warnings_df["pool"]) == {"PoolA", "PoolB"}
    assert warnings_df["message"].str.contains("Only 2 observations").all()

    top_df = pd.read_csv(paths["topN"])
    assert "realised_apy" in top_df.columns

    chain_df = pd.read_csv(paths["by_chain"])
    assert "realised_apy_wavg" in chain_df.columns
