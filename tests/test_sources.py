from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import (
    BeefySource,
    DefiLlamaSource,
    MorphoSource,
    PoolRepository,
    SchemaAwareCSVSource,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_defillama_parsing() -> None:
    src = DefiLlamaSource(cache_path=str(FIXTURES / "defillama_pools.json"))
    pools = src.fetch()
    assert pools
    assert all(p.source == "defillama" for p in pools)
    assert all(p.base_apy >= 0 for p in pools)


def test_morpho_parsing_and_filtering() -> None:
    src = MorphoSource(cache_path=str(FIXTURES / "morpho_markets.json"))
    pools = src.fetch()
    repo = PoolRepository(pools)
    filtered = repo.filter(min_tvl=1.0, min_base_apy=0.0)
    assert len(filtered) <= len(pools)
    assert all(p.tvl_usd >= 1.0 for p in filtered)


def test_beefy_parsing_and_filter_auto() -> None:
    src = BeefySource(cache_dir=str(FIXTURES / "beefy"))
    pools = src.fetch()
    assert pools
    repo = PoolRepository(pools)
    auto = repo.filter(auto_only=True)
    assert len(auto) == len(pools)
    assert all(p.is_auto for p in auto)


def test_filter_auto_only_mixed_sources() -> None:
    llama = DefiLlamaSource(cache_path=str(FIXTURES / "defillama_pools.json"))
    beefy = BeefySource(cache_dir=str(FIXTURES / "beefy"))
    repo = PoolRepository(llama.fetch() + beefy.fetch())
    auto = repo.filter(auto_only=True)
    assert all(p.is_auto for p in auto)
    assert all(p.source == "beefy" for p in auto)


def test_schema_validation_strict_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(
        """name,chain,stablecoin,tvl_usd,base_apy,reward_apy,is_auto,source,risk_score,timestamp
Invalid Pool,Ethereum,USDC,not_a_number,0.05,0.0,True,UnitTest,2.0,1700000000
"""
    )

    src = SchemaAwareCSVSource(path=str(csv_path), validation="strict")
    with pytest.raises(ValueError) as exc:
        src.fetch()

    assert "tvl_usd" in str(exc.value)


def test_schema_frequency_inference(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D").astype("int64") // 10**9
    df = pd.DataFrame(
        {
            "name": [f"Pool {i}" for i in range(4)],
            "chain": ["Ethereum"] * 4,
            "stablecoin": ["USDC"] * 4,
            "tvl_usd": [1_000_000.0] * 4,
            "base_apy": [0.05] * 4,
            "reward_apy": [0.0] * 4,
            "is_auto": [True] * 4,
            "source": ["UnitTest"] * 4,
            "risk_score": [2.0] * 4,
            "timestamp": timestamps,
        }
    )
    csv_path = tmp_path / "freq.csv"
    df.to_csv(csv_path, index=False)

    src = SchemaAwareCSVSource(path=str(csv_path), expected_frequency="D", validation="strict")
    pools = src.fetch()

    assert len(pools) == 4
    assert src.detected_frequency == "D"


def test_schema_auto_refresh(tmp_path: Path) -> None:
    csv_path = tmp_path / "refresh.csv"
    csv_path.write_text(
        """name,chain,stablecoin,tvl_usd,base_apy,reward_apy,is_auto,source,risk_score,timestamp
Stale,Ethereum,USDC,1000000,0.05,0.0,True,Cache,2.0,1700000000
"""
    )

    def refresh_callback(target: Path) -> None:
        updated = pd.DataFrame(
            {
                "name": ["Fresh"],
                "chain": ["Ethereum"],
                "stablecoin": ["USDC"],
                "tvl_usd": [2_000_000.0],
                "base_apy": [0.08],
                "reward_apy": [0.0],
                "is_auto": [True],
                "source": ["API"],
                "risk_score": [2.0],
                "timestamp": [1700000000],
            }
        )
        updated.to_csv(target, index=False)

    src = SchemaAwareCSVSource(
        path=str(csv_path),
        auto_refresh=True,
        refresh_callback=refresh_callback,
    )
    pools = src.fetch()

    assert len(pools) == 1
    assert pools[0].name == "Fresh"
