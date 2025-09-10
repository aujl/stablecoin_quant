from pathlib import Path

from stable_yield_lab import (
    BeefySource,
    DefiLlamaSource,
    MorphoSource,
    PoolRepository,
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
