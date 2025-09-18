from __future__ import annotations

from pathlib import Path

import pytest

from stable_yield_lab.core import PoolRepository
from stable_yield_lab.sources import BeefySource

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "beefy"


@pytest.fixture()
def beefy_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BeefySource:
    cache_dir = tmp_path / "beefy"
    cache_dir.mkdir()
    for name in ("vaults.json", "apy.json", "tvl.json"):
        (cache_dir / name).write_bytes((FIXTURES / name).read_bytes())

    def _unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access should use cached response")

    monkeypatch.setattr(
        "stable_yield_lab.sources.beefy.urllib.request.urlopen",
        _unexpected_network,
    )
    return BeefySource(cache_dir=str(cache_dir))


def test_beefy_uses_cache_and_preserves_casing(beefy_source: BeefySource) -> None:
    pools = beefy_source.fetch()
    assert pools
    first = next(p for p in pools if p.name == "USDf/USDC")
    assert first.chain == "ethereum"
    assert first.stablecoin == "USDf"
    assert first.base_apy == pytest.approx(0.1302727741797236)


def test_beefy_auto_only_filter(beefy_source: BeefySource) -> None:
    repo = PoolRepository(beefy_source.fetch())
    auto_only = repo.filter(auto_only=True)
    assert len(auto_only) == len(repo)
    assert all(pool.is_auto for pool in auto_only)
