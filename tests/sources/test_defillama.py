from __future__ import annotations

from pathlib import Path

import pytest

from stable_yield_lab.core import PoolRepository
from stable_yield_lab.sources import DefiLlamaSource

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def defillama_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DefiLlamaSource:
    cache = tmp_path / "defillama.json"
    cache.write_bytes((FIXTURES / "defillama_pools.json").read_bytes())

    def _unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access should use cached response")

    monkeypatch.setattr(
        "stable_yield_lab.sources.defillama.urllib.request.urlopen",
        _unexpected_network,
    )
    return DefiLlamaSource(cache_path=str(cache))


def test_defillama_converts_percentages(defillama_source: DefiLlamaSource) -> None:
    pools = defillama_source.fetch()
    assert pools
    assert pools[0].chain == "Ethereum"
    assert pools[0].stablecoin == "SUSDE"
    assert pools[0].base_apy == pytest.approx(0.0931466)
    reward_apys = {p.reward_apy for p in pools}
    assert any(pytest.approx(0.022, rel=1e-6) == val for val in reward_apys)


def test_defillama_auto_only_filter(defillama_source: DefiLlamaSource) -> None:
    repo = PoolRepository(defillama_source.fetch())
    auto_only = repo.filter(auto_only=True)
    assert len(auto_only) == 0
