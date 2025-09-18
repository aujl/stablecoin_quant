import pytest

from stable_yield_lab.core import Pool, PoolRepository


@pytest.fixture
def sample_pools() -> list[Pool]:
    """Synthetic pool universe covering assorted filter attributes."""

    return [
        Pool(
            name="HighTVL",
            chain="Ethereum",
            stablecoin="USDC",
            tvl_usd=200_000.0,
            base_apy=0.05,
            is_auto=True,
        ),
        Pool(
            name="HighAPY",
            chain="Ethereum",
            stablecoin="DAI",
            tvl_usd=50_000.0,
            base_apy=0.12,
            is_auto=False,
        ),
        Pool(
            name="AutoYield",
            chain="Polygon",
            stablecoin="USDT",
            tvl_usd=150_000.0,
            base_apy=0.09,
            is_auto=True,
        ),
        Pool(
            name="EuroPool",
            chain="Arbitrum",
            stablecoin="EURS",
            tvl_usd=300_000.0,
            base_apy=0.04,
            is_auto=True,
        ),
    ]


@pytest.fixture
def repository(sample_pools: list[Pool]) -> PoolRepository:
    return PoolRepository(sample_pools)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"min_tvl": 100_000.0}, ["HighTVL", "AutoYield", "EuroPool"]),
        ({"min_base_apy": 0.08}, ["HighAPY", "AutoYield"]),
        ({"auto_only": True}, ["HighTVL", "AutoYield", "EuroPool"]),
        ({"stablecoins": ["USDC", "DAI"]}, ["HighTVL", "HighAPY"]),
    ],
)
def test_filter_respects_criteria(
    repository: PoolRepository,
    sample_pools: list[Pool],
    kwargs: dict[str, object],
    expected: list[str],
) -> None:
    filtered = repository.filter(**kwargs)

    assert isinstance(filtered, PoolRepository)
    assert [pool.name for pool in filtered] == expected

    # Original repository order remains unchanged and deterministic ordering is preserved.
    assert [pool.name for pool in repository] == [pool.name for pool in sample_pools]


def test_filter_returns_new_repository_instance(repository: PoolRepository) -> None:
    filtered = repository.filter(min_tvl=0.0)

    assert filtered is not repository
    assert list(filtered) == list(repository)

    extra_pool = Pool(
        name="NewPool",
        chain="Base",
        stablecoin="USDC",
        tvl_usd=10_000.0,
        base_apy=0.03,
    )
    filtered.add(extra_pool)

    assert len(filtered) == len(repository) + 1
    assert len(repository) == 4
