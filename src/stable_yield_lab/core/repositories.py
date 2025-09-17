"""In-memory repositories for StableYieldLab data models."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

from .models import Pool, PoolReturn


class PoolRepository:
    """Lightweight in-memory collection with pandas export/import."""

    def __init__(self, pools: Iterable[Pool] | None = None) -> None:
        self._pools: list[Pool] = list(pools) if pools else []

    def add(self, pool: Pool) -> None:
        self._pools.append(pool)

    def extend(self, items: Iterable[Pool]) -> None:
        self._pools.extend(items)

    def filter(
        self,
        *,
        min_tvl: float = 0.0,
        min_base_apy: float = 0.0,
        chains: list[str] | None = None,
        auto_only: bool = False,
        stablecoins: list[str] | None = None,
    ) -> "PoolRepository":
        res: list[Pool] = []
        for pool in self._pools:
            if pool.tvl_usd < min_tvl:
                continue
            if pool.base_apy < min_base_apy:
                continue
            if auto_only and not pool.is_auto:
                continue
            if chains and pool.chain not in chains:
                continue
            if stablecoins and pool.stablecoin not in stablecoins:
                continue
            res.append(pool)
        return PoolRepository(res)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([pool.to_dict() for pool in self._pools])

    def __len__(self) -> int:
        return len(self._pools)

    def __iter__(self) -> Iterator[Pool]:
        return iter(self._pools)


class ReturnRepository:
    """Collection of :class:`PoolReturn` rows with pivot helper."""

    def __init__(self, rows: Iterable[PoolReturn] | None = None) -> None:
        self._rows: list[PoolReturn] = list(rows) if rows else []

    def extend(self, rows: Iterable[PoolReturn]) -> None:
        self._rows.extend(rows)

    def to_timeseries(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame([row.to_dict() for row in self._rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return (
            df.pivot(index="timestamp", columns="name", values="period_return").sort_index()
        )


__all__ = ["PoolRepository", "ReturnRepository"]

