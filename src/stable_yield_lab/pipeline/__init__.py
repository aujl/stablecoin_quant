from __future__ import annotations

"""Data orchestration pipeline for StableYieldLab adapters."""

from collections.abc import Iterable, Iterator, Sequence
import logging
from typing import Protocol, TypeVar

import pandas as pd

from ..core import Pool, PoolRepository, PoolReturn, ReturnRepository
from ..risk_scoring import score_pool
from ..sources import DataSource

logger = logging.getLogger(__name__)


T = TypeVar("T")


class HistoricalSource(Protocol):
    """Protocol for adapters returning historical :class:`PoolReturn` rows."""

    def fetch(self) -> list[PoolReturn]: ...


PipelineSource = DataSource | HistoricalSource


def _iter_instances(items: Iterable[object], cls: type[T]) -> Iterator[T]:
    for item in items:
        if isinstance(item, cls):
            yield item


class Pipeline:
    """Composable pipeline orchestrating snapshot and historical adapters."""

    def __init__(self, sources: Sequence[PipelineSource]) -> None:
        self._sources: list[PipelineSource] = list(sources)

    def run(self) -> PoolRepository:
        repo = PoolRepository()
        for source in self._sources:
            try:
                items = source.fetch()
            except Exception as exc:
                logger.warning("Source %s failed: %s", source.__class__.__name__, exc)
                continue
            for pool in _iter_instances(items, Pool):
                repo.add(score_pool(pool))
        return repo

    def run_history(self) -> pd.DataFrame:
        repo = ReturnRepository()
        for source in self._sources:
            try:
                items = source.fetch()
            except Exception as exc:
                logger.warning("Source %s failed: %s", source.__class__.__name__, exc)
                continue
            repo.extend(_iter_instances(items, PoolReturn))
        return repo.to_timeseries()


__all__ = ["Pipeline"]
