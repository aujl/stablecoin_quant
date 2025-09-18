"""Data source adapters used by :mod:`stable_yield_lab`."""

from __future__ import annotations

from typing import Protocol

from ..core import Pool, STABLE_TOKENS as _STABLE_TOKENS
from .base import HistoricalCSVSource
from .beefy import BeefySource
from .csv import CSVSource
from .defillama import DefiLlamaSource
from .morpho import MorphoSource


class DataSource(Protocol):
    """Adapter protocol returning pools compatible with :class:`PoolRepository`."""

    def fetch(self) -> list[Pool]: ...


STABLE_TOKENS = _STABLE_TOKENS

__all__ = [
    "DataSource",
    "STABLE_TOKENS",
    "CSVSource",
    "HistoricalCSVSource",
    "DefiLlamaSource",
    "MorphoSource",
    "BeefySource",
]
