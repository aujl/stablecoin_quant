"""
StableYieldLab: Modular OOP toolkit for stablecoin pool analytics & visualization.

Design goals:
- Extensible data adapters (DefiLlama, Morpho, Beefy, Yearn, Custom CSV, ...)
- Immutable data model (Pool) + light repository
- Pluggable filters and metrics
- Matplotlib visualizations (single-plot functions)
- No web access here; adapters expose a common interface; wire your own HTTP client.
"""

from __future__ import annotations

from typing import Any

import logging
import pandas as pd

from . import analytics, risk_scoring, visualization
from . import reporting as reporting_module
from .analytics.metrics import Metrics
from .analytics.performance import cumulative_return, nav_series, nav_trajectories
from .core import Pool, PoolRepository, PoolReturn, ReturnRepository
from .sources import (
    BeefySource,
    CSVSource,
    DataSource,
    DefiLlamaSource,
    HistoricalCSVSource,
    MorphoSource,
)


# -----------------
# Data Model
# -----------------

# -----------------
# Data Sources API
# -----------------

# Concrete adapters live in :mod:`stable_yield_lab.sources`.
logger = logging.getLogger(__name__)

# -----------------
# Metrics & Analytics
# -----------------

metrics = analytics.metrics
performance = analytics.performance
portfolio = analytics.portfolio
risk = analytics.risk
risk_metrics = analytics.risk
attribution = analytics.attribution


# -----------------
# Visualization
# -----------------

# Backwards-compatible re-exports for the public package namespace.
Visualizer = visualization.Visualizer
reporting = reporting_module


# -----------------
# Pipeline
# -----------------


class Pipeline:
    """Composable pipeline: fetch -> repository -> filter -> metrics -> visuals"""

    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources

    def run(self) -> PoolRepository:
        repo = PoolRepository()
        for s in self.sources:
            try:
                fetched = s.fetch()
                scored = [risk_scoring.score_pool(p) for p in fetched]
                repo.extend(scored)
            except Exception as e:
                # Log and continue
                logger.warning("Source %s failed: %s", s.__class__.__name__, e)
        return repo

    def run_history(self) -> pd.DataFrame:
        repo = ReturnRepository()
        for s in self.sources:
            try:
                repo.extend(s.fetch())
            except Exception as e:
                logger.warning("Source %s failed: %s", s.__class__.__name__, e)
        return repo.to_timeseries()


__all__ = [
    "analytics",
    "Pool",
    "PoolRepository",
    "PoolReturn",
    "ReturnRepository",
    "DataSource",
    "CSVSource",
    "HistoricalCSVSource",
    "DefiLlamaSource",
    "MorphoSource",
    "BeefySource",
    "Metrics",
    "metrics",
    "visualization",
    "Visualizer",
    "Pipeline",
    "cumulative_return",
    "nav_series",
    "performance",
    "portfolio",
    "risk",
    "risk_metrics",
    "risk_scoring",
    "attribution",
    "reporting",
]
