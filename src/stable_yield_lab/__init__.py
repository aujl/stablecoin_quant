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

from . import analytics, risk_scoring, visualization
from . import reporting as reporting_module
from .analytics.metrics import Metrics
from .analytics.performance import cumulative_return, nav_series
from .core import Pool, PoolRepository, PoolReturn, ReturnRepository
from .pipeline import Pipeline
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
