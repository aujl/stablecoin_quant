"""Backward-compatible facade for :mod:`stable_yield_lab.analytics.risk`."""

from .analytics.risk import (
    _require_riskfolio,
    efficient_frontier,
    risk_contributions,
    summary_statistics,
)

__all__ = [
    "_require_riskfolio",
    "efficient_frontier",
    "risk_contributions",
    "summary_statistics",
]
