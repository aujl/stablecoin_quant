"""Backward-compatible facade for :mod:`stable_yield_lab.analytics.portfolio`."""

from .analytics.portfolio import (
    allocate_mean_variance,
    apy_performance_summary,
    expected_apy,
    tracking_error,
    tvl_weighted_risk,
)

__all__ = [
    "allocate_mean_variance",
    "apy_performance_summary",
    "expected_apy",
    "tracking_error",
    "tvl_weighted_risk",
]
