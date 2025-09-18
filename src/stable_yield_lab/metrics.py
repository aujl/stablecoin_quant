"""Backward-compatible facade for :mod:`stable_yield_lab.analytics.metrics`."""

from .analytics.metrics import Metrics, add_net_apy_column, hhi, net_apy, weighted_mean

__all__ = [
    "Metrics",
    "add_net_apy_column",
    "hhi",
    "net_apy",
    "weighted_mean",
]
