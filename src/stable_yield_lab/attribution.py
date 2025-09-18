"""Backward-compatible facade for :mod:`stable_yield_lab.analytics.attribution`."""

from .analytics.attribution import AttributionResult, compute_attribution

__all__ = ["AttributionResult", "compute_attribution"]
