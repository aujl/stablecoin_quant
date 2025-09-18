"""Backward-compatible facade for :mod:`stable_yield_lab.analytics.performance`."""

from .analytics.performance import (
    RebalanceScenario,
    ScenarioRunResult,
    cumulative_return,
    nav_series,
    nav_trajectories,
    run_rebalance_scenarios,
    yield_trajectories,
)

__all__ = [
    "RebalanceScenario",
    "ScenarioRunResult",
    "cumulative_return",
    "nav_series",
    "nav_trajectories",
    "run_rebalance_scenarios",
    "yield_trajectories",
]
