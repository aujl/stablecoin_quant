"""Regression fixtures for portfolio rebalances with deterministic outcomes.

The JSON fixtures under ``tests/fixtures/portfolio`` provide hand-crafted
two-asset return paths and rebalance instructions so contributors can validate
NAV and APY calculations:

* ``two_asset_weekly_rebalances.json`` contains five weekly observations. The
  portfolio begins at a 70%/30% split, rebalances to a 0%/100% allocation on the
  fourth timestamp, and includes missing return values that should be interpreted
  as flat performance. Expected NAV levels and APY results for each rebalance are
  embedded in the fixture for regression checks.
* ``two_asset_missing_rebalance.json`` describes a scenario where the schedule
  attempts to shift entirely into an asset without price history. When the
  rebalance date arrives the weights collapse to zero after alignment with the
  available return columns, so the implementation should raise ``ValueError``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.performance import nav_series
from stable_yield_lab import portfolio

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "portfolio"


def _load_fixture(name: str) -> dict[str, object]:
    with (_FIXTURE_DIR / name).open() as handle:
        return json.load(handle)


def _returns_from_fixture(raw_returns: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(raw_returns)
    index = pd.to_datetime(frame.pop("date"), utc=True)
    return frame.astype(float).set_index(index)


def _schedule_from_fixture(raw_schedule: Iterable[dict[str, object]]) -> list[tuple[pd.Timestamp, pd.Series]]:
    schedule: list[tuple[pd.Timestamp, pd.Series]] = []
    for entry in raw_schedule:
        ts = pd.Timestamp(entry["date"], tz="UTC")
        weights = pd.Series(entry["weights"], dtype=float)
        schedule.append((ts, weights))
    return sorted(schedule, key=lambda item: item[0])


def _simulate_nav_with_schedule(
    returns: pd.DataFrame, schedule: list[tuple[pd.Timestamp, pd.Series]], initial: float
) -> pd.Series:
    if not schedule:
        raise ValueError("rebalance schedule is empty")

    returns = returns.sort_index()
    nav_paths: list[pd.Series] = []
    current_nav = float(initial)

    for idx, (start, weights) in enumerate(schedule):
        if idx + 1 < len(schedule):
            next_start = schedule[idx + 1][0]
            mask = (returns.index >= start) & (returns.index < next_start)
        else:
            mask = returns.index >= start
        segment = returns.loc[mask]
        if segment.empty:
            raise ValueError(f"no return observations available for rebalance at {start.isoformat()}")
        nav_segment = nav_series(segment, weights, initial=current_nav)
        nav_paths.append(nav_segment)
        current_nav = float(nav_segment.iloc[-1])

    combined = pd.concat(nav_paths)
    return combined[~combined.index.duplicated(keep="last")]


def test_weekly_rebalances_nav_and_apy_regressions() -> None:
    fixture = _load_fixture("two_asset_weekly_rebalances.json")
    returns = _returns_from_fixture(fixture["returns"])
    schedule = _schedule_from_fixture(fixture["schedule"])
    nav = _simulate_nav_with_schedule(returns, schedule, float(fixture["initial_nav"]))

    expected_nav = pd.Series(
        data=[item["nav"] for item in fixture["expected_nav"]],
        index=pd.to_datetime([item["date"] for item in fixture["expected_nav"]], utc=True),
    )
    pd.testing.assert_series_equal(nav, expected_nav, rtol=1e-12, atol=0.0, check_names=False)

    expected_apy_map = fixture["expected_apy"]
    for idx, (start, weights) in enumerate(schedule):
        if idx + 1 < len(schedule):
            next_start = schedule[idx + 1][0]
            mask = (returns.index >= start) & (returns.index < next_start)
        else:
            mask = returns.index >= start
        segment = returns.loc[mask]
        apy = portfolio.expected_apy(segment, weights)
        key = start.strftime("%Y-%m-%d")
        assert apy == pytest.approx(float(expected_apy_map[key]))


def test_missing_asset_rebalance_raises_value_error() -> None:
    fixture = _load_fixture("two_asset_missing_rebalance.json")
    returns = _returns_from_fixture(fixture["returns"])
    schedule = _schedule_from_fixture(fixture["schedule"])

    with pytest.raises(ValueError, match="weights sum to zero"):
        _simulate_nav_with_schedule(returns, schedule, float(fixture["initial_nav"]))
