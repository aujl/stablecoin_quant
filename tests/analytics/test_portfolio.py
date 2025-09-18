from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stable_yield_lab.analytics import portfolio
from stable_yield_lab.analytics.performance import nav_series


rp = pytest.importorskip("riskfolio")


def synthetic_returns() -> pd.DataFrame:
    np.random.seed(0)
    mu = [0.01, 0.012, 0.008]
    cov = [[0.0001, 0.00002, 0.000015], [0.00002, 0.00008, 0.000025], [0.000015, 0.000025, 0.00009]]
    data = np.random.multivariate_normal(mu, cov, size=100)
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_allocate_mean_variance_with_bounds() -> None:
    returns = synthetic_returns()
    bounds = {col: (0.1, 0.8) for col in returns.columns}
    w = portfolio.allocate_mean_variance(returns, bounds=bounds)

    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0)
    for asset, (lo, hi) in bounds.items():
        assert lo - 1e-6 <= w[asset] <= hi + 1e-6

    apy = portfolio.expected_apy(returns, w)
    manual_apy = ((1 + returns.mean()) ** 52 - 1).mul(w).sum()
    assert apy == pytest.approx(float(manual_apy))

    risk = portfolio.tvl_weighted_risk(returns, w, rm="MV")
    manual_risk = rp.Risk_Contribution(w, returns, returns.cov(), rm="MV").sum()
    assert risk == pytest.approx(float(manual_risk))


def test_tracking_error_matches_manual_calculation() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.012, -0.008, 0.01, 0.004, -0.002, 0.006],
            "B": [0.009, 0.011, -0.005, 0.007, 0.003, 0.008],
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})

    periodic_te, annual_te = portfolio.tracking_error(returns, weights, freq=52)

    norm_weights = weights / weights.sum()
    portfolio_returns = returns.fillna(0.0).mul(norm_weights, axis=1).sum(axis=1)
    benchmark = float(portfolio_returns.mean())
    active = portfolio_returns - benchmark
    expected_periodic = float(active.std(ddof=1))
    expected_annual = expected_periodic * np.sqrt(52)

    assert periodic_te == pytest.approx(expected_periodic)
    assert annual_te == pytest.approx(expected_annual)


def test_apy_performance_summary_with_explicit_nav() -> None:
    idx = pd.date_range("2024-03-01", periods=5, freq="W", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.004, 0.012, 0.006, -0.003],
            "B": [0.008, 0.009, -0.002, 0.007, 0.005],
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.55, "B": 0.45})

    nav = portfolio.performance.nav_series(returns, weights, initial=100.0)
    metrics, nav_path = portfolio.apy_performance_summary(
        returns,
        weights,
        freq=52,
        initial_nav=100.0,
        nav=nav,
    )

    pd.testing.assert_series_equal(nav, nav_path)

    norm_weights = weights / weights.sum()
    portfolio_returns = returns.fillna(0.0).mul(norm_weights, axis=1).sum(axis=1)
    total_growth = float((1.0 + portfolio_returns).prod())
    realised_total = total_growth - 1.0
    realised_apy = total_growth ** (52 / len(portfolio_returns)) - 1.0
    expected = portfolio.expected_apy(returns, norm_weights, freq=52)
    expected_periodic = (1.0 + expected) ** (1.0 / 52) - 1.0
    active = portfolio_returns - expected_periodic
    tracking_periodic = float(active.std(ddof=1))
    tracking_annual = tracking_periodic * np.sqrt(52)

    assert metrics["expected_apy"] == pytest.approx(expected)
    assert metrics["realized_total_return"] == pytest.approx(realised_total)
    assert metrics["realized_apy"] == pytest.approx(realised_apy)
    assert metrics["active_apy"] == pytest.approx(realised_apy - expected)
    assert metrics["tracking_error_periodic"] == pytest.approx(tracking_periodic)
    assert metrics["tracking_error_annualized"] == pytest.approx(tracking_annual)
    assert metrics["final_nav"] == pytest.approx(nav.iloc[-1])


def synthetic_nav_inputs() -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.015, -0.005, 0.012],
            "PoolB": [0.008, 0.006, 0.004, 0.007],
        },
        index=dates,
    )
    weights = pd.Series({"PoolA": 0.6, "PoolB": 0.4})
    return returns, weights


def test_apy_performance_summary_expected_vs_realised() -> None:
    returns, weights = synthetic_nav_inputs()
    freq = 52
    initial_nav = 100.0

    metrics, nav = portfolio.apy_performance_summary(
        returns,
        weights,
        freq=freq,
        initial_nav=initial_nav,
    )

    manual_nav = portfolio.performance.nav_series(returns, weights, initial=initial_nav)
    pd.testing.assert_series_equal(nav, manual_nav)

    expected = portfolio.expected_apy(returns, weights, freq=freq)

    total_return = manual_nav.iloc[-1] / initial_nav - 1.0
    realised_apy = (1.0 + total_return) ** (freq / len(manual_nav)) - 1.0

    assert metrics["expected_apy"] == pytest.approx(expected)
    assert metrics["realized_apy"] == pytest.approx(realised_apy)
    assert metrics["realized_total_return"] == pytest.approx(total_return)
    assert metrics["active_apy"] == pytest.approx(realised_apy - expected)

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    expected_periodic = (1.0 + expected) ** (1.0 / freq) - 1.0
    active_returns = portfolio_returns - expected_periodic
    manual_te_periodic = active_returns.std(ddof=1)
    manual_te_annualised = manual_te_periodic * math.sqrt(freq)

    assert metrics["tracking_error_periodic"] == pytest.approx(manual_te_periodic)
    assert metrics["tracking_error_annualized"] == pytest.approx(manual_te_annualised)
    assert metrics["horizon_periods"] == pytest.approx(len(manual_nav))
    assert metrics["horizon_years"] == pytest.approx(len(manual_nav) / freq)
    assert metrics["final_nav"] == pytest.approx(manual_nav.iloc[-1])


def test_tracking_error_accepts_dataframe_and_series() -> None:
    returns, weights = synthetic_nav_inputs()
    freq = 12

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    target = float(portfolio_returns.mean() + 0.001)

    te_series = portfolio.tracking_error(
        portfolio_returns,
        freq=freq,
        target_periodic_return=target,
    )
    te_dataframe = portfolio.tracking_error(
        returns,
        weights,
        freq=freq,
        target_periodic_return=target,
    )

    manual = (portfolio_returns - target).std(ddof=1)
    assert te_series[0] == pytest.approx(manual)
    assert te_series[1] == pytest.approx(manual * math.sqrt(freq))
    assert te_dataframe == te_series


def test_apy_performance_summary_respects_external_nav() -> None:
    returns, weights = synthetic_nav_inputs()
    nav = portfolio.performance.nav_series(returns, weights, initial=1.0)

    metrics_from_computed, nav_auto = portfolio.apy_performance_summary(returns, weights)
    metrics_from_external, nav_external = portfolio.apy_performance_summary(
        returns,
        weights,
        nav=nav,
    )

    pd.testing.assert_series_equal(nav, nav_external)
    pd.testing.assert_series_equal(nav_auto, nav_external)
    pd.testing.assert_series_equal(metrics_from_computed, metrics_from_external)


def test_tracking_error_requires_positive_frequency() -> None:
    returns, weights = synthetic_nav_inputs()

    with pytest.raises(ValueError):
        portfolio.tracking_error(returns, weights, freq=0)

    with pytest.raises(ValueError):
        portfolio.apy_performance_summary(returns, weights, freq=0)


_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio"


def _load_fixture(name: str) -> dict[str, object]:
    with (_FIXTURE_DIR / name).open() as handle:
        return json.load(handle)


def _returns_from_fixture(raw_returns: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(raw_returns)
    index = pd.to_datetime(frame.pop("date"), utc=True)
    return frame.astype(float).set_index(index)


def _schedule_from_fixture(
    raw_schedule: Iterable[dict[str, object]],
) -> list[tuple[pd.Timestamp, pd.Series]]:
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
            raise ValueError(
                f"no return observations available for rebalance at {start.isoformat()}"
            )
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
