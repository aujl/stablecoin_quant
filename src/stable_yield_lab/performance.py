from __future__ import annotations

"""Performance utilities for NAV and yield calculations."""

import pandas as pd


def nav_trajectories(returns: pd.DataFrame, *, initial_investment: float) -> pd.DataFrame:
    """Compute NAV trajectories from periodic returns.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (index=date, columns=assets) as decimal fractions.
    initial_investment:
        Starting capital per asset in USD applied equally to each series.

    Returns
    -------
    pd.DataFrame
        DataFrame of NAV values per asset over time in USD.
    """
    if returns.empty:
        return returns.copy()
    growth = (1.0 + returns.fillna(0.0)).cumprod()
    return growth * float(initial_investment)


def yield_trajectories(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative yield trajectories from periodic returns.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (index=date, columns=assets).

    Returns
    -------
    pd.DataFrame
        Cumulative return for each asset as decimal fractions, where 0.05 represents +5%.
    """
    if returns.empty:
        return returns.copy()
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0
