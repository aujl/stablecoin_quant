"""Base utilities for StableYieldLab data sources."""

from __future__ import annotations

import pandas as pd

from ..core import PoolReturn


class HistoricalCSVSource:
    """Load periodic returns from a CSV with timestamp, name and period_return."""

    def __init__(self, path: str) -> None:
        self.path = path

    def fetch(self) -> list[PoolReturn]:
        df = pd.read_csv(self.path)
        required = {"timestamp", "name", "period_return"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        rows = [
            PoolReturn(
                name=str(r["name"]),
                timestamp=pd.Timestamp(r["timestamp"]),
                period_return=float(r["period_return"]),
            )
            for _, r in df.iterrows()
        ]
        return rows


__all__ = ["HistoricalCSVSource"]
