"""CSV-backed data source implementations."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ..core import Pool


class CSVSource:
    """Load pools from a CSV mapping columns to :class:`Pool` fields."""

    def __init__(self, path: str) -> None:
        self.path = path

    def fetch(self) -> list[Pool]:
        df = pd.read_csv(self.path)
        pools: list[Pool] = []
        now = datetime.now(tz=UTC).timestamp()
        for _, r in df.iterrows():
            pools.append(
                Pool(
                    name=str(r.get("name", "")),
                    chain=str(r.get("chain", "")),
                    stablecoin=str(r.get("stablecoin", "")),
                    tvl_usd=float(r.get("tvl_usd", 0.0)),
                    base_apy=float(r.get("base_apy", 0.0)),
                    reward_apy=float(r.get("reward_apy", 0.0)),
                    is_auto=bool(r.get("is_auto", True)),
                    source=str(r.get("source", "csv")),
                    risk_score=float(r.get("risk_score", 2.0)),
                    timestamp=float(r.get("timestamp", now)),
                )
            )
        return pools


__all__ = ["CSVSource"]
