"""Immutable data models used throughout StableYieldLab."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Pool:
    """Snapshot description of a yield-bearing pool."""

    name: str
    chain: str
    stablecoin: str
    tvl_usd: float
    base_apy: float  # decimal fraction, e.g. 0.08 for 8%
    reward_apy: float = 0.0  # optional extra rewards (auto-compounded by meta protocols)
    is_auto: bool = True  # True if fully automated (no manual boosts/claims)
    source: str = "custom"
    risk_score: float = 2.0  # 1=low, 3=high (subjective / model-derived)
    timestamp: float = 0.0  # unix epoch; 0 means unknown

    def to_dict(self) -> dict[str, Any]:
        """Serialise the pool to a dictionary suitable for DataFrame creation."""

        data = asdict(self)
        # for readability in CSV outputs
        data["timestamp_iso"] = (
            datetime.fromtimestamp(self.timestamp or 0, tz=UTC).isoformat()
            if self.timestamp
            else ""
        )
        return data


@dataclass(frozen=True)
class PoolReturn:
    """Periodic return observation for a given pool."""

    name: str
    timestamp: pd.Timestamp
    period_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "period_return": self.period_return,
        }


__all__ = ["Pool", "PoolReturn"]

