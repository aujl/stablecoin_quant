"""DefiLlama adapter returning :class:`~stable_yield_lab.core.Pool` instances."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core import Pool

logger = logging.getLogger(__name__)


class DefiLlamaSource:
    """HTTP client for https://yields.llama.fi/pools."""

    URL = "https://yields.llama.fi/pools"

    def __init__(self, stable_only: bool = True, cache_path: str | None = None) -> None:
        self.stable_only = stable_only
        self.cache_path = Path(cache_path) if cache_path else None

    def _load(self) -> dict[str, Any]:
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                return json.load(f)
        with urllib.request.urlopen(self.URL) as resp:  # pragma: no cover - network path
            data = json.load(resp)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w") as f:
                json.dump(data, f)
        return data

    def fetch(self) -> list[Pool]:
        try:
            raw = self._load()
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("DefiLlama request failed: %s", exc)
            return []
        pools: list[Pool] = []
        now = datetime.now(tz=UTC).timestamp()
        for item in raw.get("data", []):
            if self.stable_only and not item.get("stablecoin"):
                continue
            base_val = item.get("apyBase")
            if base_val is None:
                base_val = item.get("apy", 0.0)
            reward_val = item.get("apyReward") or 0.0
            pools.append(
                Pool(
                    name=f"{item.get('project', '')}:{item.get('symbol', '')}",
                    chain=str(item.get("chain", "")),
                    stablecoin=str(item.get("symbol", "")),
                    tvl_usd=float(item.get("tvlUsd", 0.0)),
                    base_apy=float(base_val) / 100.0,
                    reward_apy=float(reward_val) / 100.0,
                    is_auto=False,
                    source="defillama",
                    timestamp=now,
                )
            )
        return pools


__all__ = ["DefiLlamaSource"]
