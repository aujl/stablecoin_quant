"""Morpho Blue adapter for :mod:`stable_yield_lab`."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core import Pool, STABLE_TOKENS

logger = logging.getLogger(__name__)


class MorphoSource:
    """GraphQL client for Morpho Blue markets."""

    URL = "https://blue-api.morpho.org/graphql"

    def __init__(self, cache_path: str | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None

    def _post_json(self) -> dict[str, Any]:
        payload = {
            "query": (
                "{ markets { items { uniqueKey loanAsset { symbol } "
                "collateralAsset { symbol } state { supplyApy supplyAssetsUsd } } } }"
            )
        }
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:  # pragma: no cover - network path
            return json.load(resp)

    def _load(self) -> dict[str, Any]:
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                return json.load(f)
        data = self._post_json()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("w") as f:
                json.dump(data, f)
        return data

    def fetch(self) -> list[Pool]:
        try:
            raw = self._load()
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Morpho request failed: %s", exc)
            return []
        pools: list[Pool] = []
        now = datetime.now(tz=UTC).timestamp()
        items = raw.get("data", {}).get("markets", {}).get("items", [])
        for item in items:
            sym = str(item.get("loanAsset", {}).get("symbol", ""))
            if sym.upper() not in STABLE_TOKENS:
                continue
            pools.append(
                Pool(
                    name=f"{sym}-{item.get('collateralAsset', {}).get('symbol', '')}",
                    chain="Ethereum",
                    stablecoin=sym,
                    tvl_usd=float(item.get("state", {}).get("supplyAssetsUsd", 0.0)),
                    base_apy=float(item.get("state", {}).get("supplyApy", 0.0)) / 100.0,
                    reward_apy=0.0,
                    is_auto=True,
                    source="morpho",
                    timestamp=now,
                )
            )
        return pools


__all__ = ["MorphoSource"]
