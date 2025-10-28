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

        if not isinstance(raw, dict):
            logger.warning("Morpho response is not a JSON object: %r", raw)
            return []

        data = raw.get("data") or {}
        if not isinstance(data, dict):
            logger.warning("Morpho 'data' field has unexpected type: %r", type(data))
            return []

        markets = data.get("markets") or {}
        if isinstance(markets, dict):
            items = markets.get("items", [])
        else:
            logger.warning("Morpho 'markets' field has unexpected type: %r", type(markets))
            items = []

        if not isinstance(items, list):
            logger.warning("Morpho 'items' field has unexpected type: %r", type(items))
            items = []

        pools: list[Pool] = []
        now = datetime.now(tz=UTC).timestamp()
        for item in items:
            if not isinstance(item, dict):
                logger.debug("Skipping Morpho item with unexpected type: %r", type(item))
                continue
            loan_asset = item.get("loanAsset") or {}
            collateral_asset = item.get("collateralAsset") or {}
            state = item.get("state") or {}

            if not all(isinstance(obj, dict) for obj in (loan_asset, collateral_asset, state)):
                logger.debug("Morpho item missing metadata: %r", item)
                continue

            sym = str(loan_asset.get("symbol", ""))
            if sym.upper() not in STABLE_TOKENS:
                continue

            try:
                supply_apy = float(state.get("supplyApy", 0.0)) / 100.0
            except (TypeError, ValueError):
                supply_apy = 0.0
            try:
                tvl_usd = float(state.get("supplyAssetsUsd", 0.0))
            except (TypeError, ValueError):
                tvl_usd = 0.0

            pools.append(
                Pool(
                    name=f"{sym}-{collateral_asset.get('symbol', '')}",
                    chain="Ethereum",
                    stablecoin=sym,
                    tvl_usd=tvl_usd,
                    base_apy=supply_apy,
                    reward_apy=0.0,
                    is_auto=True,
                    source="morpho",
                    timestamp=now,
                )
            )
        return pools


__all__ = ["MorphoSource"]
