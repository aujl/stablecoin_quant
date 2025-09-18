"""Beefy Finance adapter for :mod:`stable_yield_lab`."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core import Pool, STABLE_TOKENS

logger = logging.getLogger(__name__)


class BeefySource:
    """HTTP client for Beefy vault data."""

    VAULTS_URL = "https://api.beefy.finance/vaults"
    APY_URL = "https://api.beefy.finance/apy"
    TVL_URL = "https://api.beefy.finance/tvl"

    CHAIN_IDS = {
        "ethereum": "1",
        "bsc": "56",
        "polygon": "137",
        "arbitrum": "42161",
        "optimism": "10",
    }

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _get_json(self, name: str, url: str) -> Any:
        if self.cache_dir:
            path = self.cache_dir / f"{name}.json"
            if path.exists():
                with path.open() as f:
                    return json.load(f)
        with urllib.request.urlopen(url) as resp:  # pragma: no cover - network path
            data = json.load(resp)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with (self.cache_dir / f"{name}.json").open("w") as f:
                json.dump(data, f)
        return data

    def fetch(self) -> list[Pool]:
        try:
            vaults = self._get_json("vaults", self.VAULTS_URL)
            apy = self._get_json("apy", self.APY_URL)
            tvl = self._get_json("tvl", self.TVL_URL)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Beefy request failed: %s", exc)
            return []
        pools: list[Pool] = []
        now = datetime.now(tz=UTC).timestamp()
        for v in vaults:
            if v.get("status") != "active":
                continue
            assets = v.get("assets") or []
            if not assets or not all(
                a.upper() in STABLE_TOKENS or "USD" in a.upper() for a in assets
            ):
                continue
            chain = str(v.get("chain", ""))
            chain_id = self.CHAIN_IDS.get(chain.lower(), chain)
            tvl_usd = float(tvl.get(str(chain_id), {}).get(v["id"], 0.0))
            base = float(apy.get(v["id"], 0.0))
            pools.append(
                Pool(
                    name=str(v.get("name", v["id"])),
                    chain=chain,
                    stablecoin=str(assets[0]),
                    tvl_usd=tvl_usd,
                    base_apy=base,
                    reward_apy=0.0,
                    is_auto=True,
                    source="beefy",
                    timestamp=now,
                )
            )
        return pools


__all__ = ["BeefySource"]
