"""Historical yield source built on top of DefiLlama chart endpoints."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from ..core import PoolReturn

logger = logging.getLogger(__name__)


class DefiLlamaHistorySource:
    """Fetch DefiLlama yield history and emit :class:`PoolReturn` records.

    Parameters
    ----------
    pool_ids:
        Mapping of human-readable pool names to DefiLlama pool identifiers. The
        ``DefiLlamaSource`` adapter exposes these IDs via its cached payload.
    targets:
        Iterable of pool names to download. Only entries present in ``pool_ids``
        are requested.
    cache_dir:
        Directory where raw chart responses are cached. The directory is created
        automatically when missing.
    resample:
        Pandas offset alias used to aggregate daily yield data into periodic
        returns. The default weekly frequency matches the sample datasets used
        throughout the project.
    """

    def __init__(
        self,
        *,
        pool_ids: Mapping[str, str],
        targets: Iterable[str],
        cache_dir: str | Path,
        resample: str = "W-FRI",
    ) -> None:
        self._pool_ids = dict(pool_ids)
        self._targets = list(dict.fromkeys(targets))
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._resample = resample

    def fetch(self) -> list[PoolReturn]:
        rows: list[PoolReturn] = []
        for name in self._targets:
            pool_id = self._pool_ids.get(name)
            if not pool_id:
                logger.debug("DefiLlama history skipped unknown pool %s", name)
                continue
            payload = self._load_chart(pool_id)
            points = payload.get("data") if isinstance(payload, dict) else None
            if not points:
                logger.debug("DefiLlama history empty for %s (%s)", name, pool_id)
                continue

            df = pd.DataFrame(points)
            if df.empty or "timestamp" not in df:
                logger.debug("DefiLlama history missing columns for %s", pool_id)
                continue

            columns = ["timestamp", "apyBase", "apy", "apyReward"]
            df = df[[col for col in columns if col in df.columns]].copy()

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp").sort_index()
            df = df.loc[~df.index.isna()]
            if df.empty:
                continue

            apy_base = pd.to_numeric(df.get("apyBase"), errors="coerce")
            apy_total = pd.to_numeric(df.get("apy"), errors="coerce")
            apy_reward = pd.to_numeric(df.get("apyReward"), errors="coerce")

            apy_values = apy_base if apy_base is not None else pd.Series(dtype=float)
            if apy_values.empty or apy_values.isna().all():
                apy_values = apy_total if apy_total is not None else pd.Series(dtype=float)
            else:
                if apy_total is not None:
                    apy_values = apy_values.fillna(apy_total)

            if apy_values.empty or apy_values.isna().all():
                if apy_reward is not None:
                    apy_values = apy_reward

            if apy_values.empty or apy_values.isna().all():
                logger.debug("DefiLlama history lacks APY values for %s", pool_id)
                continue

            if apy_reward is not None:
                apy_values = apy_values.fillna(apy_reward)

            apy_values = apy_values.fillna(0.0)

            daily_return = (1.0 + apy_values / 100.0) ** (1.0 / 365.0) - 1.0
            weekly = (
                daily_return.resample(self._resample)
                .apply(lambda x: (1.0 + x).prod() - 1.0)
                .dropna()
            )

            for ts, value in weekly.items():
                rows.append(
                    PoolReturn(
                        name=name,
                        timestamp=pd.Timestamp(ts),
                        period_return=float(value),
                    )
                )
        return rows

    def _load_chart(self, pool_id: str) -> dict[str, object]:
        cache_path = self._cache_dir / f"{pool_id}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        url = f"https://yields.llama.fi/chart/{pool_id}"
        try:
            with urllib.request.urlopen(url) as resp:
                payload = json.load(resp)
        except urllib.error.URLError as exc:  # pragma: no cover - network issues
            logger.warning("DefiLlama chart request failed for %s: %s", pool_id, exc)
            return {}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return payload


__all__ = ["DefiLlamaHistorySource"]
