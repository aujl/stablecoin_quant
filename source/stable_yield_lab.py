
"""
StableYieldLab: Modular OOP toolkit for stablecoin pool analytics & visualization.

Design goals:
- Extensible data adapters (DefiLlama, Morpho, Beefy, Yearn, Custom CSV, ...)
- Immutable data model (Pool) + light repository
- Pluggable filters and metrics
- Matplotlib visualizations (single-plot functions)
- No web access here; adapters expose a common interface; wire your own HTTP client.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Iterable, Protocol, Any, Tuple
from datetime import datetime, timezone
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt


# -----------------
# Data Model
# -----------------

@dataclass(frozen=True)
class Pool:
    name: str
    chain: str
    stablecoin: str
    tvl_usd: float
    base_apy: float        # decimal fraction, e.g. 0.08 for 8%
    reward_apy: float = 0.0  # optional extra rewards (auto-compounded by meta protocols)
    is_auto: bool = True     # True if fully automated (no manual boosts/claims)
    source: str = "custom"
    risk_score: float = 2.0  # 1=low, 3=high (subjective / model-derived)
    timestamp: float = 0.0   # unix epoch; 0 means unknown

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # for readability in CSV
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp or 0, tz=timezone.utc).isoformat() if self.timestamp else ""
        return d


class PoolRepository:
    """Lightweight in-memory collection with pandas export/import."""
    def __init__(self, pools: Optional[Iterable[Pool]] = None) -> None:
        self._pools: List[Pool] = list(pools) if pools else []

    def add(self, pool: Pool) -> None:
        self._pools.append(pool)

    def extend(self, items: Iterable[Pool]) -> None:
        self._pools.extend(items)

    def filter(self, *, min_tvl: float = 0.0, min_base_apy: float = 0.0,
               chains: Optional[List[str]] = None,
               auto_only: bool = False,
               stablecoins: Optional[List[str]] = None) -> "PoolRepository":
        res = []
        for p in self._pools:
            if p.tvl_usd < min_tvl: 
                continue
            if p.base_apy < min_base_apy: 
                continue
            if auto_only and not p.is_auto: 
                continue
            if chains and p.chain not in chains:
                continue
            if stablecoins and p.stablecoin not in stablecoins:
                continue
            res.append(p)
        return PoolRepository(res)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self._pools])

    def __len__(self) -> int:
        return len(self._pools)

    def __iter__(self):
        return iter(self._pools)


# -----------------
# Data Sources API
# -----------------

class DataSource(Protocol):
    """Adapter protocol: produce a list of Pool objects (base APY preferred)."""
    def fetch(self) -> List[Pool]: ...


class CSVSource:
    """Load pools from a CSV with columns that map to Pool fields (see Pool.to_dict())."""
    def __init__(self, path: str) -> None:
        self.path = path

    def fetch(self) -> List[Pool]:
        df = pd.read_csv(self.path)
        pools: List[Pool] = []
        now = datetime.now(tz=timezone.utc).timestamp()
        for _, r in df.iterrows():
            pools.append(Pool(
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
            ))
        return pools


# Stubs for real adapters you can implement:
class DefiLlamaSource:
    """Stub: implement HTTP GET to yields.llama.fi/pools and map entries to Pool."""
    def __init__(self, stable_only: bool = True) -> None:
        self.stable_only = stable_only

    def fetch(self) -> List[Pool]:
        # Implement: request, parse JSON, filter stablecoin==True, map fields.
        # Return empty in this offline template.
        return []


class MorphoSource:
    """Stub: implement Graph/SDK call to Morpho Blue markets -> Pool list (base APY)."""
    def fetch(self) -> List[Pool]:
        return []


class BeefySource:
    """Stub: implement Beefy vaults endpoint -> auto-compounded vault APY as base_apy."""
    def fetch(self) -> List[Pool]:
        return []


# -----------------
# Metrics & Analytics
# -----------------

class Metrics:
    @staticmethod
    def weighted_mean(values: List[float], weights: List[float]) -> float:
        if not values or not weights or len(values) != len(weights):
            return float("nan")
        wsum = sum(weights)
        return sum(v*w for v, w in zip(values, weights)) / wsum if wsum else float("nan")

    @staticmethod
    def portfolio_apr(pools: Iterable[Pool], weights: Optional[List[float]] = None) -> float:
        arr = list(pools)
        if not arr: 
            return float("nan")
        vals = [p.base_apy for p in arr]
        if weights is None:
            weights = [p.tvl_usd for p in arr]  # TVL-weighted by default
        return Metrics.weighted_mean(vals, weights)

    @staticmethod
    def groupby_chain(repo: PoolRepository) -> pd.DataFrame:
        df = repo.to_dataframe()
        if df.empty: 
            return df
        g = df.groupby("chain").agg(
            pools=("name", "count"),
            tvl=("tvl_usd", "sum"),
            apr_avg=("base_apy", "mean"),
            apr_wavg=("base_apy", lambda x: (x * df.loc[x.index, "tvl_usd"]).sum() / df.loc[x.index, "tvl_usd"].sum())
        ).reset_index()
        return g

    @staticmethod
    def top_n(repo: PoolRepository, n: int = 10, key: str = "base_apy") -> pd.DataFrame:
        df = repo.to_dataframe()
        if df.empty: return df
        return df.sort_values(key, ascending=False).head(n)


# -----------------
# Visualization
# -----------------

class Visualizer:
    @staticmethod
    def bar_apr(df: pd.DataFrame, title: str = "Netto-APY pro Pool", x_col: str = "name", y_col: str = "base_apy") -> None:
        if df.empty: 
            return
        plt.figure(figsize=(10, 6))
        plt.bar(df[x_col], df[y_col] * 100.0)  # percentage
        plt.title(title)
        plt.ylabel("APY (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def scatter_tvl_apy(df: pd.DataFrame, title: str = "TVL vs. APY", x_col: str = "tvl_usd", y_col: str = "base_apy",
                        size_col: Optional[str] = "risk_score", annotate: bool = True) -> None:
        if df.empty:
            return
        sizes = None
        if size_col in df.columns:
            # scale bubble sizes
            sizes = (df[size_col].fillna(2.0) * 40).tolist()
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col] * 100.0, s=sizes)  # % on y-axis
        if annotate:
            for _, row in df.iterrows():
                plt.annotate(str(row.get("name", "")), (row[x_col], row[y_col] * 100.0), textcoords="offset points", xytext=(5,5))
        plt.xscale("log")
        plt.xlabel("TVL (USD, log-scale)")
        plt.ylabel("APY (%)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def bar_group_chain(df_group: pd.DataFrame, title: str = "APY (Kettenvergleich)") -> None:
        if df_group.empty: 
            return
        plt.figure(figsize=(8, 5))
        plt.bar(df_group["chain"], df_group["apr_wavg"] * 100.0)
        plt.title(title)
        plt.ylabel("TVL-gewichteter APY (%)")
        plt.tight_layout()
        plt.show()


# -----------------
# Pipeline
# -----------------

class Pipeline:
    """Composable pipeline: fetch -> repository -> filter -> metrics -> visuals"""
    def __init__(self, sources: List[DataSource]) -> None:
        self.sources = sources

    def run(self) -> PoolRepository:
        repo = PoolRepository()
        for s in self.sources:
            try:
                repo.extend(s.fetch())
            except Exception as e:
                # Log and continue
                print(f"[WARN] Source {s.__class__.__name__} failed: {e}")
        return repo
