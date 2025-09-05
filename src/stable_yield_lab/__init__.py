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

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
import pandas as pd

# -----------------
# Data Model
# -----------------


@dataclass(frozen=True)
class Pool:
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
        d = asdict(self)
        # for readability in CSV
        d["timestamp_iso"] = (
            datetime.fromtimestamp(self.timestamp or 0, tz=UTC).isoformat()
            if self.timestamp
            else ""
        )
        return d


class PoolRepository:
    """Lightweight in-memory collection with pandas export/import."""

    def __init__(self, pools: Iterable[Pool] | None = None) -> None:
        self._pools: list[Pool] = list(pools) if pools else []

    def add(self, pool: Pool) -> None:
        self._pools.append(pool)

    def extend(self, items: Iterable[Pool]) -> None:
        self._pools.extend(items)

    def filter(
        self,
        *,
        min_tvl: float = 0.0,
        min_base_apy: float = 0.0,
        chains: list[str] | None = None,
        auto_only: bool = False,
        stablecoins: list[str] | None = None,
    ) -> PoolRepository:
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

    def __iter__(self) -> Iterable[Pool]:
        return iter(self._pools)


# -----------------
# Data Sources API
# -----------------


class DataSource(Protocol):
    """Adapter protocol: produce a list of Pool objects (base APY preferred)."""

    def fetch(self) -> list[Pool]: ...


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


# Stubs for real adapters you can implement:
class DefiLlamaSource:
    """Stub: HTTP GET to yields.llama.fi/pools and map entries to :class:`Pool`."""

    def __init__(self, stable_only: bool = True) -> None:
        self.stable_only = stable_only

    def fetch(self) -> list[Pool]:
        # Implement: request, parse JSON, filter stablecoin==True, map fields.
        # Return empty in this offline template.
        return []


class MorphoSource:
    """Stub: Graph/SDK call to Morpho Blue markets -> :class:`Pool` list."""

    def fetch(self) -> list[Pool]:
        return []


class BeefySource:
    """Stub: Beefy vaults endpoint -> auto-compounded vault APY as ``base_apy``."""

    def fetch(self) -> list[Pool]:
        return []


# -----------------
# Metrics & Analytics
# -----------------


class Metrics:
    @staticmethod
    def weighted_mean(values: list[float], weights: list[float]) -> float:
        if not values or not weights or len(values) != len(weights):
            return float("nan")
        wsum = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / wsum if wsum else float("nan")

    @staticmethod
    def portfolio_apr(pools: Iterable[Pool], weights: list[float] | None = None) -> float:
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
        g = (
            df.groupby("chain")
            .agg(
                pools=("name", "count"),
                tvl=("tvl_usd", "sum"),
                apr_avg=("base_apy", "mean"),
                apr_wavg=(
                    "base_apy",
                    lambda x: (x * df.loc[x.index, "tvl_usd"]).sum()
                    / df.loc[x.index, "tvl_usd"].sum(),
                ),
            )
            .reset_index()
        )
        return g

    @staticmethod
    def top_n(repo: PoolRepository, n: int = 10, key: str = "base_apy") -> pd.DataFrame:
        df = repo.to_dataframe()
        if df.empty:
            return df
        return df.sort_values(key, ascending=False).head(n)

    @staticmethod
    def net_apy(
        base_apy: float,
        reward_apy: float = 0.0,
        *,
        perf_fee_bps: float = 0.0,
        mgmt_fee_bps: float = 0.0,
    ) -> float:
        """Compute net APY after performance/management fees.

        Fees are specified in basis points (1% = 100 bps).
        """
        gross = float(base_apy) + float(reward_apy)
        fee_frac = (perf_fee_bps + mgmt_fee_bps) / 10_000.0
        return max(gross * (1.0 - fee_frac), -1.0)

    @staticmethod
    def add_net_apy_column(
        df: pd.DataFrame,
        *,
        perf_fee_bps: float = 0.0,
        mgmt_fee_bps: float = 0.0,
        out_col: str = "net_apy",
    ) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out[out_col] = [
            Metrics.net_apy(
                row.get("base_apy", 0.0),
                row.get("reward_apy", 0.0),
                perf_fee_bps=perf_fee_bps,
                mgmt_fee_bps=mgmt_fee_bps,
            )
            for _, row in out.iterrows()
        ]
        return out

    @staticmethod
    def hhi(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> pd.DataFrame:
        """Compute Herfindahl–Hirschman Index of concentration.

        - If `group_col` is None, returns a single-row DataFrame with HHI over the whole df.
        - Otherwise, computes HHI within each group of `group_col`.
        HHI = sum_i (share_i^2), where share is value / total within the scope.
        """
        if df.empty:
            return df
        if group_col is None:
            total = df[value_col].sum()
            if total == 0:
                return pd.DataFrame({"hhi": [float("nan")]})
            shares = (df[value_col] / total) ** 2
            return pd.DataFrame({"hhi": [shares.sum()]})
        else:

            def _hhi(g: pd.Series) -> float:
                tot = g.sum()
                return float(((g / tot) ** 2).sum()) if tot else float("nan")

            res = df.groupby(group_col)[value_col].apply(_hhi).reset_index(name="hhi")
            return res


# -----------------
# Visualization
# -----------------


class Visualizer:
    @staticmethod
    def _plt():
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "matplotlib is required for visualization. Install via Poetry or pip."
            ) from exc
        return plt

    @staticmethod
    def bar_apr(
        df: pd.DataFrame,
        title: str = "Netto-APY pro Pool",
        x_col: str = "name",
        y_col: str = "base_apy",
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        if df.empty:
            return
        plt = Visualizer._plt()
        plt.figure(figsize=(10, 6))
        plt.bar(df[x_col], df[y_col] * 100.0)  # percentage
        plt.title(title)
        plt.ylabel("APY (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    @staticmethod
    def scatter_tvl_apy(
        df: pd.DataFrame,
        title: str = "TVL vs. APY",
        x_col: str = "tvl_usd",
        y_col: str = "base_apy",
        size_col: str | None = "risk_score",
        annotate: bool = True,
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        if df.empty:
            return
        sizes = None
        if size_col in df.columns:
            # scale bubble sizes
            sizes = (df[size_col].fillna(2.0) * 40).tolist()
        plt = Visualizer._plt()
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col] * 100.0, s=sizes)  # % on y-axis
        if annotate:
            for _, row in df.iterrows():
                plt.annotate(
                    str(row.get("name", "")),
                    (row[x_col], row[y_col] * 100.0),
                    textcoords="offset points",
                    xytext=(5, 5),
                )
        plt.xscale("log")
        plt.xlabel("TVL (USD, log-scale)")
        plt.ylabel("APY (%)")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    @staticmethod
    def scatter_risk_return(
        df: pd.DataFrame,
        title: str = "Volatility vs. APY",
        x_col: str = "volatility",
        y_col: str = "base_apy",
        size_col: str = "tvl_usd",
        annotate: bool = True,
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot volatility against APY with bubble sizes scaled by TVL."""
        if df.empty:
            return
        plt = Visualizer._plt()
        sizes = None
        if size_col in df.columns:
            max_val = float(df[size_col].max())
            if max_val > 0:
                sizes = (df[size_col] / max_val * 300).tolist()
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col] * 100.0, s=sizes)
        if annotate and "name" in df.columns:
            for _, row in df.iterrows():
                plt.annotate(
                    str(row.get("name", "")),
                    (row[x_col], row[y_col] * 100.0),
                    textcoords="offset points",
                    xytext=(5, 5),
                )
        plt.xlabel("Volatility")
        plt.ylabel("APY (%)")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    @staticmethod
    def bar_group_chain(
        df_group: pd.DataFrame,
        title: str = "APY (Kettenvergleich)",
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        if df_group.empty:
            return
        plt = Visualizer._plt()
        plt.figure(figsize=(8, 5))
        plt.bar(df_group["chain"], df_group["apr_wavg"] * 100.0)
        plt.title(title)
        plt.ylabel("TVL-gewichteter APY (%)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()


# -----------------
# Pipeline
# -----------------


class Pipeline:
    """Composable pipeline: fetch -> repository -> filter -> metrics -> visuals"""

    def __init__(self, sources: list[DataSource]) -> None:
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


__all__ = [
    "Pool",
    "PoolRepository",
    "DataSource",
    "CSVSource",
    "DefiLlamaSource",
    "MorphoSource",
    "BeefySource",
    "Metrics",
    "Visualizer",
    "Pipeline",
    "risk_metrics",
    "reporting",
]
