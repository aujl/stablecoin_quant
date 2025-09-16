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

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
import json
import logging
import warnings
import urllib.request
import pandas as pd

from . import performance, risk_scoring
from .performance import APYEstimate, cumulative_return, estimate_pool_apy, horizon_apys, nav_series


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
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp or 0, tz=UTC).isoformat() if self.timestamp else ""
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

    def __iter__(self) -> Iterator[Pool]:
        return iter(self._pools)


# Additional data model for time-series returns


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


class ReturnRepository:
    """Collection of :class:`PoolReturn` rows with pivot helper."""

    def __init__(self, rows: Iterable[PoolReturn] | None = None) -> None:
        self._rows: list[PoolReturn] = list(rows) if rows else []

    def extend(self, rows: Iterable[PoolReturn]) -> None:
        self._rows.extend(rows)

    def to_timeseries(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame([r.to_dict() for r in self._rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.pivot(index="timestamp", columns="name", values="period_return").sort_index()


class DataQualityWarning(UserWarning):
    """Warning raised when historical data required gap-filling."""


class DataQualityError(RuntimeError):
    """Raised when historical data does not meet minimum quality thresholds."""


@dataclass(frozen=True)
class MissingDataDiagnostics:
    """Summary statistics for the gap-filling applied to a pool's history."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    expected_periods: int
    observed_periods: int
    missing_periods: int
    filled_periods: int
    remaining_missing: int
    fill_strategy: str
    frequency: str | None


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


FillStrategy = Literal["none", "ffill", "bfill", "ffill_bfill", "zero", "ffill_zero"]


class HistoricalCSVSource:
    """Load periodic returns and normalise them to a desired sampling frequency.

    The source consolidates multiple protocol histories into a single, regularised
    panel by applying the following steps:

    1. read the CSV input (``timestamp``, ``name``, ``period_return`` columns),
    2. pivot to a wide DataFrame keyed by timestamp and protocol name,
    3. resample to ``target_frequency`` to ensure consistent spacing,
    4. fill gaps according to ``fill_strategy`` (forward/back fill or zeros), and
    5. record diagnostics about missing periods and applied imputations.

    ``period_return`` is assumed to be the return realised over the resampled
    interval (e.g. weekly APY converted to per-period return). The class does not
    attempt to annualise or otherwise transform the supplied values; it only
    normalises the time axis.
    """

    def __init__(
        self,
        path: str,
        *,
        target_frequency: str | None = "W-MON",
        fill_strategy: FillStrategy = "ffill_bfill",
        min_observations: int = 1,
    ) -> None:
        if min_observations < 0:
            raise ValueError("min_observations must be non-negative")
        self.path = path
        self.target_frequency = target_frequency
        self.fill_strategy = fill_strategy
        self.min_observations = min_observations
        self.last_diagnostics: dict[str, MissingDataDiagnostics] = {}

    def _apply_fill(self, series: pd.Series) -> pd.Series:
        """Return a copy of ``series`` with missing values imputed."""

        strategy = self.fill_strategy
        if strategy == "none":
            return series.copy()
        if strategy == "ffill":
            return series.ffill()
        if strategy == "bfill":
            return series.bfill()
        if strategy == "ffill_bfill":
            return series.ffill().bfill()
        if strategy == "zero":
            return series.fillna(0.0)
        if strategy == "ffill_zero":
            return series.ffill().fillna(0.0)
        raise ValueError(f"Unsupported fill_strategy: {strategy}")

    def fetch(self) -> list[PoolReturn]:
        df = pd.read_csv(self.path)
        required = {"timestamp", "name", "period_return"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("CSV contains invalid timestamps")
        df["name"] = df["name"].astype(str)
        df["period_return"] = pd.to_numeric(df["period_return"], errors="coerce")
        df = df.sort_values(["timestamp", "name"])

        pivot = df.pivot_table(
            index="timestamp",
            columns="name",
            values="period_return",
            aggfunc="last",
        ).sort_index()
        if pivot.empty:
            self.last_diagnostics = {}
            return []

        if self.target_frequency:
            pivot = pivot.resample(self.target_frequency).mean()

        diagnostics: dict[str, MissingDataDiagnostics] = {}
        filled_df = pd.DataFrame(index=pivot.index)
        filled_df.index.name = "timestamp"
        warning_payload: list[tuple[str, int, int, int]] = []
        freq = self.target_frequency

        for name in pivot.columns:
            series = pivot[name].astype(float)
            expected = int(len(series))
            if expected == 0:
                diagnostics[name] = MissingDataDiagnostics(
                    name=name,
                    start=pd.NaT,
                    end=pd.NaT,
                    expected_periods=0,
                    observed_periods=0,
                    missing_periods=0,
                    filled_periods=0,
                    remaining_missing=0,
                    fill_strategy=self.fill_strategy,
                    frequency=freq,
                )
                filled_df[name] = series
                continue

            observed = int(series.notna().sum())
            if observed < self.min_observations:
                raise DataQualityError(
                    f"Pool {name} has {observed} observation(s); minimum required is {self.min_observations}"
                )

            filled_series = self._apply_fill(series)
            missing_periods = expected - observed
            remaining_missing = int(filled_series.isna().sum())
            filled_periods = missing_periods - remaining_missing

            diagnostics[name] = MissingDataDiagnostics(
                name=name,
                start=series.index.min(),
                end=series.index.max(),
                expected_periods=expected,
                observed_periods=observed,
                missing_periods=missing_periods,
                filled_periods=filled_periods,
                remaining_missing=remaining_missing,
                fill_strategy=self.fill_strategy,
                frequency=freq,
            )

            filled_df[name] = filled_series
            if missing_periods > 0:
                warning_payload.append((name, missing_periods, filled_periods, remaining_missing))

        self.last_diagnostics = diagnostics

        for name, missing_periods, filled_periods, remaining_missing in warning_payload:
            warnings.warn(
                (
                    f"Pool {name} missing {missing_periods} period(s); "
                    f"filled {filled_periods} using {self.fill_strategy}; "
                    f"remaining missing {remaining_missing}"
                ),
                DataQualityWarning,
                stacklevel=2,
            )

        melted = filled_df.reset_index().melt(id_vars="timestamp", var_name="name", value_name="period_return")
        rows: list[PoolReturn] = []
        for _, row in melted.iterrows():
            value = row["period_return"]
            period_return = float(value) if pd.notna(value) else float("nan")
            rows.append(
                PoolReturn(
                    name=str(row["name"]),
                    timestamp=pd.Timestamp(row["timestamp"]),
                    period_return=period_return,
                )
            )
        return rows


logger = logging.getLogger(__name__)

STABLE_TOKENS = {
    "USDC",
    "USDT",
    "DAI",
    "FRAX",
    "LUSD",
    "GUSD",
    "TUSD",
    "USDP",
    "USDD",
    "USDR",
    "USDf",
    "USDF",
    "MAI",
    "SUSD",
    "EURS",
    "EUROE",
    "CRVUSD",
    "GHO",
    "USDC.E",
    "USDT.E",
}


class DefiLlamaSource:
    """HTTP client for yields.llama.fi/pools."""

    URL = "https://yields.llama.fi/pools"

    def __init__(self, stable_only: bool = True, cache_path: str | None = None) -> None:
        self.stable_only = stable_only
        self.cache_path = Path(cache_path) if cache_path else None

    def _load(self) -> dict[str, Any]:
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                return json.load(f)
        with urllib.request.urlopen(self.URL) as resp:
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
        with urllib.request.urlopen(req) as resp:
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
        with urllib.request.urlopen(url) as resp:
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
            if not assets or not all(a.upper() in STABLE_TOKENS or "USD" in a.upper() for a in assets):
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
                    lambda x: (x * df.loc[x.index, "tvl_usd"]).sum() / df.loc[x.index, "tvl_usd"].sum(),
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

        Fees are specified in basis points (1% = 100 bps) and reduce the annual
        growth factor, ensuring net APY never exceeds gross APY when fees are
        positive.
        """
        gross = float(base_apy) + float(reward_apy)
        fee_frac = (perf_fee_bps + mgmt_fee_bps) / 10_000.0
        net_growth = (1.0 + gross) * (1.0 - fee_frac)
        return max(net_growth - 1.0, -1.0)

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
            raise RuntimeError("matplotlib is required for visualization. Install via Poetry or pip.") from exc
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
    def line_yield(
        ts: pd.DataFrame,
        title: str = "Yield Over Time",
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot yield time series for one or multiple pools.

        Parameters
        ----------
        ts:
            DataFrame with datetime index and columns representing pools. Values
            should be in decimal form (e.g. 0.05 for 5%).
        title:
            Chart title.
        save_path:
            Optional path to save the figure.
        show:
            Whether to display the figure.
        """
        if ts.empty:
            return
        plt = Visualizer._plt()
        plt.figure(figsize=(10, 6))
        for col in ts.columns:
            plt.plot(ts.index, ts[col] * 100.0, label=str(col))
        plt.xlabel("Date")
        plt.ylabel("APY (%)")
        plt.title(title)
        if ts.shape[1] > 1:
            plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()

    @staticmethod
    def line_nav(
        nav: pd.Series,
        title: str = "Net Asset Value",
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot net asset value time series."""
        if nav.empty:
            return
        plt = Visualizer._plt()
        plt.figure(figsize=(10, 6))
        plt.plot(nav.index, nav.values)
        plt.xlabel("Date")
        plt.ylabel("NAV")
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

    @staticmethod
    def line_chart(
        data: pd.DataFrame | pd.Series,
        *,
        title: str,
        ylabel: str,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Plot time-series data as a line chart.

        Parameters
        ----------
        data:
            Series or DataFrame indexed by timestamp.
        title:
            Plot title.
        ylabel:
            Label for the y-axis.
        save_path:
            Optional path to save the figure. If ``None``, the plot is not saved.
        show:
            Display the plot window when ``True``.
        """
        df = data.to_frame() if isinstance(data, pd.Series) else data
        if df.empty:
            return
        plt = Visualizer._plt()
        plt.figure(figsize=(10, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], label=col)
        if len(df.columns) > 1:
            plt.legend()
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.title(title)
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

    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources

    def run(self) -> PoolRepository:
        repo = PoolRepository()
        for s in self.sources:
            try:
                fetched = s.fetch()
                scored = [risk_scoring.score_pool(p) for p in fetched]
                repo.extend(scored)
            except Exception as e:
                # Log and continue
                logger.warning("Source %s failed: %s", s.__class__.__name__, e)
        return repo

    def run_history(self) -> pd.DataFrame:
        repo = ReturnRepository()
        for s in self.sources:
            try:
                repo.extend(s.fetch())
            except Exception as e:
                logger.warning("Source %s failed: %s", s.__class__.__name__, e)
        return repo.to_timeseries()


__all__ = [
    "Pool",
    "PoolRepository",
    "PoolReturn",
    "ReturnRepository",
    "MissingDataDiagnostics",
    "DataQualityWarning",
    "DataQualityError",
    "DataSource",
    "CSVSource",
    "HistoricalCSVSource",
    "DefiLlamaSource",
    "MorphoSource",
    "BeefySource",
    "Metrics",
    "Visualizer",
    "Pipeline",
    "cumulative_return",
    "nav_series",
    "risk_metrics",
    "reporting",
    "portfolio",
    "risk_scoring",
    "performance",
]
