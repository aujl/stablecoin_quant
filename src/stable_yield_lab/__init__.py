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
from typing import Any, Mapping

import logging
import pandas as pd

from . import attribution, performance, risk_scoring
from .core import Pool, PoolRepository, PoolReturn, ReturnRepository
from .performance import cumulative_return, nav_series, nav_trajectories
from .sources import (
    BeefySource,
    CSVSource,
    DataSource,
    DefiLlamaSource,
    HistoricalCSVSource,
    MorphoSource,
)


# -----------------
# Data Model
# -----------------

# -----------------
# Data Sources API
# -----------------

# Concrete adapters live in :mod:`stable_yield_lab.sources`.
logger = logging.getLogger(__name__)

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

        Fees are specified in basis points (1% = 100 bps) and reduce the gross
        annual growth factor ``(1 + base_apy + reward_apy)``. The resulting
        compounding return is clamped to a minimum of ``-100%`` to avoid values
        below a total loss.
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
    def nav_with_benchmarks(
        returns: pd.DataFrame,
        initial_investment: float,
        cash_returns: float | pd.Series | None = None,
        *,
        labels: Mapping[str, str] | None = None,
        save_path: str | None = None,
        show: bool = True,
    ) -> pd.DataFrame:
        r"""Plot portfolio NAV alongside buy-and-hold and cash benchmarks.

        The rebalanced NAV :math:`\text{NAV}^{\text{rb}}_t` is computed via
        :func:`stable_yield_lab.performance.nav_series`, which applies a
        constant-weight rebalancing policy so that

        .. math::
           \text{NAV}^{\text{rb}}_t = \text{NAV}_{t-1} \Bigl(1 + \sum_i w_i r_{i,t}\Bigr).

        A buy-and-hold benchmark allocates the initial investment equally
        across assets once and compounds each asset independently before
        summing the trajectories. The cash benchmark grows at the supplied
        periodic rate ``cash_returns``.

        Parameters
        ----------
        returns:
            DataFrame of periodic simple returns (decimal form) indexed by
            timestamp.
        initial_investment:
            Starting portfolio value.
        cash_returns:
            Optional periodic simple return for the cash benchmark. A scalar
            applies the same rate each period; a Series is aligned to
            ``returns.index``.
        labels:
            Optional mapping overriding the display labels for
            ``{"rebalance", "buy_and_hold", "cash"}``.
        save_path:
            Optional file path used when persisting the generated figure.
        show:
            Display the plot window when ``True``.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the three NAV paths.
        """

        clean_returns = returns.fillna(0.0)
        index = clean_returns.index

        rebalanced_nav = nav_series(clean_returns, initial=float(initial_investment)).rename(
            "rebalance"
        )

        if clean_returns.shape[1] == 0:
            buy_and_hold_nav = pd.Series(index=index, dtype=float, name="buy_and_hold")
        else:
            num_assets = clean_returns.shape[1]
            per_asset_initial = float(initial_investment) / num_assets
            asset_navs = nav_trajectories(
                clean_returns,
                initial_investment=per_asset_initial,
            )
            buy_and_hold_nav = asset_navs.sum(axis=1).rename("buy_and_hold")

        if isinstance(cash_returns, pd.Series):
            cash_rate = cash_returns.reindex(index, fill_value=0.0)
        elif cash_returns is None:
            cash_rate = pd.Series(0.0, index=index)
        else:
            cash_rate = pd.Series(float(cash_returns), index=index)

        cash_nav = nav_series(cash_rate.to_frame(name="cash"), initial=float(initial_investment)).rename(
            "cash"
        )

        nav_df = pd.concat([rebalanced_nav, buy_and_hold_nav, cash_nav], axis=1)

        default_labels = {
            "rebalance": "Rebalanced NAV",
            "buy_and_hold": "Buy & Hold",
            "cash": "Cash Benchmark",
        }
        if labels:
            default_labels.update(labels)
        rename_map = {col: default_labels.get(col, col) for col in nav_df.columns}
        nav_df = nav_df.rename(columns=rename_map)

        Visualizer.line_chart(
            nav_df,
            title="Portfolio NAV vs Benchmarks",
            ylabel="Net Asset Value",
            save_path=save_path,
            show=show,
        )

        return nav_df

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
    "attribution",
]
