"""Matplotlib-based chart helpers for StableYieldLab."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from ..analytics.performance import nav_series, nav_trajectories


class Visualizer:
    """Collection of static helpers that turn analytics outputs into charts."""

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
        """Plot yield time series for one or multiple pools."""
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
        r"""Plot portfolio NAV alongside buy-and-hold and cash benchmarks."""

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
        """Plot time-series data as a line chart."""
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


__all__ = ["Visualizer"]
