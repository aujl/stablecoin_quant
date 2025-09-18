"""Tests for visualization helpers capturing Matplotlib interactions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from stable_yield_lab.visualization import Visualizer


class MatplotlibSpy:
    """Spy object replicating the minimal Matplotlib API used by Visualizer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, args: Iterable[Any] = (), **kwargs: Any) -> None:
        self.calls.append((name, tuple(args), dict(kwargs)))

    # plotting primitives -------------------------------------------------
    def figure(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple proxy
        self._record("figure", args, **kwargs)

    def bar(self, x: Iterable[Any], y: Iterable[Any], *args: Any, **kwargs: Any) -> None:
        self._record("bar", (list(x), list(y), *args), **kwargs)

    def scatter(
        self,
        x: Iterable[Any],
        y: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._record("scatter", (list(x), list(y), *args), **kwargs)

    def plot(self, x: Iterable[Any], y: Iterable[Any], *args: Any, **kwargs: Any) -> None:
        self._record("plot", (list(x), list(y), *args), **kwargs)

    # labelling helpers ---------------------------------------------------
    def title(self, *args: Any, **kwargs: Any) -> None:
        self._record("title", args, **kwargs)

    def ylabel(self, *args: Any, **kwargs: Any) -> None:
        self._record("ylabel", args, **kwargs)

    def xlabel(self, *args: Any, **kwargs: Any) -> None:
        self._record("xlabel", args, **kwargs)

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self._record("legend", args, **kwargs)

    def xticks(self, *args: Any, **kwargs: Any) -> None:
        self._record("xticks", args, **kwargs)

    def xscale(self, *args: Any, **kwargs: Any) -> None:
        self._record("xscale", args, **kwargs)

    def annotate(self, *args: Any, **kwargs: Any) -> None:
        self._record("annotate", args, **kwargs)

    def tight_layout(self, *args: Any, **kwargs: Any) -> None:
        self._record("tight_layout", args, **kwargs)

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        self._record("savefig", args, **kwargs)

    def show(self, *args: Any, **kwargs: Any) -> None:
        self._record("show", args, **kwargs)

    # utilities -----------------------------------------------------------
    def get_call(self, name: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        for call in self.calls:
            if call[0] == name:
                return call
        msg = f"no call named {name!r} recorded"
        raise AssertionError(msg)


@pytest.fixture()
def spy(monkeypatch: pytest.MonkeyPatch) -> MatplotlibSpy:
    canvas = MatplotlibSpy()
    monkeypatch.setattr(Visualizer, "_plt", staticmethod(lambda: canvas))
    return canvas


def test_bar_apr_uses_named_columns(spy: MatplotlibSpy) -> None:
    df = pd.DataFrame(
        {
            "name": ["Pool A", "Pool B"],
            "base_apy": [0.05, 0.08],
        }
    )

    Visualizer.bar_apr(df, title="Headline", show=False)

    bar_call = spy.get_call("bar")
    assert bar_call[1][0] == ["Pool A", "Pool B"]
    assert bar_call[1][1] == [5.0, 8.0]

    title_call = spy.get_call("title")
    assert title_call[1][0] == "Headline"


def test_scatter_tvl_apy_scales_axes_and_annotations(spy: MatplotlibSpy) -> None:
    df = pd.DataFrame(
        {
            "tvl_usd": [1_000_000, 2_500_000],
            "base_apy": [0.04, 0.06],
            "risk_score": [1.2, 0.8],
            "name": ["Vault 1", "Vault 2"],
        }
    )

    Visualizer.scatter_tvl_apy(df, title="TVL vs APY", show=False)

    scatter_call = spy.get_call("scatter")
    assert scatter_call[1][0] == [1_000_000, 2_500_000]
    assert scatter_call[1][1] == [4.0, 6.0]
    assert scatter_call[2]["s"] == pytest.approx([(1.2) * 40, (0.8) * 40])

    annotate_calls = [call for call in spy.calls if call[0] == "annotate"]
    assert len(annotate_calls) == len(df)
    assert annotate_calls[0][1][0] == "Vault 1"


def test_line_chart_plots_each_series(spy: MatplotlibSpy) -> None:
    data = pd.DataFrame(
        {
            "PoolA": [1.0, 1.1],
            "PoolB": [1.0, 0.95],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    Visualizer.line_chart(data, title="NAV", ylabel="Value", show=False, save_path=None)

    plot_calls = [call for call in spy.calls if call[0] == "plot"]
    assert len(plot_calls) == data.shape[1]
    expected_index = list(data.index)
    for column, call in zip(data.columns, plot_calls, strict=True):
        assert call[1][0] == expected_index
        assert call[1][1] == data[column].tolist()
        assert call[2]["label"] == column

    ylabel_call = spy.get_call("ylabel")
    assert ylabel_call[1][0] == "Value"

    assert any(call[0] == "legend" for call in spy.calls)
