from pathlib import Path

import matplotlib
import pandas as pd

from stable_yield_lab import Visualizer

matplotlib.use("Agg")


def test_scatter_risk_return_creates_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "volatility": [0.1, 0.2, 0.15],
            "base_apy": [0.05, 0.1, 0.07],
            "tvl_usd": [1_000_000, 2_000_000, 1_500_000],
            "name": ["A", "B", "C"],
        }
    )
    out = tmp_path / "risk_return.png"
    Visualizer.scatter_risk_return(df, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0
