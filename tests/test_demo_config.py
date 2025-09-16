from pathlib import Path

import pandas as pd
import pytest

from stable_yield_demo import load_config, main


def test_loads_config_file() -> None:
    cfg = load_config("configs/demo.toml")
    assert cfg["csv"]["path"].endswith("sample_pools.csv")
    assert cfg["output"]["show"] is False
    assert cfg["output"]["charts"] == ["bar", "scatter", "chain"]
    assert cfg["output"]["history_charts"] == ["rolling_apy", "drawdowns", "realised_vs_target"]
    history_cfg = cfg["reporting"]["history"]
    assert history_cfg["enabled"] is True
    assert history_cfg["rolling_windows"] == [4, 12]
    assert "realised_apy_lookbacks" in cfg["reporting"]
    assert cfg["reporting"]["realised_apy_lookbacks"]["last 52 weeks"] == "52W"


def test_demo_outputs_realised_apy_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("STABLE_YIELD_CONFIG", "configs/demo.toml")
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(tmp_path))

    main()

    captured = capsys.readouterr()
    assert "Realised APY (last 52 weeks)" in captured.out

    pools_path = tmp_path / "pools.csv"
    assert pools_path.is_file()
    pools_df = pd.read_csv(pools_path)
    assert "Realised APY (last 2 weeks)" in pools_df.columns
    assert "Realised APY (last 52 weeks)" in pools_df.columns
