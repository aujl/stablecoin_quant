from stable_yield_demo import load_config


def test_loads_config_file() -> None:
    cfg = load_config("configs/demo.toml")
    assert cfg["csv"]["path"].endswith("sample_pools.csv")
    assert cfg["output"]["show"] is False
    assert cfg["output"]["charts"] == ["bar", "scatter", "chain"]
    assert cfg["output"]["history_charts"] == ["rolling_apy", "drawdowns", "realised_vs_target"]
    history_cfg = cfg["reporting"]["history"]
    assert history_cfg["enabled"] is True
    assert history_cfg["rolling_windows"] == [4, 12]
