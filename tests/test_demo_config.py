from stable_yield_demo import load_config


def test_loads_config_file() -> None:
    cfg = load_config("configs/demo.toml")
    assert cfg["csv"]["path"].endswith("sample_pools.csv")
    assert cfg["output"]["show"] is False
    assert cfg["output"]["charts"] == ["bar", "scatter", "chain"]
    reporting = cfg["reporting"]
    assert reporting["realised_apy_lookback_days"] == 90
    assert reporting["realised_apy_min_observations"] == 4
