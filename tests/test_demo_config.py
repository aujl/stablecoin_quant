from stable_yield_demo import load_config


def test_loads_config_file() -> None:
    cfg = load_config("configs/demo.toml")
    assert cfg["csv"]["path"].endswith("sample_pools.csv")
    assert cfg["output"]["show"] is False
    assert cfg["output"]["charts"] == []
