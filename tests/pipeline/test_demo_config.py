from stable_yield_demo import load_config


def test_loads_config_file() -> None:
    cfg = load_config("configs/demo.toml")
    assert cfg["csv"]["path"].endswith("sample_pools.csv")
    assert cfg["yields_csv"].endswith("sample_yields.csv")
    assert cfg["output"]["show"] is False
    assert cfg["output"]["charts"] == ["bar", "scatter", "chain"]
    assert cfg["reporting"]["top_n"] == 10
    assert cfg["rebalance"]["benchmark"] == "weekly"
    assert set(cfg["rebalance"]["selected"]) == {"daily", "weekly", "monthly"}
    weekly = cfg["rebalance"]["scenarios"]["weekly"]
    assert weekly["cadence"] == "1W"
    assert weekly["cost_bps"] == 1.5
