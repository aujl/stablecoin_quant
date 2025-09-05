from stable_yield_demo import get_args


def test_loads_config_file() -> None:
    args = get_args(["configs/demo.toml"])
    assert args.csv.endswith("sample_pools.csv")
    assert args.no_show is True
    assert args.charts == []
