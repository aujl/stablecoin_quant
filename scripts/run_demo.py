#!/usr/bin/env python3
"""Execute the stable_yield_demo with a given config and validate artifacts.

This mirrors the behaviour of the `Generate Demo Artifacts` GitHub action so that
contributors can regenerate the same CSV and PNG outputs locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


REQUIRED_CSVS: tuple[str, ...] = (
    "pools.csv",
    "by_chain.csv",
    "by_source.csv",
    "by_stablecoin.csv",
    "topN.csv",
    "concentration.csv",
    "warnings.csv",
    "portfolio_performance.csv",
    "portfolio_nav.csv",
)

REQUIRED_IMAGES: tuple[str, ...] = (
    "bar_group_chain.png",
    "yield_vs_time.png",
    "nav_vs_time.png",
)

ALTERNATE_SCATTER: tuple[str, ...] = ("scatter_risk_return.png", "scatter_tvl_apy.png")


def _validate_non_empty(files: Iterable[Path]) -> None:
    missing: list[str] = []
    empty: list[str] = []
    for file_path in files:
        if not file_path.exists():
            missing.append(file_path.name)
            continue
        if file_path.stat().st_size == 0:
            empty.append(file_path.name)
    if missing:
        raise FileNotFoundError(f"Missing demo artifacts: {', '.join(missing)}")
    if empty:
        raise RuntimeError(f"Empty demo artifacts: {', '.join(empty)}")


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_demo(config: Path, outdir: Path, clean: bool) -> None:
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    if clean and outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    env_keys = {
        "STABLE_YIELD_CONFIG": os.environ.get("STABLE_YIELD_CONFIG"),
        "STABLE_YIELD_OUTDIR": os.environ.get("STABLE_YIELD_OUTDIR"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
    }

    os.environ["STABLE_YIELD_CONFIG"] = str(config)
    os.environ["STABLE_YIELD_OUTDIR"] = str(outdir)
    os.environ["MPLBACKEND"] = "Agg"

    from stable_yield_demo import main as demo_main

    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        demo_main()
    finally:
        os.chdir(original_cwd)
        _restore_env(env_keys)

    df_check_path = outdir / "pools.csv"
    if df_check_path.exists() and df_check_path.stat().st_size > 0:
        # quick sanity check: ensure expected columns present
        import pandas as pd  # deferred import to avoid hard dependency if validation skipped

        df_sample = pd.read_csv(df_check_path, nrows=1)
        if "base_apy" not in df_sample.columns:
            raise RuntimeError(
                "Generated pools.csv missing 'base_apy'. Verify the demo configuration and inputs."
            )

    required_files = [outdir / name for name in REQUIRED_CSVS]
    required_images = [outdir / name for name in REQUIRED_IMAGES]
    _validate_non_empty(required_files + required_images)

    scatter_paths = [outdir / name for name in ALTERNATE_SCATTER]
    if not any(path.exists() and path.stat().st_size > 0 for path in scatter_paths):
        raise FileNotFoundError(
            f"Expected at least one scatter plot ({', '.join(ALTERNATE_SCATTER)}) to be generated."
        )

    print(f"Demo run complete. Artifacts written to {outdir}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/demo.toml",
        help="Path to the demo configuration TOML file (default: configs/demo.toml).",
    )
    parser.add_argument(
        "--outdir",
        default="demo_artifacts",
        help="Output directory for generated artifacts (default: demo_artifacts).",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Do not remove the output directory before running.",
    )
    parser.set_defaults(clean=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    config_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    run_demo(config_path, outdir, clean=args.clean)


if __name__ == "__main__":
    main()
