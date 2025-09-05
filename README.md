# stablecoin_quant

Experimental toolkit for analyzing and visualizing yields on stablecoin pools.

## Project Layout

The repository uses a conventional layout to keep code, tests, and
documentation organized:

```
stablecoin_quant/
├── src/        # library source code and demo scripts
├── tests/      # unit tests
├── docs/       # project documentation
└── pyproject.toml
```

Run formatting, linting, and type checks with pre-commit and pytest:

```bash
poetry run pre-commit run -a
poetry run pytest -q
```

## Optional Risk Metrics

Risk analysis utilities rely on the optional [`riskfolio-lib`](https://pypi.org/project/riskfolio-lib/) package.
Install it via Poetry extras or pip:

```bash
poetry install -E risk
# or
pip install "riskfolio-lib"
```

The demo writes risk statistics and efficient frontier weights to CSV files
when the `--outdir` flag is provided. If `riskfolio-lib` is not installed,
these metrics are skipped and a message is printed.
