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


## Risk Scoring

Each pool is assigned a ``risk_score`` in the range ``1`` (lower risk) to ``3``
using ``src/stable_yield_lab/risk_scoring.py``. The score averages three
normalized factors:

1. **Chain reputation** – established networks such as Ethereum receive a
   higher reputation (lower risk), while lesser known chains start at ``0.5``.
2. **Protocol audits** – more security audits reduce risk. The contribution is
   capped at five audits.
3. **Yield volatility** – unstable historical yields increase risk. Volatility
   is expected as a 0–1 value.

The three components are combined and scaled to the ``[1, 3]`` range. During
``Pipeline.run`` the score is computed for every fetched pool.
