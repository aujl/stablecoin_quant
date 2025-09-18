"""Core constants shared across StableYieldLab modules."""

from __future__ import annotations

# Common stablecoins recognised by StableYieldLab data adapters.
#
# The list intentionally focuses on major dollar- and euro-pegged tokens to
# filter protocols when sourcing pools from aggregators such as DefiLlama or
# Morpho.  Keeping the canonical symbols in :mod:`stable_yield_lab.core`
# avoids circular imports when consumers only need the lightweight data model
# layer.
STABLE_TOKENS = {
    "USDC",
    "USDT",
    "DAI",
    "FRAX",
    "LUSD",
    "GUSD",
    "TUSD",
    "USDP",
    "USDD",
    "USDR",
    "USDf",
    "USDF",
    "MAI",
    "SUSD",
    "EURS",
    "EUROE",
    "CRVUSD",
    "GHO",
    "USDC.E",
    "USDT.E",
}

__all__ = ["STABLE_TOKENS"]
