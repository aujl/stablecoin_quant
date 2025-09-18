from __future__ import annotations

"""Heuristic risk scoring for stablecoin yield pools."""

from dataclasses import replace
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from stable_yield_lab.core import Pool

# Simple reputation mapping per chain. Values range from 0 (unknown) to 1 (blue chip).
CHAIN_REPUTATION: Mapping[str, float] = {
    "Ethereum": 1.0,
    "Arbitrum": 0.9,
    "Polygon": 0.7,
    "BSC": 0.6,
    "Tron": 0.4,
}


def calculate_risk_score(chain_rep: float, audits: int, yield_volatility: float) -> float:
    """Combine factors into a normalized risk score in the range [1, 3].

    Parameters
    ----------
    chain_rep:
        Reputation of the underlying chain, ``0`` (unknown) to ``1`` (established).
    audits:
        Number of protocol security audits. Values above ``5`` are capped.
    yield_volatility:
        Normalized measure of historical yield volatility (``0`` stable, ``1`` erratic).
    """

    # Clamp inputs to expected ranges
    chain_rep = max(0.0, min(chain_rep, 1.0))
    audits = max(0, min(audits, 5))
    yield_volatility = max(0.0, min(yield_volatility, 1.0))

    chain_component = 1.0 - chain_rep  # higher reputation lowers risk
    audit_component = 1.0 - audits / 5.0  # more audits lower risk
    vol_component = yield_volatility  # higher volatility increases risk

    raw = (chain_component + audit_component + vol_component) / 3.0
    return 1.0 + 2.0 * raw


def score_pool(
    pool: "Pool",
    *,
    chain_reputation: Mapping[str, float] | None = None,
    protocol_audits: Mapping[str, int] | None = None,
    yield_volatility: Mapping[str, float] | None = None,
) -> "Pool":
    """Return a new :class:`~stable_yield_lab.Pool` with an updated ``risk_score``.

    Missing mapping entries default to neutral values (``0.5`` reputation,
    ``0`` audits and ``0`` volatility).
    """

    chain_rep_map = chain_reputation or CHAIN_REPUTATION
    audits_map = protocol_audits or {}
    vol_map = yield_volatility or {}

    chain_rep = chain_rep_map.get(pool.chain, 0.5)
    audits = audits_map.get(pool.name, 0)
    vol = vol_map.get(pool.name, 0.0)

    score = calculate_risk_score(chain_rep, audits, vol)
    return replace(pool, risk_score=score)
