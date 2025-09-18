# Core Test Guidelines

Refer to implementation principles in `src/stable_yield_lab/core/AGENTS.md` when adding or updating tests.

## Fixtures
- Reuse shared `sample_pools` and `repository` fixtures; keep new fixtures deterministic.
- Build pools via the public constructors to verify default values and immutability semantics.

## Mocking & Determinism
- Avoid monkeypatching internals; interact with repositories through the public API.
- Any randomness (e.g., ordering) must be fixed via explicit sorting before asserting.

## Coverage Goals
- Exercise repository filtering, iteration, and mutating helpers.
- Capture regression cases for pool equality/hash behaviour when relevant.
- Keep sanity checks (`test_sanity.py`) lightweight to guard import failures.
