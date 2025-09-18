# Source Adapter Test Guidelines

Align tests with the adapter principles in `src/stable_yield_lab/sources/AGENTS.md`.

## Fixtures
- Use JSON/CSV samples from `tests/fixtures/`; add new files sparingly and document their origin.
- Prefer parametrised fixtures when validating multiple adapters against shared expectations.

## Mocking & Determinism
- Monkeypatch network calls to ensure offline execution (e.g., override `urllib.request.urlopen`).
- Validate caching by operating on `tmp_path` directories populated with fixture payloads.

## Coverage Goals
- Cover happy paths and representative failure modes (missing fields, empty payloads, schema drift).
- Assert type conversions (floats, decimals, timezone-aware timestamps) align with domain models.
- Exercise adapter-specific filters or transformations before handing data to repositories.
