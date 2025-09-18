# Source Adapter Guidelines

These instructions apply to modules inside `stable_yield_lab.sources`.

## Design Principles
- Adapters must be side-effect free and never perform network IO during import.
- Expose a uniform `fetch()` interface returning iterables of domain models.
- Cache or rate-limit network calls via explicit dependencies passed at construction.

## Implementation Notes
- Provide small deterministic fixtures or cache files for integration tests.
- Keep external service schemas isolated; parse into typed dictionaries or dataclasses early.
- Surface actionable error messages when required fields are missing or malformed.

## Testing Hooks
- All network interactions must be mockable; design constructors to accept session-like objects.
- Use local fixture data to validate parsing logic and edge cases (missing fields, empty payloads).
- Ensure CSV readers handle timezone awareness and numeric coercion consistently.
