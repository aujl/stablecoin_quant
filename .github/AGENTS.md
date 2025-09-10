# CI and Workflow Guidelines

These instructions cover files in `.github/` including GitHub Actions workflows.

## CI Workflows
- Use GitHub Actions to run pre-commit, tests, and coverage on Python 3.12.
- Keep workflows minimal and modular; prefer composite actions when steps repeat.
- When altering triggers or job matrices, document the rationale in comments.
- Validate workflow changes with `act` or via a pull request before merging.

## Codex and Automation
- Automation scripts should respect repository `AGENTS.md` instructions and maintain determinism.
- Avoid committing secrets; reference credentials through encrypted repository secrets.
- Reusable workflows should expose parameters clearly and default to safe settings.

## Reviews
- CI changes require review from maintainers familiar with infrastructure.
- Ensure any new workflow has a matching entry in documentation if it affects contributors.
