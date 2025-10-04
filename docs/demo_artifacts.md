# Daily Demo Artifacts

The repository publishes a fresh bundle of demo outputs every day via the
`generate-demo-images` GitHub Actions workflow. The job runs the
`stable_yield_demo.py` script against `configs/demo.toml` and uploads the
resulting CSV and PNG files as a single artifact named
`daily-demo-artifacts`.

## What gets published?

The artifact mirrors the contents of the demo output directory
(`demo_artifacts/`) and currently includes:

- Aggregated CSV tables (`pools.csv`, `by_chain.csv`, `by_source.csv`,
  `by_stablecoin.csv`, `topN.csv`, `concentration.csv`, `warnings.csv`,
  `portfolio_performance.csv`, `portfolio_nav.csv`).
- Chart images (`bar_group_chain.png`, either `scatter_risk_return.png`
  or `scatter_tvl_apy.png` depending on portfolio feasibility, plus
  `yield_vs_time.png` and `nav_vs_time.png`).

These files are generated without modifying the repository tree; they are
only attached to the workflow run as downloadable artifacts so the main
branch history stays clean.

## Downloading the latest artifact

1. Open the repository's **Actions** tab and select the
   `generate-demo-images` workflow.
2. Pick the most recent successful run (runs are timestamped in UTC).
3. Scroll to the **Artifacts** section and download
   `daily-demo-artifacts.zip`.
4. Extract the archive locally to view the CSV summaries and charts.

GitHub retains workflow artifacts for 90 days by default. If you need to
extend the retention period, adjust the workflow's `retention-days`
setting or mirror the artifacts to long-term storage.

## Automating downloads

The files can also be retrieved programmatically with the GitHub CLI:

```bash
# Install the CLI from https://cli.github.com/ and authenticate once
# with `gh auth login`.

# Download artifacts from the latest run on `main`
RUN_ID=$(gh run list --workflow generate-demo-images --branch main --limit 1 --json databaseId \
  --jq '.[0].databaseId')

gh run download "$RUN_ID" --name daily-demo-artifacts --dir demo_artifacts_latest
```

The command stores the extracted files under `demo_artifacts_latest/` so
you can inspect the latest daily charts and tables without rerunning the
pipeline locally.
