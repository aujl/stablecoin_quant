
from datetime import datetime, timezone
import pandas as pd
from stable_yield_lab import CSVSource, Pipeline, Metrics, Visualizer, PoolRepository

CSV_PATH = "./source/sample_pools.csv"
# Load sample data (replace CSVSource with DefiLlamaSource/MorphoSource in production)
src = CSVSource(path=CSV_PATH)
pipe = Pipeline([src])
repo = pipe.run()

# Apply your user's constraints:
# - min TVL 100k, min base APY 6%, auto-only
filtered = repo.filter(min_tvl=100_000, min_base_apy=0.06, auto_only=True)

df = filtered.to_dataframe().sort_values("base_apy", ascending=False)
print(f"Pools after filter: {len(df)}")
df.head(20)

# Compute simple summaries
by_chain = Metrics.groupby_chain(filtered)
top10 = Metrics.top_n(filtered, n=10, key="base_apy")

# Visuals (each single figure, default matplotlib styles)
Visualizer.bar_apr(top10, title="Top‑10 Stablecoin Pools – Base APY")
Visualizer.scatter_tvl_apy(df, title="TVL vs Base APY (bubble=risk)")
Visualizer.bar_group_chain(by_chain, title="TVL‑gewichteter Base‑APY je Chain")
