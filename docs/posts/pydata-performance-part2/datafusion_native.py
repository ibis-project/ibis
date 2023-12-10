from __future__ import annotations

import datafusion

with open("./datafusion_native.sql") as f:
    query = f.read()

ctx = datafusion.SessionContext()
ctx.register_parquet(name="pypi", path="/data/pypi-parquet/*.parquet")
expr = ctx.sql(query)

df = expr.to_pandas()
