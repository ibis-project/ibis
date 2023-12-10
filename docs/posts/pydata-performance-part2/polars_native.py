from __future__ import annotations

import polars as pl

expr = (
    pl.scan_parquet("/data/pypi-parquet/*.parquet")
    .filter(
        [
            pl.col("path").str.contains(
                r"\.(asm|c|cc|cpp|cxx|h|hpp|rs|[Ff][0-9]{0,2}(?:or)?|go)$"
            ),
            ~pl.col("path").str.contains(r"(^|/)test(|s|ing)"),
            ~pl.col("path").str.contains("/site-packages/", literal=True),
        ]
    )
    .with_columns(
        month=pl.col("uploaded_on").dt.truncate("1mo"),
        ext=pl.col("path")
        .str.extract(pattern=r"\.([a-z0-9]+)$", group_index=1)
        .str.replace_all(pattern=r"cxx|cpp|cc|c|hpp|h", value="C/C++")
        .str.replace_all(pattern="^f.*$", value="Fortran")
        .str.replace("rs", "Rust", literal=True)
        .str.replace("go", "Go", literal=True)
        .str.replace("asm", "Assembly", literal=True)
        .replace({"": None}),
    )
    .group_by(["month", "ext"])
    .agg(project_count=pl.col("project_name").n_unique())
    .drop_nulls(["ext"])
    .sort(["month", "project_count"], descending=True)
)

df = expr.collect(streaming=True).to_pandas()
