from __future__ import annotations

import glob
import os

import pandas as pd

df = pd.read_parquet(
    min(glob.glob("/data/pypi-parquet/*.parquet"), key=os.path.getsize),
    columns=["path", "uploaded_on", "project_name"],
)
df = df[
    df.path.str.contains(r"\.(?:asm|c|cc|cpp|cxx|h|hpp|rs|[Ff][0-9]{0,2}(?:or)?|go)$")
    & ~df.path.str.contains(r"(?:(?:^|/)test(?:|s|ing)|/site-packages/)")
]
print(
    df.assign(
        month=df.uploaded_on.dt.to_period("M").dt.to_timestamp(),
        ext=df.path.str.extract(r"\.([a-z0-9]+)$", 0)
        .iloc[:, 0]
        .str.replace(r"cxx|cpp|cc|c|hpp|h", "C/C++", regex=True)
        .str.replace("^f.*$", "Fortran", regex=True)
        .str.replace("rs", "Rust")
        .str.replace("go", "Go")
        .str.replace("asm", "Assembly"),
    )
    .groupby(["month", "ext"])
    .project_name.nunique()
    .rename("project_count")
    .reset_index()
    .sort_values(["month", "project_count"], ascending=False)
)
