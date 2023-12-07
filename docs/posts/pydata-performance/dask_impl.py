from __future__ import annotations

import logging

import dask.dataframe as dd
from dask.distributed import Client

if __name__ == "__main__":
    client = Client(silence_logs=logging.ERROR)
    df = dd.read_parquet(
        "/data/pypi-parquet/*.parquet",
        columns=["path", "uploaded_on", "project_name"],
        split_row_groups=True,
    )
    df = df[
        df.path.str.contains(
            r"\.(?:asm|c|cc|cpp|cxx|h|hpp|rs|[Ff][0-9]{0,2}(?:or)?|go)$"
        )
        & ~df.path.str.contains(r"(?:^|/)test(?:|s|ing)")
        & ~df.path.str.contains("/site-packages/")
    ]
    print(
        df.assign(
            month=df.uploaded_on.dt.to_period("M").dt.to_timestamp(),
            ext=df.path.str.extract(r"\.([a-z0-9]+)$", 0, expand=False)
            .str.replace(r"cxx|cpp|cc|c|hpp|h", "C/C++", regex=True)
            .str.replace("^f.*$", "Fortran", regex=True)
            .str.replace("rs", "Rust")
            .str.replace("go", "Go")
            .str.replace("asm", "Assembly"),
        )
        .groupby(["month", "ext"])
        .project_name.nunique()
        .rename("project_count")
        .compute()
        .reset_index()
        .sort_values(["month", "project_count"], ascending=False)
    )
    client.shutdown()
