from __future__ import annotations

import pathlib

# Versions for templated redirects
VERSIONS = ["latest", "dev", "4.1.0", "4.0.0", "3.2.0", "3.1.0"]

# Templated redirects
TEMPLATED_REDIRECTS = {
    "/docs/{version}/": "/",
    "/docs/{version}/install/": "/install/",
    "/docs/{version}/docs/": "/docs/",
    "/docs/{version}/backends/": "/backends/",
}

# Untemplated redirects
REDIRECTS = {
    "/backends/Pandas/": "/backends/pandas/",
    "/getting_started/": "/tutorial/getting_started/",
    "/ibis-for-sql-programmers/": "/tutorial/ibis-for-sql-users/",
    "/ibis-for-pandas-users/": "/tutorial/ibis-for-pandas-users/",
    "/ibis-for-dplyr-users/": "/tutorial/ibis-for-dplyr-users/",
    "/why_ibis/": "/concept/why_ibis/",
    "/user_guide/design/": "/concept/design/",
    "/user_guide/self_joins/": "/how_to/self_joins/",
    "/user_guide/configuration/": "/how_to/configuration/",
    "/user_guide/extending/": "/how_to/extending/",
    "/backends/BigQuery/": "/backends/bigquery/",
    "/backends/Clickhouse/": "/backends/clickhouse/",
    "/backends/Dask/": "/backends/dask/",
    "/backends/Datafusion/": "/backends/datafusion/",
    "/backends/Druid/": "/backends/druid/",
    "/backends/DuckDB/": "/backends/duckdb/",
    "/backends/Impala/": "/backends/impala/",
    "/backends/MSSQL/": "/backends/mssql/",
    "/backends/MySQL/": "/backends/mysql/",
    "/backends/Oracle/": "/backends/oracle/",
    "/backends/Polars/": "/backends/polars/",
    "/backends/PostgreSQL/": "/backends/postgresql/",
    "/backends/PySpark/": "/backends/pyspark/",
    "/backends/SQLite/": "/backends/sqlite/",
    "/backends/Snowflake/": "/backends/snowflake/",
    "/backends/Trino/": "/backends/trino/",
    "/backends/support_matrix": "/backends/_support_matrix/",
    "/how_to/chain-expressions/": "/how_to/chain_expressions/",
    "/how_to/memtable-join/": "/how_to/memtable_join/",
    "/docs/": "/",
    "/api/": "/reference/",
    "/api/expressions/": "/reference/expressions/",
    "/api/expressions/top_level/": "/reference/expressions/top_level/",
    "/api/expressions/tables/": "/reference/expressions/tables/",
    "/api/expressions/generic/": "/reference/expressions/generic/",
    "/api/expressions/numeric/": "/reference/expressions/numeric/",
    "/api/expressions/strings/": "/reference/expressions/strings/",
    "/api/expressions/timestamps/": "/reference/expressions/timestamps/",
    "/api/expressions/collections/": "/reference/expressions/collections/",
    "/api/expressions/geospatial/": "/reference/expressions/geospatial/",
    "/api/selectors/": "/reference/selectors/",
    "/api/datatypes/": "/reference/datatypes/",
    "/api/schemas/": "/reference/schemas/",
    "/api/config/": "/reference/config/",
    "/api/backends/": "/reference/backends/",
    "/api/backends/base/": "/reference/backends/base/",
    "/api/backends/pandas/": "/reference/backends/pandas/",
    "/api/backends/sql/": "/reference/backends/sql/",
    "/api/backends/sqlalchemy/": "/reference/backends/sqlalchemy/",
}

# Fill in templates
REDIRECTS.update(
    {
        old.format(version=version): new
        for version in VERSIONS
        for old, new in TEMPLATED_REDIRECTS.items()
    }
)

# Write all redirect files
output_dir = pathlib.Path(__file__).parent.joinpath("docs", "_output")
output_dir.mkdir(exist_ok=True, parents=True)

lines = "\n".join(f"{old} {new}" for old, new in REDIRECTS.items())
output_dir.joinpath("_redirects").write_text(f"{lines}\n")
