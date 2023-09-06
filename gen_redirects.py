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
    "/api/": "/reference/",
    "/api/backends/": "/reference/backends/",
    "/api/backends/base/": "/reference/backends/base/",
    "/api/backends/pandas/": "/reference/backends/pandas/",
    "/api/backends/sql/": "/reference/backends/sql/",
    "/api/backends/sqlalchemy/": "/reference/backends/sqlalchemy/",
    "/api/config/": "/reference/config/",
    "/api/datatypes/": "/reference/datatypes/",
    "/api/expressions/": "/reference/expressions/",
    "/api/expressions/collections/": "/reference/expressions/collections/",
    "/api/expressions/generic/": "/reference/expressions/generic/",
    "/api/expressions/geospatial/": "/reference/expressions/geospatial/",
    "/api/expressions/numeric/": "/reference/expressions/numeric/",
    "/api/expressions/strings/": "/reference/expressions/strings/",
    "/api/expressions/tables/": "/reference/expressions/tables/",
    "/api/expressions/timestamps/": "/reference/expressions/timestamps/",
    "/api/expressions/top_level/": "/reference/expressions/top_level/",
    "/api/schemas/": "/reference/schemas/",
    "/api/selectors/": "/reference/selectors/",
    "/backends/": "/support_matrix",
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
    "/backends/Pandas/": "/backends/pandas/",
    "/backends/Polars/": "/backends/polars/",
    "/backends/PostgreSQL/": "/backends/postgresql/",
    "/backends/PySpark/": "/backends/pyspark/",
    "/backends/SQLite/": "/backends/sqlite/",
    "/backends/Snowflake/": "/backends/snowflake/",
    "/backends/Trino/": "/backends/trino/",
    "/backends/_support_matrix/": "/support_matrix",
    "/backends/support_matrix": "/support_matrix",
    "/blog": "/posts",
    "/blog/Ibis-version-3.0.0-release/": "/posts/Ibis-version-3.0.0-release/",
    "/blog/Ibis-version-3.1.0-release/": "/posts/Ibis-version-3.1.0-release/",
    "/blog/ffill-and-bfill-using-ibis/": "/posts/ffill-and-bfill-using-ibis",
    "/blog/ibis-examples/": "/posts/ibis-examples/",
    "/blog/ibis-to-file/": "/posts/ibis-to-file/",
    "/blog/ibis-version-4.0.0-release/": "/posts/ibis-version-4.0.0-release/",
    "/blog/ibis_substrait_to_duckdb/": "/posts/ibis_substrait_to_duckdb/",
    "/blog/rendered/campaign-finance/": "/posts/campaign-finance/",
    "/blog/rendered/campaign-finance/campaign-finance.ipynb": "/posts/campaign-finance/",
    "/blog/rendered/ci-analysis/": "/posts/ci-analysis/",
    "/blog/rendered/ci-analysis/ci-analysis.ipynb": "/posts/ci-analysis/",
    "/blog/rendered/ibis-version-6.0.0-release/": "/posts/ibis-version-6.0.0-release/",
    "/blog/rendered/ibis-version-6.0.0-release/ibis-version-6.0.0-release.ipynb": "/posts/ibis-version-6.0.0-release/",
    "/blog/rendered/ibis-version-6.1.0-release/": "/posts/ibis-version-6.1.0-release/",
    "/blog/rendered/ibis-version-6.1.0-release/ibis-version-6.1.0-release.ipynb": "/posts/ibis-version-6.1.0-release/",
    "/blog/rendered/torch/": "/posts/torch/",
    "/blog/rendered/torch/torch.ipynb": "/posts/torch/",
    "/blog/selectors/": "/posts/selectors/",
    "/community/contribute/": "/contribute/01_environment",
    "/community/contribute/01_environment/": "/contribute/01_environment",
    "/community/contribute/02_workflow/": "/contribute/02_workflow/",
    "/community/contribute/03_style/": "/contribute/03_style/",
    "/community/contribute/05_maintainers_guide/": "/contribute/05_maintainers_guide/",
    "/concept/backends/": "/concepts/backend",
    "/concept/design/": "/concepts/internals",
    "/concept/why_ibis/": "/why",
    "/docs/": "/",
    "/docs/dev/backends/support_matrix/": "/support_matrix",
    "/docs/dev/contribute/01_environment/": "/contribute/01_environment",
    "/docs/dev/release_notes/": "/release_notes",
    "/getting_started/": "/tutorial/getting_started/",
    "/how_to/chain-expressions/": "/how-to/analytics/chain_expressions/",
    "/how_to/chain_expressions/": "/how-to/analytics/chain_expressions/",
    "/how_to/configuration/": "/how-to/configure/basics",
    "/how_to/ffill_bfill_w_window/": "/posts/ffill-and-bfill-using-ibis",
    "/how_to/streamlit/": "/how-to/visualization/streamlit",
    "/ibis-for-dplyr-users/": "/tutorial/ibis-for-dplyr-users/",
    "/ibis-for-pandas-users/": "/tutorial/ibis-for-pandas-users/",
    "/ibis-for-sql-programmers/": "/tutorial/ibis-for-sql-users/",
    "/reference/backends/pandas/": "/backends/pandas/",
    "/reference/datatypes/": "/reference/datatypes-schemas/",
    "/reference/expressions/": "/reference/",
    "/reference/expressions/collections/": "/reference/expression-collections",
    "/reference/expressions/generic/": "/reference/expression-generic",
    "/reference/expressions/geospatial/": "/reference/expression-geospatial",
    "/reference/expressions/numeric/": "/reference/expression-numeric",
    "/reference/expressions/strings/": "/reference/expression-strings",
    "/reference/expressions/tables/": "/reference/expressions-tables",
    "/reference/expressions/timestamps/": "/reference/expression-temporal",
    "/reference/expressions/top_level/": "/reference/top_level",
    "/reference/schemas/": "/reference/datatypes-schemas",
    "/tutorial/": "/tutorials/getting_started/",
    "/tutorial/getting_started/": "/tutorials/getting_started",
    "/tutorial/ibis-for-dplyr-users/": "/tutorials/ibis-for-dplyr-users/",
    "/tutorial/ibis-for-dplyr-users/ibis-for-dplyr-users.ipynb": "/tutorials/ibis-for-dplyr-users/",
    "/tutorial/ibis-for-pandas-users/": "/tutorials/ibis-for-pandas-users/",
    "/tutorial/ibis-for-pandas-users/ibis-for-pandas-users.ipynb": "/tutorials/ibis-for-pandas-users/",
    "/tutorial/ibis-for-sql-users/": "/tutorials/ibis-for-sql-users/",
    "/tutorial/ibis-for-sql-users/ibis-for-sql-users.ipynb": "/tutorials/ibis-for-sql-users/",
    "/user_guide/configuration/": "/how-to/configure/basics",
    "/user_guide/design/": "/concepts/internals",
    "/why_ibis/": "/why",
    # TODO: "/user_guide/extending/": "/how_to/extending/",
    # TODO: "/user_guide/self_joins/": "/how_to/self_joins/",
    # TODO: "/how_to/extending/elementwise/"
    # TODO: "/how_to/extending/elementwise/elementwise.ipynb"
    # TODO: "/how_to/extending/reduction/"
    # TODO: "/how_to/extending/reduction/reduction.ipynb"
    # TODO: "/community/"
    # TODO: "/how_to/duckdb_register/"
    # TODO: "/how_to/memtable-join/": "/how_to/memtable_join/",
    # TODO: "/how_to/memtable_join/"
    # TODO: "/how_to/self_joins/"
    # TODO: "/how_to/sessionize/"
    # TODO: "/how_to/topk/"
    # TODO: "/reference/backends/base/"
    # TODO: "/reference/backends/sql/"
    # TODO: "/reference/backends/sqlalchemy/"
    # TODO: "/versioning/":
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
