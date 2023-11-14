# Backends

A backend is where execution of Ibis table expressions occur after compiling into some intermediate representation. A backend is often a database and the intermediate representation often SQL, but several types of backends exist. See the [backends page](/backends/) for specific documentation on each.

## String generating backends

The first category of backends translate Ibis table expressions into query strings.

The compiler turns each table expression into a query string and passes that query
to the database through a driver API for execution.

- [Apache Impala](/backends/impala/)
- [ClickHouse](/backends/clickhouse/)
- [Google BigQuery](/backends/bigquery/)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)

## Expression generating backends

The next category of backends translates Ibis table expressions into another
system's table expression objects, for example, SQLAlchemy.

Instead of generating a query string for each table expression, these backends
produce another kind of table expression object and typically have high-level APIs
for execution.

- [Apache Arrow Datafusion](/backends/datafusion/)
- [Apache Druid](/backends/druid/)
- [Apache PySpark](/backends/pyspark/)
- [Dask](/backends/dask/)
- [DuckDB](/backends/duckdb/)
- [MS SQL Server](/backends/mssql/)
- [MySQL](/backends/mysql/)
- [Oracle](/backends/oracle/)
- [Polars](/backends/polars/)
- [PostgreSQL](/backends/postgresql/)
- [SQLite](/backends/sqlite/)
- [Snowflake](/backends/snowflake/)
- [Trino](/backends/trino/)

## Direct execution backends

The pandas backend is the only direct execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [pandas](/backends/pandas/)
