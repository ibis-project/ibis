# Backends

A backend is where execution of Ibis table expressions occur after compiling into some intermediate representation. A backend is often a database and the intermediate representation often SQL, but several types of backends exist. See the [backends page](../backends/index.md) for specific documentation on each.

## String generating backends

The first category of backends translate Ibis table expressions into query strings.

The compiler turns each table expression into a query string and passes that query
to the database through a driver API for execution.

- [Apache Impala](../backends/impala.md)
- [ClickHouse](../backends/clickhouse.md)
- [Google BigQuery](../backends/bigquery.md)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)

## Expression generating backends

The next category of backends translates Ibis table expressions into another
system's table expression objects, for example, SQLAlchemy.

Instead of generating a query string for each table expression, these backends
produce another kind of table expression object and typically have high-level APIs
for execution.

- [Apache Arrow Datafusion](../backends/datafusion.md)
- [Apache Druid](../backends/druid.md)
- [Apache PySpark](../backends/pyspark.md)
- [Dask](../backends/dask.md)
- [DuckDB](../backends/duckdb.md)
- [MS SQL Server](../backends/mssql.md)
- [MySQL](../backends/mysql.md)
- [Oracle](../backends/oracle.md)
- [Polars](../backends/polars.md)
- [PostgreSQL](../backends/postgresql.md)
- [SQLite](../backends/sqlite.md)
- [Snowflake](../backends/snowflake.md)
- [Trino](../backends/trino.md)

## Direct execution backends

The pandas backend is the only direct execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [pandas](../backends/pandas.md)
