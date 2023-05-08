# Backends

See the [configuration guide](../user_guide/configuration.md#default-backend)
to inspect or reconfigure the backend used by default.

## String Generating Backends

The first category of backend translate Ibis table expressions into query strings.

The compiler turns each table expression into a query string and passes that query
to the database through a driver API for execution.

- [Apache Impala](Impala.md)
- [ClickHouse](ClickHouse.md)
- [Google BigQuery](BigQuery.md)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)

## Expression Generating Backends

The next category of backends translates Ibis table expressions into another
system's table expression objects, for example, SQLAlchemy.

Instead of generating a query string for each table expression, these backends
produce another kind of table expression object and typically have high-level APIs
for execution.

- [Apache Arrow Datafusion](Datafusion.md)
- [Apache Druid](Druid.md)
- [Apache PySpark](PySpark.md)
- [Dask](Dask.md)
- [DuckDB](DuckDB.md)
- [MS SQL Server](MSSQL.md)
- [MySQL](MySQL.md)
- [Oracle](Oracle.md)
- [Polars](Polars.md)
- [PostgreSQL](PostgreSQL.md)
- [SQLite](SQLite.md)
- [Snowflake](Snowflake.md)
- [Trino](Trino.md)

## Direct Execution Backends

The pandas backend is the only direct execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [pandas](pandas.md)
