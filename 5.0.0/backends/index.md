# Backends

See the [configuration guide](../user_guide/configuration.md#default-backend)
to inspect or reconfigure the backend used by default.

## String Generating Backends

The first category of backend translate Ibis expressions into string queries.

The compiler turns each expression into a string query and passes that query to the
database through a driver API for execution.

- [Apache Impala](Impala.md)
- [ClickHouse](ClickHouse.md)
- [Google BigQuery](BigQuery.md)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)

## Expression Generating Backends

The next category of backends translates ibis expressions into another
system's expressions, for example, SQLAlchemy.

Instead of generating strings for each expression these backends produce
another kind of expression and typically have high-level APIs for execution.

- [Apache Arrow Datafusion](Datafusion.md)
- [Apache Druid](Druid.md)
- [Apache PySpark](PySpark.md)
- [Dask](Dask.md)
- [DuckDB](DuckDB.md)
- [MS SQL Server](MSSQL.md)
- [MySQL](MySQL.md)
- [Polars](Polars.md)
- [PostgreSQL](PostgreSQL.md)
- [SQLite](SQLite.md)
- [Snowflake](Snowflake.md)
- [Trino](Trino.md)

## Direct Execution Backends

The pandas backend is the only direct execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [Pandas](Pandas.md)
