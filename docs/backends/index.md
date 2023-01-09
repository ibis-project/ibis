# Backends

## String Generating Backends

The first category of backend translate Ibis expressions into string queries.

The compiler turns each expression into a string query and passes that query to the
database through a driver API for execution.

- [Apache Impala](Impala.md)
- [ClickHouse](ClickHouse.md)
- [Google BigQuery](https://github.com/ibis-project/ibis-bigquery/)
- [HeavyAI](https://github.com/heavyai/ibis-heavyai)

## Expression Generating Backends

The next category of backends translates ibis expressions into another
system's expressions, for example, SQLAlchemy.

Instead of generating strings for each expression these backends produce
another kind of expression and typically have high-level APIs for execution.

- [Dask](Dask.md)
- [Datafusion](Datafusion.md)
- [DuckDB](DuckDB.md)
- [MySQL](MySQL.md)
- [MS SQL Server](MSSQL.md)
- [Polars](Polars.md)
- [PostgreSQL](PostgreSQL.md)
- [PySpark](PySpark.md)
- [SQLite](SQLite.md)
- [Snowflake](Snowflake.md)
- [Trino](Trino.md)

## Direct Execution Backends

The pandas backend is the only direct execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [Pandas](Pandas.md)
