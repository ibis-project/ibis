# Backends

## String Generating Backends

The first category of backend translate Ibis expressions into strings.
Generally speaking these backends handle their own execution.

The compiler turns each expression into a string and passes that string to the
database through a driver API.

- [Apache Impala](Impala.md)
- [ClickHouse](ClickHouse.md)
- [Google BigQuery](https://github.com/ibis-project/ibis-bigquery/)
- [OmniSciDB](https://github.com/omnisci/ibis-omniscidb)

## Expression Generating Backends

The next category of backends translates ibis expressions into other
expressions.

Instead of generating strings for each expression these backends produce
another kind of expression and typically have high-level APIs for execution.

- [Dask](Dask.md)
- [Datafusion](Datafusion.md)
- [MySQL](MySQL.md)
- [PostgreSQL](PostgreSQL.md)
- [PySpark](PySpark.md)
- [SQLite](SQLite.md)

## Direct Execution Backends

The pandas backend is the only direction execution backend. A full description
of the implementation can be found in the module docstring of the pandas
backend located in `ibis/backends/pandas/core.py`.

- [Pandas](Pandas.md)
