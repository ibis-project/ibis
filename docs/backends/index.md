# Backends

This document describes the classes of backends, how they work, and any details
about each backend that are relevant to end users.

For more information on a specific backend, check the next backend pages:

There are currently three classes of backends that live in ibis.

1. String generating backends
1. Expression generating backends
1. Direct execution backends

## String Generating Backends

The first category of backend translates ibis expressions into strings.
Generally speaking these backends also need to handle their own execution. They
work by translating each node into a string, and passing the generated string
to the database through a driver API.

- [Apache Impala](https://impala.apache.org/)
- [Clickhouse](https://clickhouse.yandex/)
- [Google BigQuery](https://cloud.google.com/bigquery/)
- [Hadoop Distributed File System (HDFS)](https://hadoop.apache.org/)
- [OmniSciDB](https://www.omnisci.com/)

## Expression Generating Backends

The second category of backends translates ibis expressions into other
expressions. Currently, all expression generating backends generate [SQLAlchemy
expressions](http://docs.sqlalchemy.org/en/latest/core/tutorial.html).

Instead of generating strings at each translation step, these backends build up
an expression. These backends tend to execute their expressions directly
through the driver APIs provided by SQLAlchemy (or one of its transitive
dependencies).

- [PostgreSQL](https://www.postgresql.org/)
- [SQLite](https://www.sqlite.org/)
- [MySQL](https://www.mysql.com/)

## Direct Execution Backends

The only existing backend that directly executes ibis expressions is the pandas
backend. A full description of the implementation can be found in the module
docstring of the pandas backend located in `ibis/backends/pandas/core.py`.

- [Pandas](http://pandas.pydata.org/)
- [PySpark](https://spark.apache.org/sql/)
- [Dask](https://dask.org/) (Experimental)
- [Datafusion](https://arrow.apache.org/datafusion/)
