[![codecov.io](http://codecov.io/github/cloudera/ibis/coverage.svg?branch=master)](http://codecov.io/github/cloudera/ibis?branch=master)

# Ibis: Python data analysis framework for Hadoop and SQL engines

Install Ibis from PyPI with:

    $ pip install ibis-framework

Ibis is a Python data analysis library with a handful of related goals:

- Enable data analysts to transition analytics on SQL engines to
  Python code instead of SQL code.
- Provide high level analytics APIs and workflow tools to accelerate
  productivity.
- Provide high performance extensions for the Impala MPP query engine to enable
  high performance Python code to operate in a scalable Hadoop-like environment
- Abstract away database-specific SQL differences
- Integrate with the Python data ecosystem using the above tools

At this time, Ibis supports the following SQL-based systems:

- Impala (on HDFS)
- SQLite

Ibis is being designed and led by the creator of pandas
(github.com/pydata/pandas) and is intended to have a familiar user interface
for folks used to small data on single machines in Python.

Architecturally, Ibis features:

- A pandas-like domain specific language (DSL) designed specifically for
  analytics, aka **Ibis expressions**, that enable composable, reusable
  analytics on structured data. If you can express something with a SQL SELECT
  query, you can write it with Ibis.
- A translation system that targets multiple SQL systems
- Tools for wrapping user-defined functions in Impala and eventually other SQL
  engines

SQL engine support near on the horizon:

- PostgreSQL
- Redshift
- Vertica
- Spark SQL
- Presto
- Hive
- MySQL / MariaDB

Read the project blog at http://blog.ibis-project.org.

Learn much more at http://ibis-project.org.
