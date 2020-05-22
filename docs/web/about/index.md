# Ibis: Python Data Analysis Productivity Framework

Ibis is a toolbox to bridge the gap between local Python environments (like
pandas and scikit-learn) and remote storage and execution systems like Hadoop
components (like HDFS, Impala, Hive, Spark) and SQL databases (Postgres,
etc.). Its goal is to simplify analytical workflows and make you more
productive.

We have a handful of specific priority focus areas:

- Enable data analysts to translate local, single-node data idioms to scalable
  computation representations (e.g. SQL or Spark)
- Integration with pandas and other Python data ecosystem components
- Provide high level analytics APIs and workflow tools to enhance productivity
  and streamline common or tedious tasks.
- Integration with community standard data formats (e.g. Parquet and Avro)
- Abstract away database-specific SQL differences

As the [Apache Arrow](http://arrow.apache.org/) project develops, we will
look to use Arrow to enable computational code written in Python to be executed
natively within other systems like Apache Spark and Apache Impala (incubating).

Source code is on GitHub: <https://github.com/ibis-project/ibis>.
