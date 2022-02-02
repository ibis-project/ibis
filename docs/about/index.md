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

Architecturally, Ibis features:

- A pandas-like domain specific language (DSL) designed specifically for
  analytics, aka **expressions**, that enable composable and reusable analytics
  on structured data. If you can express something with a SQL `SELECT` query
  you should be able to write it with Ibis.
- Integrated user interfaces to HDFS and other storage systems.
- An extensible translator-compiler system that targets multiple SQL systems.
