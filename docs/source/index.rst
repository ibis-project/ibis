.. Ibis documentation master file, created by
   sphinx-quickstart on Wed Jun 10 11:06:29 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ibis: Python Data Analysis Framework
====================================

Ibis is a productivity-centric Python data analysis framework, designed to be
an ideal companion for SQL engines and distributed storage systems like
Hadoop. Ibis is being jointly developed with `Impala <http://impala.io>`_ to
deliver a complete 100% Python user experience on data of any size (small,
medium, or big).

At this time, Ibis supports the following SQL-based systems:

- Impala (on HDFS)
- SQLite

We have a handful of specific priority focus areas:

- Enable data analysts to translation analytics using SQL engines to Python
  instead of using the SQL language.
- Provide high level analytics APIs and workflow tools to enhance productivity
  and streamline common or tedious tasks.
- Provide high performance extensions for the Impala MPP query engine to enable
  high performance Python code to operate in a scalable Hadoop-like environment
- Abstract away database-specific SQL differences
- Integration with community standard data formats (e.g. Parquet and Avro)
- Integrate with the Python data ecosystem using the above tools

Architecturally, Ibis features:

- A pandas-like domain specific language (DSL) designed specifically for
  analytics, aka **Ibis expressions**, that enable composable, reusable
  analytics on structured data. If you can express something with a SQL SELECT
  query, you can write it with Ibis.
- An extensible translator-compiler system that targets multiple SQL systems
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

See the project blog http://blog.ibis-project.org for more frequent updates.

To learn more about Ibis's vision and roadmap, please visit
http://ibis-project.org.

Source code is on GitHub: http://github.com/cloudera/ibis

Since this is a young project, the documentation is definitely patchy in
places, but this will improve as things progress.

.. toctree::
   :maxdepth: 1

   getting-started
   configuration
   tutorial
   impala-udf
   api
   sql
   impala
   release
   developer
   type-system
   legal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
