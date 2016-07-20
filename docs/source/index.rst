.. Ibis documentation master file, created by
   sphinx-quickstart on Wed Jun 10 11:06:29 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ibis: Python Data Analysis Productivity Framework
=================================================

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

As the `Apache Arrow <http://arrow.apache.org/>`_ project develops, we will
look to use Arrow to enable computational code written in Python to be executed
natively within other systems like Apache Spark and Apache Impala (incubating).

To learn more about Ibis's vision, roadmap, and updates, please follow
http://ibis-project.org.

Source code is on GitHub: http://github.com/cloudera/ibis

Install Ibis from PyPI with:

::

  pip install ibis-framework

Or from `conda-forge <http://conda-forge.github.io>`_ with

::

  conda install ibis-framework -c conda-forge

At this time, Ibis offers some level of support for the following systems:

- `Apache Impala (incubating) <http://impala.io/>`_
- `Apache Kudu (incubating) <http://getkudu.io>`_
- Hadoop Distributed File System (HDFS)
- PostgreSQL (Experimental)
- SQLite

Coming from SQL? Check out :ref:`Ibis for SQL Programmers <sql>`.

Architecturally, Ibis features:

- A pandas-like domain specific language (DSL) designed specifically for
  analytics, aka **Ibis expressions**, that enable composable, reusable
  analytics on structured data. If you can express something with a SQL SELECT
  query, you can write it with Ibis.
- Integrated user interfaces to HDFS and other storage systems.
- An extensible translator-compiler system that targets multiple SQL systems

SQL engine support needing code contributors:

- Redshift
- Vertica
- Spark SQL
- Presto
- Hive
- MySQL / MariaDB

Since this is a young project, the documentation is definitely patchy in
places, but this will improve as things progress.

.. toctree::
   :maxdepth: 1

   getting-started
   configuration
   tutorial
   impala
   api
   sql
   release
   developer
   type-system
   legal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
