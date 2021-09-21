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

Source code is on GitHub: https://github.com/ibis-project/ibis

.. _install:

Installation
------------

System Dependencies
~~~~~~~~~~~~~~~~~~~

Ibis requires a working Python 3.7+ installation. We recommend using
`Anaconda <http://continuum.io/downloads>`_ to manage Python versions and
environments.

Installing the Python Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install Ibis from PyPI with:

::

  pip install ibis-framework

Or from `conda-forge <http://conda-forge.github.io>`_ with

::

  conda install ibis-framework -c conda-forge

This installs the ``ibis`` library to your configured Python environment.

Ibis can also be installed with Kerberos support for its HDFS functionality:

::

  pip install 'ibis-framework[kerberos]'

Some platforms will require that you have Kerberos installed to build properly.

* Redhat / CentOS: ``yum install krb5-devel``
* Ubuntu / Debian: ``apt-get install libkrb5-dev``
* Arch Linux     : ``pacman -S krb5``

At this time, Ibis offers some level of support for the following systems:

- `Apache Impala <https://impala.apache.org/>`_
- `Apache Kudu <https://kudu.apache.org/>`_
- `Hadoop Distributed File System (HDFS) <https://hadoop.apache.org/>`_
- `PostgreSQL <https://www.postgresql.org/>`_
- `MySQL <https://www.mysql.com/>`_ (Experimental)
- `SQLite <https://www.sqlite.org/>`_
- `Google BigQuery <https://cloud.google.com/bigquery/>`_
- `Yandex Clickhouse <https://clickhouse.yandex/>`_
- Direct execution of ibis expressions against `Pandas
  <http://pandas.pydata.org/>`_ objects
- `OmniSciDB <https://www.omnisci.com/>`_
- `PySpark/Spark SQL <https://spark.apache.org/sql/>`_ (Experimental)

Coming from SQL? Check out :ref:`Ibis for SQL Programmers <sql>`.

Architecturally, Ibis features:

- A pandas-like domain specific language (DSL) designed specifically for
  analytics, aka **Ibis expressions**, that enable composable, reusable
  analytics on structured data. If you can express something with a SQL SELECT
  query, you can write it with Ibis.
- Integrated user interfaces to HDFS and other storage systems.
- An extensible translator-compiler system that targets multiple SQL systems

SQL engine support needing code contributors:

- `Redshift <https://aws.amazon.com/redshift/>`_
- `Vertica <https://www.vertica.com/>`_
- `Presto <https://prestosql.io/>`_
- `Hive <https://hive.apache.org/>`_

.. toctree::
   :maxdepth: 1

   tutorial/index
   user_guide/index
   api
   backends/index
   release/index

Learning Resources
------------------

We collect Jupyter notebooks for learning how to use ibis here:
https://github.com/ibis-project/ibis/tree/master/docs/source/notebooks/tutorial.
Some of these notebooks will be reproduced as part of the documentation
:ref:`in the tutorial section <tutorial>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Use the meta tags to verify the site for use in Google OAuth2 consent flow.

.. meta::
    :google-site-verification: IVqzkYiD5E35oD4kkVOcTYCTfqWKU1f6zOHCnLIPkUU
