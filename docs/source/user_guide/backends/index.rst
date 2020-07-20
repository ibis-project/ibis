.. _backends:

Backends
========

This document describes the classes of backends, how they work, and any details
about each backend that are relevant to end users.

For more information on a specific backend, check the next backend pages:

.. toctree::
   :maxdepth: 1

   omnisci
   bigquery


.. _classes_of_backends:

Classes of Backends
-------------------

There are currently three classes of backends that live in ibis.

#. String generating backends
#. Expression generating backends
#. Direct execution backends

.. _string_generating_backends:

String Generating Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~

The first category of backend translates ibis expressions into strings.
Generally speaking these backends also need to handle their own execution.
They work by translating each node into a string, and passing the generated
string to the database through a driver API.

- `Apache Impala <https://impala.apache.org/>`_
- `Yandex Clickhouse <https://clickhouse.yandex/>`_
- `Google BigQuery <https://cloud.google.com/bigquery/>`_
- `Hadoop Distributed File System (HDFS) <https://hadoop.apache.org/>`_
- `OmniSciDB <https://www.omnisci.com/>`_
- `PySpark/Spark SQL <https://spark.apache.org/sql/>`_ (Experimental)

.. _expression_generating_backends:

Expression Generating Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second category of backends translates ibis expressions into other
expressions. Currently, all expression generating backends generate `SQLAlchemy
expressions <http://docs.sqlalchemy.org/en/latest/core/tutorial.html>`_.

Instead of generating strings at each translation step, these backends build up
an expression. These backends tend to execute their expressions directly
through the driver APIs provided by SQLAlchemy (or one of its transitive
dependencies).

- `PostgreSQL <https://www.postgresql.org/>`_
- `SQLite <https://www.sqlite.org/>`_
- `MySQL <https://www.mysql.com/>`_ (Experimental)

.. _direct_execution_backends:

Direct Execution Backends
~~~~~~~~~~~~~~~~~~~~~~~~~

The only existing backend that directly executes ibis expressions is the pandas
backend. A full description of the implementation can be found in the module
docstring of the pandas backend located in ``ibis/pandas/execution/core.py``.

- `Pandas <http://pandas.pydata.org/>`_
