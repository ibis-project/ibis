.. _install.spark:

`PySpark/Spark SQL <https://spark.apache.org/sql/>`_
====================================================

Install dependencies for Ibis's Spark dialect:

::

  pip install ibis-framework[spark]

Create a client by passing in the spark session as a parameter to
:func:`ibis.spark.connect`:

.. code-block:: python

   con = ibis.spark.connect(spark_session)

.. _api.spark:

API
---

SparkSQL client (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: ibis.backends.spark

The Spark SQL client is accessible through the ``ibis.spark`` namespace.

Use ``ibis.spark.connect`` to create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   SparkClient.database
   SparkClient.list_databases
   SparkClient.list_tables
   SparkClient.table

.. _api.pyspark:

PySpark client (Experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: ibis.backends.pyspark

The PySpark client is accessible through the ``ibis.pyspark`` namespace.

Use ``ibis.pyspark.connect`` to create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   PySparkClient.database
   PySparkClient.list_databases
   PySparkClient.list_tables
   PySparkClient.table
