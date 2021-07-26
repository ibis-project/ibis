.. _install.pyspark:

`PySpark <https://spark.apache.org/sql/>`_
====================================================

Install dependencies for Ibis's PySpark dialect:

::

  pip install ibis-framework[pyspark]

.. _api.pyspark:

PySpark client
~~~~~~~~~~~~~~
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
