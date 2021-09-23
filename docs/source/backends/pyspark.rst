.. _install.pyspark:

`PySpark <https://spark.apache.org/sql/>`_
====================================================

Install dependencies for Ibis's PySpark dialect:

::

  pip install 'ibis-framework[pyspark]'

.. note::

   When using the PySpark backend with PySpark 2.4.x and pyarrow >= 0.15.0, you
   need to set ``ARROW_PRE_0_15_IPC_FORMAT=1``. See `here
   <https://spark.apache.org/docs/latest/api/python/user_guide/arrow_pandas.html#compatibility-setting-for-pyarrow-0-15-0-and-spark-2-3-x-2-4-x>`_
   for details

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
