.. _install.pyspark:

`PySpark <https://spark.apache.org/sql/>`_
==========================================

Install
-------

Install dependencies for Ibis's PySpark dialect:

::

  pip install 'ibis-framework[pyspark]'

or

::

  conda install -c conda-forge ibis-pyspark

.. note::

   When using the PySpark backend with PySpark 2.3.x, 2.4.x and pyarrow >= 0.15.0, you
   need to set ``ARROW_PRE_0_15_IPC_FORMAT=1``. See `here  <https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x>`_
   for details

.. _api.pyspark:

Connect
-------

.. currentmodule:: ibis.backends.pyspark

The PySpark client is accessible through the ``ibis.pyspark`` namespace.

Use ``ibis.pyspark.connect`` to create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   Backend.database
   Backend.list_databases
   Backend.list_tables
   Backend.table
