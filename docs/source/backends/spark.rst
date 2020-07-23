.. _install.spark:

`PySpark/Spark SQL <https://spark.apache.org/sql/>`_
----------------------------------------------------

Install dependencies for Ibis's Spark dialect:

::

  pip install ibis-framework[spark]

Create a client by passing in the spark session as a parameter to
:func:`ibis.spark.connect`:

.. code-block:: python

   con = ibis.spark.connect(spark_session)
