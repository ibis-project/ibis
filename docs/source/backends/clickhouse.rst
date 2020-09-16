.. _install.clickhouse:

`Clickhouse <https://clickhouse.yandex/>`_
------------------------------------------

Install dependencies for Ibis's Clickhouse dialect(minimal supported version is `0.1.3`):

::

  pip install ibis-framework[clickhouse]

Create a client by passing in database connection parameters such as ``host``,
``port``, ``database``, and ``user`` to :func:`ibis.clickhouse.connect`:


.. code-block:: python

   con = ibis.clickhouse.connect(host='clickhouse', port=9000)
