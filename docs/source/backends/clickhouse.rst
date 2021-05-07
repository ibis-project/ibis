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

.. _api.clickhouse:

API
===
.. currentmodule:: ibis.backends.clickhouse

The ClickHouse client is accessible through the ``ibis.clickhouse`` namespace.

Use ``ibis.clickhouse.connect`` to create a client.

.. autosummary::
   :toctree: ../generated/

   connect
   ClickhouseClient.close
   ClickhouseClient.exists_table
   ClickhouseClient.exists_database
   ClickhouseClient.get_schema
   ClickhouseClient.set_database
   ClickhouseClient.list_databases
   ClickhouseClient.list_tables
