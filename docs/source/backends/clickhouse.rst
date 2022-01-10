.. _install.clickhouse:

`Clickhouse <https://clickhouse.yandex/>`_
==========================================

Install
-------

Install dependencies for Ibis's Clickhouse dialect(minimal supported version is `0.1.3`):

::

  pip install 'ibis-framework[clickhouse]'

or

::

  conda install -c conda-forge ibis-clickhouse

Connect
-------

Create a client by passing in database connection parameters such as ``host``,
``port``, ``database``, and ``user`` to :func:`ibis.clickhouse.connect`:


.. code-block:: python

   con = ibis.clickhouse.connect(host='clickhouse', port=9000)

.. _api.clickhouse:

API
---
.. currentmodule:: ibis.backends.clickhouse

The ClickHouse client is accessible through the ``ibis.clickhouse`` namespace.

Use ``ibis.clickhouse.connect`` to create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   Backend.close
   Backend.exists_table
   Backend.exists_database
   Backend.get_schema
   Backend.list_databases
   Backend.list_tables
