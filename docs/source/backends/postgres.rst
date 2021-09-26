.. _install.postgres:

`PostgreSQL <https://www.postgresql.org/>`_
===========================================

Install dependencies for Ibis's PostgreSQL dialect:

::

  pip install 'ibis-framework[postgres]'

Create a client by passing a connection string to the ``url`` parameter or
individual parameters to :func:`ibis.postgres.connect`:

.. code-block:: python

   con = ibis.postgres.connect(
       url='postgresql://postgres:postgres@postgres:5432/ibis_testing'
   )
   con = ibis.postgres.connect(
       user='postgres',
       password='postgres',
       host='postgres',
       port=5432,
       database='ibis_testing',
   )

.. _api.postgres:

API
---
.. currentmodule:: ibis.backends.postgres

The PostgreSQL client is accessible through the ``ibis.postgres`` namespace.

Use ``ibis.postgres.connect`` with a SQLAlchemy-compatible connection string to
create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   Backend.database
   Backend.list_tables
   Backend.list_databases
   Backend.table
