.. _install.mysql:

`MySQL <https://www.mysql.com/>`_
=================================

Install dependencies for Ibis's MySQL dialect:

::

  pip install 'ibis-framework[mysql]'

Create a client by passing a connection string or individual parameters to
:func:`ibis.mysql.connect`:

.. code-block:: python

   con = ibis.mysql.connect(url='mysql+pymysql://ibis:ibis@mysql/ibis_testing')
   con = ibis.mysql.connect(
       user='ibis',
       password='ibis',
       host='mysql',
       database='ibis_testing',
   )

.. _api.mysql:

API
---
.. currentmodule:: ibis.backends.mysql

The MySQL client is accessible through the ``ibis.mysql`` namespace.

Use ``ibis.mysql.connect`` with a SQLAlchemy-compatible connection string to
create a client.

.. autosummary::
   :toctree: ../generated/

   Backend.connect
   Backend.database
   Backend.list_databases
   Backend.list_tables
   Backend.table
