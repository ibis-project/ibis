.. currentmodule:: ibis
.. _impala:

*********************
Ibis for Impala users
*********************

Another goal of Ibis is to provide an integrated Python API for an Impala
cluster without requiring you to switch back and forth between Python code and
the Impala shell (where one would be using a mix of DDL and SQL statements).

Table metadata
--------------

Computing table statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Impala-backed physical tables have a method ``compute_stats`` that computes
table, column, and partition-level statistics to assist with query planning and
optimization. It is good practice to invoke this after creating a table or
loading new data:

.. code-block:: python

   table.compute_stats()

Table partition management
--------------------------

Coming soon
