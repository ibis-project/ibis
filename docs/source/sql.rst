.. currentmodule:: ibis
.. _sql:

***********************
Ibis for SQL Developers
***********************

Ibis is intended to provide a full-featured replacement for SQL ``SELECT``
queries, but expressed with Python code that is:

* Easier to write. Pythonic function calls with tab completion in IPython.
* More composable. Break complex queries down into easier-to-digest pieces
* Easier to reuse. Mix and match Ibis snippets to create expressions tailored
  for your analysis.
* Validates correctness as you go. No more debugging weird query analysis
  errors; Ibis catches your mistakes right away.

This document

.. ipython:: python
   :suppress:

   import ibis

Projections: adding/removing columns
------------------------------------

Aggregation / ``GROUP BY``
--------------------------

Sorting
-------

``LIMIT`` and ``OFFSET``
------------------------

Joins
-----

Join + aggregation
~~~~~~~~~~~~~~~~~~

Join + projection
~~~~~~~~~~~~~~~~~

Join with ``SELECT *``
~~~~~~~~~~~~~~~~~~~~~~

Self joins
~~~~~~~~~~

Column expressions
------------------

Type casts
~~~~~~~~~~

Conditional aggregates
~~~~~~~~~~~~~~~~~~~~~~

Case statements
~~~~~~~~~~~~~~~

``IN`` / ``NOT IN``
~~~~~~~~~~~~~~~~~~~

Constant columns
~~~~~~~~~~~~~~~~

``BETWEEN``
~~~~~~~~~~~

``DISTINCT`` expressions
------------------------

Window functions
----------------

Date / time data
----------------

Timedeltas
~~~~~~~~~~

Correlated / uncorrelated subqueries
------------------------------------

Comparison with scalar aggregates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional aggregates
~~~~~~~~~~~~~~~~~~~~~~

``EXISTS`` / ``NOT EXISTS`` filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IN`` / ``NOT IN`` filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common table expressions (CTEs)
-------------------------------

The simplest SQL CTE is a SQL statement that is used multiple times in a
``SELECT`` query, which can be "factored" out using the ``WITH`` keyword:

.. code-block:: sql

   WITH t0 AS (
      SELECT region, kind, sum(amount) AS total
      FROM purchases
      GROUP BY 1, 2
   )
   SELECT t0.region, t0.total - t1.total
   FROM t0
     INNER JOIN t0 t1
       ON t0.region = t1.region
   WHERE t0.kind = 'foo' AND t1.kind = 'bar'

Explicit CTEs are not necessary with Ibis. Let's look at an example involving
joining an aggregated table on itself after filtering:

.. ipython:: python

   purchases = ibis.table([('region', 'string'),
                           ('kind', 'string'),
                           ('user', 'int64'),
                           ('amount', 'double')], 'purchases')

   metric = purchases.amount.sum().name('total')
   agged = (purchases.group_by(['region', 'kind'])
            .aggregate(metric))

   left = agged[agged.kind == 'foo']
   right = agged[agged.kind == 'bar']

   result = (left.join(right, left.region == right.region)
             [left.region,
              (left.total - right.total).name('diff')])

Ibis automatically creates a CTE for ``agged``:

.. ipython:: python

   print(ibis.impala.compile(result))

``HAVING`` clause
-----------------

Unions
------
