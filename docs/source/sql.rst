.. currentmodule:: ibis
.. _sql:

************************
Ibis for SQL Programmers
************************

Ibis is intended to provide a full-featured replacement for SQL ``SELECT``
queries, but expressed with Python code that is:

* Easier to write. Pythonic function calls with tab completion in IPython.
* Type-checked and validated as you go. No more debugging cryptic database
  errors; Ibis catches your mistakes right away.
* More composable. Break complex queries down into easier-to-digest pieces
* Easier to reuse. Mix and match Ibis snippets to create expressions tailored
  for your analysis.

We intend for all ``SELECT`` queries to be fully portable to Ibis.

This document will use the Impala SQL compiler (i.e. ``ibis.impala.compile``)
for convenience, but the code here is portable to whichever system you are
using Ibis with.

.. ipython:: python
   :suppress:

   import ibis
   ibis.options.sql.default_limit = None

Projections: select/add/remove columns
--------------------------------------

All tables in Ibis are immutable. To select a subset of a table's columns, or
to add new columns, you must produce a new table by means of a *projection*.

.. ipython:: python

   t = ibis.table([('one', 'string'),
                   ('two', 'double'),
                   ('three', 'int32')], 'my_data')
   t

In SQL, you might write something like:

.. code-block:: sql

   SELECT two, one
   FROM my_data

In Ibis, this is

.. ipython:: python

   proj = t['two', 'one']

or

.. ipython:: python

   proj = t.projection(['two', 'one'])

This generates the expected SQL:

.. ipython:: python

   print(ibis.impala.compile(proj))

What about adding new columns? To form a valid projection, all column
expressions must be **named**. Let's look at the SQL:

.. code-block:: sql

   SELECT two, one, three * 2 AS new_col
   FROM my_data

The last expression is written:

.. ipython:: python

   new_col = (t.three * 2).name('new_col')

Now, we have:

.. ipython:: python

   proj = t['two', 'one', new_col]
   print(ibis.impala.compile(proj))

``mutate``: Add or modify columns easily
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since adding new columns or modifying existing columns is so common, there is a
convenience method ``mutate``:

.. ipython:: python

   mutated = t.mutate(new_col=t.three * 2)

Notice that using the ``name`` was not necessary here because we're using
Python keywords to provide the name. Indeed:

.. ipython:: python

   print(ibis.impala.compile(mutated))

If you modify an existing column with ``mutate`` it will list out all the other
columns:

.. ipython:: python

   mutated = t.mutate(two=t.two * 2)
   print(ibis.impala.compile(mutated))

``SELECT *`` equivalent
~~~~~~~~~~~~~~~~~~~~~~~

Especially in combination with relational joins, it's convenient to be able to
select all columns in a table using the ``SELECT *`` construct. To do this, use
the table expression itself in a projection:

.. ipython:: python

   proj = t[t]
   print(ibis.impala.compile(proj))

This is how ``mutate`` is implemented. The example above
``t.mutate(new_col=t.three * 2)`` can be written as a normal projection:

.. ipython:: python

   proj = t[t, new_col]
   print(ibis.impala.compile(proj))

Let's consider a table we might wish to join with ``t``:

.. ipython:: python

   t2 = ibis.table([('key', 'string'),
                    ('value', 'double')], 'dim_table')

Now let's take the SQL:

.. code-block:: sql

   SELECT t0.*, t0.two - t1.value AS diff
   FROM my_data t0
     INNER JOIN dim_table t1
       ON t0.one = t1.key

To write this with Ibis, it is:

.. ipython:: python

   diff = (t.two - t2.value).name('diff')
   joined = t.join(t2, t.one == t2.key)[t, diff]

And verify the generated SQL:

.. ipython:: python

   print(ibis.impala.compile(joined))

Using functions in projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you pass a function instead of a string or Ibis expression in any projection
context, it will be invoked with the "parent" table as its argument. This can
help significantly when `composing complex operations
<http://blog.ibis-project.org/design-composability/>`_. Consider this SQL:

.. code-block:: sql

   SELECT one, avg(abs(the_sum)) AS mad
   FROM (
     SELECT one, three, sum(two) AS the_sum
     FROM my_data
     GROUP BY 1, 2
   ) t0
   GROUP BY 1

This can be written as one chained expression:

.. ipython:: python

   expr = (t.group_by(['one', 'three'])
           .aggregate(the_sum=t.two.sum())
           .group_by('one')
           .aggregate(mad=lambda x: x.the_sum.abs().mean()))

Indeed:

.. ipython:: python

   print(ibis.impala.compile(expr))

A useful pattern you can try is that of the *function factory* which allows you
to create function that reference a field of interest:

.. code-block:: python

   def mad(field):
       def closure(table):
           return table[field].abs().mean()
       return closure


.. ipython:: python
   :suppress:

   def mad(field):
       def closure(table):
           return table[field].abs().mean()
       return closure

Now you can do:

.. ipython:: python

   expr = (t.group_by(['one', 'three'])
           .aggregate(the_sum=t.two.sum())
           .group_by('one')
           .aggregate(mad=mad('the_sum')))

Filtering / ``WHERE``
---------------------

You can add filter clauses to a table expression either by indexing with ``[]``
(like pandas) or use the ``filter`` method:

.. ipython:: python

   filtered = t[t.two > 0]
   print(ibis.impala.compile(filtered))

``filter`` can take a list of expressions, which must all be satisfied for a
row to be included in the result:

.. ipython:: python

   filtered = t.filter([t.two > 0,
                        t.one.isin(['A', 'B'])])
   print(ibis.impala.compile(filtered))

To compose boolean expressions with ``AND`` or ``OR``, use the respective ``&``
and ``|`` operators:

.. ipython:: python

   cond = (t.two < 0) | ((t.two > 0) | t.one.isin(['A', 'B']))
   filtered = t[cond]
   print(ibis.impala.compile(filtered))

Aggregation / ``GROUP BY``
--------------------------

To aggregate a table, you need:

* Zero or more grouping expressions (these can be column names)
* One or more aggregation expressions

Let's look at the ``aggregate`` method on tables:

.. ipython:: python

   stats = [t.two.sum().name('total_two'),
            t.three.mean().name('avg_three')]
   agged = t.aggregate(stats)

If you don't use any group expressions, the result will have a single row with
your statistics of interest:

.. ipython:: python

   agged.schema()

   print(ibis.impala.compile(agged))

To add groupings, use either the ``by`` argument of ``aggregate`` or use the
``group_by`` construct:

.. ipython:: python

   agged2 = t.aggregate(stats, by='one')
   agged3 = t.group_by('one').aggregate(stats)

   print(ibis.impala.compile(agged3))

Non-trivial grouping keys
~~~~~~~~~~~~~~~~~~~~~~~~~

You can use any expression (or function, like in projections) deriving from the
table you are aggregating. The only constraint is that the expressions must be
named. Let's look at an example:

.. ipython:: python

   events = ibis.table([('ts', 'timestamp'),
                        ('event_type', 'int32'),
                        ('session_id', 'int64')],
                        'web_events')

Suppose we wanted to total up event types by year and month:

.. ipython:: python

   keys = [events.ts.year().name('year'),
           events.ts.month().name('month')]

   sessions = events.session_id.nunique()
   stats = (events.group_by(keys)
            .aggregate(total=events.count(),
                       sessions=sessions))

Now we have:

.. ipython:: python

   print(ibis.impala.compile(stats))

``count(*)`` convenience: ``size()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computing group frequencies is so common that, like pandas, we have a method
``size`` that is a shortcut for the ``count(*)`` idiom:

.. ipython:: python

   freqs = events.group_by(keys).size()
   print(ibis.impala.compile(freqs))

Frequency table convenience: ``value_counts``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the SQL idiom:

.. code-block:: sql

   SELECT {{ COLUMN_EXPR }}, count(*)
   FROM table
   GROUP BY 1

This is so common that, like pandas, there is a generic array method
``value_counts`` which does this for us:

.. ipython:: python

   expr = events.ts.year().value_counts()
   print(ibis.impala.compile(expr))

``HAVING`` clause
~~~~~~~~~~~~~~~~~

The SQL ``HAVING`` clause enables you to filter the results of an aggregation
based on some group-wise condition holding true. For example, suppose we wanted
to limit our analysis to groups containing at least 1000 observations:

.. code-block:: sql

   SELECT one, sum(two) AS total
   FROM my_data
   GROUP BY 1
   HAVING count(*) >= 1000

With Ibis, you can do:

.. ipython:: python

   expr = (t.group_by('one')
           .having(t.count() >= 1000)
           .aggregate(t.two.sum().name('total')))

   print(ibis.impala.compile(expr))

``LIMIT`` and ``OFFSET``
------------------------

Sorting
-------

Joins
-----

Ways to specify join keys
~~~~~~~~~~~~~~~~~~~~~~~~~

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

Subqueries
----------

Comparison with scalar aggregates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional aggregates
~~~~~~~~~~~~~~~~~~~~~~

Correlated ``EXISTS`` / ``NOT EXISTS`` filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``IN`` / ``NOT IN`` filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DISTINCT`` expressions
------------------------

Window functions
----------------

Date / time data
----------------

Timedeltas
~~~~~~~~~~

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

Unions
------

Esoterica
---------
