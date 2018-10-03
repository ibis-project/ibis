.. _udf:

User Defined Functions
======================

Ibis provides a mechanism for writing custom scalar and aggregate functions,
with varying levels of support for different backends. UDFs/UDAFs are a complex
topic.

This section of the documentation will discuss some of the backend specific
details of user defined functions.

.. warning::

   The UDF API is provisional and subject to change.

.. _udf.pandas:

Pandas
------
Ibis supports defining three kinds of user-defined functions for operations on
expressions targeting the pandas backend: **element-wise**, **reduction**, and
**analytic**.

.. _udf.elementwise:

Element-wise Functions
~~~~~~~~~~~~~~~~~~~~~~
An **element-wise** function is a function that takes N rows as input and
produces N rows of output. ``log``, ``exp``, and ``floor`` are examples of
element-wise functions.

Here's how to define an element-wise function:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.pandas import udf

   @udf.elementwise(input_type=[dt.int64], output_type=.dtdouble)
   def add_one(x):
       return x + 1.0

.. _udf.reduction:

Reduction Functions
~~~~~~~~~~~~~~~~~~~
A **reduction** is a function that takes N rows as input and produces 1 row
as output. ``sum``, ``mean`` and ``count`` are examples of reductions. In
the context of a ``GROUP BY``, reductions produce 1 row of output *per
group*.

Here's how to define a reduction function:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.pandas import udf

   @udf.reduction(input_type=[dt.double], output_type=.dtdouble)
   def double_mean(series):
       return 2 * series.mean()

.. _udf.analytic:

Analytic Functions
~~~~~~~~~~~~~~~~~~
An **analytic** function is like an **element-wise** function in that it
takes N rows as input and produces N rows of output. The key difference is
that analytic functions can be applied *per group* using window functions.
Z-score is an example of an analytic function.

Here's how to define an analytic function:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.pandas import udf

   @udf.analytic(input_type=[dt.double], output_type=.dtdouble)
   def zscore(series):
       return (series - series.mean()) / series.std()

Details of Pandas UDFs
~~~~~~~~~~~~~~~~~~~~~~
- :ref:`Element-wise functions <udf.element>` automatically provide support for
  applying your UDF to any combination of scalar values and columns.
- :ref:`Reduction functions <udf.reduction>` automatically provide support for
  whole column aggregations, grouped aggregations, and application of your
  function over a window.
- :ref:`Analytic functions <udf.analytic>` work in both grouped and non-grouped
  settings
- The objects you receive as input arguments are either ``pandas.Series`` or
  Python/NumPy scalars.

.. note::

   Any keyword arguments must be given a default value or the function **will
   not work**.

   A common Python convention is to set the default value to ``None`` and
   handle setting it to something not ``None`` in the body of the function.

Using ``add_one`` from above as an example, the following call will receive a
``pandas.Series`` for the ``x`` argument:

.. code-block:: python

   >>> import ibis
   >>> import pandas as pd
   >>> df = pd.DataFrame({'a': [1, 2, 3]})
   >>> con = ibis.pandas.connect({'df': df})
   >>> t = con.table('df')
   >>> expr = add_one(t.a)

And this will receive the ``int`` 1:

.. code-block:: python

   >>> expr = add_one(1)

Since the pandas backend passes around ``**kwargs`` you can accept ``**kwargs``
in your function:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.pandas import udf

   @udf.elementwise([dt.int64], dt.double)
   def add_two(x, **kwargs):
       # do stuff with kwargs
       return x + 2.0

Or you can leave them out as we did in the example above. You can also
optionally accept specific keyword arguments.

For example:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.pandas import udf

   @udf.elementwise([dt.int64], dt.double)
   def add_two_with_none(x, y=None):
       if y is None:
           y = 2.0
       return x + y

BigQuery
--------

.. _udf.bigquery:

.. note::

   BigQuery only supports element-wise UDFs at this time.

BigQuery supports UDFs through JavaScript. Ibis provides support for this by
turning Python code into JavaScript.

The interface is very similar to the pandas UDF API:

.. code-block:: python

   import ibis.expr.datatypes as dt
   from ibis.bigquery import udf

   @udf([dt.double], dt.double)
   def my_bigquery_add_one(x):
       return x + 1.0

Ibis will parse the source of the function and turn the resulting Python AST
into JavaScript source code (technically, ECMAScript 2015). Most of the Python
language is supported including classes, functions and generators.

If you want to inspect the generated code you can look at the ``js`` property
of the function.

.. code-block:: python

   >>> print(my_bigquery_add_one.js)
   CREATE TEMPORARY FUNCTION my_bigquery_add_one(x FLOAT64)
   RETURNS FLOAT64
   LANGUAGE js AS """
   'use strict';
   function my_bigquery_add_one(x) {
       return (x + 1.0);
   }
   return my_bigquery_add_one(x);
   """;

When you want to use this function you call it like any other Python
function--only on an ibis expression:

.. code-block:: python

   >>> import ibis
   >>> t = ibis.table([('a', 'double')])
   >>> expr = my_bigquery_add_one(t.a)
   >>> print(ibis.bigquery.compile(expr))
   CREATE TEMPORARY FUNCTION my_bigquery_add_one(x FLOAT64)
   RETURNS FLOAT64
   LANGUAGE js AS """
   'use strict';
   function my_bigquery_add_one(x) {
       return (x + 1.0);
   }
   return my_bigquery_add_one(x);
   """;

   SELECT my_bigquery_add_one(`a`) AS `tmp`
   FROM t0

SQLite
------

.. _udf.sqlite:

TODO
