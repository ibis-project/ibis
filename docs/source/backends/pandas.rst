`pandas <https://pandas.pydata.org/>`_
======================================

Ibis's pandas backend is available in core Ibis:

Create a client by supplying a dictionary of DataFrames using
:func:`ibis.pandas.connect`. The keys become the table names:

.. code-block:: python

   import pandas as pd
   con = ibis.pandas.connect(
       {
          'A': pd._testing.makeDataFrame(),
          'B': pd._testing.makeDataFrame(),
       }
   )

.. _udf.pandas:

User Defined functions (UDF)
----------------------------

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

   @udf.elementwise(input_type=[dt.int64], output_type=dt.double)
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

   @udf.reduction(input_type=[dt.double], output_type=dt.double)
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

   @udf.analytic(input_type=[dt.double], output_type=dt.double)
   def zscore(series):
       return (series - series.mean()) / series.std()

Details of Pandas UDFs
~~~~~~~~~~~~~~~~~~~~~~
- :ref:`Element-wise functions <udf.elementwise>` automatically provide support
  for applying your UDF to any combination of scalar values and columns.
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

   import ibis
   import pandas as pd
   df = pd.DataFrame({'a': [1, 2, 3]})
   con = ibis.pandas.connect({'df': df})
   t = con.table('df')
   expr = add_one(t.a)
   expr

And this will receive the ``int`` 1:

.. code-block:: python

   expr = add_one(1)
   expr

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
