.. _udf:

User Defined Functions
======================

Ibis provides a mechanism for writing custom scalar and aggregate functions,
with varying levels of support for different backends. UDFs/UDAFs are a complex
topic.

This section of the documentation will discuss some of the backend specific
details of user defined functions.

API
---

.. warning::

   The UDF/UDAF API is quite experimental at this point and is therefore
   provisional and subject to change.

Going forward, the API for user defined *scalar* functions will look like this:

.. code-block:: python

   @udf(input_type=[double], output_type=double)
   def add_one(x):
       return x + 1.0


User defined *aggregate* functions are nearly identical, with the exception
of using the ``@udaf`` decorator instead of the ``@udf`` decorator.

Impala
------

TODO

Pandas
------

Pandas supports defining both UDFs and UDAFs.

When you define a UDF you automatically get support for applying that UDF in a
scalar context, *as well as* in any group by operation.

When you define a UDAF you automatically get support for standard scalar
aggregations, group bys, *as well as* any supported windowing operation.

The API for these functions is the same as described above.

The objects you receive as input arguments are either ``pandas.Series`` or
python or numpy scalars depending on the operation.

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

Finally, since the pandas backend passes around ``**kwargs`` you can accept
``**kwargs`` in your function:

.. code-block:: python

   @udf([double], double)
   def add_one(x, **kwargs):
       return x + 1.0

Or you can leave them out as we did in the example above. You can also
optionally accept *specific* keyword arguments. This requires knowledge of how
the pandas backend works for it to be useful:

.. note::

   Any keyword arguments (other than ``**kwargs``) must be given a default
   value or the UDF/UDAF **will not work**. A standard Python convention is to
   set the default value to ``None``.

For example:

.. code-block:: python

   @udf([double], double)
   def add_one(x, scope=None):
       return x + 1.0

BigQuery
--------

TODO

SQLite
------

TODO
