`Pandas <https://pandas.pydata.org/>`_ Quickstart
-------------------------------------------------

Ibis's Pandas backend is available in core Ibis:

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
