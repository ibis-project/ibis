.. _extending:


Extending Ibis
==============

Users typically want to extend ibis in one of two ways:

#. Add a new expression
#. Add a new backend


Below we provide notebooks showing how to extend ibis in each of these ways.


Adding a New Expression
-----------------------

Here we show how to add a ``sha1`` method to the PostgreSQL backend as well as
how to add a new ``bitwise_and`` reduction operation:

.. toctree::
   :maxdepth: 1

   extending_elementwise_expr.ipynb
   extending_reduce_expr.ipynb


Adding a New Backend
--------------------

Run test suite for separate Backend
-----------------------------------

To run the tests for specific backends you can use:

.. code:: shell

    PYTEST_BACKENDS="sqlite pandas" python -m pytest ibis/tests

Some backends may require a database server running. The CI file
`.github/workflows/main.yml` contains the configuration to run
servers for all backends using docker images.

The backends may need data to be loaded, run or check `ci/setup.py` to
see how it is loaded in the CI, and loaded for your local containers.
