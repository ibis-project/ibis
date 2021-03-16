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

Ibis backends are accessed by users calling ``ibis.backend_name``, for example
in ``ibis.sqlite.connect(fname)``.

Both, when adding a new backend to the Ibis repo, or when creating a third-party
backend, you should define a entry point in the group ``ibis.backends``, with the
name of your backend and the module where it is implemented. This is defined in
the ``setup.py`` file. The code to setup the sqlite backend could be:

.. code-block:: python

   setup(name='ibis-sqlite',
         ...
         entry_points={'ibis.backends': 'sqlite = ibis-sqlite'}
   )

In the code above, the name of the module will be ``sqlite``, as defined in the
left of the assignment ``sqlite = ibis-sqlite``. And the code should be available
in the module ``ibis-sqlite`` (the file ``ibis-sqlite/__init__.py`` will define
the ``connect`` method, as well as any other method available to the users in
``ibis.sqlite.<method>``.

For third party packages it is recommended that the name of the Python package
is ``ibis-<backend>``, since Ibis will recommend users to run ``pip install ibis-<backend>``
when a backend is not found.


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
