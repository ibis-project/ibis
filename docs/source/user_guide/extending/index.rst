.. _extending:


Extending Ibis
==============

Users typically want to extend ibis in one of two ways:

#. Add a new expression
#. Add a new backend


Below we provide notebooks showing how to extend ibis in each of these ways.


Adding a New Expression
-----------------------

.. note::

   Make sure you've run the following commands before executing the notebook

   .. code-block:: sh

      docker-compose up -d --no-build postgres dns
      docker-compose run waiter
      docker-compose run ibis ci/load-data.sh postgres

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
.. note::
   By following the steps below, you get the opportunity to run tests with one
   command: `make test BACKEND='[your added backend]'`

1) you need to add a new backend to `BACKENDS` variable in `Makefile`.

2) if backend needs to start services (implemented as docker containers and
   added into `docker-compose.yml` file) then add the services to `SERVICES`
   variable in `Makefile`, add case for switch-case construction inside
   `./ci/dockerize.sh` for proper waiting the services.

3) if backend needs to load some data then add the backend to `LOADS` variable
   in `Makefile` and implement necessary functionality in `./ci/load-data.sh`

4) the necessary markers for `pytest` will be generated inside
   `./ci/backends-markers.sh`. By default, a marker will be generated that
   matches the name of the backend (you can manually correct the generated
   name for the marker inside the file)
