.. _contrib:

********************
Contributing to Ibis
********************

.. note::

   Make sure you've read the :ref:`installation section <install>` of the docs
   before continuing.

.. _contrib.running_tests:

Running the Test Suite
----------------------

Contributor `Krisztián Szűcs <https://github.com/kszucs>`_ has spent many hours
crafting an easy-to-use `docker-compose <https://docs.docker.com/compose/>`_
setup that enables ibis developers to get up and running quickly.

Here are the steps to run clone the repo and run the test suite:

.. code-block:: sh

   # clone ibis
   git clone https://github.com/ibis-project/ibis

   # go to where the docker-compose file is
   pushd ibis/ci

   # start services, build ibis, and load data into databases
   ./build.sh

   # optionally run all tests
   ./test.sh -m 'not udf' -n auto -o cache_dir=/tmp/.pytest_cache
