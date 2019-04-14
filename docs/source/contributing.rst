.. _contrib:

********************
Contributing to Ibis
********************

.. _contrib.running_tests:

Clone the Repository
--------------------
To contribute to ibis you need to clone the repository from GitHub:

.. code-block:: sh

   git clone https://github.com/ibis-project/ibis

Set Up a Development Environment
--------------------------------
#. `Install miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
#. Create a conda environment suitable for ibis development:

   .. code-block:: sh

      conda env create -n ibis-dev --file ci/requirements-3.7-dev.yml

#. Activate the environment

   .. code-block:: sh

      conda activate ibis-dev

Run the Test Suite
------------------

Contributor `Krisztián Szűcs <https://github.com/kszucs>`_ has spent many hours
crafting an easy-to-use `docker-compose <https://docs.docker.com/compose/>`_
setup that enables ibis developers to get up and running quickly.

Here are the steps to start database services and run the test suite:

.. code-block:: sh

   make --directory ibis init
   make --directory ibis testparallel
