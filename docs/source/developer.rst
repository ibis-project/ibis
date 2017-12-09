.. _develop:

***********************************
Developing and Contributing to Ibis
***********************************

For a primer on general open source contributions, see the `pandas contribution
guide <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_. The
project will be run much like pandas has been.

Linux Test Environment Setup
============================

Conda Environment Setup
-----------------------

#. **Install the latest version of miniconda**:

   .. code:: sh

      # Download the miniconda bash installer
      curl -Ls -o $HOME/miniconda.sh \
          https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

      # Run the installer
      bash $HOME/miniconda.sh -b -p $HOME/miniconda

      # Put the conda command on your PATH
      export PATH="$HOME/miniconda/bin:$PATH"

#. **Install the development environment of your choice (Python 3.6 in this
   example), activate and install ibis in development mode**:

   .. code:: sh

      # Create a conda environment ready for ibis development
      conda env create --name ibis36 --file=ci/requirements_dev-3.6.yml

      # Activate the conda environment
      source activate ibis36

      # Install ibis
      python setup.py develop


All-in-One Command
------------------

The following command does three steps:

#. Downloads the test data
#. Starts each backend via docker-compose
#. Initializes the backends with the test tables

   .. code:: sh

      cd testing
      bash start-all.sh

To use specific backends follow the instructions below.


Download Test Dataset
---------------------

#. `Install docker <https://docs.docker.com/engine/installation/>`_
#. **Download the test data**:

   By default this will download and extract the dataset under
   testing/ibis-testing-data.

   .. code:: sh

      testing/datamgr.py download


Setting Up Test Databases
-------------------------

   .. code:: sh

      cd testing
      docker-compose up


Impala (with UDFs)
^^^^^^^^^^^^^^^^^^

#. **Start the Impala docker image in another terminal**:

   .. code:: sh

      # Keeping this running as long as you want to test ibis
      docker run --tty --rm --hostname impala cpcloud86/impala:java8

#. **Load data and UDFs into impala**:

   .. code:: sh

      testing/impalamgr.py load --data --data-dir ibis-testing-data

Clickhouse
^^^^^^^^^^

#. **Start the Clickhouse Server docker image in another terminal**:

   .. code:: sh

      # Keeping this running as long as you want to test ibis
      docker run --rm -p 9000:9000 --tty yandex/clickhouse-server

#. **Load data**:

   .. code:: sh

      testing/datamgr.py clickhouse

PostgreSQL
^^^^^^^^^^

PostgreSQL can be used from either the installation that resides on the Impala
docker image or from your machine directly.

Here's how to load test data into PostgreSQL:

   .. code:: sh

      testing/datamgr.py postgres

SQLite
^^^^^^

SQLite comes already installed on many systems. If you used the conda setup
instructions above, then SQLite will be available in the conda environment.

   .. code:: sh

      testing/datamgr.py sqlite


Running Tests
-------------

You are now ready to run the full ibis test suite:

   .. code:: sh

      pytest ibis

Contribution Ideas
==================

Here's a few ideas to think about outside of participating in the primary
development roadmap:

* Documentation
* Use cases and IPython notebooks
* Other SQL-based backends (Presto, Hive, Spark SQL)
* S3 filesytem support
* Integration with MLLib via PySpark
