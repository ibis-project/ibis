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
      # including building the documentation
      conda env create --name ibis36 --file=ci/requirements-docs-3.6.yml

      # Activate the conda environment
      source activate ibis36

      # Install ibis
      make develop

   *Note: `make develop` command also install a `pre-commit` Git hook.*


All-in-One Command
------------------

We use `docker-compose <https://docs.docker.com/compose/>`_ for
ibis development to make it easy for developers to test ibis
against databases that have traditionally requied a lot of setup,
such as Impala.

The following command does three steps:

#. Downloads the test data
#. Starts each backend as a service via docker-compose
#. Initializes the backends with the test tables

   .. code:: sh

      make init

Take a peek at the Makefile to see what else is available.

Download Test Datasets
----------------------

This step isn't necessary, but can sometimes be helpful if you
want to investigate something outside of the docker-compose setup
that ships with ibis.

#. **Download the test data**:

   By default this will download and extract the dataset under
   testing/ibis-testing-data.

   .. code:: sh

      ci/datamgr.py download


Setting Up Test Databases
-------------------------

To start every backend as a service using ``docker-compose`` and
load test datasets into each backend use this command:

   .. code:: sh

      make init

The one backend that ibis supports that can't be started as a
service running in a docker container is BigQuery.

Read the next section for details on how to get setup with
BigQuery and ibis.

BigQuery
^^^^^^^^

Before you begin, you must have a `Google Cloud Platform project
<https://cloud.google.com/docs/overview/#projects>`_ with billing set up and
the `BigQuery API enabled
<https://console.cloud.google.com/flows/enableapi?apiid=bigquery>`_.

#. **Set up application default credentials by following the `getting started with
   GCP authentication guide
   <https://cloud.google.com/docs/authentication/getting-started>`_.**

#. **Set the ``GOOGLE_BIGQUERY_PROJECT_ID`` environment variable**:

   .. code:: sh

      export GOOGLE_BIGQUERY_PROJECT_ID=your-project-id

#. **Load data into BigQuery**:

   .. code:: sh

      ci/datamgr.py bigquery

Running Tests
-------------

You are now ready to run the full ibis test suite:

   .. code:: sh

      make test

Contribution Ideas
==================

Here's a few ideas to think about outside of participating in the primary
development roadmap:

* Documentation
* Use cases and IPython notebooks
* Other SQL-based backends (Presto, Hive, Spark SQL)
* S3 filesytem support
* Integration with MLLib via PySpark
