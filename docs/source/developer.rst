.. _develop:

***********************************
Developing and Contributing to Ibis
***********************************

For a primer on general open source contributions, see the `pandas contribution
guide <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_. The
project will be run much like pandas has been.

Test environment setup
----------------------

If you do not have access to an Impala cluster, you may wish to set up the test
virtual machine. We've set up a Quickstart VM to get you up and running faster,
:ref:`see here <install.quickstart>`.

Unit tests and integration tests that use Impala require a test data load. See
``scripts/load_test_data.py`` in the source repository for the data loading
script.

Contribution Ideas
------------------

Here's a few ideas to think about outside of participating in the primary
development roadmap:

* Documentation
* Use cases and IPython notebooks
* Other SQL-based backends (Presto, Hive, Spark SQL, PostgreSQL)
* S3 filesytem support
* Integration with MLLib via PySpark

Contributor License Agreements
------------------------------

While Ibis is an Apache-licensed open source project, we require individual and
corporate contributors to execute a `contributor license agreement
<https://en.wikipedia.org/wiki/Contributor_License_Agreement>`_ to enable any
copyright issues to be avoided and to protect the user base from
disruption. This agreement only needs to be signed once.

We'll use the same CLA's that Impala uses:

* `Individual CLA <https://github.com/cloudera/Impala/wiki/Individual-Contributor-License-Agreement-(ICLA)>`_
* `Corporate CLA <https://github.com/cloudera/Impala/wiki/Corporate-Contributor-License-Agreement-(CCLA)>`_
