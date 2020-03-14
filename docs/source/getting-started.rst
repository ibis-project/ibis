.. _install:

********************************
Installation and Getting Started
********************************

Installation
------------

System Dependencies
~~~~~~~~~~~~~~~~~~~

Ibis requires a working Python 3.6+ installation. We recommend using
`Anaconda <http://continuum.io/downloads>`_ to manage Python versions and
environments.

Installing the Python Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ibis using ``pip`` or ``conda``:

::

  pip install ibis-framework

This installs the ``ibis`` library to your configured Python environment.

Ibis can also be installed with Kerberos support for its HDFS functionality:

::

  pip install ibis-framework[kerberos]

Some platforms will require that you have Kerberos installed to build properly.

* Redhat / CentOS: ``yum install krb5-devel``
* Ubuntu / Debian: ``apt-get install libkrb5-dev``
* Arch Linux     : ``pacman -S krb5``

.. _install.impala:

`Impala <https://impala.apache.org/>`_ Quickstart
-------------------------------------------------

Install dependencies for Ibis's Impala dialect:

::

  pip install ibis-framework[impala]

To create an Ibis client, you must first connect your services and assemble the
client using :func:`ibis.impala.connect`:

.. ipython:: python

   import ibis

   hdfs = ibis.hdfs_connect(host='impala', port=50070)
   con = ibis.impala.connect(
       host='impala', database='ibis_testing', hdfs_client=hdfs
   )

Both method calls can take ``auth_mechanism='GSSAPI'`` or
``auth_mechanism='LDAP'`` to connect to Kerberos clusters.  Depending on your
cluster setup, this may also include SSL. See the :ref:`API reference
<api.client>` for more, along with the Impala shell reference, as the
connection semantics are identical.

.. _install.sqlite:

`SQLite <https://www.sqlite.org/>`_ Quickstart
----------------------------------------------

Install dependencies for Ibis's SQLite dialect:

::

  pip install ibis-framework[sqlite]

Create a client by passing a path to a SQLite database to
:func:`ibis.sqlite.connect`:

.. code-block:: python

   >>> import ibis
   >>> ibis.sqlite.connect('path/to/my/sqlite.db')

See http://blog.ibis-project.org/sqlite-crunchbase-quickstart/ for a quickstart
using SQLite.

.. _install.postgres:

`PostgreSQL <https://www.postgresql.org/>`_ Quickstart
------------------------------------------------------

Install dependencies for Ibis's PostgreSQL dialect:

::

  pip install ibis-framework[postgres]

Create a client by passing a connection string to the ``url`` parameter or
individual parameters to :func:`ibis.postgres.connect`:

.. ipython:: python

   con = ibis.postgres.connect(
       url='postgresql://postgres:postgres@postgres:5432/ibis_testing'
   )
   con = ibis.postgres.connect(
       user='postgres',
       password='postgres',
       host='postgres',
       port=5432,
       database='ibis_testing',
   )

.. _install.clickhouse:

`Clickhouse <https://clickhouse.yandex/>`_ Quickstart
-----------------------------------------------------

Install dependencies for Ibis's Clickhouse dialect(minimal supported version is `0.1.3`):

::

  pip install ibis-framework[clickhouse]

Create a client by passing in database connection parameters such as ``host``,
``port``, ``database``, and ``user`` to :func:`ibis.clickhouse.connect`:


.. ipython:: python

   con = ibis.clickhouse.connect(host='clickhouse', port=9000)

.. _install.bigquery:

`BigQuery <https://cloud.google.com/bigquery/>`_ Quickstart
-----------------------------------------------------------

Install dependencies for Ibis's BigQuery dialect:

::

  pip install ibis-framework[bigquery]

Create a client by passing in the project id and dataset id you wish to operate
with:


.. code-block:: python

   >>> con = ibis.bigquery.connect(project_id='ibis-gbq', dataset_id='testing')

By default ibis assumes that the BigQuery project that's billed for queries is
also the project where the data lives.

However, it's very easy to query data that does **not** live in the billing
project.

.. note::

   When you run queries against data from other projects **the billing project
   will still be billed for any and all queries**.

If you want to query data that lives in a different project than the billing
project you can use the :meth:`ibis.bigquery.client.BigQueryClient.database`
method of :class:`ibis.bigquery.client.BigQueryClient` objects:

.. code-block:: python

   >>> db = con.database('other-data-project.other-dataset')
   >>> t = db.my_awesome_table
   >>> t.sweet_column.sum().execute()  # runs against the billing project

`Pandas <https://pandas.pydata.org/>`_ Quickstart
-------------------------------------------------

Ibis's Pandas backend is available in core Ibis:

Create a client by supplying a dictionary of DataFrames using
:func:`ibis.pandas.connect`. The keys become the table names:

.. ipython:: python

   import pandas as pd
   con = ibis.pandas.connect(
       {
          'A': pd._testing.makeDataFrame(),
          'B': pd._testing.makeDataFrame(),
       }
   )

.. _install.omniscidb:

`omniscidb <https://www.omnisci.com/>`_ Quickstart
--------------------------------------------------

Install dependencies for Ibis's omniscidb dialect:

::

  pip install ibis-framework[omniscidb]

Create a client by passing in database connection parameters such as ``host``,
``port``, ``database``,  ``user`` and ``password`` to
:func:`ibis.omniscidb.connect`:

.. ipython:: python

   con = ibis.omniscidb.connect(
       host='omniscidb',
       database='ibis_testing',
       user='admin',
       password='HyperInteractive',
   )

.. _install.mysql:

`MySQL <https://www.mysql.com/>`_ Quickstart
--------------------------------------------

Install dependencies for Ibis's MySQL dialect:

::

  pip install ibis-framework[mysql]

Create a client by passing a connection string or individual parameters to
:func:`ibis.mysql.connect`:

.. ipython:: python

   con = ibis.mysql.connect(url='mysql+pymysql://ibis:ibis@mysql/ibis_testing')
   con = ibis.mysql.connect(
       user='ibis',
       password='ibis',
       host='mysql',
       database='ibis_testing',
   )

Learning Resources
------------------

We collect Jupyter notebooks for learning how to use ibis here:
https://github.com/ibis-project/ibis/tree/master/docs/source/notebooks/tutorial.
Some of these notebooks will be reproduced as part of the documentation
:ref:`in the tutorial section <tutorial>`.
