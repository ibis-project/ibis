.. _configuration:

****************
Configuring Ibis
****************

Ibis global configuration
-------------------------

Ibis global configuration happens through the ``ibis.options``
attribute. Attributes can be get and set like class attributes.

Interactive mode
~~~~~~~~~~~~~~~~

Ibis out of the box is in *developer mode*. Expressions display their internal
details when printed to the console. For a better interactive experience, set
the ``interactive option``:

.. code-block:: python

   ibis.options.interactive = True

This will cause expressions to be executed immediately when printed to the
console (or in IPython or the IPython notebook).

SQL query execution
~~~~~~~~~~~~~~~~~~~

If an Ibis table expression has no row limit set using the ``limit`` API, a
default one is applied to prevent too much data from being retrieved from the
query engine. The default is currently 10000 rows, but this can be configured
with the ``sql.default_limit`` option:

.. code-block:: python

   ibis.options.sql.default_limit = 100

Set this to ``None`` to retrieve all rows in all queries (be careful!).

.. code-block:: python

   ibis.options.sql.default_limit = None

Verbose option and Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~

To see all internal Ibis activity (like queries being executed) set
`ibis.options.verbose`:

.. code-block:: python

    ibis.options.verbose = True

By default this information is sent to ``sys.stdout``, but you can set some
other logging function:

.. code-block:: python

   def cowsay(x):
       print("Cow says: {0}".format(x))

   ibis.options.verbose_log = cowsay

Working with secure clusters (Kerberos)
---------------------------------------

Ibis is compatible with Hadoop clusters that are secured with Kerberos (as well
as SSL and LDAP).  Just like the Impala shell and ODBC/JDBC connectors, Ibis
connects to Impala through the HiveServer2 interface (using the impyla client).
Therefore, the connection semantics are similar to the other access methods for
working with secure clusters.

Specifically, after authenticating yourself against Kerberos (e.g., by issuing
the appropriate ``kinit`` commmand), simply pass ``auth_mechanism='GSSAPI'`` or
``auth_mechanism='LDAP'`` (and set ``kerberos_service_name`` if necessary along
with ``user`` and ``password`` if necessary) to the
``ibis.impala_connect(...)`` method when instantiating an ``ImpalaConnection``.
This method also takes arguments to configure SSL (``use_ssl``, ``ca_cert``).
See the documentation for the Impala shell for more details.

Ibis also includes functionality that communicates directly with HDFS, using
the WebHDFS REST API.  When calling ``ibis.hdfs_connect(...)``, also pass
``auth_mechanism='GSSAPI'`` or ``auth_mechanism='LDAP'``, and ensure that you
are connecting to the correct port, which may likely be an SSL-secured WebHDFS
port.  Also note that you can pass ``verify=False`` to avoid verifying SSL
certificates (which may be helpful in testing).  Ibis will assume ``https``
when connecting to a Kerberized cluster. Because some Ibis commands create HDFS
directories as well as new Impala databases and/or tables, your user will
require the necessary privileges.
