.. _configuration:

****************
Configuring Ibis
****************


Working with secure clusters (Kerberos)
---------------------------------------

Ibis is compatible with Hadoop clusters that are secured with Kerberos (as well
as SSL and LDAP).  Just like the Impala shell and ODBC/JDBC connectors, Ibis
connects to Impala through the HiveServer2 interface (using the impyla client).
Therefore, the connection semantics are similar to the other access methods for
working with secure clusters.

Specifically, after authenticating yourself against Kerberos (e.g., by issuing
the appropriate ``kinit`` commmand), simply pass ``use_kerberos=True`` (and set
``kerberos_service_name`` if necessary) to the ``ibis.impala_connect(...)``
method when instantiating an ``ImpalaConnection``.  This method also takes
arguments to configure LDAP (``use_ldap``, ``ldap_user``, and
``ldap_password``) and SSL (``use_ssl``, ``ca_cert``).  See the documentation
for the Impala shell for more details.

Ibis also includes functionality that communicates directly with HDFS, using
the WebHDFS REST API.  When calling ``ibis.hdfs_connect(...)``, also pass
``use_kerberos=True``, and ensure that you are connecting to the correct port,
which may likely be an SSL-secured WebHDFS port.  Also note that you can pass
``verify=False`` to avoid verifying SSL certificates (which may be helpful in
testing).  Ibis will assume ``https`` when connecting to a Kerberized cluster.
Because some Ibis commands create HDFS directories as well as new Impala
databases and/or tables, your user will require the necessary privileges.
