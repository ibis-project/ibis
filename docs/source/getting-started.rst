.. _install:

********************************
Installation and Getting Started
********************************

Getting up and running with Ibis involves installing the Python package and
connecting to HDFS and Impala. If you don't have a Hadoop cluster available
with Impala, see :ref:`install.quickstart` below for instructions to use a VM
to get up and running quickly.

Installation
------------

System dependencies
~~~~~~~~~~~~~~~~~~~

Ibis requires a working Python 2.6, 2.7, or 3.4 installation. We recommend
`Anaconda <http://continuum.io/downloads>`_.

Installing the Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ibis using ``pip`` (or ``conda``, whenever it becomes available):

::

  pip install ibis-framework

This installs the ``ibis`` library to your configured Python environment.

Ibis can also be installed with Kerberos support for its HDFS functionality:

::

  pip install ibis-framework[kerberos]

Some platforms will require that you have Kerberos installed to build properly.

* Redhat / CentOS: ``yum install krb5-devel``
* Ubuntu / Debian: ``apt-get install libkrb5-dev``

.. _install.sqlite:

Ibis SQLite Quickstart
----------------------

See http://blog.ibis-project.org/sqlite-crunchbase-quickstart/ for a quickstart
using SQLite. Otherwise read on to try out Ibis on Impala.

Creating a client
-----------------

To create an Ibis "client", you must first connect your services and assemble
the client using ``ibis.impala.connect``:

.. code-block:: python

   import ibis

   hdfs = ibis.hdfs_connect(host=webhdfs_host, port=webhdfs_port)
   con = ibis.impala.connect(host=impala_host, port=impala_port,
                             hdfs_client=hdfs)

Both method calls can take ``auth_mechanism='GSSAPI'`` or
``auth_mechanism='LDAP'`` to connect to Kerberos clusters.  Depending on your
cluster setup, this may also include SSL. See the :ref:`API reference
<api.client>` for more, along with the Impala shell reference, as the
connection semantics are identical.

Learning resources
------------------

We are collecting IPython notebooks for learning here:
http://github.com/cloudera/ibis-notebooks. Some of these notebooks will be
reproduced as part of the documentation.

.. _install.quickstart:

Using Ibis with the Cloudera Quickstart VM
------------------------------------------

Using Ibis with Impala requires a running Impala cluster, so we have provided a
lean VirtualBox image to simplify the process for those looking to try out Ibis
(without setting up a cluster) or start contributing code to the project.

What follows are streamlined setup instructions for the VM. If you wish to
download it directly and setup from the ``ova`` file, use this `download link
<http://archive.cloudera.com/cloudera-ibis/ibis-demo.ova>`_.

The VM was built with Oracle VirtualBox 4.3.28.

TL;DR
~~~~~

::

    curl -s https://raw.githubusercontent.com/cloudera/ibis-notebooks/master/setup/bootstrap.sh | bash

Single Steps
~~~~~~~~~~~~

To use Ibis with the special Cloudera Quickstart VM follow the below
instructions:

  * Make sure Anaconda is installed. You can get it from
    http://continuum.io/downloads. Now prepend the Anaconda Python
    to your path like this ``export PATH=$ANACONDA_HOME/bin:$PATH``
  * ``pip install ibis-framework``
  * ``git clone https://github.com/cloudera/ibis-notebooks.git``
  * ``cd ibis-notebooks``
  * ``./setup/setup-ibis-demo-vm.sh``
  * ``source setup/ibis-env.sh``
  * ``ipython notebook``

VM setup
~~~~~~~~

The setup script will download a VirtualBox appliance image and import it in
VirtualBox. In addition, it will create a new host only network adapter with
DHCP. After the VM is started, it will extract the current IP address and add a
new /etc/hosts entry pointing from the IP of the VM to the hostname
``quickstart.cloudera``. The reason for this entry is that Hadoop and HDFS
require a working reverse name mapping. If you don't want to run the automated
steps make sure to check the individual steps in the file
``setup/setup-ibis-demo-vm.sh``.
