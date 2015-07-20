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

Ibis requires a working Python 2.6 or 2.7 installation (3.x support will come
in a future release). We recommend `Anaconda <http://continuum.io/downloads>`_.

Some platforms will require that you have Kerberos installed to build properly.

* Redhat / CentOS: ``yum install krb5-devel``
* Ubuntu / Debian: ``apt-get install libkrb5-dev``

Installing the Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ibis using ``pip`` (or ``conda``, whenever it becomes available):

::

  pip install ibis-framework

This installs the ``ibis`` library to your configured Python environment.

Creating a client
-----------------

To create an Ibis "client", you must first connect your services and assemble
the client using ``ibis.make_client``:

.. code-block:: python

   import ibis

   ic = ibis.impala_connect(host=impala_host, port=impala_port)
   hdfs = ibis.hdfs_connect(host=webhdfs_host, port=webhdfs_port)

   con = ibis.make_client(ic, hdfs_client=hdfs)

Depending on your cluster setup, this may be more complicated, especially if
LDAP or Kerberos is involved. See the :ref:`API reference <api.client>` for
more.

Learning resources
------------------

We are collecting IPython notebooks for learning here:
http://github.com/cloudera/ibis-notebooks. Some of these notebooks will be
reproduced as part of the documentation.

.. _install.quickstart:

Using Ibis with the Cloudera Quickstart VM
------------------------------------------

Since Ibis requires a running Impala cluster, we have provided a lean
VirtualBox image to simplify the process for those looking to try out Ibis
(without setting up a cluster) or start contributing code to the project.

TL;DR
~~~~~

::

    curl -s https://raw.githubusercontent.com/cloudera/ibis-notebooks/master/setup/bootstrap.sh | bash

Single Steps
~~~~~~~~~~~~

To use Ibis with the special Cloudera Quickstart VM follow the below
instructions:

  * Install Oracle VirtualBox
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
