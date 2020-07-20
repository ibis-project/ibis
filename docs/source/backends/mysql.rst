.. _install.mysql:

`MySQL <https://www.mysql.com/>`_ Quickstart
--------------------------------------------

Install dependencies for Ibis's MySQL dialect:

::

  pip install ibis-framework[mysql]

Create a client by passing a connection string or individual parameters to
:func:`ibis.mysql.connect`:

.. code-block:: python

   con = ibis.mysql.connect(url='mysql+pymysql://ibis:ibis@mysql/ibis_testing')
   con = ibis.mysql.connect(
       user='ibis',
       password='ibis',
       host='mysql',
       database='ibis_testing',
   )
