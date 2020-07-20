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

