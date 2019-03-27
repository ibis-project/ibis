.. _extending:


Extending Ibis
==============

Users typically want to extend ibis in one of two ways:

#. Add a new expression
#. Add a new backend


Below we provide notebooks showing how to extend ibis in each of these ways.


Adding a New Expression
-----------------------

.. note::

   Make sure you've run the following commands before executing the notebook

   .. code-block:: sh

      docker-compose up -d --no-build postgres dns
      docker-compose run waiter
      docker-compose run ibis ci/load-data.sh postgres

Here we show how to add a ``sha1`` method to the PostgreSQL backend as well as
how to add a new ``bitwise_and`` reduction operation:

.. toctree::
   :maxdepth: 1

   notebooks/tutorial/9-Adding-a-new-elementwise-expression.ipynb
   notebooks/tutorial/10-Adding-a-new-reduction-expression.ipynb


Adding a New Backend
--------------------

TBD
