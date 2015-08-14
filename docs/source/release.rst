=============
Release Notes
=============

0.4.0 (August 14, 2015)
-----------------------

New features
~~~~~~~~~~~~
* Add tooling to use Impala C++ scalar UDFs within Ibis (#262, #195)
* Support and testing for Kerberos-enabled secure HDFS clusters
* Many table functions can now accept functions as parameters (invoked on the
  calling table) to enhance composability and emulate late-binding semantics of
  languages (like R) that have non-standard evaluation (#460)
* Add ``any``, ``all``, ``notany``, and ``notall`` reductions on boolean
  arrays, as well as ``cumany`` and ``cumall``
* Using ``topk`` now produces an analytic expression that is executable (as an
  aggregation) but can also be used as a filter as before (#392, #91)
* Added experimental database object "usability layer", see
  ``ImpalaClient.database``.
* Add ``TableExpr.info``
* Add ``compute_stats`` API to table expressions referencing physical Impala
  tables
* Add ``explain`` method to ``ImpalaClient`` to show query plan for an
  expression
* Add ``chmod`` and ``chown`` APIs to ``HDFS`` interface for superusers
* Add ``convert_base`` method to strings and integer types
* Add option to ``ImpalaClient.create_table`` to create empty partitioned
  tables
* ``ibis.cross_join`` can now join more than 2 tables at once
* Add ``ImpalaClient.raw_sql`` method for running naked SQL queries
* ``ImpalaClient.insert`` now validates schemas locally prior to sending query
  to cluster, for better usability.
* Add conda installation recipes

0.3.0 (July 20, 2015)
---------------------

First public release. See http://ibis-project.org for more.

New features
~~~~~~~~~~~~
* Implement window / analytic function support
* Enable non-equijoins (join clauses with operations other than ``==``).
* Add remaining :ref:`string functions <api.string>` supported by Impala.
* Add ``pipe`` method to tables (hat-tip to the pandas dev team).
* Add ``mutate`` convenience method to tables.
* Fleshed out ``WebHDFS`` implementations: get/put directories, move files,
  etc. See the :ref:`full HDFS API <api.hdfs>`.
* Add ``truncate`` method for timestamp values
* ``ImpalaClient`` can execute scalar expressions not involving any table.
* Can also create internal Impala tables with a specific HDFS path.
* Make Ibis's temporary Impala database and HDFS paths configurable (see
  ``ibis.options``).
* Add ``truncate_table`` function to client (if the user's Impala cluster
  supports it).
* Python 2.6 compatibility
* Enable Ibis to execute concurrent queries in multithreaded applications
  (earlier versions were not thread-safe).
* Test data load script in ``scripts/load_test_data.py``
* Add an internal operation type signature API to enhance developer
  productivity.

0.2.0 (June 16, 2015)
---------------------

New features
~~~~~~~~~~~~
* ``insert`` method on Ibis client for inserting data into existing tables.
* ``parquet_file``, ``delimited_file``, and ``avro_file`` client methods for
  querying datasets not yet available in Impala
* New ``ibis.hdfs_connect`` method and ``HDFS`` client API for WebHDFS for
  writing files and directories to HDFS
* New timedelta API and improved timestamp data support
* New ``bucket`` and ``histogram`` methods on numeric expressions
* New ``category`` logical datatype for handling bucketed data, among other
  things
* Add ``summary`` API to numeric expressions
* Add ``value_counts`` convenience API to array expressions
* New string methods ``like``, ``rlike``, and ``contains`` for fuzzy and regex
  searching
* Add ``options.verbose`` option and configurable ``options.verbose_log``
  callback function for improved query logging and visibility
* Support for new SQL built-in functions

  * ``ibis.coalesce``
  * ``ibis.greatest`` and ``ibis.least``
  * ``ibis.where`` for conditional logic (see also ``ibis.case`` and
    ``ibis.cases``)
  * ``nullif`` method on value expressions
  * ``ibis.now``

* New aggregate functions: ``approx_median``, ``approx_nunique``, and
  ``group_concat``
* ``where`` argument in aggregate functions
* Add ``having`` method to ``group_by`` intermediate object
* Added group-by convenience
  ``table.group_by(exprs).COLUMN_NAME.agg_function()``
* Add default expression names to most aggregate functions
* New Impala database client helper methods

  * ``create_database``
  * ``drop_database``
  * ``exists_database``
  * ``list_databases``
  * ``set_database``

* Client ``list_tables`` searching / listing method
* Add ``add``, ``sub``, and other explicit arithmetic methods to value
  expressions

API Changes
~~~~~~~~~~~
* New Ibis client and Impala connection workflow. Client now combined from an
  Impala connection and an optional HDFS connection

Bug fixes
~~~~~~~~~
* Numerous expression API bug fixes and rough edges fixed

0.1.0 (March 26, 2015)
----------------------
