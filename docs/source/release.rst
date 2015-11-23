=============
Release Notes
=============

    **Note**: These release notes will only include notable or major bug fixes
    since most minor bug fixes tend to be esoteric and not generally
    interesting. Point (minor, e.g. 0.5.1) releases will generally not be found
    here and contain only bug fixes.

0.6 (November 25, 2015)
------------------------

This release brings expanded pandas and Impala integration, including support
for managing partitioned tables in Impala. See the new :ref:`Ibis for Impala
Users <impala>` guide for more on using Ibis with Impala.

The :ref:`Ibis for SQL Programmers <sql>` guide also was written since the 0.5
release.

This release also includes bug fixes affecting generated SQL correctness. All
users should upgrade as soon as possible.

New features
~~~~~~~~~~~~
* Improved Impala-pandas integration. Create tables or insert into existing
  tables from pandas ``DataFrame`` objects.

* New integrated Impala functionality. See :ref:`Ibis for Impala Users
  <impala>` for more details on these things.

  * Partitioned table metadata management API. Add, drop, alter, and
    insert into table partitions.
  * Add ``is_partitioned`` property to ``ImpalaTable``.
  * Added support for ``LOAD DATA`` DDL using the ``load_data`` function, also
    supporting partitioned tables.
  * Modify table metadata (location, format, SerDe properties etc.)  using
    ``ImpalaTable.alter``
  * Interrupting Impala expression execution with Control-C will attempt to
    cancel the running query with the server.
  * Set the compression codec (e.g. snappy) used with
    ``ImpalaClient.set_compression_codec``.
  * Get and set query options for a client session with
    ``ImpalaClient.get_options`` and ``ImpalaClient.set_options``.
  * Add ``ImpalaTable.metadata`` method that parses the output of the
    ``DESCRIBE FORMATTED`` DDL to simplify table metadata inspection.
  * Add ``ImpalaTable.stats`` and ``ImpalaTable.column_stats`` to see computed
    table and partition statistics.
  * Add ``CHAR`` and ``VARCHAR`` handling
  * Add ``refresh``, ``invalidate_metadata`` DDL options and add
    ``incremental`` option to ``compute_stats`` for ``COMPUTE INCREMENTAL
    STATS``.

* Add ``substitute`` method for performing multiple value substitutions in an
  array or scalar expression.
* Division is by default *true division* like Python 3 for all numeric
  data. This means for SQL systems that use C-style division semantics, the
  appropriate ``CAST`` will be automatically inserted in the generated SQL.
* Easier joins on tables with overlapping column names. See :ref:`Ibis for SQL Programmers <sql>`.
* Expressions like ``string_expr[:3]`` now work as expected.
* Add ``coalesce`` instance method to all value expressions.
* Passing ``limit=None`` to the ``execute`` method on expressions disables any
  default row limits.


Contributors
~~~~~~~~~~~~

::

    $ git log v0.5.0..v0.6.0 --pretty=format:%aN | sort | uniq -c | sort -rn

0.5 (September 10, 2015)
------------------------

Highlights in this release are the SQLite, Python 3, Impala UDA support, and an
asynchronous execution API. There are also many usability improvements, bug
fixes, and other new features.

New features
~~~~~~~~~~~~
* SQLite client and built-in function support
* Ibis now supports Python 3.4 as well as 2.6 and 2.7
* Ibis can utilize Impala user-defined aggregate (UDA) functions
* SQLAlchemy-based translation toolchain to enable more SQL engines having
  SQLAlchemy dialects to be supported
* Many window function usability improvements (nested analytic functions and
  deferred binding conveniences)
* More convenient aggregation with keyword arguments in ``aggregate`` functions
* Built preliminary wrapper API for MADLib-on-Impala
* Add ``var`` and ``std`` aggregation methods and support in Impala
* Add ``nullifzero`` numeric method for all SQL engines
* Add ``rename`` method to Impala tables (for renaming tables in the Hive
  metastore)
* Add ``close`` method to ``ImpalaClient`` for session cleanup (#533)
* Add ``relabel`` method to table expressions
* Add ``insert`` method to Impala tables
* Add ``compile`` and ``verify`` methods to all expressions to test compilation
  and ability to compile (since many operations are unavailable in SQLite, for
  example)

API changes
~~~~~~~~~~~
* Impala Ibis client creation now uses only ``ibis.impala.connect``, and
  ``ibis.make_client`` has been deprecated

Contributors
~~~~~~~~~~~~
::

    $ git log v0.4.0..v0.5.0 --pretty=format:%aN | sort | uniq -c | sort -rn
          55 Wes McKinney
          9 Uri Laserson
          1 Kristopher Overholt

0.4 (August 14, 2015)
---------------------

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

Contributors
~~~~~~~~~~~~
::

    $ git log v0.3.0..v0.4.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         38 Wes McKinney
          9 Uri Laserson
          2 Meghana Vuyyuru
          2 Kristopher Overholt
          1 Marius van Niekerk

0.3 (July 20, 2015)
-------------------

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

Contributors
~~~~~~~~~~~~
::

    $ git log v0.2.0..v0.3.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         59 Wes McKinney
         29 Uri Laserson
          4 Isaac Hodes
          2 Meghana Vuyyuru

0.2 (June 16, 2015)
-------------------

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

Contributors
~~~~~~~~~~~~
::

    $ git log v0.1.0..v0.2.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         71 Wes McKinney
          1 Juliet Hougland
          1 Isaac Hodes

0.1 (March 26, 2015)
--------------------

First Ibis release.

* Expression DSL design and type system
* Expression to ImpalaSQL compiler toolchain
* Impala built-in function wrappers

::

    $ git log 84d0435..v0.1.0 --pretty=format:%aN | sort | uniq -c | sort -rn
        78 Wes McKinney
         1 srus
         1 Henry Robinson
