=============
Release Notes
=============

    **Note**: These release notes will only include notable or major bug fixes
    since most minor bug fixes tend to be esoteric and not generally
    interesting. Point (minor, e.g. 0.5.1) releases will generally not be found
    here and contain only bug fixes.

0.8 (May 19, 2016)
------------------

This release brings initial PostgreSQL backend support along with a number of
critical bug fixes and usability improvements. As several correctness bugs with
the SQL compiler were fixed, we recommend that all users upgrade from earlier
versions of Ibis.

New features
~~~~~~~~~~~~
* Initial PostgreSQL backend contributed by Philip Cloud.
* Add ``groupby`` as an alias for ``group_by`` to table expressions

Bug fixes
~~~~~~~~~
* Fix an expression error when filtering based on a new field
* Fix Impala's SQL compilation of using ``OR`` with compound filters
* Various fixes with the ``having(...)`` function in grouped table expressions
* Fix CTE (``WITH``) extraction inside ``UNION ALL`` expressions.
* Fix ``ImportError`` on Python 2 when ``mock`` library not installed

API changes
~~~~~~~~~~~
* The deprecated ``ibis.impala_connect`` and ``ibis.make_client`` APIs have
  been removed

0.7 (March 16, 2016)
--------------------

This release brings initial Kudu-Impala integration and improved Impala and
SQLite support, along with several critical bug fixes.

New features
~~~~~~~~~~~~
* Apache Kudu (incubating) integration for Impala users. See the `blog post <http://blog.ibis-project.org/kudu-impala-ibis>`_ for now. Will add some documentation here when possible.
* Add ``use_https`` option to ``ibis.hdfs_connect`` for WebHDFS connections in
  secure (Kerberized) clusters without SSL enabled.
* Correctly compile aggregate expressions involving multiple subqueries.

To explain this last point in more detail, suppose you had:

.. code-block:: python

   table = ibis.table([('flag', 'string'),
                       ('value', 'double')],
                      'tbl')

   flagged = table[table.flag == '1']
   unflagged = table[table.flag == '0']

   fv = flagged.value
   uv = unflagged.value

   expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())

The last expression now generates the correct Impala or SQLite SQL:

.. code-block:: sql

   SELECT t0.`tmp` - t1.`tmp` AS `tmp`
   FROM (
     SELECT avg(`value`) / sum(`value`) AS `tmp`
     FROM tbl
     WHERE `flag` = '1'
   ) t0
     CROSS JOIN (
       SELECT avg(`value`) / sum(`value`) AS `tmp`
       FROM tbl
       WHERE `flag` = '0'
     ) t1

Bug fixes
~~~~~~~~~
* ``CHAR(n)`` and ``VARCHAR(n)`` Impala types now correctly map to Ibis string
  expressions
* Fix inappropriate projection-join-filter expression rewrites resulting in
  incorrect generated SQL.
* ``ImpalaClient.create_table`` correctly passes ``STORED AS PARQUET`` for
  ``format='parquet'``.
* Fixed several issues with Ibis dependencies (impyla, thriftpy, sasl,
  thrift_sasl), especially for secure clusters. Upgrading will pull in these
  new dependencies.
* Do not fail in ``ibis.impala.connect`` when trying to create the temporary
  Ibis database if no HDFS connection passed.
* Fix join predicate evaluation bug when column names overlap with table
  attributes.
* Fix handling of fully-materialized joins (aka ``select *`` joins) in
  SQLAlchemy / SQLite.

Contributors
~~~~~~~~~~~~
Thank you to all who contributed patches to this release.

::

  $ git log v0.6.0..v0.7.0 --pretty=format:%aN | sort | uniq -c | sort -rn
      21 Wes McKinney
       1 Uri Laserson
       1 Kristopher Overholt

0.6 (December 1, 2015)
----------------------

This release brings expanded pandas and Impala integration, including support
for managing partitioned tables in Impala. See the new :ref:`Ibis for Impala
Users <impala>` guide for more on using Ibis with Impala.

The :ref:`Ibis for SQL Programmers <sql>` guide also was written since the 0.5
release.

This release also includes bug fixes affecting generated SQL correctness. All
users should upgrade as soon as possible.

New features
~~~~~~~~~~~~

* New integrated Impala functionality. See :ref:`Ibis for Impala Users
  <impala>` for more details on these things.

  * Improved Impala-pandas integration. Create tables or insert into existing
    tables from pandas ``DataFrame`` objects.
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

API Changes
~~~~~~~~~~~

* ``ImpalaTable.rename`` no longer mutates the calling table expression.

Contributors
~~~~~~~~~~~~

::

    $ git log v0.5.0..v0.6.0 --pretty=format:%aN | sort | uniq -c | sort -rn
    46 Wes McKinney
     3 Uri Laserson
     1 Phillip Cloud
     1 mariusvniekerk
     1 Kristopher Overholt


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
