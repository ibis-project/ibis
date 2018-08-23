.. currentmodule:: ibis
.. _api:

*************
API Reference
*************

.. currentmodule:: ibis

.. _api.client:

Creating connections
--------------------

These methods are in the ``ibis`` module namespace, and your main point of
entry to using Ibis.

.. autosummary::
   :toctree: generated/

   hdfs_connect

Impala client
-------------
.. currentmodule:: ibis.impala.api

These methods are available on the Impala client object after connecting to
your HDFS cluster (``ibis.hdfs_connect``) and connecting to Impala with
``ibis.impala.connect``.

.. autosummary::
   :toctree: generated/

   connect
   ImpalaClient.close
   ImpalaClient.database

Database methods
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImpalaClient.set_database
   ImpalaClient.create_database
   ImpalaClient.drop_database
   ImpalaClient.list_databases
   ImpalaClient.exists_database

.. autosummary::
   :toctree: generated/

   ImpalaDatabase.create_table
   ImpalaDatabase.drop
   ImpalaDatabase.namespace
   ImpalaDatabase.table

Table methods
~~~~~~~~~~~~~

The ``ImpalaClient`` object itself has many helper utility methods. You'll find
the most methods on ``ImpalaTable``.

.. autosummary::
   :toctree: generated/

   ImpalaClient.database
   ImpalaClient.table
   ImpalaClient.sql
   ImpalaClient.raw_sql
   ImpalaClient.list_tables
   ImpalaClient.exists_table
   ImpalaClient.drop_table
   ImpalaClient.create_table
   ImpalaClient.insert
   ImpalaClient.truncate_table
   ImpalaClient.get_schema
   ImpalaClient.cache_table
   ImpalaClient.load_data
   ImpalaClient.get_options
   ImpalaClient.set_options
   ImpalaClient.set_compression_codec


The best way to interact with a single table is through the ``ImpalaTable``
object you get back from ``ImpalaClient.table``.

.. autosummary::
   :toctree: generated/

   ImpalaTable.add_partition
   ImpalaTable.alter
   ImpalaTable.alter_partition
   ImpalaTable.column_stats
   ImpalaTable.compute_stats
   ImpalaTable.describe_formatted
   ImpalaTable.drop
   ImpalaTable.drop_partition
   ImpalaTable.files
   ImpalaTable.insert
   ImpalaTable.invalidate_metadata
   ImpalaTable.load_data
   ImpalaTable.metadata
   ImpalaTable.partition_schema
   ImpalaTable.partitions
   ImpalaTable.refresh
   ImpalaTable.rename
   ImpalaTable.stats

Creating views is also possible:

.. autosummary::
   :toctree: generated/

   ImpalaClient.create_view
   ImpalaClient.drop_view
   ImpalaClient.drop_table_or_view

Accessing data formats in HDFS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImpalaClient.avro_file
   ImpalaClient.delimited_file
   ImpalaClient.parquet_file

Executing expressions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImpalaClient.execute
   ImpalaClient.disable_codegen

.. _api.postgres:

PostgreSQL client
-----------------
.. currentmodule:: ibis.sql.postgres.api

The PostgreSQL client is accessible through the ``ibis.postgres`` namespace.

Use ``ibis.postgres.connect`` with a SQLAlchemy-compatible connection string to
create a client.

.. autosummary::
   :toctree: generated/

   connect
   PostgreSQLClient.database
   PostgreSQLClient.list_tables
   PostgreSQLClient.list_databases
   PostgreSQLClient.table

.. _api.sqlite:

SQLite client
-------------
.. currentmodule:: ibis.sql.sqlite.api

The SQLite client is accessible through the ``ibis.sqlite`` namespace.

Use ``ibis.sqlite.connect`` to create a SQLite client.

.. autosummary::
   :toctree: generated/

   connect
   SQLiteClient.attach
   SQLiteClient.database
   SQLiteClient.list_tables
   SQLiteClient.table

.. _api.hdfs:

HDFS
----

Client objects have an ``hdfs`` attribute you can use to interact directly with
HDFS.

.. currentmodule:: ibis

.. autosummary::
   :toctree: generated/

   HDFS.ls
   HDFS.chmod
   HDFS.chown
   HDFS.get
   HDFS.head
   HDFS.put
   HDFS.put_tarfile
   HDFS.rm
   HDFS.rmdir
   HDFS.size
   HDFS.status

Top-level expression APIs
-------------------------

These methods are available directly in the ``ibis`` module namespace.

.. autosummary::
   :toctree: generated/

   case
   literal
   schema
   table
   timestamp
   where
   ifelse
   coalesce
   greatest
   least
   negate
   desc
   now
   NA
   null
   expr_list
   row_number
   window
   range_window
   trailing_window
   cumulative_window
   trailing_range_window

.. _api.expr:

General expression methods
--------------------------

.. currentmodule:: ibis.expr.api

.. autosummary::
   :toctree: generated/

   Expr.compile
   Expr.equals
   Expr.execute
   Expr.pipe
   Expr.verify

.. _api.table:

Table methods
-------------

.. currentmodule:: ibis.expr.api

.. autosummary::
   :toctree: generated/

   TableExpr.add_column
   TableExpr.aggregate
   TableExpr.count
   TableExpr.distinct
   TableExpr.info
   TableExpr.filter
   TableExpr.get_column
   TableExpr.get_columns
   TableExpr.group_by
   TableExpr.groupby
   TableExpr.limit
   TableExpr.mutate
   TableExpr.projection
   TableExpr.relabel
   TableExpr.schema
   TableExpr.set_column
   TableExpr.sort_by
   TableExpr.union
   TableExpr.view

   TableExpr.join
   TableExpr.cross_join
   TableExpr.inner_join
   TableExpr.left_join
   TableExpr.outer_join
   TableExpr.semi_join
   TableExpr.anti_join


Grouped table methods
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   GroupedTableExpr.aggregate
   GroupedTableExpr.count
   GroupedTableExpr.having
   GroupedTableExpr.mutate
   GroupedTableExpr.order_by
   GroupedTableExpr.over
   GroupedTableExpr.projection
   GroupedTableExpr.size

Generic value methods
---------------------

.. _api.functions:

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ValueExpr.between
   ValueExpr.cast
   ValueExpr.coalesce
   ValueExpr.fillna
   ValueExpr.isin
   ValueExpr.notin
   ValueExpr.nullif
   ValueExpr.hash
   ValueExpr.isnull
   ValueExpr.notnull
   ValueExpr.over
   ValueExpr.typeof

   ValueExpr.case
   ValueExpr.cases
   ValueExpr.substitute

Column methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ColumnExpr.distinct

   ColumnExpr.count
   ColumnExpr.min
   ColumnExpr.max
   ColumnExpr.approx_median
   ColumnExpr.approx_nunique
   ColumnExpr.group_concat
   ColumnExpr.nunique
   ColumnExpr.summary

   ColumnExpr.value_counts

   ColumnExpr.first
   ColumnExpr.last
   ColumnExpr.dense_rank
   ColumnExpr.rank
   ColumnExpr.lag
   ColumnExpr.lead
   ColumnExpr.cummin
   ColumnExpr.cummax

General numeric methods
-----------------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericValue.abs
   NumericValue.ceil
   NumericValue.floor
   NumericValue.sign
   NumericValue.exp
   NumericValue.sqrt
   NumericValue.log
   NumericValue.ln
   NumericValue.log2
   NumericValue.log10
   NumericValue.round
   NumericValue.nullifzero
   NumericValue.zeroifnull
   NumericValue.add
   NumericValue.sub
   NumericValue.mul
   NumericValue.div
   NumericValue.pow
   NumericValue.rdiv
   NumericValue.rsub



Column methods
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericColumn.sum
   NumericColumn.mean

   NumericColumn.std
   NumericColumn.var

   NumericColumn.cumsum
   NumericColumn.cummean

   NumericColumn.bottomk
   NumericColumn.topk
   NumericColumn.bucket
   NumericColumn.histogram

Integer methods
---------------

Scalar or column methods
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   IntegerValue.convert_base
   IntegerValue.to_timestamp

.. _api.string:

String methods
--------------

All string operations are valid either on scalar or array values

.. autosummary::
   :toctree: generated/

   StringValue.convert_base
   StringValue.length
   StringValue.lower
   StringValue.upper
   StringValue.reverse
   StringValue.ascii_str
   StringValue.strip
   StringValue.lstrip
   StringValue.rstrip
   StringValue.capitalize
   StringValue.contains
   StringValue.like
   StringValue.to_timestamp
   StringValue.parse_url
   StringValue.substr
   StringValue.left
   StringValue.right
   StringValue.repeat
   StringValue.find
   StringValue.translate
   StringValue.find_in_set
   StringValue.join
   StringValue.replace
   StringValue.lpad
   StringValue.rpad

   StringValue.rlike
   StringValue.re_search
   StringValue.re_extract
   StringValue.re_replace

.. _api.timestamp:

Timestamp methods
-----------------

All timestamp operations are valid either on scalar or array values

.. autosummary::
   :toctree: generated/

   TimestampValue.strftime
   TimestampValue.year
   TimestampValue.month
   TimestampValue.day
   TimestampValue.day_of_week
   TimestampValue.hour
   TimestampValue.minute
   TimestampValue.second
   TimestampValue.millisecond
   TimestampValue.truncate
   TimestampValue.time
   TimestampValue.date
   TimestampValue.add
   TimestampValue.radd
   TimestampValue.sub
   TimestampValue.rsub

.. _api.date:

Date methods
------------

.. autosummary::
   :toctree: generated/

   DateValue.strftime
   DateValue.year
   DateValue.month
   DateValue.day
   DateValue.day_of_week
   DateValue.truncate
   DateValue.add
   DateValue.radd
   DateValue.sub
   DateValue.rsub

.. _api.dow:

Day of week methods
-------------------

.. currentmodule:: ibis.expr.types

.. autosummary::
   :toctree: generated/

   DayOfWeek.index
   DayOfWeek.full_name

.. currentmodule:: ibis.expr.api

.. _api.time:

Time methods
------------

.. autosummary::
   :toctree: generated/

   TimeValue.between
   TimeValue.truncate
   TimeValue.hour
   TimeValue.minute
   TimeValue.second
   TimeValue.millisecond
   TimeValue.add
   TimeValue.radd
   TimeValue.sub
   TimeValue.rsub

.. _api.interval:

Interval methods
----------------

.. autosummary::
   :toctree: generated/

   IntervalValue.to_unit
   IntervalValue.years
   IntervalValue.quarters
   IntervalValue.months
   IntervalValue.weeks
   IntervalValue.days
   IntervalValue.hours
   IntervalValue.minutes
   IntervalValue.seconds
   IntervalValue.milliseconds
   IntervalValue.microseconds
   IntervalValue.nanoseconds
   IntervalValue.add
   IntervalValue.radd
   IntervalValue.sub
   IntervalValue.mul
   IntervalValue.rmul
   IntervalValue.floordiv
   IntervalValue.negate


Boolean methods
---------------

.. autosummary::
   :toctree: generated/

   BooleanValue.ifelse


.. autosummary::
   :toctree: generated/

   BooleanColumn.any
   BooleanColumn.all
   BooleanColumn.cumany
   BooleanColumn.cumall

Category methods
----------------

Category is a logical type with either a known or unknown cardinality. Values
are represented semantically as integers starting at 0.

.. autosummary::
   :toctree: generated/

   CategoryValue.label

Decimal methods
---------------

.. autosummary::
   :toctree: generated/

   DecimalValue.precision
   DecimalValue.scale
