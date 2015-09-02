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
your Impala cluster, HDFS cluster, and creating the client with
``ibis.make_client``.

Use ``ibis.impala.connect`` to create an Impala connection to use for
assembling a client.

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

   Database.drop
   Database.namespace
   Database.table

Table methods
~~~~~~~~~~~~~
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

.. autosummary::
   :toctree: generated/

   ImpalaTable.compute_stats
   ImpalaTable.drop
   ImpalaTable.insert
   ImpalaTable.rename

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
   trailing_window
   cumulative_window

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

Scalar or array methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ValueExpr.between
   ValueExpr.cast
   ValueExpr.fillna
   ValueExpr.isin
   ValueExpr.notin
   ValueExpr.nullif
   ValueExpr.hash
   ValueExpr.isnull
   ValueExpr.notnull
   ValueExpr.over
   ValueExpr.typeof

   ValueExpr.add
   ValueExpr.sub
   ValueExpr.mul
   ValueExpr.div
   ValueExpr.pow
   ValueExpr.rdiv
   ValueExpr.rsub

Array methods
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ArrayExpr.case
   ArrayExpr.cases
   ArrayExpr.distinct

   ArrayExpr.count
   ArrayExpr.min
   ArrayExpr.max
   ArrayExpr.approx_median
   ArrayExpr.approx_nunique
   ArrayExpr.group_concat
   ArrayExpr.nunique
   ArrayExpr.summary

   ArrayExpr.value_counts

   ArrayExpr.first
   ArrayExpr.last
   ArrayExpr.dense_rank
   ArrayExpr.rank
   ArrayExpr.lag
   ArrayExpr.lead
   ArrayExpr.cummin
   ArrayExpr.cummax

General numeric methods
-----------------------

Scalar or array methods
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericValue.abs
   NumericValue.ceil
   NumericValue.floor
   NumericValue.sign
   NumericValue.exp


Array methods
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NumericArray.sum
   NumericArray.mean

   NumericArray.cumsum
   NumericArray.cummean

   NumericArray.bottomk
   NumericArray.topk
   NumericArray.bucket
   NumericArray.histogram

Integer methods
---------------

Scalar or array methods
~~~~~~~~~~~~~~~~~~~~~~~

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
   StringValue.parse_url
   StringValue.substr
   StringValue.left
   StringValue.right
   StringValue.repeat
   StringValue.find
   StringValue.translate
   StringValue.find_in_set
   StringValue.join
   StringValue.lpad
   StringValue.rpad

   StringValue.rlike
   StringValue.re_search
   StringValue.re_extract
   StringValue.re_replace


Timestamp methods
-----------------

All timestamp operations are valid either on scalar or array values

.. autosummary::
   :toctree: generated/

   TimestampValue.truncate
   TimestampValue.year
   TimestampValue.month
   TimestampValue.day
   TimestampValue.hour
   TimestampValue.minute
   TimestampValue.second
   TimestampValue.millisecond

Boolean methods
---------------

.. autosummary::
   :toctree: generated/

   BooleanValue.ifelse


.. autosummary::
   :toctree: generated/

   BooleanArray.any
   BooleanArray.all
   BooleanArray.cumany
   BooleanArray.cumall

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
