.. currentmodule:: ibis.impala.api

.. _impala:

**********************
Using Ibis with Impala
**********************

One goal of Ibis is to provide an integrated Python API for an Impala cluster
without requiring you to switch back and forth between Python code and the
Impala shell (where one would be using a mix of DDL and SQL statements).

If you find an Impala task that you cannot perform with Ibis, please get in
touch on the `GitHub issue tracker <http://github.com/cloudera/ibis>`_.

While interoperability between the Hadoop / Spark ecosystems and pandas / the
PyData stack is overall poor (but improving), we also show some ways that you
can use pandas with Ibis and Impala.

.. ipython:: python
   :suppress:

   import ibis
   hdfs = ibis.hdfs_connect(port=5070)
   client = ibis.impala.connect(hdfs_client=hdfs)

The Impala client object
------------------------

To use Ibis with Impala, you first must connect to a cluster using the
``ibis.impala.connect`` function, *optionally* supplying an HDFS connection:

.. code-block:: python

   import ibis

   hdfs = ibis.hdfs_connect(host=webhdfs_host, port=webhdfs_port)
   client = ibis.impala.connect(host=impala_host, port=impala_port,
                                hdfs_client=hdfs)

You can accomplish many tasks directly through the client object, but we
additionally provide to streamline tasks involving a single Impala table or
database.

If you're doing analytics on a single table, you can get going by using the
``table`` method on the client:

.. code-block:: python

   table = client.table(table_name, database=db_name)

Database and Table objects
--------------------------

.. autosummary::
   :toctree: generated/

   ImpalaClient.database
   ImpalaClient.table

The client's ``table`` method allows you to create an Ibis table expression
referencing a physical Impala table:

.. ipython:: python

   table = client.table('functional_alltypes', database='ibis_testing')

While you can get by fine with only table and client objects, Ibis has a notion
of a "database object" that simplifies interactions with a single Impala
database. It also gives you IPython tab completion of table names (that are
valid Python variable names):

.. ipython:: python

   db = client.database('ibis_testing')
   db
   table = db.functional_alltypes
   db.list_tables()

So, these two lines of code are equivalent:

.. code-block:: python

   table1 = client.table(table_name, database=db)
   table2 = db.table(table_name)

``ImpalaTable`` is a Python subclass of the more general Ibis ``TableExpr``
that has additional Impala-specific methods. So you can use it interchangeably
with any code expecting a ``TableExpr``.

Like all table expressions in Ibis, ``ImpalaTable`` has a ``schema`` method you
can use to examine its schema:

.. autosummary::
   :toctree: generated/

   ImpalaTable.schema

While the client has a ``drop_table`` method you can use to drop tables, the
table itself has a method ``drop`` that you can use:

.. code-block:: python

   table.drop()

Expression execution and asynchronous queries
---------------------------------------------

Ibis expressions have an ``execute`` method with compiles and runs the
expressions on Impala or whichever backend is being referenced.

For example:

.. ipython:: python

   fa = db.functional_alltypes
   expr = fa.double_col.sum()
   expr.execute()

For longer-running queries, if you press Control-C (or whatever triggers the
Python ``KeyboardInterrupt`` on your system), Ibis will attempt to cancel the
query in progress.

As of Ibis 0.5.0, there is an explicit asynchronous API:

.. ipython:: python

   query = expr.execute(async=True)

With the returned ``AsyncQuery`` object, you have various methods available to
check on the status of the executing expression:

.. ipython:: python

   import time
   while not query.is_finished():
       time.sleep(1)
   query.is_finished()
   query.get_result()

If the query is still running, you can attempt to cancel it:

.. code-block:: python

   query.cancel()

Creating tables
---------------

There are several ways to create new Impala tables:

* From an Ibis table expression
* Empty, from a declared schema
* Empty and partitioned

In all cases, you should use the ``create_table`` method either on the
top-level client connection or a database object.

.. autosummary::
   :toctree: generated/

   ImpalaClient.create_table
   ImpalaDatabase.create_table

Creating tables from a table expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you pass an Ibis expression to ``create_table``, Ibis issues a ``CREATE
TABLE .. AS SELECT`` (CTAS) statement:

.. ipython:: python

   table = db.table('functional_alltypes')
   expr = table.group_by('string_col').size()
   db.create_table('string_freqs', expr, format='parquet')

   freqs = db.table('string_freqs')
   freqs.execute()

   files = freqs.files()
   files

   freqs.drop()

You can also choose to create an empty table and use ``insert`` (see below).

Creating an empty table
~~~~~~~~~~~~~~~~~~~~~~~

To create an empty table, you must declare an Ibis schema that will be
translated to the appopriate Impala schema and data types.

As Ibis types are simplified compared with Impala types, this may expand in the
future to include a more fine-grained schema declaration.

You can use the ``create_table`` method either on a database or client object.

.. code-block:: python

   schema = ibis.schema([('foo', 'string'),
                         ('year', 'int32'),
                         ('month', 'int16')])
   name = 'new_table'
   db.create_table(name, schema=schema)

By default, this stores the data files in the database default location. You
can force a particular path with the ``location`` option.

.. code-block:: python

   schema = ibis.schema([('foo', 'string'),
                         ('year', 'int32'),
                         ('month', 'int16')])
   name = 'new_table'
   location = '/home/wesm/new-table-data'
   db.create_table(name, schema=schema,
                   location=location)

If the schema matches a known table schema, you can always use the ``schema``
method to get a schema object:

.. ipython:: python

   t = db.table('functional_alltypes')
   t.schema()

Creating a partitioned table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create an empty partitioned table, include a list of columns to be used as
the partition keys.

.. code-block:: python

   schema = ibis.schema([('foo', 'string'),
                         ('year', 'int32'),
                         ('month', 'int16')])
   name = 'new_table'
   db.create_table(name, schema=schema, partition=['year', 'month'])

Partitioned tables
------------------

Ibis enables you to manage partitioned tables in various ways. Since each
partition behaves as its own "subtable" sharing a common schema, each partition
can have its own file format, directory path, serialization properties, and so
forth.

There are a handful of table methods for adding and removing partitions and
getting information about the partition schema and any existing partition data:

.. autosummary::
   :toctree: generated/

   ImpalaTable.add_partition
   ImpalaTable.drop_partition
   ImpalaTable.is_partitioned
   ImpalaTable.partition_schema
   ImpalaTable.partitions

For example:

.. ipython:: python

   ss = client.table('tpcds_parquet.store_sales')
   ss.is_partitioned
   ss.partitions()[:5]
   ss.partition_schema()

To address a specific partition in any method that is partition specific, you
can either use a dict with the partition key names and values, or pass a list
of the partition values:

.. code-block:: python

   schema = ibis.schema([('foo', 'string'),
                         ('year', 'int32'),
                         ('month', 'int16')])
   name = 'new_table'
   db.create_table(name, schema=schema, partition=['year', 'month'])

   table = db.table(name)

   table.add_partition({'year': 2007, 'month', 4})
   table.add_partition([2007, 5])
   table.add_partition([2007, 6])

   table.drop_partition([2007, 6])

We'll cover partition metadata management and data loading below.

Inserting data into tables
--------------------------

If the schemas are compatible, you can insert into a table directly from an
Ibis table expression:

.. ipython:: python

   t = db.functional_alltypes
   db.create_table('insert_test', schema=t.schema())
   target = db.table('insert_test')

   target.insert(t[:3])
   target.insert(t[:3])
   target.insert(t[:3])

   target.execute()

   target.drop()

If the table is partitioned, you must indicate the partition you are inserting
into:

.. code-block:: python

   part = {'year': 2007, 'month': 4}
   table.insert(expr, partition=part)

Managing table metadata
-----------------------

Ibis has functions that wrap many of the DDL commands for Impala table metadata.

Detailed table metadata: ``DESCRIBE FORMATTED``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get a handy wrangled version of ``DESCRIBE FORMATTED`` use the ``metadata``
method.

.. autosummary::
   :toctree: generated/

   ImpalaTable.metadata

The ``TableMetadata`` object that is returned has a nicer console output and
many attributes set that you can explore in IPython:

.. ipython:: python

   t = client.table('ibis_testing.functional_alltypes')
   meta = t.metadata()
   meta
   meta.location
   meta.create_time

The ``files`` function is also available to see all of the physical HDFS data
files backing a table:

.. autosummary::
   :toctree: generated/

   ImpalaTable.files

.. code-block:: ipython

    In [9]: ss = c.table('tpcds_parquet.store_sales')

    In [10]: ss.files()[:5]
    Out[10]:
                                                    path      size  \
    0  hdfs://localhost:20500/test-warehouse/tpcds.st...  160.61KB
    1  hdfs://localhost:20500/test-warehouse/tpcds.st...  123.88KB
    2  hdfs://localhost:20500/test-warehouse/tpcds.st...  139.28KB
    3  hdfs://localhost:20500/test-warehouse/tpcds.st...  139.60KB
    4  hdfs://localhost:20500/test-warehouse/tpcds.st...   62.84KB

                     partition
    0  ss_sold_date_sk=2451803
    1  ss_sold_date_sk=2451819
    2  ss_sold_date_sk=2451772
    3  ss_sold_date_sk=2451789
    4  ss_sold_date_sk=2451741

Modifying table metadata
~~~~~~~~~~~~~~~~~~~~~~~~

For unpartitioned tables, you can use the ``alter`` method to change its
location, file format, and other properties. For partitioned tables, to change
partition-specific metadata use ``alter_partition``.

.. autosummary::
   :toctree: generated/

   ImpalaTable.alter
   ImpalaTable.alter_partition

For example, if you wanted to "point" an existing table at a directory of CSV
files, you could run the following command:

.. code-block:: python

   csv_props = {
       'serialization.format': ',',
       'field.delim': ','
   }
   data_dir = '/home/wesm/my-csv-files'

   table.alter(location=data_dir, format='text',
               serde_properties=csv_props)

If the table is partitioned, you can modify only the properties of a particular
partition:

.. code-block:: python

   table.alter_partition({'year': 2007, 'month': 5},
                         location=data_dir, format='text',
                         serde_properties=csv_props)

Table statistics
----------------

Computing table and partition statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImpalaTable.compute_stats

Impala-backed physical tables have a method ``compute_stats`` that computes
table, column, and partition-level statistics to assist with query planning and
optimization. It is standard practice to invoke this after creating a table or
loading new data:

.. code-block:: python

   table.compute_stats()

If you are using a recent version of Impala, you can also access the ``COMPUTE
INCREMENTAL STATS`` DDL command:

.. code-block:: python

   table.compute_stats(incremental=True)

Seeing table and column statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   ImpalaTable.column_stats
   ImpalaTable.stats

The ``compute_stats`` and ``stats`` functions return the results of ``SHOW
COLUMN STATS`` and ``SHOW TABLE STATS``, respectively, and their output will
depend, of course, on the last ``COMPUTE STATS`` call.

.. code-block:: ipython

   In [5]: ss = c.table('tpcds_parquet.store_sales')

   In [6]: ss.compute_stats(incremental=True)

   In [7]: stats = ss.stats()
   s
   In [8]: stats[:5]
   Out[8]:
     ss_sold_date_sk  #Rows  #Files     Size Bytes Cached Cache Replication  \
   0         2450829   1071       1  78.34KB   NOT CACHED        NOT CACHED
   1         2450846    839       1  61.83KB   NOT CACHED        NOT CACHED
   2         2450860    747       1  54.86KB   NOT CACHED        NOT CACHED
   3         2450874    922       1  66.74KB   NOT CACHED        NOT CACHED
   4         2450888    856       1  63.33KB   NOT CACHED        NOT CACHED

       Format Incremental stats  \
   0  PARQUET              true
   1  PARQUET              true
   2  PARQUET              true
   3  PARQUET              true
   4  PARQUET              true

                                               Location
   0  hdfs://localhost:20500/test-warehouse/tpcds.st...
   1  hdfs://localhost:20500/test-warehouse/tpcds.st...
   2  hdfs://localhost:20500/test-warehouse/tpcds.st...
   3  hdfs://localhost:20500/test-warehouse/tpcds.st...
   4  hdfs://localhost:20500/test-warehouse/tpcds.st...

  In [9]: cstats = ss.column_stats()

  In [10]: cstats
  Out[10]:
                     Column          Type  #Distinct Values  #Nulls  Max Size  Avg Size
  0         ss_sold_time_sk        BIGINT             13879      -1       NaN         8
  1              ss_item_sk        BIGINT             17925      -1       NaN         8
  2          ss_customer_sk        BIGINT             15207      -1       NaN         8
  3             ss_cdemo_sk        BIGINT             16968      -1       NaN         8
  4             ss_hdemo_sk        BIGINT              6220      -1       NaN         8
  5              ss_addr_sk        BIGINT             14077      -1       NaN         8
  6             ss_store_sk        BIGINT                 6      -1       NaN         8
  7             ss_promo_sk        BIGINT               298      -1       NaN         8
  8        ss_ticket_number           INT             15006      -1       NaN         4
  9             ss_quantity           INT                99      -1       NaN         4
  10      ss_wholesale_cost  DECIMAL(7,2)             10196      -1       NaN         4
  11          ss_list_price  DECIMAL(7,2)             19393      -1       NaN         4
  12         ss_sales_price  DECIMAL(7,2)             15594      -1       NaN         4
  13    ss_ext_discount_amt  DECIMAL(7,2)             29772      -1       NaN         4
  14     ss_ext_sales_price  DECIMAL(7,2)            102758      -1       NaN         4
  15  ss_ext_wholesale_cost  DECIMAL(7,2)            125448      -1       NaN         4
  16      ss_ext_list_price  DECIMAL(7,2)            141419      -1       NaN         4
  17             ss_ext_tax  DECIMAL(7,2)             33837      -1       NaN         4
  18          ss_coupon_amt  DECIMAL(7,2)             29772      -1       NaN         4
  19            ss_net_paid  DECIMAL(7,2)            109981      -1       NaN         4
  20    ss_net_paid_inc_tax  DECIMAL(7,2)            132286      -1       NaN         4
  21          ss_net_profit  DECIMAL(7,2)            122436      -1       NaN         4
  22        ss_sold_date_sk        BIGINT               120       0       NaN         8


``REFRESH`` and ``INVALIDATE METADATA``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These DDL commands are available as table-level and client-level methods:

.. autosummary::
   :toctree: generated/

   ImpalaClient.invalidate_metadata
   ImpalaTable.invalidate_metadata
   ImpalaTable.refresh

You can invalidate the cached metadata for a single table or for all tables
using ``invalidate_metadata``, and similarly invoke ``REFRESH
db_name.table_name`` using the ``refresh`` method.

.. code-block:: python

   client.invalidate_metadata()

   table = db.table(table_name)
   table.invalidate_metadata()

   table.refresh()

These methods are often used in conjunction with the ``LOAD DATA`` commands and
``COMPUTE STATS``. See the Impala documentation for full details.

Issuing ``LOAD DATA`` commands
------------------------------

The ``LOAD DATA`` DDL physically moves a single data file or a directory of
files into the correct location for a table or table partition. It is
especially useful for partitioned tables as you do not have to construct the
directory path for a partition by hand, so simpler and less error-prone than
manually moving files with low level HDFS commands. It also deals with file
name conflicts so data is not lost in such cases.

.. autosummary::
   :toctree: generated/

   ImpalaClient.load_data
   ImpalaTable.load_data

To use these methods, pass the path of a single file or a directory of files
you want to load. Afterward, you may want to update the table statistics (see
Impala documentation):

.. code-block:: python

   table.load_data(path)
   table.refresh()

Like the other methods with support for partitioned tables, you can load into a
particular partition with the ``partition`` keyword argument:

.. code-block:: python

   part = [2007, 5]
   table.load_data(path, partition=part)

Parquet and other session options
---------------------------------

Ibis gives you access to Impala session-level variables that affect query
execution:

.. autosummary::
   :toctree: generated/

   ImpalaClient.disable_codegen
   ImpalaClient.get_options
   ImpalaClient.set_options
   ImpalaClient.set_compression_codec

For example:

.. ipython:: python

   client.get_options()

To enable Snappy compression for Parquet files, you could do either of:

.. ipython:: python

   client.set_options({'COMPRESSION_CODEC': 'snappy'})
   client.set_compression_codec('snappy')

   client.get_options()['COMPRESSION_CODEC']

Ingesting data from pandas
--------------------------

Overall interoperability between the Hadoop / Spark ecosystems and pandas / the
PyData stack is poor, but it will improve in time (this is a major part of the
Ibis roadmap).

Ibis's Impala tools currently interoperate with pandas in these ways:

* Ibis expressions return pandas objects (i.e. DataFrame or Series) for
  non-scalar expressions when calling their ``execute`` method
* The ``create_table`` and ``insert`` methods can accept pandas objects. This
  includes inserting into partitioned tables. It currently uses CSV as the
  ingest route.

For example:

.. code-block:: ipython

   In [2]: import pandas as pd

   In [3]: data = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd']})

   In [4]: db.create_table('pandas_table', data)

   In [5]: t = db.pandas_table

   In [6]: t.execute()
   Out[6]:
	 bar  foo
   0   a    1
   1   b    2
   2   c    3
   3   d    4

   In [7]: t.drop()

   In [8]: db.create_table('empty_for_insert', schema=t.schema())

   In [9]: to_insert = db.empty_for_insert

   In [10]: to_insert.insert(data)

   In [11]: to_insert.execute()
   Out[11]:
	 bar  foo
   0   a    1
   1   b    2
   2   c    3
   3   d    4

   In [12]: to_insert.drop()

.. .. ipython:: python

..    import pandas as pd
..    data = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd']})
..    db.create_table('pandas_table', data)
..    t = db.pandas_table
..    t.execute()
..    t.drop()
..    db.create_table('empty_for_insert', schema=t.schema())
..    to_insert = db.empty_for_insert
..    to_insert.insert(data)
..    to_insert.execute()
..    to_insert.drop()

Using Impala UDFs in Ibis
-------------------------

Impala currently supports user-defined scalar functions (known henceforth as
*UDFs*) and aggregate functions (respectively *UDAs*) via a C++ extension API.

Initial support for using C++ UDFs in Ibis came in version 0.4.0.

Using scalar functions (UDFs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's take an example to illustrate how to make a C++ UDF available to
Ibis. Here is a function that computes an approximate equality between floating
point values:

.. code-block:: c++

   #include "impala_udf/udf.h"

   #include <cctype>
   #include <cmath>

   BooleanVal FuzzyEquals(FunctionContext* ctx, const DoubleVal& x, const DoubleVal& y) {
     const double EPSILON = 0.000001f;
     if (x.is_null || y.is_null) return BooleanVal::null();
     double delta = fabs(x.val - y.val);
     return BooleanVal(delta < EPSILON);
   }

You can compile this to either a shared library (a ``.so`` file) or to LLVM
bitcode with clang (a ``.ll`` file). Skipping that step for now (will add some
more detailed instructions here later, promise).

To make this function callable, we use ``ibis.impala.wrap_udf``:

.. code-block:: python

   library = '/ibis/udfs/udftest.ll'
   inputs = ['double', 'double']
   output = 'boolean'
   symbol = 'FuzzyEquals'
   udf_db = 'ibis_testing'
   udf_name = 'fuzzy_equals'

   fuzzy_equals = ibis.impala.wrap_udf(library, inputs, output,
                                       symbol, name=udf_name)

In typical workflows, you will set up a UDF in Impala once then use it
thenceforth. So the *first time* you do this, you need to create the UDF in
Impala:

.. code-block:: python

   client.create_function(fuzzy_equals, database=udf_db)

Now, we must register this function as a new Impala operation in Ibis. This
must take place each time you load your Ibis session.

.. code-block:: python

   func.register(fuzzy_equals.name, udf_db)

The object ``fuzzy_equals`` is callable and works with Ibis expressions:

.. code-block:: python

   In [35]: db = c.database('ibis_testing')

   In [36]: t = db.functional_alltypes

   In [37]: expr = fuzzy_equals(t.float_col, t.double_col / 10)

   In [38]: expr.execute()[:10]
   Out[38]:
   0     True
   1    False
   2    False
   3    False
   4    False
   5    False
   6    False
   7    False
   8    False
   9    False
   Name: tmp, dtype: bool

Note that the call to ``register`` on the UDF object must happen each time you
use Ibis. If you have a lot of UDFs, I suggest you create a file with all of
your wrapper declarations and user APIs that you load with your Ibis session to
plug in all your own functions.

Using aggregate functions (UDAs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon.

Adding documentation to new functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fuzzy_equal.__doc__ = """\
   Approximate equals UDF

   Parameters
   ----------
   left : numeric
   right : numeric

   Returns
   -------
   is_approx_equal : boolean
   """

Adding UDF functions to Ibis types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon.

Installing the Impala UDF SDK on OS X and Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon.

Impala types to Ibis types
~~~~~~~~~~~~~~~~~~~~~~~~~~

Coming soon. See ``ibis.schema`` for now.
