---
backend_name: Impala
backend_url: https://impala.apache.org/
backend_module: impala
backend_param_style: connection parameters
intro: |
  One goal of Ibis is to provide an integrated Python API for an Impala cluster
  without requiring you to switch back and forth between Python code and the
  Impala shell.
exclude_backend_api: true
---

{% include 'backends/template.md' %}

Both method calls can take `auth_mechanism='GSSAPI'` or `auth_mechanism='LDAP'`
to connect to Kerberos clusters. Depending on your cluster setup, this may also
include SSL. See the `API reference` for more, along with the Impala shell
reference, as the connection semantics are identical.

These methods are available on the Impala client object after connecting to
your HDFS cluster (`ibis.impala.hdfs_connect`) and connecting to Impala with
`ibis.impala.connect`. See `backends.impala` for a tutorial on using this
backend.

## Database methods

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - create_database
        - drop_database
        - list_databases
        - exists_database

<!-- prettier-ignore-end -->

## Table methods

The `Backend` object itself has many helper utility methods. You'll
find the most methods on `ImpalaTable`.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - table
        - sql
        - raw_sql
        - list_tables
        - exists_table
        - drop_table
        - create_table
        - insert
        - invalidate_metadata
        - truncate_table
        - get_schema
        - cache_table
        - load_data
        - get_options
        - set_options
        - set_compression_codec

<!-- prettier-ignore-end -->

The best way to interact with a single table is through the
`ImpalaTable` object you get back from `Backend.table`.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - add_partition
        - alter
        - alter_partition
        - column_stats
        - compute_stats
        - describe_formatted
        - drop
        - drop_partition
        - files
        - insert
        - invalidate_metadata
        - is_partitioned
        - load_data
        - metadata
        - partition_schema
        - partitions
        - refresh
        - rename
        - schema
        - stats

<!-- prettier-ignore-end -->

## Creating views

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - drop_table_or_view
        - create_view

<!-- prettier-ignore-end -->

## Accessing data formats in HDFS

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - delimited_file
        - parquet_file
        - avro_file

<!-- prettier-ignore-end -->

## HDFS Interaction

Ibis delegates all HDFS interaction to the
[`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) library.

## The Impala client object

To use Ibis with Impala, you first must connect to a cluster using the
`ibis.impala.connect` function, optionally supplying an HDFS connection:

```python
import ibis

hdfs = ibis.impala.hdfs_connect(host=webhdfs_host, port=webhdfs_port)
client = ibis.impala.connect(host=impala_host, port=impala_port, hdfs_client=hdfs)
```

All examples here use the following block of code to connect to impala
using docker:

```python
import ibis

hdfs = ibis.impala.hdfs_connect(host="localhost", port=50070)
client = ibis.impala.connect(host=host, hdfs_client=hdfs)
```

You can accomplish many tasks directly through the client object, but we
additionally provide APIs to streamline tasks involving a single Impala
table or database.

## Table objects

<!-- prettier-ignore-start -->
::: ibis.backends.base.sql.BaseSQLBackend.table
    options:
      heading_level: 3
<!-- prettier-ignore-end -->

The client's `table` method allows you to create an Ibis table
expression referencing a physical Impala table:

```python
table = client.table('functional_alltypes', database='ibis_testing')
```

`ImpalaTable` is a Python subclass of the more general Ibis `Table`
that has additional Impala-specific methods. So you can use it
interchangeably with any code expecting a `Table`.

Like all table expressions in Ibis, `ImpalaTable` has a `schema` method
you can use to examine its schema:

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - schema

<!-- prettier-ignore-end -->

While the client has a `drop_table` method you can use to drop tables,
the table itself has a method `drop` that you can use:

```python
table.drop()
```

## Expression execution

Ibis expressions have execution methods like `to_pandas` that compile and run the
expressions on Impala or whichever backend is being referenced.

For example:

```python
>>> fa = db.functional_alltypes
>>> expr = fa.double_col.sum()
>>> expr.to_pandas()
331785.00000000006
```

For longer-running queries, Ibis will attempt to cancel the query in
progress if an interrupt is received.

## Creating tables

There are several ways to create new Impala tables:

- From an Ibis table expression
- Empty, from a declared schema
- Empty and partitioned

In all cases, you should use the `create_table` method either on the
top-level client connection or a database object.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend.create_table
    options:
      heading_level: 3
<!-- prettier-ignore-end -->

### Creating tables from a table expression

If you pass an Ibis expression to `create_table`, Ibis issues a
`CREATE TABLE ... AS SELECT` (CTAS) statement:

```python
>>> table = db.table('functional_alltypes')
>>> expr = table.group_by('string_col').size()
>>> db.create_table('string_freqs', expr, format='parquet')

>>> freqs = db.table('string_freqs')
>>> freqs.to_pandas()
  string_col  count
0          9    730
1          3    730
2          6    730
3          4    730
4          1    730
5          8    730
6          2    730
7          7    730
8          5    730
9          0    730

>>> files = freqs.files()
>>> files
                                                Path  Size Partition
0  hdfs://impala:8020/user/hive/warehouse/ibis_te...  584B

>>> freqs.drop()
```

You can also choose to create an empty table and use `insert` (see
below).

### Creating an empty table

To create an empty table, you must declare an Ibis schema that will be
translated to the appropriate Impala schema and data types.

As Ibis types are simplified compared with Impala types, this may expand
in the future to include a more fine-grained schema declaration.

You can use the `create_table` method either on a database or client
object.

```python
schema = ibis.schema([('foo', 'string'),
                      ('year', 'int32'),
                      ('month', 'int16')])
name = 'new_table'
db.create_table(name, schema=schema)
```

By default, this stores the data files in the database default location.
You can force a particular path with the `location` option.

```python
from getpass import getuser
schema = ibis.schema([('foo', 'string'),
                      ('year', 'int32'),
                      ('month', 'int16')])
name = 'new_table'
location = '/home/{}/new-table-data'.format(getuser())
db.create_table(name, schema=schema, location=location)
```

If the schema matches a known table schema, you can always use the
`schema` method to get a schema object:

```python
>>> t = db.table('functional_alltypes')
>>> t.schema()
ibis.Schema {
  id               int32
  bool_col         boolean
  tinyint_col      int8
  smallint_col     int16
  int_col          int32
  bigint_col       int64
  float_col        float32
  double_col       float64
  date_string_col  string
  string_col       string
  timestamp_col    timestamp
  year             int32
  month            int32
}
```

### Creating a partitioned table

To create an empty partitioned table, include a list of columns to be
used as the partition keys.

```python
schema = ibis.schema([('foo', 'string'),
                      ('year', 'int32'),
                      ('month', 'int16')])
name = 'new_table'
db.create_table(name, schema=schema, partition=['year', 'month'])
```

## Partitioned tables

Ibis enables you to manage partitioned tables in various ways. Since
each partition behaves as its own \"subtable\" sharing a common schema,
each partition can have its own file format, directory path,
serialization properties, and so forth.

There are a handful of table methods for adding and removing partitions
and getting information about the partition schema and any existing
partition data:

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - add_partition
        - drop_partition
        - is_partitioned
        - partition_schema
        - partitions

<!-- prettier-ignore-end -->

To address a specific partition in any method that is partition
specific, you can either use a dict with the partition key names and
values, or pass a list of the partition values:

```python
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
```

We'll cover partition metadata management and data loading below.

## Inserting data into tables

If the schemas are compatible, you can insert into a table directly from
an Ibis table expression:

```python
>>> t = db.functional_alltypes
>>> db.create_table('insert_test', schema=t.schema())
>>> target = db.table('insert_test')

>>> target.insert(t[:3])
>>> target.insert(t[:3])
>>> target.insert(t[:3])

>>> target.to_pandas()
     id  bool_col  tinyint_col  ...           timestamp_col  year  month
0  5770      True            0  ... 2010-08-01 00:00:00.000  2010      8
1  5771     False            1  ... 2010-08-01 00:01:00.000  2010      8
2  5772      True            2  ... 2010-08-01 00:02:00.100  2010      8
3  5770      True            0  ... 2010-08-01 00:00:00.000  2010      8
4  5771     False            1  ... 2010-08-01 00:01:00.000  2010      8
5  5772      True            2  ... 2010-08-01 00:02:00.100  2010      8
6  5770      True            0  ... 2010-08-01 00:00:00.000  2010      8
7  5771     False            1  ... 2010-08-01 00:01:00.000  2010      8
8  5772      True            2  ... 2010-08-01 00:02:00.100  2010      8

[9 rows x 13 columns]

>>> target.drop()
```

If the table is partitioned, you must indicate the partition you are
inserting into:

```python
part = {'year': 2007, 'month': 4}
table.insert(expr, partition=part)
```

## Managing table metadata

Ibis has functions that wrap many of the DDL commands for Impala table
metadata.

### Detailed table metadata: `DESCRIBE FORMATTED`

To get a handy wrangled version of `DESCRIBE FORMATTED` use the
`metadata` method.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable.metadata
    options:
      heading_level: 3
<!-- prettier-ignore-end -->

```python
>>> t = client.table('ibis_testing.functional_alltypes')
>>> meta = t.metadata()
>>> meta
<class 'ibis.backends.impala.metadata.TableMetadata'>
{'info': {'CreateTime': datetime.datetime(2021, 1, 14, 21, 23, 8),
          'Database': 'ibis_testing',
          'LastAccessTime': 'UNKNOWN',
          'Location': 'hdfs://impala:8020/__ibis/ibis-testing-data/parquet/functional_alltypes',
          'Owner': 'root',
          'Protect Mode': 'None',
          'Retention': 0,
          'Table Parameters': {'COLUMN_STATS_ACCURATE': False,
                               'EXTERNAL': True,
                               'STATS_GENERATED_VIA_STATS_TASK': True,
                               'numFiles': 3,
                               'numRows': 7300,
                               'rawDataSize': '-1',
                               'totalSize': 106278,
                               'transient_lastDdlTime': datetime.datetime(2021, 1, 14, 21, 23, 17)},
          'Table Type': 'EXTERNAL_TABLE'},
 'schema': [('id', 'int'),
            ('bool_col', 'boolean'),
            ('tinyint_col', 'tinyint'),
            ('smallint_col', 'smallint'),
            ('int_col', 'int'),
            ('bigint_col', 'bigint'),
            ('float_col', 'float'),
            ('double_col', 'double'),
            ('date_string_col', 'string'),
            ('string_col', 'string'),
            ('timestamp_col', 'timestamp'),
            ('year', 'int'),
            ('month', 'int')],
 'storage info': {'Bucket Columns': '[]',
                  'Compressed': False,
                  'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                  'Num Buckets': 0,
                  'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                  'SerDe Library': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
                  'Sort Columns': '[]'}}

>>> meta.location
'hdfs://impala:8020/__ibis/ibis-testing-data/parquet/functional_alltypes'

>>> meta.create_time
datetime.datetime(2021, 1, 14, 21, 23, 8)
```

The `files` function is also available to see all of the physical HDFS
data files backing a table:

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - files

<!-- prettier-ignore-end -->

```python
>>> ss = c.table('tpcds_parquet.store_sales')

>>> ss.files()[:5]
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
```

### Modifying table metadata

For unpartitioned tables, you can use the `alter` method to change its
location, file format, and other properties. For partitioned tables, to
change partition-specific metadata use `alter_partition`.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - alter
        - alter_partition

<!-- prettier-ignore-end -->

For example, if you wanted to \"point\" an existing table at a directory
of CSV files, you could run the following command:

```python
from getpass import getuser

csv_props = {
    'serialization.format': ',',
    'field.delim': ',',
}
data_dir = '/home/{}/my-csv-files'.format(getuser())

table.alter(location=data_dir, format='text', serde_properties=csv_props)
```

If the table is partitioned, you can modify only the properties of a
particular partition:

```python
table.alter_partition(
    {'year': 2007, 'month': 5},
    location=data_dir,
    format='text',
    serde_properties=csv_props
)
```

## Table statistics

### Computing table and partition statistics

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - compute_stats

<!-- prettier-ignore-end -->

Impala-backed physical tables have a method `compute_stats` that
computes table, column, and partition-level statistics to assist with
query planning and optimization. It is standard practice to invoke this
after creating a table or loading new data:

```python
table.compute_stats()
```

If you are using a recent version of Impala, you can also access the
`COMPUTE INCREMENTAL STATS` DDL command:

```python
table.compute_stats(incremental=True)
```

### Seeing table and column statistics

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - column_stats
        - stats

<!-- prettier-ignore-end -->

The `compute_stats` and `stats` functions return the results of
`SHOW COLUMN STATS` and `SHOW TABLE STATS`, respectively, and their
output will depend, of course, on the last `COMPUTE STATS` call.

```python
>>> ss = c.table('tpcds_parquet.store_sales')
>>> ss.compute_stats(incremental=True)
>>> stats = ss.stats()
>>> stats[:5]
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

>>> cstats = ss.column_stats()
>>> cstats
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
```

### `REFRESH` and `INVALIDATE METADATA`

These DDL commands are available as table-level and client-level
methods:

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - invalidate_metadata

<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - invalidate_metadata
        - refresh

<!-- prettier-ignore-end -->

You can invalidate the cached metadata for a single table or for all
tables using `invalidate_metadata`, and similarly invoke
`REFRESH db_name.table_name` using the `refresh` method.

```python
client.invalidate_metadata()

table = db.table(table_name)
table.invalidate_metadata()

table.refresh()
```

These methods are often used in conjunction with the `LOAD DATA`
commands and `COMPUTE STATS`. See the Impala documentation for full
details.

## Issuing `LOAD DATA` commands

The `LOAD DATA` DDL physically moves a single data file or a directory
of files into the correct location for a table or table partition. It is
especially useful for partitioned tables as you do not have to construct
the directory path for a partition by hand, so simpler and less
error-prone than manually moving files with low level HDFS commands. It
also deals with file name conflicts so data is not lost in such cases.

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - load_data

<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
::: ibis.backends.impala.client.ImpalaTable
    options:
      heading_level: 3
      members:
        - load_data

<!-- prettier-ignore-end -->

To use these methods, pass the path of a single file or a directory of
files you want to load. Afterward, you may want to update the table
statistics (see Impala documentation):

```python
table.load_data(path)
table.refresh()
```

Like the other methods with support for partitioned tables, you can load
into a particular partition with the `partition` keyword argument:

```python
part = [2007, 5]
table.load_data(path, partition=part)
```

## Parquet and other session options

Ibis gives you access to Impala session-level variables that affect
query execution:

<!-- prettier-ignore-start -->
::: ibis.backends.impala.Backend
    options:
      heading_level: 3
      members:
        - disable_codegen
        - get_options
        - set_options
        - set_compression_codec

<!-- prettier-ignore-end -->

For example:

```python
>>> client.get_options()
{'ABORT_ON_ERROR': '0',
 'APPX_COUNT_DISTINCT': '0',
 'BUFFER_POOL_LIMIT': '',
 'COMPRESSION_CODEC': '',
 'COMPUTE_STATS_MIN_SAMPLE_SIZE': '1073741824',
 'DEFAULT_JOIN_DISTRIBUTION_MODE': '0',
 'DEFAULT_SPILLABLE_BUFFER_SIZE': '2097152',
 'DISABLE_CODEGEN': '0',
 'DISABLE_CODEGEN_ROWS_THRESHOLD': '50000',
 'DISABLE_ROW_RUNTIME_FILTERING': '0',
 'DISABLE_STREAMING_PREAGGREGATIONS': '0',
 'DISABLE_UNSAFE_SPILLS': '0',
 'ENABLE_EXPR_REWRITES': '1',
 'EXEC_SINGLE_NODE_ROWS_THRESHOLD': '100',
 'EXEC_TIME_LIMIT_S': '0',
 'EXPLAIN_LEVEL': '1',
 'HBASE_CACHE_BLOCKS': '0',
 'HBASE_CACHING': '0',
 'IDLE_SESSION_TIMEOUT': '0',
 'MAX_ERRORS': '100',
 'MAX_NUM_RUNTIME_FILTERS': '10',
 'MAX_ROW_SIZE': '524288',
 'MEM_LIMIT': '0',
 'MIN_SPILLABLE_BUFFER_SIZE': '65536',
 'MT_DOP': '',
 'NUM_SCANNER_THREADS': '0',
 'OPTIMIZE_PARTITION_KEY_SCANS': '0',
 'PARQUET_ANNOTATE_STRINGS_UTF8': '0',
 'PARQUET_ARRAY_RESOLUTION': '2',
 'PARQUET_DICTIONARY_FILTERING': '1',
 'PARQUET_FALLBACK_SCHEMA_RESOLUTION': '0',
 'PARQUET_FILE_SIZE': '0',
 'PARQUET_READ_STATISTICS': '1',
 'PREFETCH_MODE': '1',
 'QUERY_TIMEOUT_S': '0',
 'REPLICA_PREFERENCE': '0',
 'REQUEST_POOL': '',
 'RUNTIME_BLOOM_FILTER_SIZE': '1048576',
 'RUNTIME_FILTER_MAX_SIZE': '16777216',
 'RUNTIME_FILTER_MIN_SIZE': '1048576',
 'RUNTIME_FILTER_MODE': '2',
 'RUNTIME_FILTER_WAIT_TIME_MS': '0',
 'S3_SKIP_INSERT_STAGING': '1',
 'SCHEDULE_RANDOM_REPLICA': '0',
 'SCRATCH_LIMIT': '-1',
 'SEQ_COMPRESSION_MODE': '',
 'SYNC_DDL': '0'}
```

To enable Snappy compression for Parquet files, you could do either of:

```python
>>> client.set_options({'COMPRESSION_CODEC': 'snappy'})
>>> client.set_compression_codec('snappy')

>>> client.get_options()['COMPRESSION_CODEC']
'SNAPPY'
```

## Ingesting data from pandas

Overall interoperability between the Hadoop / Spark ecosystems and
pandas / the PyData stack is poor, but it will improve in time (this is
a major part of the Ibis roadmap).

Ibis's Impala tools currently interoperate with pandas in these ways:

- Ibis expressions return pandas objects (i.e. DataFrame or Series)
  for non-scalar expressions when calling their `to_pandas` method
- The `create_table` and `insert` methods can accept pandas objects.
  This includes inserting into partitioned tables. It currently uses
  CSV as the ingest route.

For example:

```python
>>> import pandas as pd

>>> data = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd']})

>>> db.create_table('pandas_table', data)
>>> t = db.pandas_table
>>> t.to_pandas()
  bar  foo
0   a    1
1   b    2
2   c    3
3   d    4

>>> t.drop()

>>> db.create_table('empty_for_insert', schema=t.schema())

>>> to_insert = db.empty_for_insert
>>> to_insert.insert(data)
>>> to_insert.to_pandas()
  bar  foo
0   a    1
1   b    2
2   c    3
3   d    4

>>> to_insert.drop()
```

```python
>>> import pandas as pd

>>> data = pd.DataFrame({'foo': [1, 2, 3, 4], 'bar': ['a', 'b', 'c', 'd']})

>>> db.create_table('pandas_table', data)
>>> t = db.pandas_table
>>> t.to_pandas()
   foo bar
0    1   a
1    2   b
2    3   c
3    4   d

>>> t.drop()
>>> db.create_table('empty_for_insert', schema=t.schema())
>>> to_insert = db.empty_for_insert
>>> to_insert.insert(data)
>>> to_insert.to_pandas()
   foo bar
0    1   a
1    2   b
2    3   c
3    4   d

>>> to_insert.drop()
```

## Uploading / downloading data from HDFS

If you've set up an HDFS connection, you can use the Ibis HDFS interface
to look through your data and read and write files to and from HDFS:

```python
>>> hdfs = con.hdfs
>>> hdfs.ls('/__ibis/ibis-testing-data')
['README.md',
 'avro',
 'awards_players.csv',
 'batting.csv',
 'csv',
 'diamonds.csv',
 'functional_alltypes.csv',
 'functional_alltypes.parquet',
 'geo.csv',
 'ibis_testing.db',
 'parquet',
 'struct_table.avro',
 'udf']
```

```python
>>> hdfs.ls('/__ibis/ibis-testing-data/parquet')
['functional_alltypes',
 'tpch_customer',
 'tpch_lineitem',
 'tpch_nation',
 'tpch_orders',
 'tpch_part',
 'tpch_partsupp',
 'tpch_region',
 'tpch_supplier']
```

Suppose we wanted to download
`/__ibis/ibis-testing-data/parquet/functional_alltypes`, which is a
directory. We need only do:

```bash
$ rm -rf parquet_dir/
```

```python
>>> hdfs.get('/__ibis/ibis-testing-data/parquet/functional_alltypes',
...          'parquet_dir',
...           recursive=True)
'/ibis/docs/source/tutorial/parquet_dir'
```

Now we have that directory locally:

```bash
$ ls parquet_dir/
9a41de519352ab07-4e76bc4d9fb5a789_1624886651_data.0.parq
9a41de519352ab07-4e76bc4d9fb5a78a_778826485_data.0.parq
9a41de519352ab07-4e76bc4d9fb5a78b_1277612014_data.0.parq
```

Files and directories can be written to HDFS just as easily using `put`:

```python
>>> path = '/__ibis/dir-write-example'
>>> hdfs.rm(path, recursive=True)
>>> hdfs.put(path, 'parquet_dir', recursive=True)
```

```python
>>> hdfs.ls('/__ibis/dir-write-example')
['9a41de519352ab07-4e76bc4d9fb5a789_1624886651_data.0.parq',
 '9a41de519352ab07-4e76bc4d9fb5a78a_778826485_data.0.parq',
 '9a41de519352ab07-4e76bc4d9fb5a78b_1277612014_data.0.parq']
```

Delete files and directories with `rm`:

```python
>>> hdfs.rm('/__ibis/dir-write-example', recursive=True)
```

```bash
rm -rf parquet_dir/
```

## Queries on Parquet, Avro, and Delimited files in HDFS

Ibis can easily create temporary or persistent Impala tables that
reference data in the following formats:

- Parquet (`parquet_file`)
- Avro (`avro_file`)
- Delimited text formats (CSV, TSV, etc.) (`delimited_file`)

Parquet is the easiest because the schema can be read from the data
files:

```python
>>> path = '/__ibis/ibis-testing-data/parquet/tpch_lineitem'
>>> lineitem = con.parquet_file(path)
>>> lineitem.limit(2)
   l_orderkey  l_partkey  l_suppkey  l_linenumber l_quantity l_extendedprice  \
0           1     155190       7706             1      17.00        21168.23
1           1      67310       7311             2      36.00        45983.16

  l_discount l_tax l_returnflag l_linestatus  l_shipdate l_commitdate  \
0       0.04  0.02            N            O  1996-03-13   1996-02-12
1       0.09  0.06            N            O  1996-04-12   1996-02-28

  l_receiptdate     l_shipinstruct l_shipmode  \
0    1996-03-22  DELIVER IN PERSON      TRUCK
1    1996-04-20   TAKE BACK RETURN       MAIL

                            l_comment
0             egular courts above the
1  ly final dependencies: slyly bold
```

```python
>>> lineitem.l_extendedprice.sum()
Decimal('229577310901.20')
```

If you want to query a Parquet file and also create a table in Impala
that remains after your session, you can pass more information to
`parquet_file`:

```python
>>> table = con.parquet_file(path, name='my_parquet_table',
...                          database='ibis_testing',
...                          persist=True)
>>> table.l_extendedprice.sum()
Decimal('229577310901.20')
```

```python
>>> con.table('my_parquet_table').l_extendedprice.sum()
Decimal('229577310901.20')
```

```python
>>> con.drop_table('my_parquet_table')
```

To query delimited files, you need to write down an Ibis schema. At some
point we'd like to build some helper tools that will infer the schema
for you, all in good time.

There's some CSV files in the test folder, so let's use those:

```python
>>> hdfs.get('/__ibis/ibis-testing-data/csv', 'csv-files', recursive=True)
'/ibis/docs/source/tutorial/csv-files'
```

```bash
$ cat csv-files/0.csv
63IEbRheTh,0.679388707915,6
mG4hlqnjeG,2.80710565922,15
JTPdX9SZH5,-0.155126406372,55
2jcl6FypOl,1.03787834032,21
k3TbJLaadQ,-1.40190801103,23
rP5J4xvinM,-0.442092712869,22
WniUylixYt,-0.863748033806,27
znsDuKOB1n,-0.566029637098,47
4SRP9jlo1M,0.331460412318,88
KsfjPyDf5e,-0.578930506363,70
```

```bash
$ rm -rf csv-files/
```

The schema here is pretty simple (see `ibis.schema` for more):

```python
>>> schema = ibis.schema([('foo', 'string'),
...                       ('bar', 'double'),
...                       ('baz', 'int32')])

>>> table = con.delimited_file('/__ibis/ibis-testing-data/csv',
...                            schema)
>>> table.limit(10)
          foo       bar  baz
0  63IEbRheTh  0.679389    6
1  mG4hlqnjeG  2.807106   15
2  JTPdX9SZH5 -0.155126   55
3  2jcl6FypOl  1.037878   21
4  k3TbJLaadQ -1.401908   23
5  rP5J4xvinM -0.442093   22
6  WniUylixYt -0.863748   27
7  znsDuKOB1n -0.566030   47
8  4SRP9jlo1M  0.331460   88
9  KsfjPyDf5e -0.578931   70
```

```python
>>> table.bar.summary()
   count  nulls       min       max       sum    mean  approx_nunique
0    100      0 -1.401908  2.807106  8.479978  0.0848              10
```

For functions like `parquet_file` and `delimited_file`, an HDFS
directory must be passed (we'll add support for S3 and other filesystems
later) and the directory must contain files all having the same schema.

If you have Avro data, you can query it too if you have the full avro
schema:

```python
>>> avro_schema = {
...     "fields": [
...         {"type": ["int", "null"], "name": "R_REGIONKEY"},
...         {"type": ["string", "null"], "name": "R_NAME"},
...         {"type": ["string", "null"], "name": "R_COMMENT"}],
...     "type": "record",
...     "name": "a"
... }

>>> path = '/__ibis/ibis-testing-data/avro/tpch.region'

>>> hdfs.mkdir(path, create_parents=True)
>>> table = con.avro_file(path, avro_schema)
>>> table
Empty DataFrame
Columns: [r_regionkey, r_name, r_comment]
Index: []
```

## Other helper functions for interacting with the database

We're adding a growing list of useful utility functions for interacting
with an Impala cluster on the client object. The idea is that you should
be able to do any database-admin-type work with Ibis and not have to
switch over to the Impala SQL shell. Any ways we can make this more
pleasant, please let us know.

Here's some of the features, which we'll give examples for:

- Listing and searching for available databases and tables
- Creating and dropping databases
- Getting table schemas

```python
>>> con.list_databases(like='ibis*')
['ibis_testing', 'ibis_testing_tmp_db']
```

```python
>>> con.list_tables(database='ibis_testing', like='tpch*')
['tpch_customer',
 'tpch_lineitem',
 'tpch_nation',
 'tpch_orders',
 'tpch_part',
 'tpch_partsupp',
 'tpch_region',
 'tpch_region_avro',
 'tpch_supplier']
```

```python
>>> schema = con.get_schema('functional_alltypes')
>>> schema
ibis.Schema {
  id               int32
  bool_col         boolean
  tinyint_col      int8
  smallint_col     int16
  int_col          int32
  bigint_col       int64
  float_col        float32
  double_col       float64
  date_string_col  string
  string_col       string
  timestamp_col    timestamp
  year             int32
  month            int32
}
```

Databases can be created, too, and you can set the storage path in HDFS
you want for the data files

```python
>>> db = 'ibis_testing2'
>>> con.create_database(db, path='/__ibis/my-test-database', force=True)

>>> # you may or may not have to give the impala user write and execute permissions to '/__ibis/my-test-database'
>>> hdfs.chmod('/__ibis/my-test-database', 0o777)
```

```python
>>> con.create_table('example_table', con.table('functional_alltypes'),
...                  database=db, force=True)
```

Hopefully, there will be data files in the indicated spot in HDFS:

```python
>>> hdfs.ls('/__ibis/my-test-database')
['example_table']
```

To drop a database, including all tables in it, you can use
`drop_database` with `force=True`:

```python
>>> con.drop_database(db, force=True)
```

## Faster queries on small data in Impala

Since Impala internally uses LLVM to compile parts of queries (aka
"codegen") to make them faster on large data sets there is a certain
amount of overhead with running many kinds of queries, even on small
datasets. You can disable LLVM code generation when using Ibis, which
may significantly speed up queries on smaller datasets:

```python
>>> from numpy.random import rand
>>> con.disable_codegen()
>>> t = con.table('ibis_testing.functional_alltypes')
```

```bash
$ time python -c "(t.double_col + rand()).sum().to_pandas()"
27.7 ms ± 996 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
# Turn codegen back on
con.disable_codegen(False)
```

```bash
$ time python -c "(t.double_col + rand()).sum().to_pandas()"
27 ms ± 1.62 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

It's important to remember that codegen is a fixed overhead and will
significantly speed up queries on big data

## User Defined functions (UDF)

Impala currently supports user-defined scalar functions (known
henceforth as _UDFs_) and aggregate functions (respectively _UDAs_) via
a C++ extension API.

Initial support for using C++ UDFs in Ibis came in version 0.4.0.

### Using scalar functions (UDFs)

Let's take an example to illustrate how to make a C++ UDF available to
Ibis. Here is a function that computes an approximate equality between
floating point values:

```c++
#include "impala_udf/udf.h"

#include <cctype>
#include <cmath>

BooleanVal FuzzyEquals(FunctionContext* ctx, const DoubleVal& x, const DoubleVal& y) {
  const double EPSILON = 0.000001f;
  if (x.is_null || y.is_null) return BooleanVal::null();
  double delta = fabs(x.val - y.val);
  return BooleanVal(delta < EPSILON);
}
```

You can compile this to either a shared library (a `.so` file) or to
LLVM bitcode with clang (a `.ll` file). Skipping that step for now (will
add some more detailed instructions here later, promise).

To make this function callable, we use `ibis.impala.wrap_udf`:

```python
library = '/ibis/udfs/udftest.ll'
inputs = ['double', 'double']
output = 'boolean'
symbol = 'FuzzyEquals'
udf_db = 'ibis_testing'
udf_name = 'fuzzy_equals'

fuzzy_equals = ibis.impala.wrap_udf(
    library, inputs, output, symbol, name=udf_name
)
```

In typical workflows, you will set up a UDF in Impala once then use it
thenceforth. So the _first time_ you do this, you need to create the UDF
in Impala:

```python
client.create_function(fuzzy_equals, database=udf_db)
```

Now, we must register this function as a new Impala operation in Ibis.
This must take place each time you load your Ibis session.

```python
func.register(fuzzy_equals.name, udf_db)
```

The object `fuzzy_equals` is callable and works with Ibis expressions:

```python
>>> db = c.database('ibis_testing')

>>> t = db.functional_alltypes

>>> expr = fuzzy_equals(t.float_col, t.double_col / 10)

>>> expr.to_pandas()[:10]
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
```

Note that the call to `register` on the UDF object must happen each time
you use Ibis. If you have a lot of UDFs, I suggest you create a file
with all of your wrapper declarations and user APIs that you load with
your Ibis session to plug in all your own functions.

## Working with secure clusters (Kerberos)

Ibis is compatible with Hadoop clusters that are secured with Kerberos (as well
as SSL and LDAP). Note that to enable this support, you'll also need to install
the `kerberos` package.

```sh
$ pip install kerberos
```

Just like the Impala shell and ODBC/JDBC connectors, Ibis connects to Impala
through the HiveServer2 interface (using the impyla client). Therefore, the
connection semantics are similar to the other access methods for working with
secure clusters.

Specifically, after authenticating yourself against Kerberos (e.g., by issuing
the appropriate `kinit` command), simply pass `auth_mechanism='GSSAPI'` or
`auth_mechanism='LDAP'` (and set `kerberos_service_name` if necessary along
with `user` and `password` if necessary) to the
`ibis.impala_connect(...)` method when instantiating an `ImpalaConnection`.
This method also takes arguments to configure SSL (`use_ssl`, `ca_cert`).
See the documentation for the Impala shell for more details.

Ibis also includes functionality that communicates directly with HDFS, using
the WebHDFS REST API. When calling `ibis.impala.hdfs_connect(...)`, also pass
`auth_mechanism='GSSAPI'` or `auth_mechanism='LDAP'`, and ensure that you
are connecting to the correct port, which may likely be an SSL-secured WebHDFS
port. Also note that you can pass `verify=False` to avoid verifying SSL
certificates (which may be helpful in testing). Ibis will assume `https`
when connecting to a Kerberized cluster. Because some Ibis commands create HDFS
directories as well as new Impala databases and/or tables, your user will
require the necessary privileges.

## Default Configuration Values for CDH Components

Cloudera CDH ships with HDFS, Impala, Hive and many other components.
Sometimes it's not obvious what default configuration values these tools are
using or should be using.

Check out [this
link](https://www.cloudera.com/documentation/enterprise/latest/topics/cdh_ig_ports_cdh5.html#topic_9_1)
to see the default configuration values for every component of CDH.
