Release Notes
---

# [2.1.0](https://github.com/ibis-project/ibis/compare/2.0.0...2.1.0) (2022-01-12)


### Bug Fixes

* consider all packages' entry points ([b495cf6](https://github.com/ibis-project/ibis/commit/b495cf6c9f568ab5fd45f4d5a8a80dde2f14d897))
* **datatypes:** infer bytes literal as binary [#2915](https://github.com/ibis-project/ibis/issues/2915) ([#3124](https://github.com/ibis-project/ibis/issues/3124)) ([887efbd](https://github.com/ibis-project/ibis/commit/887efbd4c9d0657f5639638eebea53906044b78f))
* **deps:** bump minimum dask version to 2021.10.0 ([e6b5c09](https://github.com/ibis-project/ibis/commit/e6b5c095e562a0f1f1386efe578ca4562062154a))
* **deps:** constrain numpy to ensure wheels are used on windows ([70c308b](https://github.com/ibis-project/ibis/commit/70c308b7d2dcd5e8b537b2f5ccf496ace6328979))
* **deps:** update dependency clickhouse-driver to ^0.1 || ^0.2.0 ([#3061](https://github.com/ibis-project/ibis/issues/3061)) ([a839d54](https://github.com/ibis-project/ibis/commit/a839d544f1bf74b4eea8e2e99d81de0be8cd8aa7))
* **deps:** update dependency geoalchemy2 to >=0.6,<0.11 ([4cede9d](https://github.com/ibis-project/ibis/commit/4cede9d7a08ce9914c048b25a5858480d6a40254))
* **deps:** update dependency pyarrow to v6 ([#3092](https://github.com/ibis-project/ibis/issues/3092)) ([61e52b5](https://github.com/ibis-project/ibis/commit/61e52b51478354354dc59878a5a9987ad312b19b))
* don't force backends to override do_connect until 3.0.0 ([4b46973](https://github.com/ibis-project/ibis/commit/4b46973930e113dce345240a47906a92bb1cf24e))
* execute materialized joins in the pandas and dask backends ([#3086](https://github.com/ibis-project/ibis/issues/3086)) ([9ed937a](https://github.com/ibis-project/ibis/commit/9ed937a9aef71acaf5df86c88e013d9fe3ff7cce))
* **literal:** allow creating ibis literal with uuid ([#3131](https://github.com/ibis-project/ibis/issues/3131)) ([b0f4f44](https://github.com/ibis-project/ibis/commit/b0f4f44a182644bd2389c1f52338e690f7d50da7))
* restore the ability to have more than two option levels ([#3151](https://github.com/ibis-project/ibis/issues/3151)) ([fb4a944](https://github.com/ibis-project/ibis/commit/fb4a9449022bde322e7f17996e339638af40335e))
* **sqlalchemy:** fix correlated subquery compilation ([43b9010](https://github.com/ibis-project/ibis/commit/43b9010c600d30ac8fdb79b59c05391fed75e589))
* **sqlite:** defer db connection until needed ([#3127](https://github.com/ibis-project/ibis/issues/3127)) ([5467afa](https://github.com/ibis-project/ibis/commit/5467afaf22ded02bc79efe4c7956957cc1457e96)), closes [#64](https://github.com/ibis-project/ibis/issues/64)


### Features

* allow column_of to take a column expression ([dbc34bb](https://github.com/ibis-project/ibis/commit/dbc34bbe7c3506f1f4e881eaabf928890d9477ca))
* **ci:** More readable workflow job titles  ([#3111](https://github.com/ibis-project/ibis/issues/3111)) ([d8fd7d9](https://github.com/ibis-project/ibis/commit/d8fd7d9691612379c29d6e745b4041a3dab85636))
* **datafusion:** initial implementation for Arrow Datafusion backend ([3a67840](https://github.com/ibis-project/ibis/commit/3a67840155a928bb9c0feff5d1bf9e2cbfe70d91)), closes [#2627](https://github.com/ibis-project/ibis/issues/2627)
* **datafusion:** initial implementation for Arrow Datafusion backend ([75876d9](https://github.com/ibis-project/ibis/commit/75876d9718e22de46763d01a6e272f95645d60bc)), closes [#2627](https://github.com/ibis-project/ibis/issues/2627)
* make dayofweek impls conform to pandas semantics ([#3161](https://github.com/ibis-project/ibis/issues/3161)) ([9297828](https://github.com/ibis-project/ibis/commit/92978286f1fd009ee490befab236442fd1c7a095))


### Reverts

* "ci: install gdal for fiona" ([8503361](https://github.com/ibis-project/ibis/commit/850336100a271ee2b6043b92a1ceeb1d1d7b30f2))

# [2.0.0](https://github.com/ibis-project/ibis/releases/tag/2.0.0) (2021-10-06)

## Features

* Serialization-deserialization of Node via pickle is now byte compatible between different processes ([#2938](https://github.com/ibis-project/ibis/issues/2938))
* Support joining on different columns in ClickHouse backend ([#2916](https://github.com/ibis-project/ibis/issues/2916))
* Support summarization of empty data in Pandas backend ([#2908](https://github.com/ibis-project/ibis/issues/2908))
* Unify implementation of fillna and isna in Pyspark backend ([#2882](https://github.com/ibis-project/ibis/issues/2882))
* Support binary operation with Timedelta in Pyspark backend ([#2873](https://github.com/ibis-project/ibis/issues/2873))
* Add `group_concat` operation for Clickhouse backend ([#2839](https://github.com/ibis-project/ibis/issues/2839))
* Support comparison of ColumnExpr to timestamp literal ([#2808](https://github.com/ibis-project/ibis/issues/2808))
* Make op schema a cached property ([#2805](https://github.com/ibis-project/ibis/issues/2805))
* Implement `.insert()` for SQLAlchemy backends ([#2613](https://github.com/ibis-project/ibis/issues/2613), [#2613](https://github.com/ibis-project/ibis/issues/2778))
* Infer categorical and decimal Series to more specific Ibis types in Pandas backend ([#2792](https://github.com/ibis-project/ibis/issues/2792))
* Add `startswith` and `endswith` operations ([#2790](https://github.com/ibis-project/ibis/issues/2790))
* Allow more flexible return type for UDFs ([#2776](https://github.com/ibis-project/ibis/issues/2776), [#2797](https://github.com/ibis-project/ibis/issues/2776))
* Implement Clip in the Pyspark backend ([#2779](https://github.com/ibis-project/ibis/issues/2779))
* Use `ndarray` as array representation in Pandas backend ([#2753](https://github.com/ibis-project/ibis/issues/2753))
* Support Spark filter with window operation ([#2687](https://github.com/ibis-project/ibis/issues/2687))
* Support context adjustment for udfs for pandas backend ([#2646](https://github.com/ibis-project/ibis/issues/2646))
* Add `auth_local_webserver`, `auth_external_data`, and `auth_cache` parameters to BigQuery connect method. Set `auth_local_webserver` to use a local server instead of copy-pasting an authorization code. Set `auth_external_data` to true to request additional scopes required to query Google Drive and Sheets. Set `auth_cache` to `reauth` or `none` to force reauthentication. ([#2655](https://github.com/ibis-project/ibis/issues/2655))
* Add `bit_and`, `bit_or`, and `bit_xor` integer column aggregates (BigQuery and MySQL backends) ([#2641](https://github.com/ibis-project/ibis/issues/2641))
* Backends are defined as entry points ([#2379](https://github.com/ibis-project/ibis/issues/2379))
* Add `ibis.array` for creating array expressions ([#2615](https://github.com/ibis-project/ibis/issues/2615))
* Implement Not operation in PySpark backend ([#2607](https://github.com/ibis-project/ibis/issues/2607))
* Added support for case/when in PySpark backend ([#2610](https://github.com/ibis-project/ibis/issues/2610))
* Add support for np.array as literals for backends that already support lists as literals ([#2603](https://github.com/ibis-project/ibis/issues/2603))

## Bugs

* Fix data races in impala connection pool accounting ([#2991](https://github.com/ibis-project/ibis/issues/2991))
* Fix null literal compilation in the Clickhouse backend ([#2985](https://github.com/ibis-project/ibis/issues/2985))
* Fix order of limit and offset parameters in the Clickhouse backend ([#2984](https://github.com/ibis-project/ibis/issues/2984))
* Replace `equals` operation for geospatial datatype to `geo_equals` ([#2956](https://github.com/ibis-project/ibis/issues/2956))
* Fix .drop(fields). The argument can now be either a list of strings or a string. ([#2829](https://github.com/ibis-project/ibis/issues/2829))
* Fix projection on differences and intersections for SQL backends ([#2845](https://github.com/ibis-project/ibis/issues/2845))
* Backends are loaded in a lazy way, so third-party backends can import Ibis without circular imports ([#2827](https://github.com/ibis-project/ibis/issues/2827))
* Disable aggregation optimization due to N squared performance ([#2830](https://github.com/ibis-project/ibis/issues/2830))
* Fix `.cast()` to array outputting list instead of np.array in Pandas backend ([#2821](https://github.com/ibis-project/ibis/issues/2821))
* Fix aggregation with mixed reduction datatypes (array + scalar) on Dask backend ([#2820](https://github.com/ibis-project/ibis/issues/2820))
* Fix error when using reduction UDF that returns np.array in a grouped aggregation ([#2770](https://github.com/ibis-project/ibis/issues/2770))
* Fix time context trimming error for multi column udfs in pandas backend ([#2712](https://github.com/ibis-project/ibis/issues/2712))
* Fix error during compilation of range_window in base_sql backends (:issue:`2608`) ([#2710](https://github.com/ibis-project/ibis/issues/2710))
* Fix wrong row indexing in the result for 'window after filter' for timecontext adjustment ([#2696](https://github.com/ibis-project/ibis/issues/2696))
* Fix `aggregate` exploding the output of Reduction ops that return a list/ndarray ([#2702](https://github.com/ibis-project/ibis/issues/2702))
* Fix issues with context adjustment for filter with PySpark backend ([#2693](https://github.com/ibis-project/ibis/issues/2693))
* Add temporary struct col in pyspark backend to ensure that UDFs are executed only once ([#2657](https://github.com/ibis-project/ibis/issues/2657))
* Fix BigQuery connect bug that ignored project ID parameter ([#2588](https://github.com/ibis-project/ibis/issues/2588))
* Fix overwrite logic to account for DestructColumn inside mutate API ([#2636](https://github.com/ibis-project/ibis/issues/2636))
* Fix fusion optimization bug that incorrectly changes operation order ([#2635](https://github.com/ibis-project/ibis/issues/2635))
* Fixes a NPE issue with substr in PySpark backend ([#2610](https://github.com/ibis-project/ibis/issues/2610))
* Fixes binary data type translation into BigQuery bytes data type ([#2354](https://github.com/ibis-project/ibis/issues/2354))
* Make StructValue picklable ([#2577](https://github.com/ibis-project/ibis/issues/2577))

## Support

* Improvement of the backend API. The former `Client` subclasses have been replaced by a `Backend` class that must subclass `ibis.backends.base.BaseBackend`. The `BaseBackend` class contains abstract methods for the minimum subset of methods that backends must implement, and their signatures have been standardized across backends. The Ibis compiler has been refactored, and backends don't need to implement all compiler classes anymore if the default works for them. Only a subclass of `ibis.backends.base.sql.compiler.Compiler` is now required. Backends now need to register themselves as entry points. ([#2678](https://github.com/ibis-project/ibis/issues/2678))
* Deprecate `exists_table(table)` in favor of `table in list_tables()` ([#2905](https://github.com/ibis-project/ibis/issues/2905))
* Remove handwritten type parser; parsing errors that were previously `IbisTypeError` are now `parsy.ParseError`. `parsy` is now a hard requirement. ([#2977](https://github.com/ibis-project/ibis/issues/2977))
* Methods `current_database` and `list_databases` raise an exception for backends that do not support databases ([#2962](https://github.com/ibis-project/ibis/issues/2962))
* Method `set_database` has been deprecated, in favor of creating a new connection to a different database ([#2913](https://github.com/ibis-project/ibis/issues/2913))
* Removed `log` method of clients, in favor of `verbose_log` option ([#2914](https://github.com/ibis-project/ibis/issues/2914))
* Output of `Client.version` returned as a string, instead of a setuptools `Version` ([#2883](https://github.com/ibis-project/ibis/issues/2883))
* Deprecated `list_schemas` in SQLAlchemy backends in favor of `list_databases` ([#2862](https://github.com/ibis-project/ibis/issues/2862))
* Deprecated `ibis.<backend>.verify()` in favor of capturing exception in `ibis.<backend>.compile()` ([#2865](https://github.com/ibis-project/ibis/issues/2865))
* Simplification of data fetching. Backends don't need to implement `Query` anymore ([#2789](https://github.com/ibis-project/ibis/issues/2789))
* Move BigQuery backend to a `separate repository <https://github.com/ibis-project/ibis-bigquery>`_. The backend will be released separately, use `pip install ibis-bigquery` or `conda install ibis-bigquery` to install it, and then use as before. ([#2665](https://github.com/ibis-project/ibis/issues/2665))
* Supporting SQLAlchemy 1.4, and requiring minimum 1.3 ([#2689](https://github.com/ibis-project/ibis/issues/2689))
* Namespace time_col config, fix type check for trim_with_timecontext for pandas window execution ([#2680](https://github.com/ibis-project/ibis/issues/2680))
* Remove deprecated `ibis.HDFS`, `ibis.WebHDFS` and `ibis.hdfs_connect` ([#2505](https://github.com/ibis-project/ibis/issues/2505))


# [1.4.0](https://github.com/ibis-project/ibis/releases/tag/1.4.0) (2020-11-07)

## Features

* Add Struct.from_dict ([#2514](https://github.com/ibis-project/ibis/issues/2514))
* Add hash and hashbytes support for BigQuery backend ([#2310](https://github.com/ibis-project/ibis/issues/2310))
* Support reduction UDF without groupby to return multiple columns for Pandas backend ([#2511](https://github.com/ibis-project/ibis/issues/2511))
* Support analytic and reduction UDF to return multiple columns for Pandas backend ([#2487](https://github.com/ibis-project/ibis/issues/2487))
* Support elementwise UDF to return multiple columns for Pandas and PySpark backend ([#2473](https://github.com/ibis-project/ibis/issues/2473))
* FEAT: Support Ibis interval for window in pyspark backend ([#2409](https://github.com/ibis-project/ibis/issues/2409))
* Use Scope class for scope in pyspark backend ([#2402](https://github.com/ibis-project/ibis/issues/2402))
* Add PySpark support for ReductionVectorizedUDF ([#2366](https://github.com/ibis-project/ibis/issues/2366))
* Add time context in `scope` in execution for pandas backend ([#2306](https://github.com/ibis-project/ibis/issues/2306))
* Add ``start_point`` and ``end_point`` to PostGIS backend. ([#2081](https://github.com/ibis-project/ibis/issues/2081))
* Add set difference to general ibis api ([#2347](https://github.com/ibis-project/ibis/issues/2347))
* Add ``rowid`` expression, supported by SQLite and OmniSciDB ([#2251](https://github.com/ibis-project/ibis/issues/2251))
* Add intersection to general ibis api ([#2230](https://github.com/ibis-project/ibis/issues/2230))
* Add ``application_name`` argument to ``ibis.bigquery.connect`` to allow attributing Google API requests to projects that use Ibis. ([#2303](https://github.com/ibis-project/ibis/issues/2303))
* Add support for casting category dtype in pandas backend ([#2285](https://github.com/ibis-project/ibis/issues/2285))
* Add support for Union in the PySpark backend ([#2270](https://github.com/ibis-project/ibis/issues/2270))
* Add support for implementign custom window object for pandas backend ([#2260](https://github.com/ibis-project/ibis/issues/2260))
* Implement two level dispatcher for execute_node ([#2246](https://github.com/ibis-project/ibis/issues/2246))
* Add ibis.pandas.trace module to log time and call stack information. ([#2233](https://github.com/ibis-project/ibis/issues/2233))
* Validate that the output type of a UDF is a single element ([#2198](https://github.com/ibis-project/ibis/issues/2198))
* ZeroIfNull and NullIfZero implementation for OmniSciDB ([#2186](https://github.com/ibis-project/ibis/issues/2186))
* IsNan implementation for OmniSciDB ([#2093](https://github.com/ibis-project/ibis/issues/2093))
* [OmnisciDB] Support add_columns and drop_columns for OmnisciDB table ([#2094](https://github.com/ibis-project/ibis/issues/2094))
* Create ExtractQuarter operation and add its support to Clickhouse, CSV, Impala, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark ([#2175](https://github.com/ibis-project/ibis/issues/2175))
* Add translation rules for isnull() and notnull() for pyspark backend ([#2126](https://github.com/ibis-project/ibis/issues/2126))
* Add window operations support to SQLite ([#2232](https://github.com/ibis-project/ibis/issues/2232))
* Implement read_csv for omniscidb backend ([#2062](https://github.com/ibis-project/ibis/issues/2062))
* [OmniSciDB] Add support to week extraction ([#2171](https://github.com/ibis-project/ibis/issues/2171))
* Date, DateDiff and TimestampDiff implementations for OmniSciDB ([#2097](https://github.com/ibis-project/ibis/issues/2097))
* Create ExtractWeekOfYear operation and add its support to Clickhouse, CSV, MySQL, Pandas, Parquet, PostgreSQL, PySpark and Spark ([#2177](https://github.com/ibis-project/ibis/issues/2177))
* Add initial support for ibis.random function ([#2060](https://github.com/ibis-project/ibis/issues/2060))
* Added epoch_seconds extraction operation to Clickhouse, CSV, Impala, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite, Spark and BigQuery :issue:`2273` ([#2178](https://github.com/ibis-project/ibis/issues/2178))
* [OmniSciDB] Add "method" parameter to load_data ([#2165](https://github.com/ibis-project/ibis/issues/2165))
* Add non-nullable info to schema output ([#2117](https://github.com/ibis-project/ibis/issues/2117))
* fillna and nullif implementations for OmnisciDB ([#2083](https://github.com/ibis-project/ibis/issues/2083))
* Add load_data to sqlalchemy's backends and fix database parameter for load/create/drop when database parameter is the same than the current database ([#1981](https://github.com/ibis-project/ibis/issues/1981))
* [OmniSciDB] Add support for within, d_fully_within and point ([#2125](https://github.com/ibis-project/ibis/issues/2125))
* OmniSciDB - Refactor DDL and Client; Add temporary parameter to create_table and "force" parameter to drop_view ([#2086](https://github.com/ibis-project/ibis/issues/2086))
* Create ExtractDayOfYear operation and add its support to Clickhouse, CSV, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark ([#2173](https://github.com/ibis-project/ibis/issues/2173))
* Implementations of Log Log2 Log10 for OmniSciDB backend ([#2095](https://github.com/ibis-project/ibis/issues/2095))

## Bugs

* Table expressions do not recognize inet datatype (Postgres backend) ([#2462](https://github.com/ibis-project/ibis/issues/2462))
* Table expressions do not recognize macaddr datatype (Postgres backend) ([#2461](https://github.com/ibis-project/ibis/issues/2461))
* Fix ``aggcontext.Summarize`` not always producing scalar (Pandas backend) ([#2410](https://github.com/ibis-project/ibis/issues/2410))
* Fix same window op with different window size on table lead to incorrect results for pyspark backend ([#2414](https://github.com/ibis-project/ibis/issues/2414))
* Fix same column with multiple aliases not showing properly in repr ([#2229](https://github.com/ibis-project/ibis/issues/2229))
* Fix reduction UDFs over ungrouped, bounded windows on Pandas backend ([#2395](https://github.com/ibis-project/ibis/issues/2395))
* FEAT: Support rolling window UDF with non numeric inputs for pandas backend. ([#2386](https://github.com/ibis-project/ibis/issues/2386))
* Fix scope get to use hashmap lookup instead of list lookup ([#2386](https://github.com/ibis-project/ibis/issues/2386))
* Fix equality behavior for Literal ops ([#2387](https://github.com/ibis-project/ibis/issues/2387))
* Fix analytic ops over ungrouped and unordered windows on Pandas backend ([#2376](https://github.com/ibis-project/ibis/issues/2376))
* Fix the covariance operator in the BigQuery backend. ([#2367](https://github.com/ibis-project/ibis/issues/2367))
* Update impala kerberos dependencies ([#2342](https://github.com/ibis-project/ibis/issues/2342))
* Added verbose logging to SQL backends ([#1320](https://github.com/ibis-project/ibis/issues/1320))
* Fix issue with sql_validate call to OmnisciDB. ([#2256](https://github.com/ibis-project/ibis/issues/2256))
* Add missing float types to pandas backend ([#2237](https://github.com/ibis-project/ibis/issues/2237))
* Allow group_by and order_by as window operation input in pandas backend ([#2252](https://github.com/ibis-project/ibis/issues/2252))
* Fix PySpark compiler error when elementwise UDF output_type is Decimal or Timestamp ([#2223](https://github.com/ibis-project/ibis/issues/2223))
* Fix interactive mode returning a expression instead of the value when used in Jupyter ([#2157](https://github.com/ibis-project/ibis/issues/2157))
* Fix PySpark error when doing alias after selection ([#2127](https://github.com/ibis-project/ibis/issues/2127))
* Fix millisecond issue for OmniSciDB :issue:`2167`, MySQL :issue:`2169`, PostgreSQL :issue:`2166`, Pandas :issue:`2168`, BigQuery :issue:`2273` backends ([#2170](https://github.com/ibis-project/ibis/issues/2170))
* [OmniSciDB] Fix TopK when used as filter ([#2134](https://github.com/ibis-project/ibis/issues/2134))

## Support

* Move `ibis.HDFS`, `ibis.WebHDFS` and `ibis.hdfs_connect` to `ibis.impala.*` ([#2497](https://github.com/ibis-project/ibis/issues/2497))
* Drop support to Python 3.6 ([#2288](https://github.com/ibis-project/ibis/issues/2288))
* Simplifying tests directories structure ([#2351](https://github.com/ibis-project/ibis/issues/2351))
* Update ``google-cloud-bigquery`` dependency minimum version to 1.12.0 ([#2304](https://github.com/ibis-project/ibis/issues/2304))
* Remove "experimental" mentions for OmniSciDB and Pandas backends ([#2234](https://github.com/ibis-project/ibis/issues/2234))
* Use an OmniSciDB image stable on CI ([#2244](https://github.com/ibis-project/ibis/issues/2244))
* Added fragment_size to table creation for OmniSciDB ([#2107](https://github.com/ibis-project/ibis/issues/2107))
* Added round() support for OmniSciDB ([#2096](https://github.com/ibis-project/ibis/issues/2096))
* Enabled cumulative ops support for OmniSciDB ([#2113](https://github.com/ibis-project/ibis/issues/2113))


# [1.3.0](https://github.com/ibis-project/ibis/releases/tag/1.3.0) (2020-02-27)

## Features

* Improve many arguments UDF performance in pandas backend. ([#2071](https://github.com/ibis-project/ibis/issues/2071))
* Add DenseRank, RowNumber, MinRank, Count, PercentRank/CumeDist window operations to OmniSciDB ([#1976](https://github.com/ibis-project/ibis/issues/1976))
* Introduce a top level vectorized UDF module (experimental). Implement element-wise UDF for pandas and PySpark backend. ([#2047](https://github.com/ibis-project/ibis/issues/2047))
* Add support for  multi arguments window UDAF for the pandas backend ([#2035](https://github.com/ibis-project/ibis/issues/2035))
* Clean up window translation logic in pyspark backend ([#2004](https://github.com/ibis-project/ibis/issues/2004))
* Add docstring check to CI for an initial subset files ([#1996](https://github.com/ibis-project/ibis/issues/1996))
* Pyspark backend bounded windows ([#2001](https://github.com/ibis-project/ibis/issues/2001))
* Add more POSTGIS operations ([#1987](https://github.com/ibis-project/ibis/issues/1987))
* SQLAlchemy Default precision and scale to decimal types for PostgreSQL and MySQL ([#1969](https://github.com/ibis-project/ibis/issues/1969))
* Add support for array operations in PySpark backend ([#1983](https://github.com/ibis-project/ibis/issues/1983))
* Implement sort, if_null, null_if and notin for PySpark backend ([#1978](https://github.com/ibis-project/ibis/issues/1978))
* Add support for date/time operations in PySpark backend ([#1974](https://github.com/ibis-project/ibis/issues/1974))
* Add support for params, query_schema, and sql in PySpark backend ([#1973](https://github.com/ibis-project/ibis/issues/1973))
* Implement join for PySpark backend ([#1967](https://github.com/ibis-project/ibis/issues/1967))
* Validate AsOfJoin tolerance and attempt interval unit conversion ([#1952](https://github.com/ibis-project/ibis/issues/1952))
* filter for PySpark backend ([#1943](https://github.com/ibis-project/ibis/issues/1943))
* window operations for pyspark backend ([#1945](https://github.com/ibis-project/ibis/issues/1945))
* Implement IntervalSub for pandas backend ([#1951](https://github.com/ibis-project/ibis/issues/1951))
* PySpark backend string and column ops ([#1942](https://github.com/ibis-project/ibis/issues/1942))
* PySpark backend ([#1913](https://github.com/ibis-project/ibis/issues/1913))
* DDL support for Spark backend ([#1908](https://github.com/ibis-project/ibis/issues/1908))
* Support timezone aware arrow timestamps ([#1923](https://github.com/ibis-project/ibis/issues/1923))
* Add shapely geometries as input for literals ([#1860](https://github.com/ibis-project/ibis/issues/1860))
* Add geopandas as output for omniscidb ([#1858](https://github.com/ibis-project/ibis/issues/1858))
* Spark UDFs ([#1885](https://github.com/ibis-project/ibis/issues/1885))
* Add support for Postgres UDFs ([#1871](https://github.com/ibis-project/ibis/issues/1871))
* Spark tests ([#1830](https://github.com/ibis-project/ibis/issues/1830))
* Spark client ([#1807](https://github.com/ibis-project/ibis/issues/1807))
* Use pandas rolling apply to implement rows_with_max_lookback ([#1868](https://github.com/ibis-project/ibis/issues/1868))

## Bugs

* Pin "clickhouse-driver" to ">=0.1.3" ([#2089](https://github.com/ibis-project/ibis/issues/2089))
* Fix load data stage for Linux CI ([#2069](https://github.com/ibis-project/ibis/issues/2069))
* Fix datamgr.py fail if IBIS_TEST_OMNISCIDB_DATABASE=omnisci ([#2057](https://github.com/ibis-project/ibis/issues/2057))
* Change pymapd connection parameter from "session_id" to "sessionid" ([#2041](https://github.com/ibis-project/ibis/issues/2041))
* Fix pandas backend to treat trailing_window preceding arg as window bound rather than window size (e.g. preceding=0 now indicates current row rather than window size 0) ([#2009](https://github.com/ibis-project/ibis/issues/2009))
* Fix handling of Array types in Postgres UDF ([#2015](https://github.com/ibis-project/ibis/issues/2015))
* Fix pydocstyle config ([#2010](https://github.com/ibis-project/ibis/issues/2010))
* Pinning clickhouse-driver<0.1.2 ([#2006](https://github.com/ibis-project/ibis/issues/2006))
* Fix CI log for database ([#1984](https://github.com/ibis-project/ibis/issues/1984))
* Fixes explain operation ([#1933](https://github.com/ibis-project/ibis/issues/1933))
* Fix incorrect assumptions about attached SQLite databases ([#1937](https://github.com/ibis-project/ibis/issues/1937))
* Upgrade to JDK11 ([#1938](https://github.com/ibis-project/ibis/issues/1938))
* `sql` method doesn't work when the query uses LIMIT clause ([#1903](https://github.com/ibis-project/ibis/issues/1903))
* Fix union implementation ([#1910](https://github.com/ibis-project/ibis/issues/1910))
* Fix failing com imports on master ([#1912](https://github.com/ibis-project/ibis/issues/1912))
* OmniSci/MapD - Fix reduction for bool ([#1901](https://github.com/ibis-project/ibis/issues/1901))
* Pass scope to grouping execution in the pandas backend ([#1899](https://github.com/ibis-project/ibis/issues/1899))
* Fix various Spark backend issues ([#1888](https://github.com/ibis-project/ibis/issues/1888))
* Make Nodes enforce the proper signature ([#1891](https://github.com/ibis-project/ibis/issues/1891))
* Fix according to bug in pd.to_datetime when passing the unit flag ([#1893](https://github.com/ibis-project/ibis/issues/1893))
* Fix small formatting buglet in PR merge tool ([#1883](https://github.com/ibis-project/ibis/issues/1883))
* Fix the case where we do not have an index when using preceding with intervals ([#1876](https://github.com/ibis-project/ibis/issues/1876))
* Fixed issues with geo data ([#1872](https://github.com/ibis-project/ibis/issues/1872))
* Remove -x from pytest call in linux CI ([#1869](https://github.com/ibis-project/ibis/issues/1869))
* Fix return type of Struct.from_tuples ([#1867](https://github.com/ibis-project/ibis/issues/1867))

## Support

* Add support to Python 3.8 ([#2066](https://github.com/ibis-project/ibis/issues/2066))
* Pin back version of isort ([#2079](https://github.com/ibis-project/ibis/issues/2079))
* Use user-defined port variables for Omnisci and PostgreSQL tests ([#2082](https://github.com/ibis-project/ibis/issues/2082))
* Change omniscidb image tag from v5.0.0 to v5.1.0 on docker-compose recipe ([#2077](https://github.com/ibis-project/ibis/issues/2077))
* [Omnisci] The same SRIDs for test_geo_spatial_binops ([#2051](https://github.com/ibis-project/ibis/issues/2051))
* Unpin rtree version ([#2078](https://github.com/ibis-project/ibis/issues/2078))
* Link pandas issues with xfail tests in pandas/tests/test_udf.py ([#2074](https://github.com/ibis-project/ibis/issues/2074))
* Disable Postgres tests on Windows CI. ([#2075](https://github.com/ibis-project/ibis/issues/2075))
* use conda for installation black and isort tools ([#2068](https://github.com/ibis-project/ibis/issues/2068))
* CI: Fix CI builds related to new pandas 1.0 compatibility ([#2061](https://github.com/ibis-project/ibis/issues/2061))
* Fix data map for int8 on OmniSciDB backend ([#2056](https://github.com/ibis-project/ibis/issues/2056))
* Add possibility to run tests for separate backend via `make test BACKENDS=[YOUR BACKEND]` ([#2052](https://github.com/ibis-project/ibis/issues/2052))
* Fix "cudf" import on OmniSciDB backend ([#2055](https://github.com/ibis-project/ibis/issues/2055))
* CI: Drop table only if it exists (OmniSciDB) ([#2050](https://github.com/ibis-project/ibis/issues/2050))
* Add initial documentation for OmniSciDB, MySQL, PySpark and SparkSQL backends, add initial documentation for geospatial methods and add links to Ibis wiki page ([#2034](https://github.com/ibis-project/ibis/issues/2034))
* Implement covariance for bigquery backend ([#2044](https://github.com/ibis-project/ibis/issues/2044))
* Add Spark to supported backends list ([#2046](https://github.com/ibis-project/ibis/issues/2046))
* Ping dependency of rtree to fix CI failure ([#2043](https://github.com/ibis-project/ibis/issues/2043))
* Drop support for Python 3.5 ([#2037](https://github.com/ibis-project/ibis/issues/2037))
* HTML escape column names and types in png repr. ([#2023](https://github.com/ibis-project/ibis/issues/2023))
* Add geospatial tutorial notebook ([#1991](https://github.com/ibis-project/ibis/issues/1991))
* Change omniscidb image tag from v4.7.0 to v5.0.0 on docker-compose recipe ([#2031](https://github.com/ibis-project/ibis/issues/2031))
* Pin "semantic_version" to "<2.7" in the docs build CI, fix "builddoc" and "doc" section inside "Makefile" and skip mysql tzinfo on CI to allow to run MySQL using docker container on a hard disk drive. ([#2030](https://github.com/ibis-project/ibis/issues/2030))
* Fixed impala start up issues ([#2012](https://github.com/ibis-project/ibis/issues/2012))
* cache all ops in translate() ([#1999](https://github.com/ibis-project/ibis/issues/1999))
* Add black step to CI ([#1988](https://github.com/ibis-project/ibis/issues/1988))
* Json UUID any ([#1962](https://github.com/ibis-project/ibis/issues/1962))
* Add log for database services ([#1982](https://github.com/ibis-project/ibis/issues/1982))
* Fix BigQuery backend fixture so batting and awards_players fixture re… ([#1972](https://github.com/ibis-project/ibis/issues/1972))
* Disable BigQuery explicitly in all/test_join.py ([#1971](https://github.com/ibis-project/ibis/issues/1971))
* Re-formatting all files using pre-commit hook ([#1963](https://github.com/ibis-project/ibis/issues/1963))
* Disable codecov report upload during CI builds ([#1961](https://github.com/ibis-project/ibis/issues/1961))
* Developer doc enhancements ([#1960](https://github.com/ibis-project/ibis/issues/1960))
* Missing geospatial ops for OmniSciDB ([#1958](https://github.com/ibis-project/ibis/issues/1958))
* Remove pandas deprecation warnings ([#1950](https://github.com/ibis-project/ibis/issues/1950))
* Add developer docs to get docker setup ([#1948](https://github.com/ibis-project/ibis/issues/1948))
* More informative IntegrityError on duplicate columns ([#1949](https://github.com/ibis-project/ibis/issues/1949))
* Improve geospatial literals and smoke tests ([#1928](https://github.com/ibis-project/ibis/issues/1928))
* PostGIS enhancements ([#1925](https://github.com/ibis-project/ibis/issues/1925))
* Rename mapd to omniscidb backend ([#1866](https://github.com/ibis-project/ibis/issues/1866))
* Fix failing BigQuery tests ([#1926](https://github.com/ibis-project/ibis/issues/1926))
* Added missing null literal op ([#1917](https://github.com/ibis-project/ibis/issues/1917))
* Update link to Presto website ([#1895](https://github.com/ibis-project/ibis/issues/1895))
* Removing linting from windows ([#1896](https://github.com/ibis-project/ibis/issues/1896))
* Fix link to NUMFOCUS CoC ([#1884](https://github.com/ibis-project/ibis/issues/1884))
* Added CoC section ([#1882](https://github.com/ibis-project/ibis/issues/1882))
* Remove pandas exception for rows_with_max_lookback ([#1859](https://github.com/ibis-project/ibis/issues/1859))
* Move CI pipelines to Azure ([#1856](https://github.com/ibis-project/ibis/issues/1856))


# [1.2.0](https://github.com/ibis-project/ibis/releases/tag/1.2.0) (2019-06-24)

## Features

* Add new geospatial functions to OmniSciDB backend ([#1836](https://github.com/ibis-project/ibis/issues/1836))
* allow pandas timedelta in rows_with_max_lookback ([#1838](https://github.com/ibis-project/ibis/issues/1838))
* Accept rows-with-max-lookback as preceding parameter ([#1825](https://github.com/ibis-project/ibis/issues/1825))
* PostGIS support ([#1787](https://github.com/ibis-project/ibis/issues/1787))

## Bugs

* Fix call to psql causing failing CI ([#1855](https://github.com/ibis-project/ibis/issues/1855))
* Fix nested array literal repr ([#1851](https://github.com/ibis-project/ibis/issues/1851))
* Fix repr of empty schema ([#1850](https://github.com/ibis-project/ibis/issues/1850))
* Add max_lookback to window replace and combine functions ([#1843](https://github.com/ibis-project/ibis/issues/1843))
* Partially revert #1758 ([#1837](https://github.com/ibis-project/ibis/issues/1837))

## Support

* Skip SQLAlchemy backend tests in connect method in backends.py ([#1847](https://github.com/ibis-project/ibis/issues/1847))
* Validate order_by when using rows_with_max_lookback window ([#1848](https://github.com/ibis-project/ibis/issues/1848))
* Generate release notes from commits ([#1845](https://github.com/ibis-project/ibis/issues/1845))
* Raise exception on backends where rows_with_max_lookback can't be implemented ([#1844](https://github.com/ibis-project/ibis/issues/1844))
* Tighter version spec for pytest ([#1840](https://github.com/ibis-project/ibis/issues/1840))
* Allow passing a branch to ci/feedstock.py ([#1826](https://github.com/ibis-project/ibis/issues/1826))


# [1.1.0](https://github.com/ibis-project/ibis/releases/tag/1.1.0) (2019-06-09)

## Features

* Conslidate trailing window functions ([#1809](https://github.com/ibis-project/ibis/issues/1809))
* Call to_interval when casting integers to intervals ([#1766](https://github.com/ibis-project/ibis/issues/1766))
* Add session feature to mapd client API ([#1796](https://github.com/ibis-project/ibis/issues/1796))
* Add min periods parameter to Window ([#1792](https://github.com/ibis-project/ibis/issues/1792))
* Allow strings for types in pandas UDFs ([#1785](https://github.com/ibis-project/ibis/issues/1785))
* Add missing date operations and struct field operation for the pandas backend ([#1790](https://github.com/ibis-project/ibis/issues/1790))
* Add window operations to the OmniSci backend ([#1771](https://github.com/ibis-project/ibis/issues/1771))
* Reimplement the pandas backend using topological sort ([#1758](https://github.com/ibis-project/ibis/issues/1758))
* Add marker for xfailing specific backends ([#1778](https://github.com/ibis-project/ibis/issues/1778))
* Enable window function tests where possible ([#1777](https://github.com/ibis-project/ibis/issues/1777))
* is_computable_arg dispatcher ([#1743](https://github.com/ibis-project/ibis/issues/1743))
* Added float32 and geospatial types for create table from schema ([#1753](https://github.com/ibis-project/ibis/issues/1753))

## Bugs

* Fix group_concat test and implementations ([#1819](https://github.com/ibis-project/ibis/issues/1819))
* Fix failing strftime tests on Python 3.7 ([#1818](https://github.com/ibis-project/ibis/issues/1818))
* Remove unnecessary (and erroneous in some cases) frame clauses ([#1757](https://github.com/ibis-project/ibis/issues/1757))
* Chained mutate operations are buggy ([#1799](https://github.com/ibis-project/ibis/issues/1799))
* Allow projections from joins to attempt fusion ([#1783](https://github.com/ibis-project/ibis/issues/1783))
* Fix Python 3.5 dependency versions ([#1798](https://github.com/ibis-project/ibis/issues/1798))
* Fix compatibility and bugs associated with pandas toposort reimplementation ([#1789](https://github.com/ibis-project/ibis/issues/1789))
* Fix outer_join generating LEFT join instead of FULL OUTER ([#1772](https://github.com/ibis-project/ibis/issues/1772))
* NullIf should enforce that its arguments are castable to a common type ([#1782](https://github.com/ibis-project/ibis/issues/1782))
* Fix conda create command in documentation ([#1775](https://github.com/ibis-project/ibis/issues/1775))
* Fix preceding and following with ``None`` ([#1765](https://github.com/ibis-project/ibis/issues/1765))
* PostgreSQL interval type not recognized ([#1661](https://github.com/ibis-project/ibis/issues/1661))

## Support

* Remove decorator hacks and add custom markers ([#1820](https://github.com/ibis-project/ibis/issues/1820))
* Add development deps to setup.py ([#1814](https://github.com/ibis-project/ibis/issues/1814))
* Fix design and developer docs ([#1805](https://github.com/ibis-project/ibis/issues/1805))
* Pin sphinx version to 2.0.1 ([#1810](https://github.com/ibis-project/ibis/issues/1810))
* Add pep8speaks integration ([#1793](https://github.com/ibis-project/ibis/issues/1793))
* Fix typo in UDF signature specification ([#1821](https://github.com/ibis-project/ibis/issues/1821))
* Clean up most xpassing tests ([#1779](https://github.com/ibis-project/ibis/issues/1779))
* Update omnisci container version ([#1781](https://github.com/ibis-project/ibis/issues/1781))
* Constrain PyMapD version to get passing builds ([#1776](https://github.com/ibis-project/ibis/issues/1776))
* Remove warnings and clean up some docstrings ([#1763](https://github.com/ibis-project/ibis/issues/1763))
* Add StringToTimestamp as unsupported ([#1638](https://github.com/ibis-project/ibis/issues/1638))
* Add isort pre-commit hooks ([#1759](https://github.com/ibis-project/ibis/issues/1759))
* Add Python 3.5 testing back to CI ([#1750](https://github.com/ibis-project/ibis/issues/1750))
* Re-enable CI for building step ([#1700](https://github.com/ibis-project/ibis/issues/1700))
* Update README reference to MapD to say OmniSci ([#1749](https://github.com/ibis-project/ibis/issues/1749))


# [1.0.0](https://github.com/ibis-project/ibis/releases/tag/1.0.0) (2019-03-26)

## Features

* Add black as a pre-commit hook ([#1735](https://github.com/ibis-project/ibis/issues/1735))
* Add support for the arbitrary aggregate in the mapd backend ([#1680](https://github.com/ibis-project/ibis/issues/1680))
* Add SQL method for the MapD backend ([#1731](https://github.com/ibis-project/ibis/issues/1731))
* Clean up merge PR script and use the actual merge feature of GitHub ([#1744](https://github.com/ibis-project/ibis/issues/1744))
* Add cross join to the pandas backend ([#1723](https://github.com/ibis-project/ibis/issues/1723))
* Implement default handler for multiple client ``pre_execute`` ([#1727](https://github.com/ibis-project/ibis/issues/1727))
* Implement BigQuery auth using ``pydata_google_auth`` ([#1728](https://github.com/ibis-project/ibis/issues/1728))
* Timestamp literal accepts a timezone parameter ([#1712](https://github.com/ibis-project/ibis/issues/1712))
* Remove support for passing integers to ``ibis.timestamp`` ([#1725](https://github.com/ibis-project/ibis/issues/1725))
* Add ``find_nodes`` to lineage ([#1704](https://github.com/ibis-project/ibis/issues/1704))
* Remove a bunch of deprecated APIs and clean up warnings ([#1714](https://github.com/ibis-project/ibis/issues/1714))
* Implement table distinct for the pandas backend ([#1716](https://github.com/ibis-project/ibis/issues/1716))
* Implement geospatial functions for MapD ([#1678](https://github.com/ibis-project/ibis/issues/1678))
* Implement geospatial types for MapD ([#1666](https://github.com/ibis-project/ibis/issues/1666))
* Add pre commit hook ([#1685](https://github.com/ibis-project/ibis/issues/1685))
* Getting started with mapd, mysql and pandas ([#1686](https://github.com/ibis-project/ibis/issues/1686))
* Support column names with special characters in mapd ([#1675](https://github.com/ibis-project/ibis/issues/1675))
* Allow operations to hide arguments from display ([#1669](https://github.com/ibis-project/ibis/issues/1669))
* Remove implicit ordering requirements in the PostgreSQL backend ([#1636](https://github.com/ibis-project/ibis/issues/1636))
* Add cross join operator to MapD ([#1655](https://github.com/ibis-project/ibis/issues/1655))
* Fix UDF bugs and add support for non-aggregate analytic functions ([#1637](https://github.com/ibis-project/ibis/issues/1637))
* Support string slicing with other expressions ([#1627](https://github.com/ibis-project/ibis/issues/1627))
* Publish the ibis roadmap ([#1618](https://github.com/ibis-project/ibis/issues/1618))
* Implement ``approx_median`` in BigQuery ([#1604](https://github.com/ibis-project/ibis/issues/1604))
* Make ibis node instances hashable ([#1611](https://github.com/ibis-project/ibis/issues/1611))
* Add ``range_window`` and ``trailing_range_window`` to docs ([#1608](https://github.com/ibis-project/ibis/issues/1608))

## Bugs

* Make ``dev/merge-pr.py`` script handle PR branches ([#1745](https://github.com/ibis-project/ibis/issues/1745))
* Fix ``NULLIF`` implementation for the pandas backend ([#1742](https://github.com/ibis-project/ibis/issues/1742))
* Fix casting to float in the MapD backend ([#1737](https://github.com/ibis-project/ibis/issues/1737))
* Fix testing for BigQuery after auth flow update ([#1741](https://github.com/ibis-project/ibis/issues/1741))
* Fix skipping for new BigQuery auth flow ([#1738](https://github.com/ibis-project/ibis/issues/1738))
* Fix bug in ``TableExpr.drop`` ([#1732](https://github.com/ibis-project/ibis/issues/1732))
* Filter the ``raw`` warning from newer pandas to support older pandas ([#1729](https://github.com/ibis-project/ibis/issues/1729))
* Fix BigQuery credentials link ([#1706](https://github.com/ibis-project/ibis/issues/1706))
* Add Union as an unsuppoted operation for MapD ([#1639](https://github.com/ibis-project/ibis/issues/1639))
* Fix visualizing an ibis expression when showing a selection after a table join ([#1705](https://github.com/ibis-project/ibis/issues/1705))
* Fix MapD exception for ``toDateTime`` ([#1659](https://github.com/ibis-project/ibis/issues/1659))
* Use ``==`` to compare strings ([#1701](https://github.com/ibis-project/ibis/issues/1701))
* Resolves joining with different column names ([#1647](https://github.com/ibis-project/ibis/issues/1647))
* Fix map get with compatible types ([#1643](https://github.com/ibis-project/ibis/issues/1643))
* Fixed where operator for MapD ([#1653](https://github.com/ibis-project/ibis/issues/1653))
* Remove parameters from mapd ([#1648](https://github.com/ibis-project/ibis/issues/1648))
* Make sure we cast when NULL is else in CASE expressions ([#1651](https://github.com/ibis-project/ibis/issues/1651))
* Fix equality ([#1600](https://github.com/ibis-project/ibis/issues/1600))

## Support

* Do not build universal wheels ([#1748](https://github.com/ibis-project/ibis/issues/1748))
* Remove tag prefix from versioneer ([#1747](https://github.com/ibis-project/ibis/issues/1747))
* Use releases to manage documentation ([#1746](https://github.com/ibis-project/ibis/issues/1746))
* Use cudf instead of pygdf ([#1694](https://github.com/ibis-project/ibis/issues/1694))
* Fix multiple CI issues ([#1696](https://github.com/ibis-project/ibis/issues/1696))
* Update mapd ci to v4.4.1 ([#1681](https://github.com/ibis-project/ibis/issues/1681))
* Enabled mysql CI on azure pipelines ([#1672](https://github.com/ibis-project/ibis/issues/1672))
* Remove support for Python 2 ([#1670](https://github.com/ibis-project/ibis/issues/1670))
* Fix flake8 and many other warnings ([#1667](https://github.com/ibis-project/ibis/issues/1667))
* Update README.md for impala and kudu ([#1664](https://github.com/ibis-project/ibis/issues/1664))
* Remove defaults as a channel from azure pipelines ([#1660](https://github.com/ibis-project/ibis/issues/1660))
* Fixes a very typo in the pandas/core.py docstring ([#1658](https://github.com/ibis-project/ibis/issues/1658))
* Unpin clickhouse-driver version ([#1657](https://github.com/ibis-project/ibis/issues/1657))
* Add test for reduction returning lists ([#1650](https://github.com/ibis-project/ibis/issues/1650))
* Fix Azure VM image name ([#1646](https://github.com/ibis-project/ibis/issues/1646))
* Updated MapD server-CI ([#1641](https://github.com/ibis-project/ibis/issues/1641))
* Add TableExpr.drop to API documentation ([#1645](https://github.com/ibis-project/ibis/issues/1645))
* Fix Azure deployment step ([#1642](https://github.com/ibis-project/ibis/issues/1642))
* Set up CI with Azure Pipelines ([#1640](https://github.com/ibis-project/ibis/issues/1640))
* Fix conda builds ([#1609](https://github.com/ibis-project/ibis/issues/1609))

# v0.14.0 (2018-08-23)

This release brings refactored, more composable core components and rule
system to ibis. We also focused quite heavily on the BigQuery backend
this release.

## New Features

-   Allow keyword arguments in Node subclasses ([#968](https://github.com/ibis-project/ibis/issues/968))
-   Splat args into Node subclasses instead of requiring a list
    ([#969](https://github.com/ibis-project/ibis/issues/969))
-   Add support for `UNION` in the BigQuery backend
    ([#1408](https://github.com/ibis-project/ibis/issues/1408), [#1409](https://github.com/ibis-project/ibis/issues/1409))
-   Support for writing UDFs in BigQuery ([#1377](https://github.com/ibis-project/ibis/issues/1377)). See the BigQuery UDF docs for more details.
-   Support for cross-project expressions in the BigQuery backend.
    ([#1427](https://github.com/ibis-project/ibis/issues/1427), [#1428](https://github.com/ibis-project/ibis/issues/1428))
-   Add `strftime` and `to_timestamp` support for BigQuery
    ([#1422](https://github.com/ibis-project/ibis/issues/1422), [#1410](https://github.com/ibis-project/ibis/issues/1410))
-   Require `google-cloud-bigquery >=1.0` ([#1424](https://github.com/ibis-project/ibis/issues/1424))
-   Limited support for interval arithmetic in the pandas backend
    ([#1407](https://github.com/ibis-project/ibis/issues/1407))
-   Support for subclassing `TableExpr` ([#1439](https://github.com/ibis-project/ibis/issues/1439))
-   Fill out pandas backend operations ([#1423](https://github.com/ibis-project/ibis/issues/1423))
-   Add common DDL APIs to the pandas backend ([#1464](https://github.com/ibis-project/ibis/issues/1464))
-   Implement the `sql` method for BigQuery ([#1463](https://github.com/ibis-project/ibis/issues/1463))
-   Add `to_timestamp` for BigQuery ([#1455](https://github.com/ibis-project/ibis/issues/1455))
-   Add the `mapd` backend ([#1419](https://github.com/ibis-project/ibis/issues/1419))
-   Implement range windows ([#1349](https://github.com/ibis-project/ibis/issues/1349))
-   Support for map types in the pandas backend
    ([#1498](https://github.com/ibis-project/ibis/issues/1498))
-   Add `mean` and `sum` for `boolean` types in BigQuery
    ([#1516](https://github.com/ibis-project/ibis/issues/1516))
-   All recent versions of SQLAlchemy are now suppported
    ([#1384](https://github.com/ibis-project/ibis/issues/1384))
-   Add support for `NUMERIC` types in the BigQuery backend
    ([#1534](https://github.com/ibis-project/ibis/issues/1534))
-   Speed up grouped and rolling operations in the pandas backend
    ([#1549](https://github.com/ibis-project/ibis/issues/1549))
-   Implement `TimestampNow` for BigQuery and pandas
    ([#1575](https://github.com/ibis-project/ibis/issues/1575))

## Bug Fixes

-   Nullable property is now propagated through value types
    ([#1289](https://github.com/ibis-project/ibis/issues/1289))
-   Implicit casting between signed and unsigned integers checks
    boundaries
-   Fix precedence of case statement ([#1412](https://github.com/ibis-project/ibis/issues/1412))
-   Fix handling of large timestamps ([#1440](https://github.com/ibis-project/ibis/issues/1440))
-   Fix `identical_to` precedence ([#1458](https://github.com/ibis-project/ibis/issues/1458))
-   Pandas 0.23 compatibility ([#1458](https://github.com/ibis-project/ibis/issues/1458))
-   Preserve timezones in timestamp-typed literals
    ([#1459](https://github.com/ibis-project/ibis/issues/1459))
-   Fix incorrect topological ordering of `UNION` expressions
    ([#1501](https://github.com/ibis-project/ibis/issues/1501))
-   Fix projection fusion bug when attempting to fuse columns of the
    same name ([#1496](https://github.com/ibis-project/ibis/issues/1496))
-   Fix output type for some decimal operations
    ([#1541](https://github.com/ibis-project/ibis/issues/1541))

## API Changes

-   The previous, private rules API has been rewritten
    ([#1366](https://github.com/ibis-project/ibis/issues/1366))
-   Defining input arguments for operations happens in a more readable
    fashion instead of the previous [input_type]{.title-ref} list.
-   Removed support for async query execution (only Impala supported)
-   Remove support for Python 3.4 ([#1326](https://github.com/ibis-project/ibis/issues/1326))
-   BigQuery division defaults to using `IEEE_DIVIDE`
    ([#1390](https://github.com/ibis-project/ibis/issues/1390))
-   Add `tolerance` parameter to `asof_join` ([#1443](https://github.com/ibis-project/ibis/issues/1443))

# v0.13.0 (2018-03-30)

This release brings new backends, including support for executing
against files, MySQL, Pandas user defined scalar and aggregations along
with a number of bug fixes and reliability enhancements. We recommend
that all users upgrade from earlier versions of Ibis.

## New Backends

-   File Support for CSV & HDF5 ([#1165](https://github.com/ibis-project/ibis/issues/1165), [#1194](https://github.com/ibis-project/ibis/issues/1194))
-   File Support for Parquet Format ([#1175](https://github.com/ibis-project/ibis/issues/1175), [#1194](https://github.com/ibis-project/ibis/issues/1194))
-   Experimental support for `MySQL` thanks to \@kszucs
    ([#1224](https://github.com/ibis-project/ibis/issues/1224))

## New Features

-   Support for Unsigned Integer Types ([#1194](https://github.com/ibis-project/ibis/issues/1194))
-   Support for Interval types and expressions with support for
    execution on the Impala and Clickhouse backends
    ([#1243](https://github.com/ibis-project/ibis/issues/1243))
-   Isnan, isinf operations for float and double values
    ([#1261](https://github.com/ibis-project/ibis/issues/1261))
-   Support for an interval with a quarter period
    ([#1259](https://github.com/ibis-project/ibis/issues/1259))
-   `ibis.pandas.from_dataframe` convenience function
    ([#1155](https://github.com/ibis-project/ibis/issues/1155))
-   Remove the restriction on `ROW_NUMBER()` requiring it to have an
    `ORDER BY` clause ([#1371](https://github.com/ibis-project/ibis/issues/1371))
-   Add `.get()` operation on a Map type ([#1376](https://github.com/ibis-project/ibis/issues/1376))
-   Allow visualization of custom defined expressions
-   Add experimental support for pandas UDFs/UDAFs
    ([#1277](https://github.com/ibis-project/ibis/issues/1277))
-   Functions can be used as groupby keys ([#1214](https://github.com/ibis-project/ibis/issues/1214), [#1215](https://github.com/ibis-project/ibis/issues/1215))
-   Generalize the use of the `where` parameter to reduction operations
    ([#1220](https://github.com/ibis-project/ibis/issues/1220))
-   Support for interval operations thanks to \@kszucs
    ([#1243](https://github.com/ibis-project/ibis/issues/1243), [#1260](https://github.com/ibis-project/ibis/issues/1260), [#1249](https://github.com/ibis-project/ibis/issues/1249))
-   Support for the `PARTITIONTIME` column in the BigQuery backend
    ([#1322](https://github.com/ibis-project/ibis/issues/1322))
-   Add `arbitrary()` method for selecting the first non null value in a
    column ([#1230](https://github.com/ibis-project/ibis/issues/1230),
    [#1309](https://github.com/ibis-project/ibis/issues/1309))
-   Windowed `MultiQuantile` operation in the pandas backend thanks to
    \@DiegoAlbertoTorres ([#1343](https://github.com/ibis-project/ibis/issues/1343))
-   Rules for validating table expressions thanks to
    \@DiegoAlbertoTorres ([#1298](https://github.com/ibis-project/ibis/issues/1298))
-   Complete end-to-end testing framework for all supported backends
    ([#1256](https://github.com/ibis-project/ibis/issues/1256))
-   `contains`/`not contains` now supported in the pandas backend
    ([#1210](https://github.com/ibis-project/ibis/issues/1210), [#1211](https://github.com/ibis-project/ibis/issues/1211))
-   CI builds are now reproducible *locally* thanks to \@kszucs
    ([#1121](https://github.com/ibis-project/ibis/issues/1121), [#1237](https://github.com/ibis-project/ibis/issues/1237), [#1255](https://github.com/ibis-project/ibis/issues/1255),
    [#1311](https://github.com/ibis-project/ibis/issues/1311))
-   `isnan`/`isinf` operations thanks to \@kszucs
    ([#1261](https://github.com/ibis-project/ibis/issues/1261))
-   Framework for generalized dtype and schema inference, and implicit
    casting thanks to \@kszucs ([#1221](https://github.com/ibis-project/ibis/issues/1221), [#1269](https://github.com/ibis-project/ibis/issues/1269))
-   Generic utilities for expression traversal thanks to \@kszucs
    ([#1336](https://github.com/ibis-project/ibis/issues/1336))
-   `day_of_week` API ([#306](https://github.com/ibis-project/ibis/issues/306),
    [#1047](https://github.com/ibis-project/ibis/issues/1047))
-   Design documentation for ibis ([#1351](https://github.com/ibis-project/ibis/issues/1351))

## Bug Fixes

-   Unbound parameters were failing in the simple case of a
    `ibis.expr.types.TableExpr.mutate`
    call with no operation ([#1378](https://github.com/ibis-project/ibis/issues/1378))
-   Fix parameterized subqueries ([#1300](https://github.com/ibis-project/ibis/issues/1300), [#1331](https://github.com/ibis-project/ibis/issues/1331),
    [#1303](https://github.com/ibis-project/ibis/issues/1303), [#1378](https://github.com/ibis-project/ibis/issues/1378))
-   Fix subquery extraction, which wasn\'t happening in topological
    order ([#1342](https://github.com/ibis-project/ibis/issues/1342))
-   Fix parenthesization if `isnull` ([#1307](https://github.com/ibis-project/ibis/issues/1307))
-   Calling drop after mutate did not work ([#1296](https://github.com/ibis-project/ibis/issues/1296), [#1299](https://github.com/ibis-project/ibis/issues/1299))
-   SQLAlchemy backends were missing an implementation of
    `ibis.expr.operations.NotContains`.
-   Support `REGEX_EXTRACT` in PostgreSQL 10 ([#1276](https://github.com/ibis-project/ibis/issues/1276), [#1278](https://github.com/ibis-project/ibis/issues/1278))

## API Changes

-   Fixing [#1378](https://github.com/ibis-project/ibis/issues/1378) required the removal
    of the `name` parameter to the `ibis.param` function. Use the
    `ibis.expr.types.Expr.name` method
    instead.

# v0.12.0 (2017-10-28)

This release brings Clickhouse and BigQuery SQL support along with a
number of bug fixes and reliability enhancements. We recommend that all
users upgrade from earlier versions of Ibis.

## New Backends

-   BigQuery backend ([#1170](https://github.com/ibis-project/ibis/issues/1170)), thanks
    to \@tsdlovell.
-   Clickhouse backend ([#1127](https://github.com/ibis-project/ibis/issues/1127)),
    thanks to \@kszucs.

## New Features

-   Add support for `Binary` data type ([#1183](https://github.com/ibis-project/ibis/issues/1183))
-   Allow users of the BigQuery client to define their own API proxy
    classes ([#1188](https://github.com/ibis-project/ibis/issues/1188))
-   Add support for HAVING in the pandas backend
    ([#1182](https://github.com/ibis-project/ibis/issues/1182))
-   Add struct field tab completion ([#1178](https://github.com/ibis-project/ibis/issues/1178))
-   Add expressions for Map/Struct types and columns
    ([#1166](https://github.com/ibis-project/ibis/issues/1166))
-   Support Table.asof_join ([#1162](https://github.com/ibis-project/ibis/issues/1162))
-   Allow right side of arithmetic operations to take over
    ([#1150](https://github.com/ibis-project/ibis/issues/1150))
-   Add a data_preload step in pandas backend ([#1142](https://github.com/ibis-project/ibis/issues/1142))
-   expressions in join predicates in the pandas backend
    ([#1138](https://github.com/ibis-project/ibis/issues/1138))
-   Scalar parameters ([#1075](https://github.com/ibis-project/ibis/issues/1075))
-   Limited window function support for pandas ([#1083](https://github.com/ibis-project/ibis/issues/1083))
-   Implement Time datatype ([#1105](https://github.com/ibis-project/ibis/issues/1105))
-   Implement array ops for pandas ([#1100](https://github.com/ibis-project/ibis/issues/1100))
-   support for passing multiple quantiles in `.quantile()`
    ([#1094](https://github.com/ibis-project/ibis/issues/1094))
-   support for clip and quantile ops on DoubleColumns
    ([#1090](https://github.com/ibis-project/ibis/issues/1090))
-   Enable unary math operations for pandas, sqlite
    ([#1071](https://github.com/ibis-project/ibis/issues/1071))
-   Enable casting from strings to temporal types
    ([#1076](https://github.com/ibis-project/ibis/issues/1076))
-   Allow selection of whole tables in pandas joins
    ([#1072](https://github.com/ibis-project/ibis/issues/1072))
-   Implement comparison for string vs date and timestamp types
    ([#1065](https://github.com/ibis-project/ibis/issues/1065))
-   Implement isnull and notnull for pandas ([#1066](https://github.com/ibis-project/ibis/issues/1066))
-   Allow like operation to accept a list of conditions to match
    ([#1061](https://github.com/ibis-project/ibis/issues/1061))
-   Add a pre_execute step in pandas backend ([#1189](https://github.com/ibis-project/ibis/issues/1189))

## Bug Fixes

-   Remove global expression caching to ensure repeatable code
    generation ([#1179](https://github.com/ibis-project/ibis/issues/1179),
    [#1181](https://github.com/ibis-project/ibis/issues/1181))
-   Fix `ORDER BY` generation without a `GROUP BY`
    ([#1180](https://github.com/ibis-project/ibis/issues/1180), [#1181](https://github.com/ibis-project/ibis/issues/1181))
-   Ensure that `~ibis.expr.datatypes.DataType` and subclasses hash properly ([#1172](https://github.com/ibis-project/ibis/issues/1172))
-   Ensure that the pandas backend can deal with unary operations in
    groupby
-   ([#1182](https://github.com/ibis-project/ibis/issues/1182))
-   Incorrect impala code generated for NOT with complex argument
    ([#1176](https://github.com/ibis-project/ibis/issues/1176))
-   BUG/CLN: Fix predicates on Selections on Joins
    ([#1149](https://github.com/ibis-project/ibis/issues/1149))
-   Don\'t use SET LOCAL to allow redshift to work
    ([#1163](https://github.com/ibis-project/ibis/issues/1163))
-   Allow empty arrays as arguments ([#1154](https://github.com/ibis-project/ibis/issues/1154))
-   Fix column renaming in groupby keys ([#1151](https://github.com/ibis-project/ibis/issues/1151))
-   Ensure that we only cast if timezone is not None
    ([#1147](https://github.com/ibis-project/ibis/issues/1147))
-   Fix location of conftest.py ([#1107](https://github.com/ibis-project/ibis/issues/1107))
-   TST/Make sure we drop tables during postgres testing
    ([#1101](https://github.com/ibis-project/ibis/issues/1101))
-   Fix misleading join error message ([#1086](https://github.com/ibis-project/ibis/issues/1086))
-   BUG/TST: Make hdfs an optional dependency ([#1082](https://github.com/ibis-project/ibis/issues/1082))
-   Memoization should include expression name where available
    ([#1080](https://github.com/ibis-project/ibis/issues/1080))

## Performance Enhancements

-   Speed up imports ([#1074](https://github.com/ibis-project/ibis/issues/1074))
-   Fix execution perf of groupby and selection
    ([#1073](https://github.com/ibis-project/ibis/issues/1073))
-   Use normalize for casting to dates in pandas
    ([#1070](https://github.com/ibis-project/ibis/issues/1070))
-   Speed up pandas groupby ([#1067](https://github.com/ibis-project/ibis/issues/1067))

## Contributors

The following people contributed to the 0.12.0 release :

    $ git shortlog -sn --no-merges v0.11.2..v0.12.0
    63  Phillip Cloud
     8  Jeff Reback
     2  Krisztián Szűcs
     2  Tory Haavik
     1  Anirudh
     1  Szucs Krisztian
     1  dlovell
     1  kwangin

# 0.11.0 (2017-06-28)

This release brings initial Pandas backend support along with a number
of bug fixes and reliability enhancements. We recommend that all users
upgrade from earlier versions of Ibis.

## New Features

-   Experimental pandas backend to allow execution of ibis expression
    against pandas DataFrames
-   Graphviz visualization of ibis expressions. Implements `_repr_png_`
    for Jupyter Notebook functionality
-   Ability to create a partitioned table from an ibis expression
-   Support for missing operations in the SQLite backend: sqrt, power,
    variance, and standard deviation, regular expression functions, and
    missing power support for PostgreSQL
-   Support for schemas inside databases with the PostgreSQL backend
-   Appveyor testing on core ibis across all supported Python versions
-   Add `year`/`month`/`day` methods to `date` types
-   Ability to sort, group by and project columns according to
    positional index rather than only by name
-   Added a `type` parameter to `ibis.literal` to allow user
    specification of literal types

## Bug Fixes

-   Fix broken conda recipe
-   Fix incorrectly typed fillna operation
-   Fix postgres boolean summary operations
-   Fix kudu support to reflect client API Changes
-   Fix equality of nested types and construction of nested types when
    the value type is specified as a string

## API Changes

-   Deprecate passing integer values to the `ibis.timestamp` literal
    constructor, this will be removed in 0.12.0
-   Added the `admin_timeout` parameter to the kudu client `connect`
    function

## Contributors

    $ git shortlog --summary --numbered v0.10.0..v0.11.0

      58 Phillip Cloud
       1 Greg Rahn
       1 Marius van Niekerk
       1 Tarun Gogineni
       1 Wes McKinney

# 0.8 (2016-05-19)

This release brings initial PostgreSQL backend support along with a
number of critical bug fixes and usability improvements. As several
correctness bugs with the SQL compiler were fixed, we recommend that all
users upgrade from earlier versions of Ibis.

## New Features

-   Initial PostgreSQL backend contributed by Phillip Cloud.
-   Add `groupby` as an alias for `group_by` to table expressions

## Bug Fixes

-   Fix an expression error when filtering based on a new field
-   Fix Impala\'s SQL compilation of using `OR` with compound filters
-   Various fixes with the `having(...)` function in grouped table
    expressions
-   Fix CTE (`WITH`) extraction inside `UNION ALL` expressions.
-   Fix `ImportError` on Python 2 when `mock` library not installed

## API Changes

-   The deprecated `ibis.impala_connect` and `ibis.make_client` APIs
    have been removed

# 0.7 (2016-03-16)

This release brings initial Kudu-Impala integration and improved Impala
and SQLite support, along with several critical bug fixes.

## New Features

-   Apache Kudu (incubating) integration for Impala users. See the [blog
    post](http://blog.ibis-project.org/kudu-impala-ibis) for now. Will
    add some documentation here when possible.
-   Add `use_https` option to `ibis.hdfs_connect` for WebHDFS
    connections in secure (Kerberized) clusters without SSL enabled.
-   Correctly compile aggregate expressions involving multiple
    subqueries.

To explain this last point in more detail, suppose you had:

``` python
table = ibis.table([('flag', 'string'),
                    ('value', 'double')],
                   'tbl')

flagged = table[table.flag == '1']
unflagged = table[table.flag == '0']

fv = flagged.value
uv = unflagged.value

expr = (fv.mean() / fv.sum()) - (uv.mean() / uv.sum())
```

The last expression now generates the correct Impala or SQLite SQL:

``` sql
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
```

## Bug Fixes

-   `CHAR(n)` and `VARCHAR(n)` Impala types now correctly map to Ibis
    string expressions
-   Fix inappropriate projection-join-filter expression rewrites
    resulting in incorrect generated SQL.
-   `ImpalaClient.create_table` correctly passes `STORED AS PARQUET` for
    `format='parquet'`.
-   Fixed several issues with Ibis dependencies (impyla, thriftpy, sasl,
    thrift_sasl), especially for secure clusters. Upgrading will pull in
    these new dependencies.
-   Do not fail in `ibis.impala.connect` when trying to create the
    temporary Ibis database if no HDFS connection passed.
-   Fix join predicate evaluation bug when column names overlap with
    table attributes.
-   Fix handling of fully-materialized joins (aka `select *` joins) in
    SQLAlchemy / SQLite.

## Contributors

Thank you to all who contributed patches to this release.

    $ git log v0.6.0..v0.7.0 --pretty=format:%aN | sort | uniq -c | sort -rn
        21 Wes McKinney
         1 Uri Laserson
         1 Kristopher Overholt

# 0.6 (2015-12-01)

This release brings expanded pandas and Impala integration, including
support for managing partitioned tables in Impala. See the new
`Ibis for Impala Users` guide for more on
using Ibis with Impala.

The `Ibis for SQL Programmers` guide
also was written since the 0.5 release.

This release also includes bug fixes affecting generated SQL
correctness. All users should upgrade as soon as possible.

## New Features

-   New integrated Impala functionality. See `Ibis for Impala Users` for more details on
    these things.
    -   Improved Impala-pandas integration. Create tables or insert into
        existing tables from pandas `DataFrame` objects.
    -   Partitioned table metadata management API. Add, drop, alter, and
        insert into table partitions.
    -   Add `is_partitioned` property to `ImpalaTable`.
    -   Added support for `LOAD DATA` DDL using the `load_data`
        function, also supporting partitioned tables.
    -   Modify table metadata (location, format, SerDe properties etc.)
        using `ImpalaTable.alter`
    -   Interrupting Impala expression execution with Control-C will
        attempt to cancel the running query with the server.
    -   Set the compression codec (e.g. snappy) used with
        `ImpalaClient.set_compression_codec`.
    -   Get and set query options for a client session with
        `ImpalaClient.get_options` and `ImpalaClient.set_options`.
    -   Add `ImpalaTable.metadata` method that parses the output of the
        `DESCRIBE FORMATTED` DDL to simplify table metadata inspection.
    -   Add `ImpalaTable.stats` and `ImpalaTable.column_stats` to see
        computed table and partition statistics.
    -   Add `CHAR` and `VARCHAR` handling
    -   Add `refresh`, `invalidate_metadata` DDL options and add
        `incremental` option to `compute_stats` for
        `COMPUTE INCREMENTAL STATS`.
-   Add `substitute` method for performing multiple value substitutions
    in an array or scalar expression.
-   Division is by default *true division* like Python 3 for all numeric
    data. This means for SQL systems that use C-style division
    semantics, the appropriate `CAST` will be automatically inserted in
    the generated SQL.
-   Easier joins on tables with overlapping column names. See `Ibis for SQL Programmers`.
-   Expressions like `string_expr[:3]` now work as expected.
-   Add `coalesce` instance method to all value expressions.
-   Passing `limit=None` to the `execute` method on expressions disables
    any default row limits.

## API Changes

-   `ImpalaTable.rename` no longer mutates the calling table expression.

## Contributors

    $ git log v0.5.0..v0.6.0 --pretty=format:%aN | sort | uniq -c | sort -rn
    46 Wes McKinney
     3 Uri Laserson
     1 Phillip Cloud
     1 mariusvniekerk
     1 Kristopher Overholt

# 0.5 (2015-09-10)

Highlights in this release are the SQLite, Python 3, Impala UDA support,
and an asynchronous execution API. There are also many usability
improvements, bug fixes, and other new features.

## New Features

-   SQLite client and built-in function support
-   Ibis now supports Python 3.4 as well as 2.6 and 2.7
-   Ibis can utilize Impala user-defined aggregate (UDA) functions
-   SQLAlchemy-based translation toolchain to enable more SQL engines
    having SQLAlchemy dialects to be supported
-   Many window function usability improvements (nested analytic
    functions and deferred binding conveniences)
-   More convenient aggregation with keyword arguments in `aggregate`
    functions
-   Built preliminary wrapper API for MADLib-on-Impala
-   Add `var` and `std` aggregation methods and support in Impala
-   Add `nullifzero` numeric method for all SQL engines
-   Add `rename` method to Impala tables (for renaming tables in the
    Hive metastore)
-   Add `close` method to `ImpalaClient` for session cleanup (#533)
-   Add `relabel` method to table expressions
-   Add `insert` method to Impala tables
-   Add `compile` and `verify` methods to all expressions to test
    compilation and ability to compile (since many operations are
    unavailable in SQLite, for example)

## API Changes

-   Impala Ibis client creation now uses only `ibis.impala.connect`, and
    `ibis.make_client` has been deprecated

## Contributors

    $ git log v0.4.0..v0.5.0 --pretty=format:%aN | sort | uniq -c | sort -rn
          55 Wes McKinney
          9 Uri Laserson
          1 Kristopher Overholt

# 0.4 (2015-08-14)

## New Features

-   Add tooling to use Impala C++ scalar UDFs within Ibis (#262, #195)
-   Support and testing for Kerberos-enabled secure HDFS clusters
-   Many table functions can now accept functions as parameters (invoked
    on the calling table) to enhance composability and emulate
    late-binding semantics of languages (like R) that have non-standard
    evaluation (#460)
-   Add `any`, `all`, `notany`, and `notall` reductions on boolean
    arrays, as well as `cumany` and `cumall`
-   Using `topk` now produces an analytic expression that is executable
    (as an aggregation) but can also be used as a filter as before
    (#392, #91)
-   Added experimental database object \"usability layer\", see
    `ImpalaClient.database`.
-   Add `TableExpr.info`
-   Add `compute_stats` API to table expressions referencing physical
    Impala tables
-   Add `explain` method to `ImpalaClient` to show query plan for an
    expression
-   Add `chmod` and `chown` APIs to `HDFS` interface for superusers
-   Add `convert_base` method to strings and integer types
-   Add option to `ImpalaClient.create_table` to create empty
    partitioned tables
-   `ibis.cross_join` can now join more than 2 tables at once
-   Add `ImpalaClient.raw_sql` method for running naked SQL queries
-   `ImpalaClient.insert` now validates schemas locally prior to sending
    query to cluster, for better usability.
-   Add conda installation recipes

## Contributors

    $ git log v0.3.0..v0.4.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         38 Wes McKinney
          9 Uri Laserson
          2 Meghana Vuyyuru
          2 Kristopher Overholt
          1 Marius van Niekerk

# 0.3 (2015-07-20)

First public release. See <http://ibis-project.org> for more.

## New Features

-   Implement window / analytic function support
-   Enable non-equijoins (join clauses with operations other than `==`).
-   Add remaining `string functions` supported by Impala.
-   Add `pipe` method to tables (hat-tip to the pandas dev team).
-   Add `mutate` convenience method to tables.
-   Fleshed out `WebHDFS` implementations: get/put directories, move
    files, etc. See the `full HDFS API`.
-   Add `truncate` method for timestamp values
-   `ImpalaClient` can execute scalar expressions not involving any
    table.
-   Can also create internal Impala tables with a specific HDFS path.
-   Make Ibis\'s temporary Impala database and HDFS paths configurable
    (see `ibis.options`).
-   Add `truncate_table` function to client (if the user\'s Impala
    cluster supports it).
-   Python 2.6 compatibility
-   Enable Ibis to execute concurrent queries in multithreaded
    applications (earlier versions were not thread-safe).
-   Test data load script in `scripts/load_test_data.py`
-   Add an internal operation type signature API to enhance developer
    productivity.

## Contributors

    $ git log v0.2.0..v0.3.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         59 Wes McKinney
         29 Uri Laserson
          4 Isaac Hodes
          2 Meghana Vuyyuru

# 0.2 (2015-06-16)

## New Features

-   `insert` method on Ibis client for inserting data into existing
    tables.
-   `parquet_file`, `delimited_file`, and `avro_file` client methods for
    querying datasets not yet available in Impala
-   New `ibis.hdfs_connect` method and `HDFS` client API for WebHDFS for
    writing files and directories to HDFS
-   New timedelta API and improved timestamp data support
-   New `bucket` and `histogram` methods on numeric expressions
-   New `category` logical datatype for handling bucketed data, among
    other things
-   Add `summary` API to numeric expressions
-   Add `value_counts` convenience API to array expressions
-   New string methods `like`, `rlike`, and `contains` for fuzzy and
    regex searching
-   Add `options.verbose` option and configurable `options.verbose_log`
    callback function for improved query logging and visibility
-   Support for new SQL built-in functions
    -   `ibis.coalesce`
    -   `ibis.greatest` and `ibis.least`
    -   `ibis.where` for conditional logic (see also `ibis.case` and
        `ibis.cases`)
    -   `nullif` method on value expressions
    -   `ibis.now`
-   New aggregate functions: `approx_median`, `approx_nunique`, and
    `group_concat`
-   `where` argument in aggregate functions
-   Add `having` method to `group_by` intermediate object
-   Added group-by convenience
    `table.group_by(exprs).COLUMN_NAME.agg_function()`
-   Add default expression names to most aggregate functions
-   New Impala database client helper methods
    -   `create_database`
    -   `drop_database`
    -   `exists_database`
    -   `list_databases`
    -   `set_database`
-   Client `list_tables` searching / listing method
-   Add `add`, `sub`, and other explicit arithmetic methods to value
    expressions

## API Changes

-   New Ibis client and Impala connection workflow. Client now combined
    from an Impala connection and an optional HDFS connection

## Bug Fixes

-   Numerous expression API bug fixes and rough edges fixed

## Contributors

    $ git log v0.1.0..v0.2.0 --pretty=format:%aN | sort | uniq -c | sort -rn
         71 Wes McKinney
          1 Juliet Hougland
          1 Isaac Hodes

# 0.1 (2015-03-26)

First Ibis release.

-   Expression DSL design and type system
-   Expression to ImpalaSQL compiler toolchain
-   Impala built-in function wrappers

    $ git log 84d0435..v0.1.0 --pretty=format:%aN | sort | uniq -c | sort -rn
        78 Wes McKinney
         1 srus
         1 Henry Robinson
