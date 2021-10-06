=============
Release Notes
=============

.. toctree::
   :hidden:

   release-pre-1.0

.. note::

   These release notes are for versions of ibis **1.0 and later**. Release
   notes for pre-1.0 versions of ibis can be found at :doc:`release-pre-1.0`

* :support:`2678` Improvement of the backend API. The former `Client` subclasses have been replaced by a `Backend` class that must
  subclass `ibis.backends.base.BaseBackend`. The `BaseBackend` class contains abstract methods for the minimum subset of methods that
  backends must implement, and their signatures have been standardized across backends. The Ibis compiler has been refactored, and
  backends don't need to implement all compiler classes anymore if the default works for them. Only a subclass of
  `ibis.backends.base.sql.compiler.Compiler` is now required. Backends now need to register themselves as entry points.
* :support:`2905` Deprecate `exists_table(table)` in favor of `table in list_tables()`
* :bug:`2991` Fix data races in impala connection pool accounting
* :bug:`2985` Fix null literal compilation in the Clickhouse backend
* :bug:`2984` Fix order of limit and offset parameters in the Clickhouse backend
* :support:`2977` Remove handwritten type parser; parsing errors that were previously `IbisTypeError` are now `parsy.ParseError`. `parsy` is now a hard requirement.
* :support:`2962` Methods `current_database` and `list_databases` raise an exception for backends that do not support databases
* :bug:`2956` Replace `equals` operation for geospatial datatype to `geo_equals` 
* :support:`2913` Method `set_database` has been deprecated, in favor of creating a new connection to a different database
* :feature:`2938` Serialization-deserialization of Node via pickle is now byte compatible between different processes
* :support:`2914` Removed `log` method of clients, in favor of `verbose_log` option
* :feature:`2916` Support joining on different columns in ClickHouse backend
* :feature:`2908` Support summarization of empty data in Pandas backend
* :support:`2883` Output of `Client.version` returned as a string, instead of a setuptools `Version`
* :feature:`2882` Unify implementation of fillna and isna in Pyspark backend
* :support:`2862` Deprecated `list_schemas` in SQLAlchemy backends in favor of `list_databases`
* :bug:`2829` Fix .drop(fields). The argument can now be either a list of strings or a string.
* :feature:`2873` Support binary operation with Timedelta in Pyspark backend
* :support:`2865` Deprecated `ibis.<backend>.verify()` in favor of capturing exception in `ibis.<backend>.compile()`
* :bug:`2845` Fix projection on differences and intersections for SQL backends
* :feature:`2839`: Add `group_concat` operation for Clickhouse backend
* :bug:`2827` Backends are loaded in a lazy way, so third-party backends can import Ibis without circular imports
* :bug:`2830` Disable aggregation optimization due to N squared performance
* :bug:`2821` Fix `.cast()` to array outputting list instead of np.array in Pandas backend
* :bug:`2820` Fix aggregation with mixed reduction datatypes (array + scalar) on Dask backend
* :feature:`2808` Support comparison of ColumnExpr to timestamp literal
* :support:`2789` Simplification of data fetching. Backends don't need to implement `Query` anymore
* :feature:`2805` Make op schema a cached property
* :feature:`2613` :feature:`2778` Implement `.insert()` for SQLAlchemy backends
* :feature:`2792` Infer categorical and decimal Series to more specific Ibis types in Pandas backend
* :feature:`2790` Add `startswith` and `endswith` operations
* :feature:`2776` :feature:`2797` Allow more flexible return type for UDFs
* :feature:`2779` Implement Clip in the Pyspark backend
* :bug:`2770` Fix error when using reduction UDF that returns np.array in a grouped aggregation
* :feature:`2753` Use `ndarray` as array representation in Pandas backend
* :support:`2665` Move BigQuery backend to a `separate repository <https://github.com/ibis-project/ibis-bigquery>`_.
  The backend will be released separately, use `pip install ibis-bigquery` or `conda install ibis-bigquery` to
  install it, and then use as before.
* :bug:`2712` Fix time context trimming error for multi column udfs in pandas backend
* :bug:`2710` Fix error during compilation of range_window in base_sql backends (:issue:`2608`)
* :feature:`2687` Support Spark filter with window operation
* :bug:`2696` Fix wrong row indexing in the result for 'window after filter' for timecontext adjustment
* :bug:`2702` Fix `aggregate` exploding the output of Reduction ops that return a list/ndarray
* :bug:`2693` Fix issues with context adjustment for filter with PySpark backend
* :support:`2689` Supporting SQLAlchemy 1.4, and requiring minimum 1.3
* :support:`2680` Namespace time_col config, fix type check for trim_with_timecontext for pandas window execution
* :feature:`2646` Support context adjustment for udfs for pandas backend
* :feature:`2655` Add `auth_local_webserver`, `auth_external_data`, and
  `auth_cache` parameters to BigQuery connect method. Set
  `auth_local_webserver` to use a local server instead of copy-pasting an
  authorization code. Set `auth_external_data` to true to request additional
  scopes required to query Google Drive and Sheets. Set `auth_cache` to
  `reauth` or `none` to force reauthentication.
* :bug:`2657` Add temporary struct col in pyspark backend to ensure that UDFs are executed only once
* :bug:`2588` Fix BigQuery connect bug that ignored project ID parameter
* :bug:`2636` Fix overwrite logic to account for DestructColumn inside mutate API
* :feature:`2641` Add `bit_and`, `bit_or`, and `bit_xor` integer column aggregates (BigQuery and MySQL backends)
* :feature:`2379` Backends are defined as entry points
* :bug:`2635` Fix fusion optimization bug that incorrectly changes operation order
* :feature:`2615` Add `ibis.array` for creating array expressions
* :feature:`2607` Implement Not operation in PySpark backend
* :feature:`2610` Added support for case/when in PySpark backend
* :bug:`2610` Fixes a NPE issue with substr in PySpark backend
* :feature:`2603` Add support for np.array as literals for backends that already support lists as literals
* :bug:`2354` Fixes binary data type translation into BigQuery bytes data type
* :bug:`2577` Make StructValue picklable
* :support:`2505` Remove deprecated `ibis.HDFS`, `ibis.WebHDFS` and `ibis.hdfs_connect`
* :feature:`2514` Add Struct.from_dict
* :feature:`2310` Add hash and hashbytes support for BigQuery backend
* :feature:`2511` Support reduction UDF without groupby to return multiple columns for Pandas backend
* :feature:`2487` Support analytic and reduction UDF to return multiple columns for Pandas backend
* :support:`2497` Move `ibis.HDFS`, `ibis.WebHDFS` and `ibis.hdfs_connect` to `ibis.impala.*`
* :feature:`2473` Support elementwise UDF to return multiple columns for Pandas and PySpark backend
* :bug:`2462` Table expressions do not recognize inet datatype (Postgres backend)
* :bug:`2461` Table expressions do not recognize macaddr datatype (Postgres backend)
* :bug:`2410` Fix ``aggcontext.Summarize`` not always producing scalar (Pandas backend)
* :bug:`2414` Fix same window op with different window size on table lead to incorrect results for pyspark backend
* :feature:`2409` FEAT: Support Ibis interval for window in pyspark backend
* :bug:`2229` Fix same column with multiple aliases not showing properly in repr
* :feature:`2402` Use Scope class for scope in pyspark backend
* :bug:`2395` Fix reduction UDFs over ungrouped, bounded windows on Pandas backend
* :bug:`2386` FEAT: Support rolling window UDF with non numeric inputs for pandas backend.
* :bug:`2386` Fix scope get to use hashmap lookup instead of list lookup
* :bug:`2387` Fix equality behavior for Literal ops
* :bug:`2376` Fix analytic ops over ungrouped and unordered windows on Pandas backend
* :support:`2288` Drop support to Python 3.6
* :bug:`2367` Fix the covariance operator in the BigQuery backend.
* :feature:`2366` Add PySpark support for ReductionVectorizedUDF
* :bug:`2342` Update impala kerberos dependencies
* :feature:`2306` Add time context in `scope` in execution for pandas backend
* :support:`2351` Simplifying tests directories structure
* :feature:`2081` Add ``start_point`` and ``end_point`` to PostGIS backend.
* :feature:`2347` Add set difference to general ibis api
* :feature:`2251` Add ``rowid`` expression, supported by SQLite and OmniSciDB
* :feature:`2230` Add intersection to general ibis api
* :support:`2304` Update ``google-cloud-bigquery`` dependency minimum version to 1.12.0
* :feature:`2303` Add ``application_name`` argument to ``ibis.bigquery.connect`` to allow attributing Google API requests to projects that use Ibis.
* :bug:`1320` Added verbose logging to SQL backends
* :feature:`2285` Add support for casting category dtype in pandas backend
* :feature:`2270` Add support for Union in the PySpark backend
* :bug:`2256` Fix issue with sql_validate call to OmnisciDB.
* :feature:`2260` Add support for implementign custom window object for pandas backend
* :bug:`2237` Add missing float types to pandas backend
* :bug:`2252` Allow group_by and order_by as window operation input in pandas backend
* :feature:`2246` Implement two level dispatcher for execute_node
* :feature:`2233` Add ibis.pandas.trace module to log time and call stack information.
* :feature:`2198` Validate that the output type of a UDF is a single element
* :bug:`2223` Fix PySpark compiler error when elementwise UDF output_type is Decimal or Timestamp
* :feature:`2186` ZeroIfNull and NullIfZero implementation for OmniSciDB
* :bug:`2157` Fix interactive mode returning a expression instead of the value when used in Jupyter
* :feature:`2093` IsNan implementation for OmniSciDB
* :feature:`2094` [OmnisciDB] Support add_columns and drop_columns for OmnisciDB table
* :support:`2234` Remove "experimental" mentions for OmniSciDB and Pandas backends
* :bug:`2127` Fix PySpark error when doing alias after selection
* :support:`2244` Use an OmniSciDB image stable on CI
* :feature:`2175` Create ExtractQuarter operation and add its support to Clickhouse, CSV, Impala, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark
* :feature:`2126` Add translation rules for isnull() and notnull() for pyspark backend
* :feature:`2232` Add window operations support to SQLite
* :feature:`2062` Implement read_csv for omniscidb backend
* :feature:`2171` [OmniSciDB] Add support to week extraction
* :feature:`2097` Date, DateDiff and TimestampDiff implementations for OmniSciDB
* :bug:`2170` Fix millisecond issue for OmniSciDB :issue:`2167`, MySQL :issue:`2169`, PostgreSQL :issue:`2166`, Pandas :issue:`2168`, BigQuery :issue:`2273` backends
* :feature:`2177` Create ExtractWeekOfYear operation and add its support to Clickhouse, CSV, MySQL, Pandas, Parquet, PostgreSQL, PySpark and Spark
* :feature:`2060` Add initial support for ibis.random function
* :support:`2107` Added fragment_size to table creation for OmniSciDB
* :feature:`2178` Added epoch_seconds extraction operation to Clickhouse, CSV, Impala, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite, Spark and BigQuery :issue:`2273`
* :feature:`2165` [OmniSciDB] Add "method" parameter to load_data
* :feature:`2117` Add non-nullable info to schema output
* :feature:`2083` fillna and nullif implementations for OmnisciDB
* :feature:`1981` Add load_data to sqlalchemy's backends and fix database parameter for load/create/drop when database parameter is the same than the current database
* :support:`2096` Added round() support for OmniSciDB
* :feature:`2125` [OmniSciDB] Add support for within, d_fully_within and point
* :feature:`2086` OmniSciDB - Refactor DDL and Client; Add temporary parameter to create_table and "force" parameter to drop_view
* :support:`2113` Enabled cumulative ops support for OmniSciDB
* :bug:`2134` [OmniSciDB] Fix TopK when used as filter
* :feature:`2173` Create ExtractDayOfYear operation and add its support to Clickhouse, CSV, MySQL, OmniSciDB, Pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark
* :feature:`2095` Implementations of Log Log2 Log10 for OmniSciDB backend
* :release:`1.3.0 <2020-02-27>`
* :support:`2066` Add support to Python 3.8
* :bug:`2089 major` Pin "clickhouse-driver" to ">=0.1.3"
* :support:`2079` Pin back version of isort
* :support:`2082` Use user-defined port variables for Omnisci and PostgreSQL tests
* :support:`2077` Change omniscidb image tag from v5.0.0 to v5.1.0 on docker-compose recipe
* :support:`2051` [Omnisci] The same SRIDs for test_geo_spatial_binops
* :support:`2078` Unpin rtree version
* :feature:`2071` Improve many arguments UDF performance in pandas backend.
* :bug:`2069 major` Fix load data stage for Linux CI
* :support:`2074` Link pandas issues with xfail tests in pandas/tests/test_udf.py
* :support:`2075` Disable Postgres tests on Windows CI.
* :support:`2068` use conda for installation black and isort tools
* :bug:`2057 major` Fix datamgr.py fail if IBIS_TEST_OMNISCIDB_DATABASE=omnisci
* :support:`2061` CI: Fix CI builds related to new pandas 1.0 compatibility
* :support:`2056` Fix data map for int8 on OmniSciDB backend
* :feature:`1976` Add DenseRank, RowNumber, MinRank, Count, PercentRank/CumeDist window operations to OmniSciDB
* :support:`2052` Add possibility to run tests for separate backend via `make test BACKENDS=[YOUR BACKEND]`
* :support:`2055` Fix "cudf" import on OmniSciDB backend
* :feature:`2047` Introduce a top level vectorized UDF module (experimental). Implement element-wise UDF for pandas and PySpark backend.
* :support:`2050` CI: Drop table only if it exists (OmniSciDB)
* :support:`2034` Add initial documentation for OmniSciDB, MySQL, PySpark and SparkSQL backends, add initial documentation for geospatial methods and add links to Ibis wiki page
* :support:`2044` Implement covariance for bigquery backend
* :feature:`2035` Add support for  multi arguments window UDAF for the pandas backend
* :bug:`2041 major` Change pymapd connection parameter from "session_id" to "sessionid"
* :support:`2046` Add Spark to supported backends list
* :support:`2043` Ping dependency of rtree to fix CI failure
* :support:`2037` Drop support for Python 3.5
* :support:`2023` HTML escape column names and types in png repr.
* :support:`1991` Add geospatial tutorial notebook
* :support:`2031` Change omniscidb image tag from v4.7.0 to v5.0.0 on docker-compose recipe
* :support:`2030` Pin "semantic_version" to "<2.7" in the docs build CI, fix "builddoc" and "doc" section inside "Makefile" and skip mysql tzinfo on CI to allow to run MySQL using docker container on a hard disk drive.
* :bug:`2009 major` Fix pandas backend to treat trailing_window preceding arg as window bound rather than window size (e.g. preceding=0 now indicates current row rather than window size 0)
* :feature:`2004` Clean up window translation logic in pyspark backend
* :bug:`2015 major` Fix handling of Array types in Postgres UDF
* :feature:`1996` Add docstring check to CI for an initial subset files
* :bug:`2010 major` Fix pydocstyle config
* :support:`2012` Fixed impala start up issues
* :feature:`2001` Pyspark backend bounded windows
* :bug:`2006 major` Pinning clickhouse-driver<0.1.2
* :support:`1999` cache all ops in translate()
* :feature:`1987` Add more POSTGIS operations
* :feature:`1969` SQLAlchemy Default precision and scale to decimal types for PostgreSQL and MySQL
* :support:`1988` Add black step to CI
* :support:`1962` Json UUID any
* :bug:`1984 major` Fix CI log for database
* :feature:`1983` Add support for array operations in PySpark backend
* :feature:`1978` Implement sort, if_null, null_if and notin for PySpark backend
* :support:`1982` Add log for database services
* :feature:`1974` Add support for date/time operations in PySpark backend
* :feature:`1973` Add support for params, query_schema, and sql in PySpark backend
* :support:`1972` Fix BigQuery backend fixture so batting and awards_players fixture re…
* :support:`1971` Disable BigQuery explicitly in all/test_join.py
* :feature:`1967` Implement join for PySpark backend
* :feature:`1952` Validate AsOfJoin tolerance and attempt interval unit conversion
* :support:`1963` Re-formatting all files using pre-commit hook
* :support:`1961` Disable codecov report upload during CI builds
* :support:`1960` Developer doc enhancements
* :feature:`1943` filter for PySpark backend
* :feature:`1945` window operations for pyspark backend
* :support:`1958` Missing geospatial ops for OmniSciDB
* :feature:`1951` Implement IntervalSub for pandas backend
* :support:`1950` Remove pandas deprecation warnings
* :support:`1948` Add developer docs to get docker setup
* :support:`1949` More informative IntegrityError on duplicate columns
* :feature:`1942` PySpark backend string and column ops
* :support:`1928` Improve geospatial literals and smoke tests
* :support:`1925` PostGIS enhancements
* :bug:`1933 major` Fixes explain operation
* :feature:`1913` PySpark backend
* :bug:`1937 major` Fix incorrect assumptions about attached SQLite databases
* :bug:`1938 major` Upgrade to JDK11
* :support:`1866` Rename mapd to omniscidb backend
* :support:`1926` Fix failing BigQuery tests
* :feature:`1908` DDL support for Spark backend
* :support:`1917` Added missing null literal op
* :feature:`1923` Support timezone aware arrow timestamps
* :bug:`1903 major` `sql` method doesn't work when the query uses LIMIT clause
* :feature:`1860` Add shapely geometries as input for literals
* :bug:`1910 major` Fix union implementation
* :bug:`1912 major` Fix failing com imports on master
* :feature:`1858` Add geopandas as output for omniscidb
* :bug:`1901 major` OmniSci/MapD - Fix reduction for bool
* :feature:`1885` Spark UDFs
* :feature:`1871` Add support for Postgres UDFs
* :bug:`1899 major` Pass scope to grouping execution in the pandas backend
* :support:`1895` Update link to Presto website
* :support:`1896` Removing linting from windows
* :bug:`1888 major` Fix various Spark backend issues
* :bug:`1891 major` Make Nodes enforce the proper signature
* :bug:`1893 major` Fix according to bug in pd.to_datetime when passing the unit flag
* :feature:`1830` Spark tests
* :support:`1884` Fix link to NUMFOCUS CoC
* :bug:`1883 major` Fix small formatting buglet in PR merge tool
* :support:`1882` Added CoC section
* :bug:`1876 major` Fix the case where we do not have an index when using preceding with intervals
* :feature:`1807` Spark client
* :bug:`1872 major` Fixed issues with geo data
* :feature:`1868` Use pandas rolling apply to implement rows_with_max_lookback
* :bug:`1869 major` Remove -x from pytest call in linux CI
* :bug:`1867 major` Fix return type of Struct.from_tuples
* :support:`1859` Remove pandas exception for rows_with_max_lookback
* :support:`1856` Move CI pipelines to Azure
* :release:`1.2.0 <2019-06-24>`
* :feature:`1836` Add new geospatial functions to OmniSciDB backend
* :support:`1847` Skip SQLAlchemy backend tests in connect method in backends.py
* :bug:`1855 major` Fix call to psql causing failing CI
* :bug:`1851 major` Fix nested array literal repr
* :support:`1848` Validate order_by when using rows_with_max_lookback window
* :bug:`1850 major` Fix repr of empty schema
* :support:`1845` Generate release notes from commits
* :support:`1844` Raise exception on backends where rows_with_max_lookback can't be implemented
* :bug:`1843 major` Add max_lookback to window replace and combine functions
* :bug:`1837 major` Partially revert #1758
* :support:`1840` Tighter version spec for pytest
* :feature:`1838` allow pandas timedelta in rows_with_max_lookback
* :feature:`1825` Accept rows-with-max-lookback as preceding parameter
* :feature:`1787` PostGIS support
* :support:`1826` Allow passing a branch to ci/feedstock.py
* :support:`-` Bugs go into feature releases
* :support:`-` No space after :release:
* :release:`1.1.0 <2019-06-09>`
* :bug:`1819 major` Fix group_concat test and implementations
* :support:`1820` Remove decorator hacks and add custom markers
* :bug:`1818 major` Fix failing strftime tests on Python 3.7
* :bug:`1757 major` Remove unnecessary (and erroneous in some cases) frame clauses
* :support:`1814` Add development deps to setup.py
* :feature:`1809` Conslidate trailing window functions
* :bug:`1799 major` Chained mutate operations are buggy
* :support:`1805` Fix design and developer docs
* :support:`1810` Pin sphinx version to 2.0.1
* :feature:`1766` Call to_interval when casting integers to intervals
* :bug:`1783 major` Allow projections from joins to attempt fusion
* :feature:`1796` Add session feature to mapd client API
* :bug:`1798 major` Fix Python 3.5 dependency versions
* :feature:`1792` Add min periods parameter to Window
* :support:`1793` Add pep8speaks integration
* :support:`1821` Fix typo in UDF signature specification
* :feature:`1785` Allow strings for types in pandas UDFs
* :feature:`1790` Add missing date operations and struct field operation for the pandas backend
* :bug:`1789 major` Fix compatibility and bugs associated with pandas toposort reimplementation
* :bug:`1772 major` Fix outer_join generating LEFT join instead of FULL OUTER
* :feature:`1771` Add window operations to the OmniSci backend
* :feature:`1758` Reimplement the pandas backend using topological sort
* :support:`1779` Clean up most xpassing tests
* :bug:`1782 major` NullIf should enforce that its arguments are castable to a common type
* :support:`1781` Update omnisci container version
* :feature:`1778` Add marker for xfailing specific backends
* :feature:`1777` Enable window function tests where possible
* :bug:`1775 major` Fix conda create command in documentation
* :support:`1776` Constrain PyMapD version to get passing builds
* :bug:`1765 major` Fix preceding and following with ``None``
* :support:`1763` Remove warnings and clean up some docstrings
* :support:`1638` Add StringToTimestamp as unsupported
* :feature:`1743` is_computable_arg dispatcher
* :support:`1759` Add isort pre-commit hooks
* :feature:`1753` Added float32 and geospatial types for create table from schema
* :bug:`1661 major` PostgreSQL interval type not recognized
* :support:`1750` Add Python 3.5 testing back to CI
* :support:`1700` Re-enable CI for building step
* :support:`1749` Update README reference to MapD to say OmniSci
* :release:`1.0.0 <2019-03-26>`
* :support:`1748` Do not build universal wheels
* :support:`1747` Remove tag prefix from versioneer
* :support:`1746` Use releases to manage documentation
* :feature:`1735` Add black as a pre-commit hook
* :feature:`1680` Add support for the arbitrary aggregate in the mapd backend
* :bug:`1745` Make ``dev/merge-pr.py`` script handle PR branches
* :feature:`1731` Add SQL method for the MapD backend
* :feature:`1744` Clean up merge PR script and use the actual merge feature of GitHub
* :bug:`1742` Fix ``NULLIF`` implementation for the pandas backend
* :bug:`1737` Fix casting to float in the MapD backend
* :bug:`1741` Fix testing for BigQuery after auth flow update
* :feature:`1723` Add cross join to the pandas backend
* :bug:`1738` Fix skipping for new BigQuery auth flow
* :bug:`1732` Fix bug in ``TableExpr.drop``
* :feature:`1727` Implement default handler for multiple client ``pre_execute``
* :feature:`1728` Implement BigQuery auth using ``pydata_google_auth``
* :bug:`1729` Filter the ``raw`` warning from newer pandas to support older pandas
* :bug:`1706` Fix BigQuery credentials link
* :feature:`1712` Timestamp literal accepts a timezone parameter
* :feature:`1725` Remove support for passing integers to ``ibis.timestamp``
* :feature:`1704` Add ``find_nodes`` to lineage
* :feature:`1714` Remove a bunch of deprecated APIs and clean up warnings
* :feature:`1716` Implement table distinct for the pandas backend
* :feature:`1678` Implement geospatial functions for MapD
* :feature:`1666` Implement geospatial types for MapD
* :support:`1694` Use cudf instead of pygdf
* :bug:`1639` Add Union as an unsuppoted operation for MapD
* :bug:`1705` Fix visualizing an ibis expression when showing a selection after a table join
* :bug:`1659` Fix MapD exception for ``toDateTime``
* :bug:`1701` Use ``==`` to compare strings
* :support:`1696` Fix multiple CI issues
* :feature:`1685` Add pre commit hook
* :support:`1681` Update mapd ci to v4.4.1
* :feature:`1686` Getting started with mapd, mysql and pandas
* :support:`1672` Enabled mysql CI on azure pipelines
* :support:`-` Update docs to reflect Apache Impala and Kudu as ASF TLPs
* :feature:`1675` Support column names with special characters in mapd
* :support:`1670` Remove support for Python 2
* :feature:`1669` Allow operations to hide arguments from display
* :bug:`1647` Resolves joining with different column names
* :bug:`1643` Fix map get with compatible types
* :feature:`1636` Remove implicit ordering requirements in the PostgreSQL backend
* :feature:`1655` Add cross join operator to MapD
* :support:`1667` Fix flake8 and many other warnings
* :bug:`1653` Fixed where operator for MapD
* :support:`1664` Update README.md for impala and kudu
* :support:`1660` Remove defaults as a channel from azure pipelines
* :support:`1658` Fixes a very typo in the pandas/core.py docstring
* :support:`1657` Unpin clickhouse-driver version
* :bug:`1648` Remove parameters from mapd
* :bug:`1651` Make sure we cast when NULL is else in CASE expressions
* :support:`1650` Add test for reduction returning lists
* :feature:`1637` Fix UDF bugs and add support for non-aggregate analytic functions
* :support:`1646` Fix Azure VM image name
* :support:`1641` Updated MapD server-CI
* :support:`1645` Add TableExpr.drop to API documentation
* :support:`1642` Fix Azure deployment step
* :support:`-` Update README.md
* :support:`1640` Set up CI with Azure Pipelines
* :feature:`1627` Support string slicing with other expressions
* :feature:`1618` Publish the ibis roadmap
* :feature:`1604` Implement ``approx_median`` in BigQuery
* :feature:`1611` Make ibis node instances hashable
* :bug:`1600` Fix equality
* :feature:`1608` Add ``range_window`` and ``trailing_range_window`` to docs
* :support:`1609` Fix conda builds
* :release:`0.14.0 <2018-08-23>`
