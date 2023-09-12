Release notes
---

## [6.2.0](https://github.com/ibis-project/ibis/compare/6.1.0...6.2.0) (2023-08-31)

### Features

* **trino:** add source application to trino backend ([cf5fdb9](https://github.com/ibis-project/ibis/commit/cf5fdb9d29567c72680d94f66c9445d76d9ccbef))

### Bug Fixes

* **bigquery,impala:** escape all ASCII escape sequences in string literals ([402f5ca](https://github.com/ibis-project/ibis/commit/402f5ca756acc85e1ddddbc60af2417f371e4ee0))
* **bigquery:** correctly escape ASCII escape sequences in regex patterns ([a455203](https://github.com/ibis-project/ibis/commit/a455203d7dc6f5410ad2627123ec18e7126d6bc9))
* **release:** pin conventional-changelog-conventionalcommits to 6.1.0 ([d6526b8](https://github.com/ibis-project/ibis/commit/d6526b83c2250f4f620ab9e71f0421a65e89d55b))
* **trino:** ensure that list_databases look at all catalogs not just the current one ([cfbdbf1](https://github.com/ibis-project/ibis/commit/cfbdbf19426e9771cbda9f9e665173da7286e3c5))
* **trino:** override incorrect base sqlalchemy `list_schemas` implementation ([84d38a1](https://github.com/ibis-project/ibis/commit/84d38a1e9cd67eb480ed9faa4fc506f98247dea3))

### Documentation

* **trino:** add connection docstring ([507a00e](https://github.com/ibis-project/ibis/commit/507a00e95784fd4054bd5146f49c2e6e168793db))

## [6.1.0](https://github.com/ibis-project/ibis/compare/6.0.0...6.1.0) (2023-08-03)


### Features

* **api:** add `ibis.dtype` top-level API ([867e5f1](https://github.com/ibis-project/ibis/commit/867e5f1e3dc94fc40d075da34e8db63f690bb048))
* **api:** add `table.nunique()` for counting unique table rows ([adcd762](https://github.com/ibis-project/ibis/commit/adcd7628a7c2637e0bde1d924d11f7da5bd1659d))
* **api:** allow mixing literals and columns in `ibis.array` ([3355dd8](https://github.com/ibis-project/ibis/commit/3355dd88268c27642ab6dba7b1091caddd542130))
* **api:** improve efficiency of `__dataframe__` protocol ([15e27da](https://github.com/ibis-project/ibis/commit/15e27da1d0020a58c222ccab95b97d39a2c3e3c7))
* **api:** support boolean literals in join API ([c56376f](https://github.com/ibis-project/ibis/commit/c56376f6b209a1961ab50a7a3de24e9b20475196))
* **arrays:** add `concat` method equivalent to `__add__`/`__radd__` ([0ed0ab1](https://github.com/ibis-project/ibis/commit/0ed0ab14f93621c5d647338c5b2cb882abe36a85))
* **arrays:** add `repeat` method equivalent to `__mul__`/`__rmul__` ([b457c7b](https://github.com/ibis-project/ibis/commit/b457c7b1e02bb7d0e20accc1e145f9f8af3d09cf))
* **backends:** add `current_schema` API ([955a9d0](https://github.com/ibis-project/ibis/commit/955a9d0e55b19db0edec5fb891c22e964b15ff78))
* **bigquery:** fill out `CREATE TABLE` DDL options including support for `overwrite` ([5dac7ec](https://github.com/ibis-project/ibis/commit/5dac7eccd43fb8fbeb146340864837b5a1084a65))
* **datafusion:** add count_distinct, median, approx_median, stddev and var aggregations ([45089c4](https://github.com/ibis-project/ibis/commit/45089c4c9137971a3bc3b1743ca61493b93301a2))
* **datafusion:** add extract url fields functions ([4f5ea98](https://github.com/ibis-project/ibis/commit/4f5ea9896d8a9ca91f631ddd37c1eb1d775e6770))
* **datafusion:** add functions sign, power, nullifzero, log ([ef72e40](https://github.com/ibis-project/ibis/commit/ef72e403200ff0ef6eb0c87ab1d5e8484de4d636))
* **datafusion:** add RegexSearch, StringContains and StringJoin ([4edaab5](https://github.com/ibis-project/ibis/commit/4edaab587d7b5fc6d64f72bbeb121c0cdccb45ce))
* **datafusion:** implement in-memory table ([d4ec5c2](https://github.com/ibis-project/ibis/commit/d4ec5c244e1812041794f9c5abc04421c23ecb9e))
* **flink:** add tests and translation rules for additional operators ([fc2aa5d](https://github.com/ibis-project/ibis/commit/fc2aa5d7bcc63a82dd0069e3daa81b5146e18902))
* **flink:** implement translation rules and tests for over aggregation in Flink backend ([e173cd7](https://github.com/ibis-project/ibis/commit/e173cd799b2fb5a2a19e81b3fa67721d28ca11a3))
* **flink:** implement translation rules for literal expressions in flink compiler ([a8f4880](https://github.com/ibis-project/ibis/commit/a8f4880b44f7c58a41a23a00b8171c54c392f0a1))
* improved error messages when missing backend dependencies ([2fe851b](https://github.com/ibis-project/ibis/commit/2fe851b9dde5e30dc5597d6c64c4eed4a1186aa1))
* make output of `to_sql` a proper `str` subclass ([084bdb9](https://github.com/ibis-project/ibis/commit/084bdb9bcaf79dffa8dfafe950194606cd5f2ffb))
* **pandas:** add ExtractURLField functions ([e369333](https://github.com/ibis-project/ibis/commit/e36933326a8206baf15bf38a11eb2b21655f32ee))
* **polars:** implement `ops.SelfReference` ([983e393](https://github.com/ibis-project/ibis/commit/983e393ad9d2a3cf6d45a21bb664173fbda0784b))
* **pyspark:** read/write delta tables ([d403187](https://github.com/ibis-project/ibis/commit/d40318777f8ecd2d75b745e5a3895bac439caf77))
* refactor ddl for create_database and add create_schema where relevant ([d7a857c](https://github.com/ibis-project/ibis/commit/d7a857ca3eb4359cec800d45d2f47d6685c4ebe8))
* **sqlite:** add scalar python udf support to sqlite ([92f29e6](https://github.com/ibis-project/ibis/commit/92f29e6b820f207b16a306e5a39f91d396fe818a))
* **sqlite:** implement extract url field functions ([cb1956f](https://github.com/ibis-project/ibis/commit/cb1956ff25fb0e9d6d8a0c3dba9ddf512a6c42aa))
* **trino:** implement support for `.sql` table expression method ([479bc60](https://github.com/ibis-project/ibis/commit/479bc602b6c302753d34fa1e6c9ee544b591b61f))
* **trino:** support table properties when creating a table ([b9d65ef](https://github.com/ibis-project/ibis/commit/b9d65efebce5f5b750a447b3cdf00f505eec0111))


### Bug Fixes

* **api:** allow scalar window order keys ([3d3f4f3](https://github.com/ibis-project/ibis/commit/3d3f4f344880c2401f970373889348f6f5ce4150))
* **backends:** make `current_database` implementation and API consistent across all backends ([eeeeee0](https://github.com/ibis-project/ibis/commit/eeeeee057ba4e94a35721747493c18281c169b30))
* **bigquery:** respect the fully qualified table name at the init ([a25f460](https://github.com/ibis-project/ibis/commit/a25f460b06a2c9f47c0ad3ea2fb9e894c5ad3f4e))
* **clickhouse:** check dispatching instead of membership in the registry for `has_operation` ([acb7f3f](https://github.com/ibis-project/ibis/commit/acb7f3ff2d3b7f224307548698e003fac85bdff4))
* **datafusion:** always quote column names to prevent datafusion from normalizing case ([310db2b](https://github.com/ibis-project/ibis/commit/310db2bf9416b73b1e741570350034bdd9d4337f))
* **deps:** update dependency datafusion to v27 ([3a311cd](https://github.com/ibis-project/ibis/commit/3a311cd9d50fc12f527e7ba244169542a2796227))
* **druid:** handle conversion issues from string, binary, and timestamp ([b632063](https://github.com/ibis-project/ibis/commit/b632063dc9b7fcc722dcb4e5bb98b10aa4f7d54e))
* **duckdb:** avoid double escaping backslashes for bind parameters ([8436f57](https://github.com/ibis-project/ibis/commit/8436f573b5819001595a5a8fe6803fbfaf282b2f))
* **duckdb:** cast read_only to string for connection ([27e17d6](https://github.com/ibis-project/ibis/commit/27e17d6200740615b583dca267eb1517abf7ac41))
* **duckdb:** deduplicate results from `list_schemas()` ([172520e](https://github.com/ibis-project/ibis/commit/172520e52b946a3688af0bcfe93d621c160781e9))
* **duckdb:** ensure that current_database returns the correct value ([2039b1e](https://github.com/ibis-project/ibis/commit/2039b1e7cce182dbbc901480037a58488b3730cb))
* **duckdb:** handle conversion from duckdb_engine unsigned int aliases ([e6fd0cc](https://github.com/ibis-project/ibis/commit/e6fd0cc2d726668d4e645274b7637df3f886d08e))
* **duckdb:** map hugeint to decimal to avoid information loss ([4fe91d4](https://github.com/ibis-project/ibis/commit/4fe91d49f00ed76e9b677e5dfbdb1dbeda96a8f7))
* **duckdb:** run pre-execute-hooks in duckdb before file export ([5bdaa1d](https://github.com/ibis-project/ibis/commit/5bdaa1d6339c221c76016a6f04ded1f20b61e017))
* **duckdb:** use regexp_matches to ensure that matching checks containment instead of a full match ([0a0cda6](https://github.com/ibis-project/ibis/commit/0a0cda6f05c624cf4af58ed08209b79c1a7ca877))
* **examples:** remove example datasets that are incompatible with case-insensitive file systems ([4048826](https://github.com/ibis-project/ibis/commit/4048826714efdd41cc233a79603756ab66c813b1))
* **exprs:** ensure that left_semi and semi are equivalent ([bbc1eb7](https://github.com/ibis-project/ibis/commit/bbc1eb7ac2573aa0e762caae1cdc87137da2ac8b))
* forward arguments through `__dataframe__` protocol ([50f3be9](https://github.com/ibis-project/ibis/commit/50f3be972b794e767b00fb25b35b6e5e18f4d4c0))
* **ir:** change "it not a" to "is not a" in errors ([d0d463f](https://github.com/ibis-project/ibis/commit/d0d463febd56d966917788ab2e480d8e11608ad4))
* **memtable:** implement support for translation of empty memtable ([05b02da](https://github.com/ibis-project/ibis/commit/05b02da6f856e2939a3f65437a94306e60015330))
* **mysql:** fix UUID type reflection for sqlalchemy 2.0.18 ([12d4039](https://github.com/ibis-project/ibis/commit/12d4039b619fdbdb30e13dc3724ee8b3b1a6f1cf))
* **mysql:** pass-through kwargs to connect_args ([e3f3e2d](https://github.com/ibis-project/ibis/commit/e3f3e2d9d693aaa224d1baf96bfa706a7f447d09))
* **ops:** ensure that name attribute is always valid for `ops.SelfReference` ([9068aca](https://github.com/ibis-project/ibis/commit/9068aca381169e865b89945e802d0020b9c1e2e3))
* **polars:** ensure that `pivot_longer` works with more than one column ([822c912](https://github.com/ibis-project/ibis/commit/822c912b919a7049a32a3f01cae0abdb2433cb1f))
* **polars:** fix collect implementation ([c1182be](https://github.com/ibis-project/ibis/commit/c1182be6d3ee2bba866a39f0894bd3f2cd0d64ea))
* **postgres:** by default use domain socket ([e44fdfb](https://github.com/ibis-project/ibis/commit/e44fdfb46bea79aed126e24f034bcc00e6adae40))
* **pyspark:** make `has_operation` method a `[@classmethod](https://github.com/classmethod)` ([c1b7dbc](https://github.com/ibis-project/ibis/commit/c1b7dbc02d2ed215870138405aa39d20da90f00d))
* **release:** use @google/semantic-release-replace-plugin@1.2.0 to avoid module loading bug ([673aab3](https://github.com/ibis-project/ibis/commit/673aab3d01783c771f74a8e4650d94acf93baf56))
* **snowflake:** fix broken unnest functionality ([207587c](https://github.com/ibis-project/ibis/commit/207587cc778433ab03cb490e08d32243ef4842a4))
* **snowflake:** reset the schema and database to the original schema after creating them ([54ce26a](https://github.com/ibis-project/ibis/commit/54ce26a4b00d06b1bb6c8fd38dc9edaa1672aac0))
* **snowflake:** reset to original schema when resetting the database ([32ff832](https://github.com/ibis-project/ibis/commit/32ff8329e8905727772956875776f264e8fbe1d6))
* **snowflake:** use `regexp_instr != 0` instead of `REGEXP` keyword ([06e2be4](https://github.com/ibis-project/ibis/commit/06e2be4e2019b6fa714e1fcb34485860ef1ede79))
* **sqlalchemy:** add support for sqlalchemy string subclassed types ([8b33b35](https://github.com/ibis-project/ibis/commit/8b33b352d8e88cfd6aaa6c318e5eb41d5482b362))
* **sql:** handle parsing aliases ([3645cf4](https://github.com/ibis-project/ibis/commit/3645cf4119620e8b01e57c7f9b5965400476f7d1))
* **trino:** handle all remaining common datatype parsing ([b3778c7](https://github.com/ibis-project/ibis/commit/b3778c781c78d5cdaf66a2a7286ef9e57ec519db))
* **trino:** remove filter index warning in Trino dialect ([a2ae7ae](https://github.com/ibis-project/ibis/commit/a2ae7ae328d8f1e468686ba2505e101b42a6df6c))


### Documentation

* add conda/mamba install instructions for specific backends ([c643fca](https://github.com/ibis-project/ibis/commit/c643fcaf57a46a36b5d2bdb24088d18bd386a1bf))
* add docstrings to `DataType.is_*` methods ([ed40fdb](https://github.com/ibis-project/ibis/commit/ed40fdb9e961cdf70784aa665736f203661767c1))
* **backend-matrix:** add ability to select a specific subset of backends ([f663066](https://github.com/ibis-project/ibis/commit/f6630664ef941709bd4b52bc4050fc8f5c70ff42))
* **backends:** document memtable support and performance for each backend ([b321733](https://github.com/ibis-project/ibis/commit/b321733f4ae27d938f4a2e4fc30a2d38a57a2230))
* **blog:** v6.0.0 release blog ([21fc5da](https://github.com/ibis-project/ibis/commit/21fc5daa86b8738469973b027c9ded8228fc2275))
* document versioning policy ([242ea15](https://github.com/ibis-project/ibis/commit/242ea1527ee5bf8ca44b8adcac9e1f26adb7c51a))
* **dot-sql:** add examples of mixing ibis expressions and SQL strings ([5abd30e](https://github.com/ibis-project/ibis/commit/5abd30ebc92b70bf58afedba7f7b071b2840ef91))
* **dplyr:** small fixes to the dplyr getting started guide ([4b57f7f](https://github.com/ibis-project/ibis/commit/4b57f7fc4fdd3e839cdc2e1014955149bd4f14ca))
* expand docstring for `dtype` function ([39b7a24](https://github.com/ibis-project/ibis/commit/39b7a241cb9d6bf5d6f2b0235262726fc5fdb398))
* fix functions names in examples of extract url fields ([872445e](https://github.com/ibis-project/ibis/commit/872445e4b14a2035721287dbe1f7e77b5c25382e))
* fix heading in 6.0.0 blog ([0ad3ce2](https://github.com/ibis-project/ibis/commit/0ad3ce258af4258dbb2f29f3c48895980aba314a))
* **oracle:** add note about old password checks in oracle ([470b90b](https://github.com/ibis-project/ibis/commit/470b90baac1ce395052bf975b23b030427326a5d))
* **postgres:** fix postgres memtable docs ([7423eb9](https://github.com/ibis-project/ibis/commit/7423eb98ec03accf43471d07f4002aca400d2840))
* **release-notes:** fix typo ([a319e3a](https://github.com/ibis-project/ibis/commit/a319e3a7ab61dc9dc6299984b1d1f415d4ed5287))
* **social:** add social media preview cards ([e98a0a6](https://github.com/ibis-project/ibis/commit/e98a0a6e12582bcad672b975bd97e8fe15538890))
* update imports/exports for pyspark backend ([16d73c4](https://github.com/ibis-project/ibis/commit/16d73c4dab95fe4ce005140516a81341747df801))


### Refactors

* **pyarrow:** remove unnecessary calls to combine_chunks ([c026d2d](https://github.com/ibis-project/ibis/commit/c026d2d0768f94e3bc2afff24ed6fe0d6f54df75))
* **pyarrow:** use `schema.empty_table()` instead of manually constructing empty tables ([c099302](https://github.com/ibis-project/ibis/commit/c0993020ae2d141c58a1f33f94378c272f95b421))
* **result-handling:** remove `result_handler` in favor of expression specific methods ([3dc7143](https://github.com/ibis-project/ibis/commit/3dc7143402d9a31c2655d2d09f637d75598afefb))
* **snowflake:** enable multiple statements and clean up duplicated parameter setting code ([75824a6](https://github.com/ibis-project/ibis/commit/75824a6d302afd53988bfb4684e18d59354cd04f))
* **tests:** clean up backend test setup to make non-data-loading steps atomic ([16b4632](https://github.com/ibis-project/ibis/commit/16b4632ba73a2c88be5a31662aba9f094602354e))

## [6.0.0](https://github.com/ibis-project/ibis/compare/5.1.0...6.0.0) (2023-07-05)


### ⚠ BREAKING CHANGES

* **imports:** Use of `ibis.udf` as a module is removed. Use `ibis.legacy.udf` instead.
* The minimum supported Python version is now Python 3.9
* **api:** `group_by().count()` no longer automatically names the count aggregation `count`. Use `relabel` to rename columns.
* **backends:** `Backend.ast_schema` is removed. Use `expr.as_table().schema()` instead.
* **snowflake/postgres:** Postgres UDFs now use the new `@udf.scalar.python` API. This should be a low-effort replacement for the existing API.
* **ir:** `ops.NullLiteral` is removed
* **datatypes:** `dt.Interval` has no longer a default unit, `dt.interval` is removed
* **deps:** `snowflake-connector-python`'s lower bound was increased to 3.0.2, the minimum version needed to avoid a high-severity vulnerability. Please upgrade `snowflake-connector-python` to at least version 3.0.2.
* **api:** `Table.difference()`, `Table.intersection()`, and `Table.union()` now require at least one argument.
* **postgres:** Ibis no longer automatically defines `first`/`last` reductions on connection to the postgres backend. Use DDL shown in https://wiki.postgresql.org/wiki/First/last_(aggregate) or one of the `pgxn` implementations instead.
* **api:** `ibis.examples.<example-name>.fetch` no longer forwards arbitrary keyword arguments to `read_csv`/`read_parquet`.
* **datatypes:** `dt.Interval.value_type` attribute is removed
* **api:** `Table.count()` is no longer automatically named `"count"`. Use `Table.count().name("count")` to achieve the previous behavior.
* **trino:** The trino backend now requires at least version 0.321 of the `trino` Python package.
* **backends:** removed `AlchemyTable`, `AlchemyDatabase`, `DaskTable`, `DaskDatabase`, `PandasTable`, `PandasDatabase`, `PySparkDatabaseTable`, use `ops.DatabaseTable` instead
* **dtypes:** temporal unit enums are now available under `ibis.common.temporal` instead of `ibis.common.enums`.
* **clickhouse:** `external_tables` can no longer be passed in `ibis.clickhouse.connect`. Pass `external_tables` directly in `raw_sql`/`execute`/`to_pyarrow`/`to_pyarrow_batches()`.
* **datatypes:** `dt.Set` is now an alias for `dt.Array`
* **bigquery:** Before this change, ibis timestamp is mapping to Bigquery TIMESTAMP type and no timezone supports. However, it's not correct, BigQuery TIMESTAMP type should have UTC timezone, while DATETIME type is the no timezone version. Hence, this change is breaking the ibis timestamp mapping to BigQuery: If ibis timestamp has the UTC timezone, will map to BigQuery TIMESTAMP type. If ibis timestamp has no timezone, will map to BigQuery DATETIME type.
* **impala:** Cursors are no longer returned from DDL operations to prevent resource leakage. Use `raw_sql` if you need specialized operations that return a cursor. Additionally, table-based DDL operations now return the table they're operating on.
* **api:** `Column.first()`/`Column.last()` are now reductions by default. Code running these expressions in isolation will no longer be windowed over the entire table. Code using this function in `select`-based APIs should function unchanged.
* **bigquery:** when using the bigquery backend, casting float to int
will no longer round floats to the nearest integer
* **ops.Hash:** The `hash` method on table columns on longer accepts
the `how` argument.  The hashing functions available are highly
backend-dependent and the intention of the hash operation is to provide
a fast, consistent (on the same backend, only) integer value.
If you have been passing in a value for `how`, you can remove it and you
will get the same results as before, as there were no backends with
multiple hash functions working.
* **duckdb:** Some CSV files may now have headers that did not have them previously. Set `header=False` to get the previous behavior.
* **deps:** New environments will have a different default setting for `compression` in the ClickHouse backend due to removal of optional dependencies. Ibis is still capable of using the optional dependencies but doesn't include them by default. Install `clickhouse-cityhash` and `lz4` to preserve the previous behavior.
* **api:** `Table.set_column()` is removed; use `Table.mutate(name=expr)` instead
* **api:** the `suffixes` argument in all join methods has been removed in favor of `lname`/`rname` args. The default renaming scheme for duplicate columns has also changed. To get the exact same behavior as before, pass in `lname="{name}_x", rname="{name}_y"`.
* **ir:** `IntervalType.unit` is now an enum instead of a string
* **type-system:** Inferred types of Python objects may be slightly different. Ibis now use `pyarrow` to infer the column types of pandas DataFrame and other types.
* **backends:** `path` argument of `Backend.connect()` is removed, use the `database` argument instead
* **api:** removed `Table.sort_by()` and `Table.groupby()`, use `.order_by()` and `.group_by()` respectively
* **datatypes:** `DataType.scalar` and `column` class attributes are now strings.
* **backends:** `Backend.load_data()`, `Backend.exists_database()` and `Backend.exists_table()` are removed
* **ir:** `Value.summary()` and `NumericValue.summary()` are removed
* **schema:** `Schema.merge()` is removed, use the union operator `schema1 | schema2` instead
* **api:** `ibis.sequence()` is removed

* drop support for Python 3.8 ([747f4ca](https://github.com/ibis-project/ibis/commit/747f4caca1cf8ab6d4ea908e5d7d7d31eae99f9d))


### Features

* add dask windowing ([9cb920a](https://github.com/ibis-project/ibis/commit/9cb920a35305ddd32decabbeae96dbaff7691ed2))
* add easy type hints to GroupBy ([da330b1](https://github.com/ibis-project/ibis/commit/da330b100ffcf2230e152c311329601e3e3af278))
* add microsecond method to TimestampValue and TimeValue ([e9df2da](https://github.com/ibis-project/ibis/commit/e9df2dafd1208900aa9bda7d2a81c4bb8a99ea40))
* **api:** add `__dataframe__` implementation ([b3d9619](https://github.com/ibis-project/ibis/commit/b3d961998b6cf6a6fcff5e1405e1b37b144dd5f6))
* **api:** add ALL_CAPS option to Table.relabel ([c0b30e2](https://github.com/ibis-project/ibis/commit/c0b30e296cf2156320e7b9718754d637c2d89532))
* **api:** add first/last reduction APIs ([8c01980](https://github.com/ibis-project/ibis/commit/8c01980ed5a30912ea35fa47903347ebf44c8b12))
* **api:** add zip operation and api ([fecf695](https://github.com/ibis-project/ibis/commit/fecf69504ff48651ce707bbbad23539e48e46e4e))
* **api:** allow passing multiple keyword arguments to `ibis.interval` ([22ee854](https://github.com/ibis-project/ibis/commit/22ee854fac49f654faa53575a52f74d61c68e4dd))
* **api:** better repr and pickle support for deferred expressions ([2b1ec9c](https://github.com/ibis-project/ibis/commit/2b1ec9ca28685a34cafeb2771d0e09791a831e96))
* **api:** exact median ([c53031c](https://github.com/ibis-project/ibis/commit/c53031c9069ea9f1e121a9ce3f139ba69eee7fdb))
* **api:** raise better error on column name collision in joins ([e04c38c](https://github.com/ibis-project/ibis/commit/e04c38c9c5cfe95c4015efde3d14f2d786665970))
* **api:** replace `suffixes` in `join` with `lname`/`rname` ([3caf3a1](https://github.com/ibis-project/ibis/commit/3caf3a12469d017428d5e2bb94143185e8770038))
* **api:** support abstract type names in `selectors.of_type` ([f6d2d56](https://github.com/ibis-project/ibis/commit/f6d2d56b3859e59035e4e8b95e93a272f572445f))
* **api:** support list of strings and single strings in the `across` selector ([a6b60e7](https://github.com/ibis-project/ibis/commit/a6b60e78ee626eb2bcb22abb96a65fae20581b08))
* **api:** use `create_table` to load example data ([42e09a4](https://github.com/ibis-project/ibis/commit/42e09a4b3fa910d7c5184611d78afa765356dc60))
* **bigquery:** add client and storage_client params to connect ([4cf1354](https://github.com/ibis-project/ibis/commit/4cf1354945273a66f96f9fcbdb6b03ab69e14433))
* **bigquery:** enable group_concat over windows ([d6a1117](https://github.com/ibis-project/ibis/commit/d6a1117e6ee126d8f69dbc5fafa04c255063d077))
* **cast:** add table-level try_cast ([5e4d16b](https://github.com/ibis-project/ibis/commit/5e4d16b3338fa22f13ff06503c0dfadbe6c03abb))
* **clickhouse:** add array zip impl ([efba835](https://github.com/ibis-project/ibis/commit/efba835aa2ff00a74ad4f94c45da5eabe5b74430))
* **clickhouse:** move to clickhouse supported Python client ([012557a](https://github.com/ibis-project/ibis/commit/012557a737959c31d9c1717e1a431ac96bb5f3b4))
* **clickhouse:** set default engine to native file ([29815fa](https://github.com/ibis-project/ibis/commit/29815fa500f137429592176ace70f9d0e83e3353))
* **clickhouse:** support pyarrow decimal types ([7472dd5](https://github.com/ibis-project/ibis/commit/7472dd5b4377cb3930683b6091ffd9c0208a7b30))
* **common:** add a pure python egraph implementation ([aed2ed0](https://github.com/ibis-project/ibis/commit/aed2ed05e66a2b0b9160d651252f299d42cc6c57))
* **common:** add pattern matchers ([b515d5c](https://github.com/ibis-project/ibis/commit/b515d5cadac8522e388fc89dc19dc6b1b9e0b287))
* **common:** add support for start parameter in StringFind ([31ce741](https://github.com/ibis-project/ibis/commit/31ce741cc9a7c7c890c3552e3e4f466845cf556f))
* **common:** add Topmost and Innermost pattern matchers ([90b48fc](https://github.com/ibis-project/ibis/commit/90b48fc43eeda6df541c239f11927c237f21748a))
* **common:** implement copy protocol for Immutable base class ([e61c66b](https://github.com/ibis-project/ibis/commit/e61c66b874d74b95d0f79c0255579ec595eb1a76))
* **create_table:** support pyarrow Table in table creation ([9dbb25c](https://github.com/ibis-project/ibis/commit/9dbb25cd9a50df8ab64eb4028fc3c8e5911e15ad))
* **datafusion:** add string functions ([66c0afb](https://github.com/ibis-project/ibis/commit/66c0afb9a3ec522b1fdfa67097c085285acd7fa6))
* **datafusion:** add support for scalar pyarrow UDFs ([45935b7](https://github.com/ibis-project/ibis/commit/45935b78922f09ab5be60aef1a1efaf204bc7f4d))
* **datafusion:** minimal decimal support ([c550780](https://github.com/ibis-project/ibis/commit/c550780cdf2b1e0c2c32e017c66dace1db625170))
* **datafusion:** register tables and datasets in datafusion ([cb2cc58](https://github.com/ibis-project/ibis/commit/cb2cc58eac2ca4e7307acb7608dc0d35a92dfd70))
* **datatypes:** add support for decimal values with arrow-based APIs ([b4ba6b9](https://github.com/ibis-project/ibis/commit/b4ba6b98cb175b58238a6873e7894c30729ea8cc))
* **datatypes:** support creating Timestamp from units ([66f2ff0](https://github.com/ibis-project/ibis/commit/66f2ff05dbe945dc900321c3c5cd9ca89c790596))
* **deps:** load examples lazily ([4ea0ddb](https://github.com/ibis-project/ibis/commit/4ea0ddbd2696411e4f717e30d6846d881ae66af5))
* **duckdb:** add attach_sqlite method ([bd32649](https://github.com/ibis-project/ibis/commit/bd326493992845283ceba6d9e45126d9ea395b4a))
* **duckdb:** add support for native and pyarrow UDFs ([7e56fc4](https://github.com/ibis-project/ibis/commit/7e56fc456b13777173598628f804ee4ef806899a))
* **duckdb:** expand map support to `.values()` and map concatenation ([ad49a09](https://github.com/ibis-project/ibis/commit/ad49a099b0feeedc673c7ff1b0a4ba0ef43a1ae1))
* **duckdb:** set `header=True` by default ([e4b515d](https://github.com/ibis-project/ibis/commit/e4b515daa6f96eaa6826c8623e5a885d05a4b3c4))
* **duckdb:** support 0.8.0 ([ae9ae7d](https://github.com/ibis-project/ibis/commit/ae9ae7d07deb3f8e5802365fedfc54c56d04af2e))
* **duckdb:** support array zip operation ([2d14ccc](https://github.com/ibis-project/ibis/commit/2d14ccca308f03f8a246a6d7e61bb463519514ac))
* **duckdb:** support motherduck ([053dc7e](https://github.com/ibis-project/ibis/commit/053dc7ec3ee4f0a38708c9f75fda5b594cc5937a))
* **duckdb:** warn when querying an already consumed RecordBatchReader ([5a013ff](https://github.com/ibis-project/ibis/commit/5a013ffd3032e596a16a6a71476c2754f9bc9c8b))
* **flink:** add initial flink SQL compiler ([053a6d2](https://github.com/ibis-project/ibis/commit/053a6d21e87aa009e284f571efa3b29abd10c5fa))
* **formats:** support timestamps in delta output; default to micros for pyarrow conversion ([d8d5710](https://github.com/ibis-project/ibis/commit/d8d5710f12f220b932f55d41484f779b0303c2bf))
* implement read_delta and to_delta for some backends ([74fc863](https://github.com/ibis-project/ibis/commit/74fc8635a25b45116daea02d04bd9bd959e9d9ae))
* implement read_delta for datafusion ([eb4602f](https://github.com/ibis-project/ibis/commit/eb4602fada06b5f2b64d496c42f769598ef5d99e))
* implement try_cast for a few backends ([f488f0e](https://github.com/ibis-project/ibis/commit/f488f0ec634d91517e94db942486494c89484668))
* **io:** add `to_torch` API ([685c8fc](https://github.com/ibis-project/ibis/commit/685c8fcddb0a68e00109103f74c483acb1c095f0))
* **io:** add az/gs prefixes to normalize_filename in utils ([e9eebba](https://github.com/ibis-project/ibis/commit/e9eebbae0d38854197e77f455bb63095b5dea033))
* **mysql:** add re_extract ([5ed40e1](https://github.com/ibis-project/ibis/commit/5ed40e1d9eea4573cb5626de248759412c5980cb))
* **oracle:** add oracle backend ([c9b038b](https://github.com/ibis-project/ibis/commit/c9b038b773b0b7177d0d5d3109e2f3d6d8584a70))
* **oracle:** support temporary tables ([6e64cd0](https://github.com/ibis-project/ibis/commit/6e64cd0d811a1abb74170d46d695fd56cbb250e7))
* **pandas:** add approx_median ([6714b9f](https://github.com/ibis-project/ibis/commit/6714b9f25d5034cf30ed08e3cfc45ea274f539c7))
* **pandas:** support passing memtables to `create_table` ([3ea9a21](https://github.com/ibis-project/ibis/commit/3ea9a21ad49662dfa78ddbe059a3f74113d8f4d6))
* **polars:** add any and all reductions ([0bd3c01](https://github.com/ibis-project/ibis/commit/0bd3c01f0f492aa83987f5b0dd2cbc05a0eef128))
* **polars:** add argmin and argmax ([78562d3](https://github.com/ibis-project/ibis/commit/78562d346dbde11cc6f12d90fecdedb2922ee6fb))
* **polars:** add correlation operation ([05ff488](https://github.com/ibis-project/ibis/commit/05ff4883c762a4a26ae560cebbb88dc9ebbdb82b))
* **polars:** add polars support for `identical_to` ([aab3bae](https://github.com/ibis-project/ibis/commit/aab3baeca5e5d88c0b2c5ea6081757180d275e55))
* **polars:** add support for `offset`, binary literals, and `dropna(how='all')` ([d2298e9](https://github.com/ibis-project/ibis/commit/d2298e9b24e1537ab733c711daa515c32a215723))
* **polars:** allow seamless connection for DataFrame as well as LazyFrame ([a2a3e45](https://github.com/ibis-project/ibis/commit/a2a3e452e7e87c16ce411199598fe35d1779ee34))
* **polars:** implement `.sql` methods ([86f2a34](https://github.com/ibis-project/ibis/commit/86f2a3472f0d50aa6b20231794f3a35363c29fc7))
* **polars:** lower-latency column return for non-temporal results ([b009563](https://github.com/ibis-project/ibis/commit/b009563e69d1f10823a1f29670e16be1d61b8821))
* **polars:** support pyarrow decimal types ([7e6c365](https://github.com/ibis-project/ibis/commit/7e6c3656ee832b26f502e2f820cfd82cbc79db9b))
* **polars:** support SQL dialect translation ([c87f695](https://github.com/ibis-project/ibis/commit/c87f6950f861bad874b83f92291d3739c23de930))
* **polars:** support table registration from multiple parquet files ([9c0a8be](https://github.com/ibis-project/ibis/commit/9c0a8befacf218e941ac65635883adc9feaa5ff8))
* **postgres:** add ApproxMedian aggregation ([887f572](https://github.com/ibis-project/ibis/commit/887f572a4228d9bb7de16cfd3b367a5cb90cd62d))
* **pyspark:** add zip array impl ([6c00cbc](https://github.com/ibis-project/ibis/commit/6c00cbc9ff5f874d057ade34884a2482346244d0))
* **snowflake/postgres:** scalar UDFs ([dbf5b62](https://github.com/ibis-project/ibis/commit/dbf5b6285fc83647cf689fbda7f0f87d58cc9529))
* **snowflake:** implement array zip ([839e1f0](https://github.com/ibis-project/ibis/commit/839e1f0a036f6dffc85845ea828a71bd1859f837))
* **snowflake:** implement proper approx median ([b15a6fe](https://github.com/ibis-project/ibis/commit/b15a6feb64fc519d5df9bbb59c8ec3443298e6ef))
* **snowflake:** support SSO and other forms of passwordless authentication ([23ac53d](https://github.com/ibis-project/ibis/commit/23ac53d3f122a04de29f25e53c2434de6d34e544))
* **snowflake:** use the client python version as the UDF runtime where possible ([69a9101](https://github.com/ibis-project/ibis/commit/69a91017d403baeddd07e662190d6d1438fbd00e))
* **sql:** allow any SQL dialect accepted by sqlgllot in `Table.sql` and `Backend.sql` ([f38c447](https://github.com/ibis-project/ibis/commit/f38c44789f99c284b0990965ad6b95b1e13d6272))
* **sqlite:** add argmin and argmax functions ([c8af9d4](https://github.com/ibis-project/ibis/commit/c8af9d49263a1c13e8b91e67c3e5256522264572))
* **sqlite:** add arithmetic mode aggregation ([6fcac44](https://github.com/ibis-project/ibis/commit/6fcac4412683d0fc883066518d732a1863f9e86f))
* **sqlite:** add ops.DateSub, ops.DateAdd, ops.DateDiff ([cfd65a0](https://github.com/ibis-project/ibis/commit/cfd65a0216384173ac74228afc7a6a5928698223))
* **streamlit:** add support for streamlit connection interface ([05c9449](https://github.com/ibis-project/ibis/commit/05c9449c158c949bd93543678aad41e7adecf8d9))
* **trino:** implement zip ([cd11daa](https://github.com/ibis-project/ibis/commit/cd11daa136842b5818762b306d8918db4336df35))


### Bug Fixes

* add issue write permission to assign.yml ([9445cee](https://github.com/ibis-project/ibis/commit/9445cee0d3621ae8721c9674ce9abbffc6643865))
* **alchemy:** close the cursor on error during dataframe construction ([cc7dffb](https://github.com/ibis-project/ibis/commit/cc7dffbfccb7a25767ed3e3ff23cf6d6d3c44cdb))
* **backends:** fix capitalize to lowercase subsequent characters ([49978f9](https://github.com/ibis-project/ibis/commit/49978f937a50cbe0d5dd4da6cb7d82178bbea02c))
* **backends:** fix notall/notany translation ([56b56b3](https://github.com/ibis-project/ibis/commit/56b56b363cea406df67d807a2c2067bb71c226ab))
* **bigquery:** add srid=4326 to the geography dtype mapping ([57a825b](https://github.com/ibis-project/ibis/commit/57a825bf7f60c488295703a085bf47509126dbfe))
* **bigquery:** allow passing both schema and obj in create_table ([49cc2c4](https://github.com/ibis-project/ibis/commit/49cc2c40a6c6ae6a32b57208660c51c4435c4551))
* **bigquery:** bigquery timestamp and datetime dtypes ([067e8a5](https://github.com/ibis-project/ibis/commit/067e8a50e7f5d3cc4b0ae0198345f9ae4f868200))
* **bigquery:** ensure that bigquery temporal ops work with the new timeunit/dateunit/intervalunit enums ([0e00d86](https://github.com/ibis-project/ibis/commit/0e00d86eede64170130de0af67025f3fed6873a9))
* **bigquery:** ensure that generated names are used when compiling columns and allow flexible column names ([c7044fe](https://github.com/ibis-project/ibis/commit/c7044fee12ab68b5cb32e49ccaf891d3dbe44ed0))
* **bigquery:** fix table naming from `count` rename removal refactor ([5b009d2](https://github.com/ibis-project/ibis/commit/5b009d2e2775f263628af12723cce6714cad6d22))
* **bigquery:** raise OperationNotDefinedError for IntervalAdd and IntervalSubtract ([501aaf7](https://github.com/ibis-project/ibis/commit/501aaf77b1d7cb7ff9a82275369c023494344adf))
* **bigquery:** support capture group functionality ([3f4f05b](https://github.com/ibis-project/ibis/commit/3f4f05bbc927a55152cc6badfd213f13ab76dc8b))
* **bigquery:** truncate when casting float to int ([267d8e1](https://github.com/ibis-project/ibis/commit/267d8e116ede329407f82e86185261ae49a733a8))
* **ci:** use mariadb-admin instead of mysqladmin in mariadb 11.x ([d4ccd3d](https://github.com/ibis-project/ibis/commit/d4ccd3db72dc90117323b0f9a0f79a3d22a05cdf))
* **clickhouse:** avoid generating names for structs ([5d11f48](https://github.com/ibis-project/ibis/commit/5d11f4823172201497a1a2fbb1b7c376f17e0a7f))
* **clickhouse:** clean up external tables per query to avoid leaking them across queries ([6d32edd](https://github.com/ibis-project/ibis/commit/6d32edd3ec84a468718c96c90c0fa4637de730df))
* **clickhouse:** close cursors more aggressively ([478a40f](https://github.com/ibis-project/ibis/commit/478a40f5d084978466d2ddafe4e1cc20c3c9244c))
* **clickhouse:** use correct functions for milli and micro extraction ([49b3136](https://github.com/ibis-project/ibis/commit/49b313600c696a325b7b1d38de045643832ad201))
* **clickhouse:** use named rather than positional group by ([1f7e309](https://github.com/ibis-project/ibis/commit/1f7e309994c32c4cbc543d0618becd3ff749860a))
* **clickhouse:** use the correct dialect to generate subquery string for Contains operation ([f656bd5](https://github.com/ibis-project/ibis/commit/f656bd57eee83ec0ff6ee2d5dd0c38b97e0e162d))
* **common:** fix bug in re_extract ([6ebaeab](https://github.com/ibis-project/ibis/commit/6ebaeab8ef970dc3b93545590acddaca7c31ba2d)), closes [#6167](https://github.com/ibis-project/ibis/issues/6167)
* **core:** interval resolution should upcast to smallest unit ([f7f844d](https://github.com/ibis-project/ibis/commit/f7f844dd64d5201a64f6c1b5651e85fd7199b0ea)), closes [#6139](https://github.com/ibis-project/ibis/issues/6139)
* **datafusion:** fix incorrect order of predicate -> select compilation ([0092304](https://github.com/ibis-project/ibis/commit/009230421b2bc1f86591e8b850d37a489e8e4f06))
* **deps:** make pyarrow a required dependency ([b217cde](https://github.com/ibis-project/ibis/commit/b217cdef5ace577029191271d089398e4d7be04b))
* **deps:** prevent vulnerable snowflake-connector-python versions ([6dedb45](https://github.com/ibis-project/ibis/commit/6dedb45c9ffb6ef44c65c36aedd98ac0b146b5f9))
* **deps:** support multipledispatch version 1 ([805a7d7](https://github.com/ibis-project/ibis/commit/805a7d7f29054fd53b487f3e9c7741f1fe54346b))
* **deps:** update dependency atpublic to v4 ([3a44755](https://github.com/ibis-project/ibis/commit/3a44755a4224394cfe9128e18a5fe17a04253f40))
* **deps:** update dependency datafusion to v22 ([15d8d11](https://github.com/ibis-project/ibis/commit/15d8d11cdc0bba5d7ce8a8dd06e681bdbe216014))
* **deps:** update dependency datafusion to v23 ([e4d666d](https://github.com/ibis-project/ibis/commit/e4d666da5254debf46507c3d4bad8ed4558cf146))
* **deps:** update dependency datafusion to v24 ([c158b78](https://github.com/ibis-project/ibis/commit/c158b781d9b43f6b873f4273fb30fdbdc336393c))
* **deps:** update dependency datafusion to v25 ([c3a6264](https://github.com/ibis-project/ibis/commit/c3a626470e9efaf235b144d0859ddbfa7ca24a88))
* **deps:** update dependency datafusion to v26 ([7e84ffe](https://github.com/ibis-project/ibis/commit/7e84ffe2dbb003a7e88c6bdae84389c583591f35))
* **deps:** update dependency deltalake to >=0.9.0,<0.11.0 ([9817a83](https://github.com/ibis-project/ibis/commit/9817a8329b9bd048e6711e62baaf5eb24372eefb))
* **deps:** update dependency pyarrow to v12 ([3cbc239](https://github.com/ibis-project/ibis/commit/3cbc23909dbd2db8a7f6abd59c50b790fbbe4c22))
* **deps:** update dependency sqlglot to v12 ([5504bd4](https://github.com/ibis-project/ibis/commit/5504bd40984911c691fc30efaa31cb198e5b0aa9))
* **deps:** update dependency sqlglot to v13 ([1485dd0](https://github.com/ibis-project/ibis/commit/1485dd0336cf6d3e6ce31b545a9e57c6b7babced))
* **deps:** update dependency sqlglot to v14 ([9c40c06](https://github.com/ibis-project/ibis/commit/9c40c066d61781bca9ca010e4e412af0eaa250d3))
* **deps:** update dependency sqlglot to v15 ([f149729](https://github.com/ibis-project/ibis/commit/f14972958e04d5b4558cbad587468bbfa56cfcac))
* **deps:** update dependency sqlglot to v16 ([46601ef](https://github.com/ibis-project/ibis/commit/46601ef03db89ae56cef00b449f9a10a71a9198a))
* **deps:** update dependency sqlglot to v17 ([9b50fb4](https://github.com/ibis-project/ibis/commit/9b50fb4b243e146904eaac7f478697a59be80abc))
* **docs:** fix failing doctests ([04b9f19](https://github.com/ibis-project/ibis/commit/04b9f19074e6d47bac800d0c553c75b95efccf3e))
* **docs:** typo in code without selectors ([b236893](https://github.com/ibis-project/ibis/commit/b23689372fa3d9c30651b69541c8945c1be38152))
* **docs:** typo in docstrings and comments ([0d3ed86](https://github.com/ibis-project/ibis/commit/0d3ed863aab7c9a6768986d236dfcefd53976678))
* **docs:** typo in snowflake do_connect kwargs ([671bc31](https://github.com/ibis-project/ibis/commit/671bc317f276a2a62bcacb40a513c5861decabd7))
* **duckdb:** better types for null literals ([7b9d85e](https://github.com/ibis-project/ibis/commit/7b9d85eab7abe19a1890d479197332c475e791af))
* **duckdb:** disable map values and map merge for columns ([b5472b3](https://github.com/ibis-project/ibis/commit/b5472b3c55afaa891104a2e780ef2b0087ce9aad))
* **duckdb:** ensure `to_timestamp` returns a UTC timestamp ([0ce0b9f](https://github.com/ibis-project/ibis/commit/0ce0b9f59f0275c64af050eb0aa0de06bcf71ca9))
* **duckdb:** ensure connection lifetime is greater than or equal to record batch reader lifetime ([6ed353e](https://github.com/ibis-project/ibis/commit/6ed353e5ffeee30b741e87a099f829fdf1c6d15f))
* **duckdb:** ensure that quoted struct field names work ([47de1c3](https://github.com/ibis-project/ibis/commit/47de1c3e89015f4f778231de182db2b307c4b8ae))
* **duckdb:** ensure that types are inferred correctly across `duckdb_engine` versions ([9c3d173](https://github.com/ibis-project/ibis/commit/9c3d173c1de88e4a489a6c4c39f7302e03fdabe1))
* **duckdb:** fix check for literal maps ([b2b229b](https://github.com/ibis-project/ibis/commit/b2b229bbe75193cf4b65d3bad6f396b70611b685))
* **duckdb:** fix exporting pyarrow record batches by bumping duckdb to 0.8.1 ([aca52ab](https://github.com/ibis-project/ibis/commit/aca52abb03dbd02c4e194da2b43b95104554eff8))
* **duckdb:** fix read_csv problem with kwargs ([6f71735](https://github.com/ibis-project/ibis/commit/6f7173566017267b5925738f6a62ccd98d4a027f)), closes [#6190](https://github.com/ibis-project/ibis/issues/6190)
* **examples:** move lockfile creation to data directory ([b8f6e6b](https://github.com/ibis-project/ibis/commit/b8f6e6b7e52d0773d5dfec7bbcb0aa36d6bc0b39))
* **examples:** use filelock to prevent pooch from clobbering files when fetching concurrently ([e14662e](https://github.com/ibis-project/ibis/commit/e14662e34b720cb2662832819def08f72d50f325))
* **expr:** fix graphviz rendering ([6d4a34f](https://github.com/ibis-project/ibis/commit/6d4a34f74bf4880d33495ad94d44d58fdd97c2b0))
* **impala:** do not cast `ca_cert` `None` value to string ([bfdfb0e](https://github.com/ibis-project/ibis/commit/bfdfb0e8c3848127f1ef146fcc69aa1b5f7aae6d))
* **impala:** expose `hdfs_connect` function as `ibis.impala.hdfs_connect` ([27a0d12](https://github.com/ibis-project/ibis/commit/27a0d12a356bb4e9ac997a33d032755533ba7763))
* **impala:** more aggressively clean up cursors internally ([bf5687e](https://github.com/ibis-project/ibis/commit/bf5687e0d350cf307d9a7d226b9eb7981c601d3c))
* **impala:** replace `time_mapping` with `TIME_MAPPING` and backwards compatible check ([4c3ca20](https://github.com/ibis-project/ibis/commit/4c3ca2003c03639e6bc78ca75ec78fe38708c019))
* **ir:** force an alias if projecting or aggregating columns ([9fb1e88](https://github.com/ibis-project/ibis/commit/9fb1e8861a6daaf3b37c4a4d11eabf12ffa6cd89))
* **ir:** raise Exception for group by with no keys ([845f7ab](https://github.com/ibis-project/ibis/commit/845f7abf75fe2bd42901f7cf230a0527f99f0bf1)), closes [#6237](https://github.com/ibis-project/ibis/issues/6237)
* **mssql:** dont yield from inside a cursor ([4af0731](https://github.com/ibis-project/ibis/commit/4af0731397638664463fe93b318f14abdd0db4e1))
* **mysql:** do not fail when we cannot set the session timezone ([930f8ab](https://github.com/ibis-project/ibis/commit/930f8abc2035fe1d8716fde0da5138127cfe9b5c))
* **mysql:** ensure enum string functions are coerced to the correct type ([e499c7f](https://github.com/ibis-project/ibis/commit/e499c7fcbcc2ed96e99657f16a39dfaee8ee41b7))
* **mysql:** ensure that floats and double do not come back as Python Decimal objects ([a3c329f](https://github.com/ibis-project/ibis/commit/a3c329fd175a24f200d30e32e37f697d54294161))
* **mysql:** fix binary literals ([e081252](https://github.com/ibis-project/ibis/commit/e081252f515fe52c1480656fa7687f8075a58f71))
* **mysql:** handle the zero timestamp value ([9ac86fd](https://github.com/ibis-project/ibis/commit/9ac86fdfe29c34a36eb667e30cea8873fd68a4e2))
* **operations:** ensure that self refs have a distinct name from the table they are referencing ([bd8eb88](https://github.com/ibis-project/ibis/commit/bd8eb884679947542c83254ddf41d6e1c6ae0c28))
* **oracle:** disable autoload when cleaning up temp tables ([b824142](https://github.com/ibis-project/ibis/commit/b8241429bd9ab705334dde9e14175b9fce17ba9a))
* **oracle:** disable statement cache ([41d3857](https://github.com/ibis-project/ibis/commit/41d3857902a750fb6b92d3ace071a92a2c392a0a))
* **oracle:** disable temp tables to get inserts working ([f9985fe](https://github.com/ibis-project/ibis/commit/f9985fe8b9fda88740a82bb036ce8d87eafecc3d))
* **pandas, dask:** allow overlapping non-predicate columns in asof join ([09e26a0](https://github.com/ibis-project/ibis/commit/09e26a00aa484a21265cfe4f21feb6e4128b56f7))
* **pandas:** fix first and last over windows ([9079bc4](https://github.com/ibis-project/ibis/commit/9079bc4fec31721d5cca7f1091aee26624d1e692)), closes [#5417](https://github.com/ibis-project/ibis/issues/5417)
* **pandas:** fix string translate function ([12b9569](https://github.com/ibis-project/ibis/commit/12b956914568b88a6e5d2572ecd2b454e9892735)), closes [#6157](https://github.com/ibis-project/ibis/issues/6157)
* **pandas:** grouped aggregation using a case statement ([d4ac345](https://github.com/ibis-project/ibis/commit/d4ac345de11c8b58e6a3650ed1eed9091796cbad))
* **pandas:** preserve RHS values in asof join when column names collide ([4514668](https://github.com/ibis-project/ibis/commit/4514668afc4745200e6648bdc688f43e0f41cb57))
* **pandas:** solve problem with first and last window function ([dfdede5](https://github.com/ibis-project/ibis/commit/dfdede5585c0bfd2c665895357569afa6586cb9e)), closes [#4918](https://github.com/ibis-project/ibis/issues/4918)
* **polars:** avoid `implode` deprecation warning ([ce3bdad](https://github.com/ibis-project/ibis/commit/ce3bdad86ba47e8447842436c16c7eb6417bb895))
* **polars:** ensure that `to_pyarrow` is called from the backend ([41bacf2](https://github.com/ibis-project/ibis/commit/41bacf2a53e351a2935a6aad8967fa102104c743))
* **polars:** make list column operations backwards compatible ([35fc5f7](https://github.com/ibis-project/ibis/commit/35fc5f72b50fb478ced886fa4a8b22c4f4961136))
* **postgres:** ensure that `alias` method overwrites view even if types are different ([7d5845b](https://github.com/ibis-project/ibis/commit/7d5845b2a18aef74e05e1ae3bd95c5a1a1b6e286))
* **postgres:** ensure that backend still works when create/drop first/last aggregates fails ([eb5d534](https://github.com/ibis-project/ibis/commit/eb5d534b6f3a9f595cc884a5e18440c28dda0f67))
* **pyspark:** enable joining on columns with different names as well as complex predicates ([dcee821](https://github.com/ibis-project/ibis/commit/dcee821ec89b9591faa2860ceebb8e8491bfb550))
* **snowflake:** always use pyarrow for memtables ([da34d6f](https://github.com/ibis-project/ibis/commit/da34d6f03396f85c477a1415ae70e6140d07d48b))
* **snowflake:** ensure connection lifetime is greater than or equal to record batch reader lifetime ([34a0c59](https://github.com/ibis-project/ibis/commit/34a0c5918a3b18db611537977fa8a854e925fe67))
* **snowflake:** ensure that `_pandas_converter` attribute is resolved correctly ([9058bbe](https://github.com/ibis-project/ibis/commit/9058bbe11ba7714976f3f6383d1f9a9d6ce88722))
* **snowflake:** ensure that temp tables are only created once ([43b8152](https://github.com/ibis-project/ibis/commit/43b81529c947518aaa07ec3276502334796ce2f9))
* **snowflake:** ensure unnest works for nested struct/object types ([fc6ffc2](https://github.com/ibis-project/ibis/commit/fc6ffc25834b1778dfcf25c7603ab38359beed23))
* **snowflake:** ensure use of the right timezone value ([40426bf](https://github.com/ibis-project/ibis/commit/40426bfff23777d71d5b12c96a4444b8829415df))
* **snowflake:** fix `tmpdir` construction for python <3.10 ([a507ae2](https://github.com/ibis-project/ibis/commit/a507ae2722ab4814a1a9e758385ad95184c67e00))
* **snowflake:** fix incorrect arguments to snowflake regexp_substr ([9261f70](https://github.com/ibis-project/ibis/commit/9261f708f28d53d30001147dbfe06db403f6073d))
* **snowflake:** fix invalid attribute access when using pyarrow ([bfd90a8](https://github.com/ibis-project/ibis/commit/bfd90a88459f542d523f8e578fbc56793f15580a))
* **snowflake:** handle broken upstream behavior when a table can't be found ([31a8366](https://github.com/ibis-project/ibis/commit/31a836655675899543589450a40fc4125f81bd9b))
* **snowflake:** resolve import error from interval datatype refactor ([3092012](https://github.com/ibis-project/ibis/commit/3092012c020225fdfdb3bd6033353a1976ea01f3))
* **snowflake:** use `convert_timezone` for timezone conversion instead of invalid postgres `AT TIME ZONE` syntax ([1595e7b](https://github.com/ibis-project/ibis/commit/1595e7b7ceeeb29fc5305dbb243fc326393fec32))
* **sqlalchemy:** ensure that backends don't clobber tables needed by inputs ([76e38a3](https://github.com/ibis-project/ibis/commit/76e38a320fbb1fc5a7ff60e403975cc214cb8c70))
* **sqlalchemy:** ensure that union_all-generated memtables use the correct column names ([a4f546b](https://github.com/ibis-project/ibis/commit/a4f546bf09e3c30d692bb07e8c77774ab3adf11d))
* **sqlalchemy:** prepend the table's schema when querying metadata ([d8818e2](https://github.com/ibis-project/ibis/commit/d8818e2da26bb4089c1b5f2cbd1169a65d7a6c50))
* **sqlalchemy:** quote struct field names ([f5c91fc](https://github.com/ibis-project/ibis/commit/f5c91fcec8560cab49c38f815c06409c4e8571b2))
* **tests:** ensure that record batch readers are cleaned up ([d230a8d](https://github.com/ibis-project/ibis/commit/d230a8dea38a416ffd4f09d405c1746b0b08373c))
* **trino:** bump lower bound to avoid having to handle `experimental_python_types` ([bf6eeab](https://github.com/ibis-project/ibis/commit/bf6eeabade897898dd06ddf57ff8bd3b6eec835e))
* **trino:** ensure that nested array types are inferred correctly ([030f76d](https://github.com/ibis-project/ibis/commit/030f76d7950c5d57977950233e4ba491b3a60b19))
* **trino:** fix incorrect `version` computation ([04d3a89](https://github.com/ibis-project/ibis/commit/04d3a897fb7269bfaa83e3a804655d2b7a9d2768))
* **trino:** support trino 0.323 special tuple type for struct results ([ea1529d](https://github.com/ibis-project/ibis/commit/ea1529d14aec891974076c4b093ac6c7c6bc9854))
* **type-system:** infer in-memory object types using pyarrow ([f7018ee](https://github.com/ibis-project/ibis/commit/f7018ee8a3c2df29e05400f9dc33a2f0bb6ab509))
* **typehint:** update type hint for class instance ([2e1e14f](https://github.com/ibis-project/ibis/commit/2e1e14ff3109e6c864ebbfbd128b565ad2c25a30))


### Documentation

* **across:** add documentation for across ([b8941d3](https://github.com/ibis-project/ibis/commit/b8941d3562dcd98c25678517e61e5e89f63b6eae))
* add allowed input for memtable constructor ([69cdee5](https://github.com/ibis-project/ibis/commit/69cdee5c731c77db01bacdd5a3dd2b109f31ff04))
* add disclaimer on no row order guarantees ([75dd8b0](https://github.com/ibis-project/ibis/commit/75dd8b039439cf890fd37c4d314b4613759d4bef))
* add examples to `if_any` and `if_all` ([5015677](https://github.com/ibis-project/ibis/commit/5015677d78909473014a61725d371b4bf772cdff))
* add platform comment in conda env creation ([e38eacb](https://github.com/ibis-project/ibis/commit/e38eacb6f5da2dfc67cc9923cfa3324c3a741a66))
* add read_delta and related to backends docs ([90eaed2](https://github.com/ibis-project/ibis/commit/90eaed265f52deda08cda2374c7ed390c16dfd04))
* **api:** ensure all top-level items have a description ([c83d783](https://github.com/ibis-project/ibis/commit/c83d783fd90d384e1b6447d6e9a2640bc24876a7))
* **api:** hide dunder methods in API docs ([6724b7b](https://github.com/ibis-project/ibis/commit/6724b7be92f82343aa74608492ebdf1eb6a801a2))
* **api:** manually add inherited mixin methods to timey classes ([7dbc96d](https://github.com/ibis-project/ibis/commit/7dbc96d162c357221e88770e59ae7fc76fd51a5d))
* **api:** show source for classes to allow dunder method inspection ([4cef0f8](https://github.com/ibis-project/ibis/commit/4cef0f82ee9cc3a1bc143b20f4c394f4ebcc0af7))
* **backends:** fix typo in pip install command ([6a7207c](https://github.com/ibis-project/ibis/commit/6a7207c3beae5ad38f0e4b67c3f49dc21c22e412))
* **bigquery:** add connection explainer to bigquery backend docs ([84caa5b](https://github.com/ibis-project/ibis/commit/84caa5b7b7272cd3917a351e87758a2961c66a7d))
* **blog:** add Ibis + PyTorch + DuckDB blog post ([1ad946c](https://github.com/ibis-project/ibis/commit/1ad946ce23ac123c993b4750916ee4ea7f01dec5))
* change plural variable name cols to col ([c33a3ed](https://github.com/ibis-project/ibis/commit/c33a3ed0048faefb35c6b5b6b1b86398d7f9b80a)), closes [#6115](https://github.com/ibis-project/ibis/issues/6115)
* clarify map refers to Python Mapping container ([f050a61](https://github.com/ibis-project/ibis/commit/f050a611a31331ca1247407b7a018a571246dbb8))
* **css:** enable code block copy button, don't select prompt ([3510abe](https://github.com/ibis-project/ibis/commit/3510abe63a7fb611898f028a4ba8b94ae15604b4))
* de-template remaining backends (except pandas, dask, impala) ([82b7408](https://github.com/ibis-project/ibis/commit/82b740875329e786aa59a427b2ef204d42621d37))
* describe NULL differences with pandas ([688b293](https://github.com/ibis-project/ibis/commit/688b2934c487ffc8ad4678c8a841cf45fde6becd))
* **dev-env:** remove python 3.8 from environment support matrix ([4f89565](https://github.com/ibis-project/ibis/commit/4f89565345cdb9a9afe055cf01637a57d46749f9))
* drop `docker-compose` install for conda dev env setup ([e19924d](https://github.com/ibis-project/ibis/commit/e19924dac362caa683cf5c9efbfa10a4b479428e))
* **duckdb:** add quick explainer on connecting to motherduck ([4ef710e](https://github.com/ibis-project/ibis/commit/4ef710e85b4a5f51e2694ab2329df5fc1aca0bcb))
* **file support:** add badge and docstrings for `read_*` methods ([0767b7c](https://github.com/ibis-project/ibis/commit/0767b7c85e4b2190aebd5bb7d6cd1fa89f4f2b0c))
* fill out more docstrings ([dc0289c](https://github.com/ibis-project/ibis/commit/dc0289c3e96291a95f9a7016538db29311028224))
* fix errors and add 'table' before 'expression' ([096b568](https://github.com/ibis-project/ibis/commit/096b568a801b0baae830ece70a9b1cdd8b82191d))
* fix some redirects ([3a23c1f](https://github.com/ibis-project/ibis/commit/3a23c1f8e19055a5a88711a0a8fbe52c9033bbc9))
* fix typo in Table.relabel return description ([05cc51e](https://github.com/ibis-project/ibis/commit/05cc51e4cf1225674a28a9d15b4c4f971dfb1f09))
* **generic:** add docstring examples in types/generic ([1d87292](https://github.com/ibis-project/ibis/commit/1d872923ba08be8495b7c6a1e4c534021877c9e1))
* **guides:** add brief installation instructions at top of notebooks ([dc3e694](https://github.com/ibis-project/ibis/commit/dc3e694dec2222a890612b7e3445d3bf53d605d0))
* **guides:** update ibis-for-dplyr-users.ipynb with latest ([1aa172e](https://github.com/ibis-project/ibis/commit/1aa172e543cd76d83810db66a139681fc7cbf2b2)), closes [#6125](https://github.com/ibis-project/ibis/issues/6125)
* improve docstrings for BooleanValue and BoleanColumn ([30c1009](https://github.com/ibis-project/ibis/commit/30c1009102826875bf530123f49935920e8645b1))
* improve docstrings to map types ([72a49b0](https://github.com/ibis-project/ibis/commit/72a49b0e3771870f061f35e9ec60dbb55c50c46c))
* **install:** add quotes to all bracketed installs for shell compatibility ([bb5c075](https://github.com/ibis-project/ibis/commit/bb5c075174f517427a061581728c55c65c7b8c38))
* **intersphinx:** add mapping to autolink pyarrow and pandas refs ([cd92019](https://github.com/ibis-project/ibis/commit/cd92019d517a63b5e583711b7ead310ccfa1aa48))
* **intro:** create Ibis for dplyr users document ([e02a6f2](https://github.com/ibis-project/ibis/commit/e02a6f28dc6be2cfe04d8179839e8e77215f33a6))
* **introguides:** use DuckDB for intro pandas notebook, remove iris ([a7e845a](https://github.com/ibis-project/ibis/commit/a7e845ac9a2c2069135c38a923babc1f1d6de65c))
* link to Ibis for dplyr users ([6e7c6a2](https://github.com/ibis-project/ibis/commit/6e7c6a23048c24b7598147f759bbf30a62069e37))
* make pandas.md filename lowercase ([4937d45](https://github.com/ibis-project/ibis/commit/4937d459c96467cba12303642f6a189c3adbc773))
* more group_by() and NULL in pandas guide ([486b696](https://github.com/ibis-project/ibis/commit/486b6964f6254e335c42e44f9443a84672bcc741))
* more spelling fixes ([564abbe](https://github.com/ibis-project/ibis/commit/564abbe88b99e1455103c58f21ef1eb6f40b9a86))
* move API docs to top-level ([dcc409f](https://github.com/ibis-project/ibis/commit/dcc409f1ab408f7650d8346297f2cadf5fd0ad10))
* **numeric:** add examples to numeric methods ([39b470f](https://github.com/ibis-project/ibis/commit/39b470faec0288581627774d4a313a7511a74f76))
* **oracle:** add basic backend documentation ([c871790](https://github.com/ibis-project/ibis/commit/c8717908537ed299d16aadad55cceadc8272613b))
* **oracle:** add oracle to matrix ([89aecf2](https://github.com/ibis-project/ibis/commit/89aecf2d42cb68bbecb27891318c1eea7dc7cc97))
* **python-versions:** document how we decide to drop support for Python versions ([3474dbc](https://github.com/ibis-project/ibis/commit/3474dbc9f9fae60cb6fd52b98de2112cb55a5bf4))
* redirect Pandas to pandas ([4074284](https://github.com/ibis-project/ibis/commit/40742848b41fd3b4b6e7709f0d48eb2cc080aca7))
* remove trailing whitespace ([63db643](https://github.com/ibis-project/ibis/commit/63db64333e0a7a178008663657fec7fa2425c439))
* reorder sections in pandas guide ([3b66093](https://github.com/ibis-project/ibis/commit/3b6609328f0b6d6385f8658a02d5c59e88fc1cfa))
* restructure and consistency ([351d424](https://github.com/ibis-project/ibis/commit/351d4249539339f165eeb4b137b5018013021a01))
* **snowflake:** add connection explainer to snowflake backend docs ([a62bbcd](https://github.com/ibis-project/ibis/commit/a62bbcda61e62dbde2c4c760918cff2a3a9d7e02))
* **streamlit:** fix ibis-framework install ([a8cf773](https://github.com/ibis-project/ibis/commit/a8cf7730c3d3448a9975fcc561a9fce3b7d8b7c0))
* update copyright and some minor edits ([b9aed44](https://github.com/ibis-project/ibis/commit/b9aed4404a182e6dcfc33e5f3027cf50658db06c))
* update notany/notall docstrings with  arg ([a5ec986](https://github.com/ibis-project/ibis/commit/a5ec9862a87cdefe9fe99bf6a72007e180f5ff63)), closes [#5993](https://github.com/ibis-project/ibis/issues/5993)
* update structs and fix constructor docstrings ([493437a](https://github.com/ibis-project/ibis/commit/493437ad84e3ff3cc3db793ed2d95154749a0e57))
* use lowercase pandas ([19b5d10](https://github.com/ibis-project/ibis/commit/19b5d10df7b05995905a552571f0e4440a3615de))
* use to_pandas instead of execute ([882949e](https://github.com/ibis-project/ibis/commit/882949e0e59a13778af4470682b0ab278752dd3d))


### Refactors

* **alchemy:** abstract out custom type mapping and fix sqlite ([d712e2e](https://github.com/ibis-project/ibis/commit/d712e2ec23fa13e659c813e2298f846775e335f6))
* **api:** consolidate `ibis.date()`, `ibis.time()` and `ibis.timestamp()` functions ([20f71bf](https://github.com/ibis-project/ibis/commit/20f71bfed5aabdd43cbcdca5fbd32534cf5ae9e5))
* **api:** enforce at least one argument for `Table` set operations ([57e948f](https://github.com/ibis-project/ibis/commit/57e948fc5777eb77aa9fc986ac4b13cdc31fe51b))
* **api:** remove automatic `count` name from relations ([2cb19ec](https://github.com/ibis-project/ibis/commit/2cb19ec5ab05d81ba4701f490ad62002fffe3d6a))
* **api:** remove automatic group by count naming ([15d9e50](https://github.com/ibis-project/ibis/commit/15d9e50676093efce438f02180c869c3957eb521))
* **api:** remove deprecated `ibis.sequence()` function ([de0bf69](https://github.com/ibis-project/ibis/commit/de0bf69ca68dd0c8ee22bc3cbb63c574721e43d0))
* **api:** remove deprecated `Table.set_column()` method ([aa5ed94](https://github.com/ibis-project/ibis/commit/aa5ed94d5d5aff5141d72ff60e08a972b71f1f68))
* **api:** remove deprecated `Table.sort_by()` and `Table.groupby()` methods ([1316635](https://github.com/ibis-project/ibis/commit/1316635d36a185c547d2c9d658b888b1ce80f387))
* **backends:** remove `ast_schema` method ([51b5ef8](https://github.com/ibis-project/ibis/commit/51b5ef8a2331ebaa03f5b573856c4f88bd297a34))
* **backends:** remove backend specific `DatabaseTable` operations ([d1bab97](https://github.com/ibis-project/ibis/commit/d1bab97e3388e19e3b2235512e895b60cb52434c))
* **backends:** remove deprecated `Backend.load_data()`, `.exists_database()` and `.exists_table()` methods ([755555f](https://github.com/ibis-project/ibis/commit/755555f5a61245d0fdb5da1d98283a67f350f468))
* **backends:** remove deprecated `path` argument of `Backend.connect()` ([6737ea8](https://github.com/ibis-project/ibis/commit/6737ea88c36d97170ea369edc1d73d910d832716))
* **bigquery:** align datatype conversions with the new convention ([70b8232](https://github.com/ibis-project/ibis/commit/70b823264ebf894c1e10e7acd389ca57d8a64c32))
* **bigquery:** support a broader range of interval units in temporal binary operations ([f78ce73](https://github.com/ibis-project/ibis/commit/f78ce731e3bf36688c2bb38d95553708048c4591))
* **common:** add sanity checks for creating ENodes and Patterns ([fc89cc3](https://github.com/ibis-project/ibis/commit/fc89cc3a7f7354019362ba18ffe842eb1955f3ce))
* **common:** cleanup unit conversions ([73de24e](https://github.com/ibis-project/ibis/commit/73de24eb73355d4a917aa64b1049d9a489ba6a8c))
* **common:** disallow unit conversions between days and hours ([5619ce0](https://github.com/ibis-project/ibis/commit/5619ce05fd61f8cdbe5196e4e7b53b05825ff93f))
* **common:** move `ibis.collections.DisjointSet` to `ibis.common.egraph` ([07dde21](https://github.com/ibis-project/ibis/commit/07dde215c21dcc8b9336262ceda5476ead1c64de))
* **common:** move tests for re_extract to general suite ([acd1774](https://github.com/ibis-project/ibis/commit/acd17745c5d098dc467836f4ac4d5d35d0b334bc))
* **common:** use an enum as a sentinel value instead of NoMatch class ([6674353](https://github.com/ibis-project/ibis/commit/667435355365102aab9eab1d8655b73530ba69eb)), closes [#6049](https://github.com/ibis-project/ibis/issues/6049)
* **dask/pandas:** align datatype conversions with the new convention ([cecc24c](https://github.com/ibis-project/ibis/commit/cecc24cafe710e6247c110c6dd2a63f874440626))
* **datatypes:** make pandas conversion backend specific if needed ([544d27c](https://github.com/ibis-project/ibis/commit/544d27c258c82ac028960342bf4385e8e83263e4))
* **datatypes:** normalize interval values to integers ([80a40ab](https://github.com/ibis-project/ibis/commit/80a40abb7116156e6091c5d8e2e3609d483b60db))
* **datatypes:** remove `Set()` in favor of `Array()` datatype ([30a4f7e](https://github.com/ibis-project/ibis/commit/30a4f7e3531df57aba0d7391b4cf2a1605a4ea18))
* **datatypes:** remove `value_type` parametrization of the Interval datatype ([463cdc3](https://github.com/ibis-project/ibis/commit/463cdc3328a61414a07328399d0cb51c2690904a))
* **datatypes:** remove direct `ir` dependency from `datatypes` ([d7f0be0](https://github.com/ibis-project/ibis/commit/d7f0be0433fbab1c4e35c749cbe016f978b751bc))
* **datatypes:** use typehints instead of rules ([704542e](https://github.com/ibis-project/ibis/commit/704542e05bb7fea1d6dd3a9bd9f552771a1c3c8d))
* **deps:** remove optional dependency on clickhouse-cityhash and lz4 ([736fe26](https://github.com/ibis-project/ibis/commit/736fe26275b7bce00cbbcbd8f7e7a16acffc3b07))
* **dtypes:** add `normalize_datetime()` and `normalize_timezone()` common utilities ([c00ab38](https://github.com/ibis-project/ibis/commit/c00ab38c82d2273e030adabbe66ac0a294ca9cad))
* **dtypes:** turn dt.dtype() into lazily dispatched factory function ([5261003](https://github.com/ibis-project/ibis/commit/52610037915f623cb08a6bf112de3b71d4b72731))
* **formats:** consolidate the dataframe conversion logic ([53ed88e](https://github.com/ibis-project/ibis/commit/53ed88e92f7166914df488871649537a356d922c))
* **formats:** encapsulate conversions to TypeMapper, SchemaMapper and DataMapper subclasses ([ab35311](https://github.com/ibis-project/ibis/commit/ab35311bf9f487d965869d4d775242eec83ae921))
* **formats:** introduce a standalone subpackage to deal with common in-memory formats ([e8f45f5](https://github.com/ibis-project/ibis/commit/e8f45f560491b58dc3559996fcffea88657a019f))
* **impala:** rely on impyla cursor for _wait_synchronous ([a1b8736](https://github.com/ibis-project/ibis/commit/a1b8736e97c8edc674b17603084613d0dc3d5beb))
* **imports:** move old UDF implementation to ibis.legacy module ([cf93d5d](https://github.com/ibis-project/ibis/commit/cf93d5dbaf5b89e16f5ff65dfef995f8f328af09))
* **ir:** encapsulate temporal unit handling in enums ([1b8fa7b](https://github.com/ibis-project/ibis/commit/1b8fa7bd96a61b5fbe46f8913a9d0ce686f1426a))
* **ir:** remove `rlz.column_from`, `rlz.base_table_of` and `rlz.function_of` rules ([ed71d51](https://github.com/ibis-project/ibis/commit/ed71d510612b098174d6bb9a8eb2d0dd42d5df3f))
* **ir:** remove deprecated `Value.summary()` and `NumericValue.summary()` expression methods ([6cd8050](https://github.com/ibis-project/ibis/commit/6cd80508ca0a8e7bf2f7345a4ede61cdfca43587))
* **ir:** remove redundant `ops.NullLiteral()` operation ([a881703](https://github.com/ibis-project/ibis/commit/a881703e35d62062b47b59b1084cf85c7b669ce7))
* **ir:** simplify `Expr._find_backends()` implementation by using the `ibis.common.graph` utilities ([91ff8d4](https://github.com/ibis-project/ibis/commit/91ff8d44f45d358e7a85051f907d0853bc001776))
* **ir:** use `dt.normalize()` to construct literals ([bf72f16](https://github.com/ibis-project/ibis/commit/bf72f163f68eee9f6195238fff5e293b0ce74389))
* **ops.Hash:** remove `how` from backend-specific hash operation ([46a55fc](https://github.com/ibis-project/ibis/commit/46a55fcf2060326f8a0bd279a4a6c49edb89dda2))
* **pandas:** solve and remove stale TODOs ([92d979e](https://github.com/ibis-project/ibis/commit/92d979ea726e3d76d2a96be13938412ca1d1a024))
* **polars:** align datatype conversion functions with the new convention ([5d61159](https://github.com/ibis-project/ibis/commit/5d611591f83c82faf95440465922acdfc4122dad))
* **postgres:** fail at execute time for UDFs to avoid db connections in `.compile()` ([e3a4d4d](https://github.com/ibis-project/ibis/commit/e3a4d4d2fbe6ca0016cf54bebf45114c8b093cf0))
* **pyspark:** align datatype conversion functions with the new convention ([3437bb6](https://github.com/ibis-project/ibis/commit/3437bb670e9fae20676cf39d13b304ff057bf349))
* **pyspark:** remove useless window branching in compiler ([ad08da4](https://github.com/ibis-project/ibis/commit/ad08da480bca7090dcd0090b731137da8a04923c))
* replace custom `_merge` using `pd.merge` ([fe74f76](https://github.com/ibis-project/ibis/commit/fe74f767148d801f1faae8e90ac5268248090e6d))
* **schema:** remove deprecated `Schema.merge()` method ([d307722](https://github.com/ibis-project/ibis/commit/d307722d8b30e22646804010ff47cf8491b62a7c))
* **schema:** use type annotations instead of rules ([98cd539](https://github.com/ibis-project/ibis/commit/98cd5396ce7f52d299b1748c458974be4522db40))
* **snowflake:** add flags to supplemental JavaScript UDFs ([054add4](https://github.com/ibis-project/ibis/commit/054add461b1b0b90e865782e41a1ab00bcdcd3f2))
* **sql:** align datatype conversions with the new convention ([0ef145b](https://github.com/ibis-project/ibis/commit/0ef145b64a8ac8893ea52cb5bdf9b14952ffc91c))
* **sqlite:** remove roundtripping for DayOfWeekIndex and DayOfWeekName ([b5a2bc5](https://github.com/ibis-project/ibis/commit/b5a2bc5527f34cd3f4568b253da6cfcc222f4ac9))
* **test:** cleanup test data ([7ae2b24](https://github.com/ibis-project/ibis/commit/7ae2b24223b0501c0844693c50dedeae72f6c658))
* **to-pyarrow-batches:** ensure that batch readers are always closed and exhausted ([35a391f](https://github.com/ibis-project/ibis/commit/35a391f15cbea0b6da089cf8ab71fe3f6d839da7))
* **trino:** always clean up prepared statements created when accessing query metadata ([4f3a4cd](https://github.com/ibis-project/ibis/commit/4f3a4cda10ac02b4cd9093f82a63be1f0b70b55e))
* **util:** use base32 to compress uuid table names ([ba039a3](https://github.com/ibis-project/ibis/commit/ba039a3bbc52bdb6f34aa70eacd99933aed10c47))


### Performance

* **imports:** speed up checking for geospatial support ([aa601af](https://github.com/ibis-project/ibis/commit/aa601afc24964b4ce3f8c0106950fe22036b6f4e))
* **snowflake:** use pyarrow for all transport ([1fb89a1](https://github.com/ibis-project/ibis/commit/1fb89a13bed15ece0301d92206b2bd6c167cf283))
* **sqlalchemy:** lazily construct the inspector object ([8db5624](https://github.com/ibis-project/ibis/commit/8db56240efd3deac88e85b666ae4304a6f1e0191))


### Deprecations

* **api:** deprecate tuple syntax for order by keys ([5ed5110](https://github.com/ibis-project/ibis/commit/5ed51107a866fcfac01b9f7aadb3f38607eb3f04))

## [5.1.0](https://github.com/ibis-project/ibis/compare/5.0.0...5.1.0) (2023-04-11)


### Features

* **api:** expand `distinct` API for dropping duplicates based on column subsets ([3720ea5](https://github.com/ibis-project/ibis/commit/3720ea5d86b45b5455870ade991d2de755254760))
* **api:** implement pyarrow memtables ([9d4fbbd](https://github.com/ibis-project/ibis/commit/9d4fbbd0e9da85146a15f698be34cd7bb63bfe05))
* **api:** support passing a format string to `Table.relabel` ([0583959](https://github.com/ibis-project/ibis/commit/05839595ca203715e36d528dae83cd68a804a05d))
* **api:** thread kwargs around properly to support more complex connection arguments ([7e0e15b](https://github.com/ibis-project/ibis/commit/7e0e15b0f226b523968384a59942dda840f60911))
* **backends:** add more array functions ([5208801](https://github.com/ibis-project/ibis/commit/5208801cc38ff2ccb02fc9f7e83ad40c44222af1))
* **bigquery:** make `to_pyarrow_batches()` smarter ([42f5987](https://github.com/ibis-project/ibis/commit/42f5987762da25a63a598b5fdf0abc3eea5ae8e0))
* **bigquery:** support bignumeric type ([d7c0f49](https://github.com/ibis-project/ibis/commit/d7c0f49eebae9649b968a9554f62c7175742a3c2))
* default repr to showing all columns in Jupyter notebooks ([91a0811](https://github.com/ibis-project/ibis/commit/91a08113a9913971abba49683953b7e342bbb8aa))
* **druid:** add re_search support ([946202b](https://github.com/ibis-project/ibis/commit/946202b8058ee8df83a8cb4ac326d4c9c94ff1a5))
* **duckdb:** add map operations ([a4c4e77](https://github.com/ibis-project/ibis/commit/a4c4e77be24e3692608d6f1773bf563bf9a39085))
* **duckdb:** support sqlalchemy 2 ([679bb52](https://github.com/ibis-project/ibis/commit/679bb52daa72793ab5ea598e1f0399780c2a0801))
* **mssql:** implement ops.StandardDev, ops.Variance ([e322f1d](https://github.com/ibis-project/ibis/commit/e322f1d5c36d725c9e96b764f86242482e4f577a))
* **pandas:** support memtable in pandas backend ([6e4d621](https://github.com/ibis-project/ibis/commit/6e4d621754557c08e60fd46212ee9eac39594aed)), closes [#5467](https://github.com/ibis-project/ibis/issues/5467)
* **polars:** implement count distinct ([aea4ccd](https://github.com/ibis-project/ibis/commit/aea4ccd7bc85b020d23506c610daf3fe5b204c2f))
* **postgres:** implement `ops.Arbitrary` ([ee8dbab](https://github.com/ibis-project/ibis/commit/ee8dbabfa210257102010d76f7ae7f0cd4d01e05))
* **pyspark:** `pivot_longer` ([f600c90](https://github.com/ibis-project/ibis/commit/f600c90f5abc06f167a1110bcecf66c6b7ed35f2))
* **pyspark:** add ArrayFilter operation ([2b1301e](https://github.com/ibis-project/ibis/commit/2b1301e14fe6abf80ce0d942fbaa5ba5427b9b7a))
* **pyspark:** add ArrayMap operation ([e2c159c](https://github.com/ibis-project/ibis/commit/e2c159cb497eb5268dc253dc50348b86020bd21c))
* **pyspark:** add DateDiff operation ([bfd6109](https://github.com/ibis-project/ibis/commit/bfd61093feb7da46edf56ba9d4b9dbc151700629))
* **pyspark:** add partial support for interval types ([067120d](https://github.com/ibis-project/ibis/commit/067120df1f3c0d2b45af60fc1b28cb4b0463d5fb))
* **pyspark:** add read_csv, read_parquet, and register ([7bd22af](https://github.com/ibis-project/ibis/commit/7bd22af6158ff60d2aa70d41a7d5156a67e14bfa))
* **pyspark:** implement count distinct ([db29e10](https://github.com/ibis-project/ibis/commit/db29e10fe7fd26ef8e9024686fef38fc137546b0))
* **pyspark:** support basic caching ([ab0df7a](https://github.com/ibis-project/ibis/commit/ab0df7aac6f06727d857c7c23a3279f341bdee64))
* **snowflake:** add optional 'connect_args' param ([8bf2043](https://github.com/ibis-project/ibis/commit/8bf2043d1a1a5517e51f0638686596d0428a3f3c))
* **snowflake:** native pyarrow support ([ce3d6a4](https://github.com/ibis-project/ibis/commit/ce3d6a450e780e5efcc65ca056ec3e2ec41a2001))
* **sqlalchemy:** support unknown types ([fde79fa](https://github.com/ibis-project/ibis/commit/fde79fa0ee48b0b87bc7e82023e2b81d621c8cef))
* **sqlite:** implement `ops.Arbitrary` ([9bcdf77](https://github.com/ibis-project/ibis/commit/9bcdf77ddafce75f0e5d8714d01dde81ed0b90f2))
* **sql:** use temp views where possible ([5b9d8c0](https://github.com/ibis-project/ibis/commit/5b9d8c0db244e2742f2eddeca0787661aa516642))
* **table:** implement `pivot_wider` API ([60e7731](https://github.com/ibis-project/ibis/commit/60e7731f5e58c1236c6bf50a29cebe807fa08c77))
* **ux:** move `ibis.expr.selectors` to `ibis.selectors` and deprecate for removal in 6.0 ([0ae639d](https://github.com/ibis-project/ibis/commit/0ae639d6ca8f91cd098f069b4bb118b9c6d05059))


### Bug Fixes

* **api:** disambiguate attribute errors from a missing `resolve` method ([e12c4df](https://github.com/ibis-project/ibis/commit/e12c4df0436a7e8f13f8e63d0af1e7a780012ca6))
* **api:** support filter on literal followed by aggregate ([68d65c8](https://github.com/ibis-project/ibis/commit/68d65c89098563e7a48073837ad8d73ec0f2bdba))
* **clickhouse:** do not render aliases when compiling aggregate expression components ([46caf3b](https://github.com/ibis-project/ibis/commit/46caf3b877a4c7e0457dd78a1085bfcff082a2d9))
* **clickhouse:** ensure that clickhouse depends on sqlalchemy for `make_url` usage ([ea10a27](https://github.com/ibis-project/ibis/commit/ea10a2752e02ab8e30396f7e5244b6a20c80ddd3))
* **clickhouse:** ensure that truncate works ([1639914](https://github.com/ibis-project/ibis/commit/163991453e233cb1a9c92bb571327bf3840d10a1))
* **clickhouse:** fix `create_table` implementation ([5a54489](https://github.com/ibis-project/ibis/commit/5a544898ab221731a6f6a847c24afda64c39e44c))
* **clickhouse:** workaround sqlglot issue with calling `match` ([762f4d6](https://github.com/ibis-project/ibis/commit/762f4d64a61c8e961682bdb7f8ac5832973625fb))
* **deps:** support pandas 2.0 ([4f1d9fe](https://github.com/ibis-project/ibis/commit/4f1d9fefb991733d14d03677bb63fdee93a7fc60))
* **duckdb:** branch to avoid unnecessary dataframe construction ([9d5d943](https://github.com/ibis-project/ibis/commit/9d5d94311889dd99f29affc4524b02f362c55f0f))
* **duckdb:** disable the progress bar by default ([1a1892c](https://github.com/ibis-project/ibis/commit/1a1892c1f567672a16140943d9406c6a0796b4a5))
* **duckdb:** drop use of experimental parallel csv reader ([47d8b92](https://github.com/ibis-project/ibis/commit/47d8b92cc021b8f5fd1b284bab8c1ac8e1e71a07))
* **duckdb:** generate `SIMILAR TO` instead of tilde to workaround sqlglot issue ([434da27](https://github.com/ibis-project/ibis/commit/434da2755a2ed41884b035bcec3810143c6ccd05))
* improve typing signature of .dropna() ([e11de3f](https://github.com/ibis-project/ibis/commit/e11de3fe0aab878fe88ea5a2cf65ebbf6374cc99))
* **mssql:** improve aggregation on expressions ([58aa78d](https://github.com/ibis-project/ibis/commit/58aa78d7ac78c3b4dc7159e184740c6daa8cdb54))
* **mssql:** remove invalid aggregations ([1ce3ef9](https://github.com/ibis-project/ibis/commit/1ce3ef963ec3caf22a2a985873f93666fc17dcb0))
* **polars:** backwards compatibility for the `time_zone` and `time_unit` properties ([3a2c4df](https://github.com/ibis-project/ibis/commit/3a2c4df9c73b2ce99be2b7730b529df53d617be1))
* **postgres:** allow inference of unknown types ([343fb37](https://github.com/ibis-project/ibis/commit/343fb375d11e612807061818c612d387903ca469))
* **pyspark:** fail when aggregation contains a `having` filter ([bd81a9f](https://github.com/ibis-project/ibis/commit/bd81a9ff5f8fafb674a238eb6a0abb31849eb39e))
* **pyspark:** raise proper error when trying to generate sql ([51afc13](https://github.com/ibis-project/ibis/commit/51afc134ecb99fcd81d496e9c2660147320a6c2f))
* **snowflake:** fix new array operations; remove `ArrayRemove` operation ([772668b](https://github.com/ibis-project/ibis/commit/772668bec9d535a21f1c5435e80f47753429182f))
* **snowflake:** make sure ephemeral tables following backend quoting rules ([9a845df](https://github.com/ibis-project/ibis/commit/9a845df3e0c473c129c94c45b3683aaf68863410))
* **snowflake:** make sure pyarrow is used when possible ([01f5154](https://github.com/ibis-project/ibis/commit/01f5154e76c1f24a1e2bf5b71133eecc43ab7c59))
* **sql:** ensure that set operations resolve to a single relation ([3a02965](https://github.com/ibis-project/ibis/commit/3a029653e01ea853ba16200210d7e462618d151b))
* **sql:** generate consistent `pivot_longer` semantics in the presence of multiple `unnest`s ([6bc301a](https://github.com/ibis-project/ibis/commit/6bc301ab07c33b2afb3d66750367e086ddc96400))
* **sqlglot:** work with newer versions ([6f7302d](https://github.com/ibis-project/ibis/commit/6f7302d52cfdb37c4d0b73fbfa0fa9defbf034cb))
* **trino,duckdb,postgres:** make cumulative `notany`/`notall` aggregations work ([c2e985f](https://github.com/ibis-project/ibis/commit/c2e985f493fcb3306cc74793ac457661ee942967))
* **trino:** only support `how='first'` with `arbitrary` reduction ([315b5e7](https://github.com/ibis-project/ibis/commit/315b5e73b264c0f47a4c2be229386b790763de58))
* **ux:** use guaranteed length-1 characters for `NULL` values ([8618789](https://github.com/ibis-project/ibis/commit/86187899803a34921f53063ec114caf693cfdff4))


### Refactors

* **api:** remove explicit use of `.projection` in favor of the shorter `.select` ([73df8df](https://github.com/ibis-project/ibis/commit/73df8df279127021fa42f996225d1f38696e5c14))
* **cache:** factor out ref counted cache ([c816f00](https://github.com/ibis-project/ibis/commit/c816f006779cbeb04bd835e9c14b206ec9bb1f06))
* **duckdb:** simplify `to_pyarrow_batches` implementation ([d6235ee](https://github.com/ibis-project/ibis/commit/d6235ee0d12a0c3136ef09e19832d82169b7bb40))
* **duckdb:** source loaded and installed extensions from duckdb ([fb06262](https://github.com/ibis-project/ibis/commit/fb0626281b03aa0c4239b9b0bd813f2fe0bf1ce2))
* **duckdb:** use native duckdb parquet reader unless auth required ([e9f57eb](https://github.com/ibis-project/ibis/commit/e9f57eb19934c3e84a4a72246a6f2874ad88dcd8))
* generate uuid-based names for temp tables ([a1164df](https://github.com/ibis-project/ibis/commit/a1164df5d1bc4fa454371626a0527d30ca8dd296))
* **memtable:** clean up dispatch code ([9a19302](https://github.com/ibis-project/ibis/commit/9a1930226f22b5707f03e64309e31dd8c59e37ff))
* **memtable:** dedup table proxy code ([3bccec0](https://github.com/ibis-project/ibis/commit/3bccec06ac736674ed72865b1275b6cae6b190fd))
* **sqlalchemy:** remove unused `_meta` instance attributes ([523e198](https://github.com/ibis-project/ibis/commit/523e1981732ca4c36c2c4c6be0a7e871c566dcb7))


### Deprecations

* **api:** deprecate `Table.set_column` in favor of `Table.mutate` ([954a6b7](https://github.com/ibis-project/ibis/commit/954a6b7a88e2727d598d4d2e5c3ac4d7cccd5f7e))


### Documentation

* add a getting started guide ([8fd03ce](https://github.com/ibis-project/ibis/commit/8fd03cefc49eab144e94344711e805bf6b5be40e))
* add warning about comparisons to `None` ([5cf186a](https://github.com/ibis-project/ibis/commit/5cf186acad6578d84ad5b8d63ff9180d3c4a8df6))
* **blog:** add campaign finance blog post ([383c708](https://github.com/ibis-project/ibis/commit/383c708ed0aa4dab61482157c371d32a28ac733d))
* **blog:** add campaign finance to `SUMMARY.md` ([0bdd093](https://github.com/ibis-project/ibis/commit/0bdd093f9ca3728d4c40a8f7ce8abe5301c2f0d2))
* clean up agg argument descriptions and add join examples ([93d3059](https://github.com/ibis-project/ibis/commit/93d3059f33280f625326866d291d7d206fe2077e))
* **comparison:** add a "why ibis" page ([011cc19](https://github.com/ibis-project/ibis/commit/011cc1939eef94e45d688e1ae3dc45a8c3649863))
* move conda before nix in dev setup instructions ([6b2cbaa](https://github.com/ibis-project/ibis/commit/6b2cbaaa33f0c9cc7b0174e082d5155f045cc2fe))
* **nth:** improve docstring for nth() ([fb7b34b](https://github.com/ibis-project/ibis/commit/fb7b34b424cb333cb956d0c20f145e5d2f52f5bc))
* patch docs build to fix anchor links ([51be459](https://github.com/ibis-project/ibis/commit/51be4592473e99b20dcdb56d67c4fece40a0a1b0))
* **penguins:** add citation for palmer penguins data ([679848d](https://github.com/ibis-project/ibis/commit/679848d0c28d1c18156a929344330902e585fb10))
* **penguins:** change to flipper ([eec3706](https://github.com/ibis-project/ibis/commit/eec370606bf25ec2fbb53451036d5000f5b24afe))
* refresh environment setup pages ([b609571](https://github.com/ibis-project/ibis/commit/b6095714b8cfed2c11092fefd330470075b77f66))
* **selectors:** make doctests more complete and actually run them ([c8f2964](https://github.com/ibis-project/ibis/commit/c8f2964e224b07d5d676fd57106043883da4d207))
* style and review fixes in getting started guide ([3b0f8db](https://github.com/ibis-project/ibis/commit/3b0f8dbe93b6112101e084737e8733fa61c6558a))

## [5.0.0](https://github.com/ibis-project/ibis/compare/4.1.0...5.0.0) (2023-03-15)


### ⚠ BREAKING CHANGES

* **api:** Snowflake identifiers are now kept **as is** from the database. Many table names and column names may now be in SHOUTING CASE. Adjust code accordingly.
* **backend:** Backends now raise `ibis.common.exceptions.UnsupportedOperationError` in more places during compilation. You may need to catch this error type instead of the previous type, which differed between backends.
* **ux:** `Table.info` now returns an expression
* **ux:** Passing a sequence of column names to `Table.drop` is removed. Replace `drop(cols)` with `drop(*cols)`.
* The `spark` plugin alias is removed. Use `pyspark` instead
* **ir:** removed `ibis.expr.scope` and `ibis.expr.timecontext` modules, access them under `ibis.backends.base.df.<module>`
* some methods have been removed from the top-level `ibis.<backend>` namespaces, access them on a connected backend instance instead.
* **common:** removed `ibis.common.geospatial`, import the functions from `ibis.backends.base.sql.registry.geospatial`
* **datatypes:** `JSON` is no longer a subtype of `String`
* **datatype:** `Category`, `CategoryValue`/`Column`/`Scalar` are removed. Use string types instead.
* **ux:** The `metric_name` argument to `value_counts` is removed. Use `Table.relabel` to change the metric column's name.
* **deps:** the minimum version of `parsy` is now 2.0
* **ir/backends:** removed the following symbols:
- `ibis.backends.duckdb.parse_type()` function
- `ibis.backends.impala.Backend.set_database()` method
- `ibis.backends.pyspark.Backend.set_database()` method
- `ibis.backends.impala.ImpalaConnection.ping()` method
- `ibis.expr.operations.DatabaseTable.change_name()` method
- `ibis.expr.operations.ParseURL` class
- `ibis.expr.operations.Value.to_projection()` method
- `ibis.expr.types.Table.get_column()` method
- `ibis.expr.types.Table.get_columns()` method
- `ibis.expr.types.StringValue.parse_url()` method
* **schema:** `Schema.from_dict()`, `.delete()` and `.append()` methods are removed
* **datatype:** `struct_type.pairs` is removed, use `struct_type.fields` instead
* **datatype:** `Struct(names, types)` is not supported anymore, pass a dictionary to `Struct` constructor instead

### Features

* add `max_columns` option for table repr ([a3aa236](https://github.com/ibis-project/ibis/commit/a3aa23684346b88aa05d76d0a712f98310ed4eaa))
* add examples API ([b62356e](https://github.com/ibis-project/ibis/commit/b62356e7bcb4d4202b8ebf48643c1d54222ad771))
* **api:** add `map`/`array` accessors for easy conversion of JSON to stronger-typed values ([d1e9d11](https://github.com/ibis-project/ibis/commit/d1e9d113ab063154a36ef5a90c4bcc8dd6f5425b))
* **api:** add array to string join operation ([74de349](https://github.com/ibis-project/ibis/commit/74de349c4285e85bc5d3257ef481d5d7255ea477))
* **api:** add builtin support for relabeling columns to snake case ([1157273](https://github.com/ibis-project/ibis/commit/1157273fa0b047f51a987d6332601a9910536d18))
* **api:** add support for passing a mapping to `ibis.map` ([d365fd4](https://github.com/ibis-project/ibis/commit/d365fd472f0e3ac83b8167e3934a09b4b88dfaa4))
* **api:** allow single argument set operations ([bb0a6f0](https://github.com/ibis-project/ibis/commit/bb0a6f0fdc249dbc1d47dc9830e7fff6bf10fea1))
* **api:** implement `to_pandas()` API for ecosystem compatibility ([cad316c](https://github.com/ibis-project/ibis/commit/cad316c3f33697128d6cf4ca24067d0dc466c310))
* **api:** implement isin ([ac31db2](https://github.com/ibis-project/ibis/commit/ac31db2c9784f3d028dad36dcaa62557a1008712))
* **api:** make `cache` evaluate only once per session per expression ([5a8ffe9](https://github.com/ibis-project/ibis/commit/5a8ffe9164de9ae97d14716f1bfa107fef74a181))
* **api:** make create_table uniform ([833c698](https://github.com/ibis-project/ibis/commit/833c6989a4ff8f3a7baeab9a326d399f2edf2e96))
* **api:** more selectors ([5844304](https://github.com/ibis-project/ibis/commit/58443042aed5fc1e33a232f2025f9dbd4015464f))
* **api:** upcast pandas DataFrames to memtables in `rlz.table` rule ([8dcfb8d](https://github.com/ibis-project/ibis/commit/8dcfb8d7cb7b84ffe1f8cd42a09be27b77d7d777))
* **backends:** implement `ops.Time` for sqlalchemy backends ([713cd33](https://github.com/ibis-project/ibis/commit/713cd33a8dad1be7846384698f490ff0a14c3d0b))
* **bigquery:** add `BIGNUMERIC` type support ([5c98ea4](https://github.com/ibis-project/ibis/commit/5c98ea462cde8033591f6a6841d7d34cf03a167a))
* **bigquery:** add UUID literal support ([ac47c62](https://github.com/ibis-project/ibis/commit/ac47c62e9c2016fabbade6366d3e82a73bc90f14))
* **bigquery:** enable subqueries in select statements ([ef4dc86](https://github.com/ibis-project/ibis/commit/ef4dc8611c776ca597ffee420f42628cbe1d24a7))
* **bigquery:** implement create and drop table method ([5f3c22c](https://github.com/ibis-project/ibis/commit/5f3c22c3cb6745159d759f7114d20fc5d8d52782))
* **bigquery:** implement create_view and drop_view method ([a586473](https://github.com/ibis-project/ibis/commit/a586473126f1bb7638852ab4331f07d7518f69c6))
* **bigquery:** support creating tables from in-memory tables ([c3a25f1](https://github.com/ibis-project/ibis/commit/c3a25f1c483b9eb157fdc6382a9392e5115ca1b1))
* **bigquery:** support in-memory tables ([37e3279](https://github.com/ibis-project/ibis/commit/37e327961b59a1a6185d5483a97879bf1deb4934))
* change Rich repr of dtypes from blue to dim ([008311f](https://github.com/ibis-project/ibis/commit/008311f5a866adea5a402483f89634f374245a0e))
* **clickhouse:** implement `ArrayFilter` translation ([f2144b6](https://github.com/ibis-project/ibis/commit/f2144b643503897e68cd0a29cb2d7e2ea67177b6))
* **clickhouse:** implement `ops.ArrayMap` ([45000e7](https://github.com/ibis-project/ibis/commit/45000e70ff5d9969ba5a6a58a8083c79df051638))
* **clickhouse:** implement `ops.MapLength` ([fc82eaa](https://github.com/ibis-project/ibis/commit/fc82eaa7bcab167b9386b456b5a6515d97ebf86b))
* **clickhouse:** implement ops.Capitalize ([914c64c](https://github.com/ibis-project/ibis/commit/914c64c2feea8c9d5e50eca76ae7619e0e0d3d39))
* **clickhouse:** implement ops.ExtractMillisecond ([ee74e3a](https://github.com/ibis-project/ibis/commit/ee74e3a67363ee81ae6e755ce55b8b7045cf1f90))
* **clickhouse:** implement ops.RandomScalar ([104aeed](https://github.com/ibis-project/ibis/commit/104aeedb54754e9564e348f3514dc1a37af940c3))
* **clickhouse:** implement ops.StringAscii ([a507d17](https://github.com/ibis-project/ibis/commit/a507d17e45dafbe8b50eee71e8b87de0cfcd3931))
* **clickhouse:** implement ops.TimestampFromYMDHMS, ops.DateFromYMD ([05f5ae5](https://github.com/ibis-project/ibis/commit/05f5ae541d73dc27b02e6050b1f196eda102859f))
* **clickhouse:** improve error message for invalid types in literal ([e4d7799](https://github.com/ibis-project/ibis/commit/e4d7799774cda873fc00c5d09005deb30bc20992))
* **clickhouse:** support asof_join ([7ed5143](https://github.com/ibis-project/ibis/commit/7ed5143d7f08e336a2f6e11336ae85d6bfbb47e8))
* **common:** add abstract mapping collection with support for set operations ([7d4aa0f](https://github.com/ibis-project/ibis/commit/7d4aa0fa65f0c078b2686f05ebd23e54b4b75985))
* **common:** add support for variadic positional and variadic keyword annotations ([baea1fa](https://github.com/ibis-project/ibis/commit/baea1fade57c194baae2a514be58195e9f606016))
* **common:** hold typehint in the annotation objects ([b3601c6](https://github.com/ibis-project/ibis/commit/b3601c642a22ee296c1bcd92874c7a2837b9e64c))
* **common:** support `Callable` arguments and return types in `Validator.from_annotable()` ([ae57c36](https://github.com/ibis-project/ibis/commit/ae57c367da307880a2d067ffb1081fecd42a83bb))
* **common:** support positional only and keyword only arguments in annotations ([340dca1](https://github.com/ibis-project/ibis/commit/340dca1dd046d1ae1c8e42b92016296137021e59))
* **dask/pandas:** raise OperationNotDefinedError exc for not defined operations ([2833685](https://github.com/ibis-project/ibis/commit/2833685dff46239d09986d689498b59804e04d59))
* **datafusion:** implement ops.Degrees, ops.Radians ([7e61391](https://github.com/ibis-project/ibis/commit/7e61391819f17ce87f1f63123fd8862a18eb3098))
* **datafusion:** implement ops.Exp ([7cb3ade](https://github.com/ibis-project/ibis/commit/7cb3adef86a781ae422c8774a2da7a96313b22ea))
* **datafusion:** implement ops.Pi, ops.E ([5a74cb4](https://github.com/ibis-project/ibis/commit/5a74cb4e409593ded3572df6383fd273c17d44f3))
* **datafusion:** implement ops.RandomScalar ([5d1cd0f](https://github.com/ibis-project/ibis/commit/5d1cd0f0658f9f2efcb46949febb2d59028bb18d))
* **datafusion:** implement ops.StartsWith ([8099014](https://github.com/ibis-project/ibis/commit/8099014c087396e00dc40afd46a8ecbc9196b2d2))
* **datafusion:** implement ops.StringAscii ([b1d7672](https://github.com/ibis-project/ibis/commit/b1d76728b1c44ad3092c8f3edaa8c83e145f3216))
* **datafusion:** implement ops.StrRight ([016a082](https://github.com/ibis-project/ibis/commit/016a082b3dfd41b9860c92f722378ca2725e3e0b))
* **datafusion:** implement ops.Translate ([2fe3fc4](https://github.com/ibis-project/ibis/commit/2fe3fc401e4e9823271d298a3025f782e2031867))
* **datafusion:** support substr without end ([a19fd87](https://github.com/ibis-project/ibis/commit/a19fd871ea060f2f18477ec58d14cca2b07fffeb))
* **datatype/schema:** support datatype and schema declaration using type annotated classes ([6722c31](https://github.com/ibis-project/ibis/commit/6722c31f98608bef3be4dfcb7e3c06568d464ddb))
* **datatype:** enable inference of `Decimal` type ([8761732](https://github.com/ibis-project/ibis/commit/87617320375d749c8df0f1741052861af431e047))
* **datatype:** implement `Mapping` abstract base class for `StructType` ([5df2022](https://github.com/ibis-project/ibis/commit/5df2022634ec6421b36db546af9c8acff7b12dc9))
* **deps:** add Python 3.11 support and tests ([6f3f759](https://github.com/ibis-project/ibis/commit/6f3f759e21a69c5b4050a371d9da788dd5551676))
* **druid:** add Apache Druid backend ([c4cc2a6](https://github.com/ibis-project/ibis/commit/c4cc2a652b26b7c1263141d545fde56c48bf4c08))
* **druid:** implement bitwise operations ([3ac7447](https://github.com/ibis-project/ibis/commit/3ac74478cb5c4c157b2e475445978d5c630f4ba7))
* **druid:** implement ops.Pi, ops.Modulus, ops.Power, ops.Log10 ([090ff03](https://github.com/ibis-project/ibis/commit/090ff03c7731a79a919750a71d97b5b127142bfd))
* **druid:** implement ops.Sign ([35f52cc](https://github.com/ibis-project/ibis/commit/35f52cc6ecbcbbb8db7e0e9661312ac676b117b2))
* **druid:** implement ops.StringJoin ([42cd9a3](https://github.com/ibis-project/ibis/commit/42cd9a31c7fd923e91a5e423bd9aa489b24acdd9))
* **duckdb:** add support for reading tables from sqlite databases ([9ba2211](https://github.com/ibis-project/ibis/commit/9ba2211fce61f5981436126cefa41e3e7fecf0f8))
* **duckdb:** add UUID type support ([5cd6d76](https://github.com/ibis-project/ibis/commit/5cd6d76ad1637c36729e0dea0c3ba16ca832bdc8))
* **duckdb:** implement `ArrayFilter` translation ([5f35d5c](https://github.com/ibis-project/ibis/commit/5f35d5c5529aa80c75ae2236bd985caf77fcaa77))
* **duckdb:** implement `ops.ArrayMap` ([063602d](https://github.com/ibis-project/ibis/commit/063602de1df1f1652a577fab04c7cb91996b554d))
* **duckdb:** implement create_view and drop_view method ([4f73953](https://github.com/ibis-project/ibis/commit/4f73953024ff3bdc7914bb535380323735b96bfc))
* **duckdb:** implement ops.Capitalize ([b17116e](https://github.com/ibis-project/ibis/commit/b17116e538409a54264158dc45f120e259682320))
* **duckdb:** implement ops.TimestampDiff, ops.IntervalAdd, ops.IntervalSubtract ([a7fd8fb](https://github.com/ibis-project/ibis/commit/a7fd8fb54edf53ba1e4b95c7e299a2ad9e6cf80b))
* **duckdb:** implement uuid result type ([3150333](https://github.com/ibis-project/ibis/commit/31503337fda8621c78439cf8e99f8e29c5034dbc))
* **duckdb:** support dt.MACADDR, dt.INET as string ([c4739c7](https://github.com/ibis-project/ibis/commit/c4739c74c3ab1355db36cbb5ed83d7c9433d30ae))
* **duckdb:** use `read_json_auto` when reading json ([4193867](https://github.com/ibis-project/ibis/commit/41938674a7ed8c5b1415a652a6d09627c277fc19))
* **examples:** add imdb dataset examples ([3d63203](https://github.com/ibis-project/ibis/commit/3d632035f597254355951b5c552d43e1eda3bdea))
* **examples:** add movielens small dataset ([5f7c15c](https://github.com/ibis-project/ibis/commit/5f7c15cfb9afe3b451dc9e8720d92e71be97fa12))
* **examples:** add wowah_data data to examples ([bf9a7cc](https://github.com/ibis-project/ibis/commit/bf9a7cc390991e29cc0552197b4b5f9a4973ee1a))
* **examples:** enable progressbar and faster hashing ([4adfe29](https://github.com/ibis-project/ibis/commit/4adfe292307d042175b4221a33ebbfc3d53029e8))
* **impala:** implement ops.Clip ([279fd78](https://github.com/ibis-project/ibis/commit/279fd78e9eed00492f9c07037b148edada2e918c))
* **impala:** implement ops.Radians, ops.Degrees ([a794ace](https://github.com/ibis-project/ibis/commit/a794ace76daa781afec83db816d45a317d27e16b))
* **impala:** implement ops.RandomScalar ([874f2ff](https://github.com/ibis-project/ibis/commit/874f2ffaed9ab200268b88c27b3d798cd430eaf1))
* **io:** add to_parquet, to_csv to backends ([fecca42](https://github.com/ibis-project/ibis/commit/fecca421ad405465c4c001afa4e2976c07164530))
* **ir:** add `ArrayFilter` operation ([e719d60](https://github.com/ibis-project/ibis/commit/e719d601cbd9cc86d181f28a3658146985c052dc))
* **ir:** add `ArrayMap` operation ([49e5f7a](https://github.com/ibis-project/ibis/commit/49e5f7ab4dd9e32b3edbac9c3932fdd3ebc49274))
* **mysql:** support in-memory tables ([4dfabbd](https://github.com/ibis-project/ibis/commit/4dfabbd156490c0ff2cbf5fdb384903e3be7331d))
* **pandas/dask:** implement bitwise operations ([4994add](https://github.com/ibis-project/ibis/commit/4994add3fe719660ef991daa64c97cc9bf46506a))
* **pandas/dask:** implement ops.Pi, ops.E ([091be3c](https://github.com/ibis-project/ibis/commit/091be3c3b6f94590922136676959d56d3efe7b75))
* **pandas:** add basic unnest support ([dd36b9d](https://github.com/ibis-project/ibis/commit/dd36b9d4fde6348ad0670ddceebcbbe9ba066f64))
* **pandas:** implement ops.StartsWith, ops.EndsWith ([2725423](https://github.com/ibis-project/ibis/commit/2725423bb14b196318f93050a92630156c9d7e47))
* **pandas:** support more pandas extension dtypes ([54818ef](https://github.com/ibis-project/ibis/commit/54818ef8e3a523318512073138659b4824519738))
* **polars:** implement `ops.Union` ([17c6011](https://github.com/ibis-project/ibis/commit/17c60116496907bdae586339e87d744662b9c75f))
* **polars:** implement ops.Pi, ops.E ([6d8fc4a](https://github.com/ibis-project/ibis/commit/6d8fc4a9fc4561431c6f92da1d48db0ba2f68f70))
* **postgres:** allow connecting with an explicit `schema` ([39c9ea8](https://github.com/ibis-project/ibis/commit/39c9ea88a426ba2acedb630aaceb1a4a1a132140))
* **postgres:** fix interval literal ([c0fa933](https://github.com/ibis-project/ibis/commit/c0fa9339c92d36f7cba98052ff258df13e994af2))
* **postgres:** implement `argmin`/`argmax` ([82668ec](https://github.com/ibis-project/ibis/commit/82668ec6be4b460ea737cabe6edee53a7b42f3fd))
* **postgres:** parse tsvector columns as strings ([fac8c47](https://github.com/ibis-project/ibis/commit/fac8c478b4166c6f47bd9b9047592acd1ec936d2)), closes [#5402](https://github.com/ibis-project/ibis/issues/5402)
* **pyspark:** add support for `ops.ArgMin` and `ops.ArgMax` ([a3fa57c](https://github.com/ibis-project/ibis/commit/a3fa57cb7e956ceb8cd9aa93b78eeceab18df1de))
* **pyspark:** implement ops.Between ([ed83465](https://github.com/ibis-project/ibis/commit/ed83465a9f0680547e08c7f637d573dd73f09020))
* return Table from create_table(), create_view() ([e4ea597](https://github.com/ibis-project/ibis/commit/e4ea59752ecc1f7ee5a54c11a15250a1f51a9671))
* **schema:** implement `Mapping` abstract base class for `Schema` ([167d85a](https://github.com/ibis-project/ibis/commit/167d85a088156eab688a1903e0f94c4ed869d68f))
* **selectors:** support ranges ([e10caf4](https://github.com/ibis-project/ibis/commit/e10caf44111f6efb1d5d84adf5dd8718661887c8))
* **snowflake:** add support for alias in snowflake ([b1b947a](https://github.com/ibis-project/ibis/commit/b1b947aa80c3af06762b1a505cf9cdf31a4e8b96))
* **snowflake:** add support for bulk upload for temp tables in snowflake ([6cc174f](https://github.com/ibis-project/ibis/commit/6cc174f206ef0682389041bf92b2a81cb8fe29d1))
* **snowflake:** add UUID literal support ([436c781](https://github.com/ibis-project/ibis/commit/436c781e383fb419f08f0401817511e42e0e01a8))
* **snowflake:** implement argmin/argmax ([8b998a5](https://github.com/ibis-project/ibis/commit/8b998a5ece33f5d1beec2d41f55bfdaeddd73688))
* **snowflake:** implement ops.BitwiseAnd, ops.BitwiseNot, ops.BitwiseOr, ops.BitwiseXor ([1acd4b7](https://github.com/ibis-project/ibis/commit/1acd4b745752df2e0e0a8d97941986577fafdea2))
* **snowflake:** implement ops.GroupConcat ([2219866](https://github.com/ibis-project/ibis/commit/22198669b41ca348b13dec52112cffd0906391f0))
* **snowflake:** implement remaining map functions ([c48c9a6](https://github.com/ibis-project/ibis/commit/c48c9a62e3d5bebf2006bb6d6d5635db6bf9f842))
* **snowflake:** support binary variance reduction with filters ([eeabdee](https://github.com/ibis-project/ibis/commit/eeabdee8dfc6c0d188480e12f3871a5e6cd937e5))
* **snowflake:** support cross-database table access ([79cb445](https://github.com/ibis-project/ibis/commit/79cb44580b05ef37e7946c86e4133bb826c6e7be))
* **sqlalchemy:** generalize unnest to work on backends that don't support it ([5943ce7](https://github.com/ibis-project/ibis/commit/5943ce73852ca7edc624bf5aea15e505cdea7e08))
* **sqlite:** add sqlite type support ([addd6a9](https://github.com/ibis-project/ibis/commit/addd6a9931de371a0b775c294348419ee147fee7))
* **sqlite:** support in-memory tables ([1b24848](https://github.com/ibis-project/ibis/commit/1b24848d95b25ae9d765ab38abe579a04bb9a5f7))
* **sql:** support for creating temporary tables in sql based backends ([466cf35](https://github.com/ibis-project/ibis/commit/466cf35a5975c41c365308e4135163cd19c206b8))
* **tables:** cast table using schema ([96ce109](https://github.com/ibis-project/ibis/commit/96ce109bf85162782cc2f3a1fe88f645bb1b8483))
* **tables:** implement `pivot_longer` API ([11c5736](https://github.com/ibis-project/ibis/commit/11c5736de56ab486756d1a75bd18035d89faa7d9))
* **trino:** enable `MapLength` operation ([a7ad1db](https://github.com/ibis-project/ibis/commit/a7ad1db2d1e311016277b1196d1b77abf053227a))
* **trino:** implement `ArrayFilter` translation ([50f6fcc](https://github.com/ibis-project/ibis/commit/50f6fccd791bf7cee1d86865719b0ed7f4bd397f))
* **trino:** implement `ops.ArrayMap` ([657bf61](https://github.com/ibis-project/ibis/commit/657bf61ef38194bb109b3774f09b9e1d8a170c05))
* **trino:** implement `ops.Between` ([d70b9c0](https://github.com/ibis-project/ibis/commit/d70b9c0b4305d9e42d55a913c1c3cc985ab514b1))
* **trino:** support sqlalchemy 2 ([0d078c1](https://github.com/ibis-project/ibis/commit/0d078c154ce2d788b986d4e2dd31b88fbf5bec16))
* **ux:** accept selectors in `Table.drop` ([325140f](https://github.com/ibis-project/ibis/commit/325140ff0293a83ef55272b0299c2865cb8e4aaa))
* **ux:** allow creating unbound tables using annotated class definitions ([d7bf6a2](https://github.com/ibis-project/ibis/commit/d7bf6a27f7c74c91ae3a9682b3d5c88b312dee13))
* **ux:** easy interactive setup ([6850146](https://github.com/ibis-project/ibis/commit/6850146479f3e90a27a1200c85c35cf35efb2d2e))
* **ux:** expose `between`, `rows` and `range` keyword arguments in `value.over()` ([5763063](https://github.com/ibis-project/ibis/commit/57630637f7829d9f8d38e0eff5a521757232819d))


### Bug Fixes

* **analysis:** extract `Limit` subqueries ([62f6e14](https://github.com/ibis-project/ibis/commit/62f6e14e12a1f48809e9284744f4adb0e55c23af))
* **api:** add a `name` attribute to backend proxy modules ([d6d8e7e](https://github.com/ibis-project/ibis/commit/d6d8e7ebc5b11ac16819cc0fd06390062f7120e9))
* **api:** fix broken `__radd__` array concat operation ([121d9a0](https://github.com/ibis-project/ibis/commit/121d9a0a65227d3a71f5fa19d888facaead5eb67))
* **api:** only include valid python identifiers in struct tab completion ([8f33775](https://github.com/ibis-project/ibis/commit/8f33775b72249125d928470389dd91baad633473))
* **api:** only include valid python identifiers in table tab completion ([031a48c](https://github.com/ibis-project/ibis/commit/031a48c27e085a926cd0226e5388c84c1e4f1f73))
* **backend:** provide useful error if default backend is unavailable ([1dbc682](https://github.com/ibis-project/ibis/commit/1dbc6828327d2d18ca2aa882796ebc33588bd48a))
* **backends:** fix capitalize implementations across all backends ([d4f0275](https://github.com/ibis-project/ibis/commit/d4f0275faee2b1aeee432734f599d88643113e0c))
* **backends:** fix null literal handling ([7f46342](https://github.com/ibis-project/ibis/commit/7f46342a2253ca391c3c4246adc9290e2eab025c))
* **bigquery:** ensure that memtables are translated correctly ([d6e56c5](https://github.com/ibis-project/ibis/commit/d6e56c50c072fabf970fd5fe8bdacede582f4864))
* **bigquery:** fix decimal literals ([4a04c9b](https://github.com/ibis-project/ibis/commit/4a04c9b7fde2750618442f6eb27b8fdce58f029a))
* **bigquery:** regenerate negative string index sql snapshots ([3f02c73](https://github.com/ibis-project/ibis/commit/3f02c73a1a06012df09da546f41b821fb1f1ed43))
* **bigquery:** regenerate sql for predicate pushdown fix ([509806f](https://github.com/ibis-project/ibis/commit/509806f11e629d29c91e540dd3288771f4e098a4))
* **cache:** remove bogus schema argument and validate database argument type ([c4254f6](https://github.com/ibis-project/ibis/commit/c4254f64a1c74f3d8751f684c22e4db88e9d0366))
* **ci:** fix invalid test id ([f70de1d](https://github.com/ibis-project/ibis/commit/f70de1d5b743a43a70385cd00b51c82dea8c1e87))
* **clickhouse:** fix decimal literal ([4dcd2cb](https://github.com/ibis-project/ibis/commit/4dcd2cb0e55170b57ed541bfe7b4f0566ba89f70))
* **clickhouse:** fix set ops with table operands ([86bcf32](https://github.com/ibis-project/ibis/commit/86bcf3259971fd2b538d6a5a9abac8dd836b6ee9))
* **clickhouse:** raise OperationNotDefinedError if operation is not supported ([71e2570](https://github.com/ibis-project/ibis/commit/71e2570d506184dd91dbea35e7bab078b5a48b4e))
* **clickhouse:** register in-memory tables in pyarrow-related calls ([09a045c](https://github.com/ibis-project/ibis/commit/09a045c4216e86eee6d57e40e45edbe42b894890))
* **clickhouse:** use a bool type supported by `clickhouse_driver` ([ab8f064](https://github.com/ibis-project/ibis/commit/ab8f064cae7c8863b4b766f8da6a9575c0a78248))
* **clickhouse:** workaround sqlglot's insistence on uppercasing ([6151f37](https://github.com/ibis-project/ibis/commit/6151f37ecc5f4b5799f1bc3f2fa99ac0e74ae0f3))
* **compiler:** generate aliases in a less clever way ([04a4aa5](https://github.com/ibis-project/ibis/commit/04a4aa54cfe37a6ff7d0b1dded0fc34395a8813d))
* **datafusion:** support sum aggregation on bool column ([9421400](https://github.com/ibis-project/ibis/commit/9421400b1c125e57a8b12f8b3207fa3154e8c75c))
* **deps:** bump duckdb to 0.7.0 ([38d2276](https://github.com/ibis-project/ibis/commit/38d227668441c3154039001abd8a15f9351432a4))
* **deps:** bump snowflake-connector-python upper bound ([b368b04](https://github.com/ibis-project/ibis/commit/b368b046a946f5d2e0bcd0d48ce501da98738d30))
* **deps:** ensure that pyspark depends on sqlalchemy ([60c7382](https://github.com/ibis-project/ibis/commit/60c73824610dbbf677e6ac2a4c80a7cf16864d9b))
* **deps:** update dependency pyarrow to v11 ([2af5d8d](https://github.com/ibis-project/ibis/commit/2af5d8d8c17f680dac586c4f2038cf678f9076ea))
* **deps:** update dependency sqlglot to v11 ([e581e2f](https://github.com/ibis-project/ibis/commit/e581e2ff21503c9d09359c2d463f759465e6f527))
* don't expose backend methods on `ibis.<backend>` directly ([5a16431](https://github.com/ibis-project/ibis/commit/5a164312eb3d6267a00e09f1c8e35fb8b10bf9d8))
* **druid:** remove invalid operations ([19f214c](https://github.com/ibis-project/ibis/commit/19f214c110d7f717f1ef8fbac8c7b4a9a9f22002))
* **duckdb:** add `null` to duckdb datatype parser ([07d2a86](https://github.com/ibis-project/ibis/commit/07d2a863b86f415eea7aba2c3e889876be8ce3dc))
* **duckdb:** ensure that `temp_directory` exists ([00ba6cb](https://github.com/ibis-project/ibis/commit/00ba6cb6ed4956943b94059a9cd72e547a5ee474))
* **duckdb:** explicitly set timezone to UTC on connection ([6ae4a06](https://github.com/ibis-project/ibis/commit/6ae4a060d506c237bf7918e99e3ecd48b4d42657))
* **duckdb:** fix blob type in literal ([f66e8a1](https://github.com/ibis-project/ibis/commit/f66e8a1b1088ecc38e8085939e86fc22eb109c21))
* **duckdb:** fix memtable `to_pyarrow`/`to_pyarrow_batches` ([0e8b066](https://github.com/ibis-project/ibis/commit/0e8b06643bfe81fd8694c5578324f263e49b9890))
* **duckdb:** in-memory objects registered with duckdb show up in list_tables ([7772f79](https://github.com/ibis-project/ibis/commit/7772f792db7c193d86b51792187a781f88f9b558))
* **duckdb:** quote identifiers if necessary in `struct_pack` ([6e598cc](https://github.com/ibis-project/ibis/commit/6e598cc92e4aa591673eae88ce047b655add95e3))
* **duckdb:** support casting to unsigned integer types ([066c158](https://github.com/ibis-project/ibis/commit/066c1582cab394896625b93bf8b05d07c2a321db))
* **duckdb:** treat `g` `re_replace` flag as literal text ([aa3c31c](https://github.com/ibis-project/ibis/commit/aa3c31c5e1d46e0c68c9e79651e703ba0d50e542))
* **duckdb:** workaround an ownership bug at the interaction of duckdb, pandas and pyarrow ([2819cff](https://github.com/ibis-project/ibis/commit/2819cffdfbbe9da8fffb9c87d4b689cffae8be88))
* **duckdb:** workaround duckdb bug that prevents multiple substitutions ([0e09220](https://github.com/ibis-project/ibis/commit/0e092200adc50cfe9feb16abdbbb1d1e9d453509))
* **imports:** remove top-level import of sqlalchemy from base backend ([b13cf25](https://github.com/ibis-project/ibis/commit/b13cf2576146621f111bf87873dec19cc07edb65))
* **io:** add `read_parquet` and `read_csv` to base backend mixin ([ce80d36](https://github.com/ibis-project/ibis/commit/ce80d360233942bad1582d404551fe29e7029e4d)), closes [#5420](https://github.com/ibis-project/ibis/issues/5420)
* **ir:** incorrect predicate pushdown ([9a9204f](https://github.com/ibis-project/ibis/commit/9a9204ff495bf9cb106bbb7484340a0794b4ab31))
* **ir:** make `find_subqueries` return in topological order ([3587910](https://github.com/ibis-project/ibis/commit/3587910a2cba5a61b05a3a713401b56b8e52d118))
* **ir:** properly raise error if literal cannot be coerced to a datatype ([e16b91f](https://github.com/ibis-project/ibis/commit/e16b91ffac3863688270f82156e1312b67ba2144))
* **ir:** reorder the right schema of set operations to align with the left schema ([58e60ae](https://github.com/ibis-project/ibis/commit/58e60ae0bfd8818efa69e388a1ba32d7ee84603a))
* **ir:** use `rlz.map_to()` rule instead of `isin` to normalize temporal units ([a1c46a2](https://github.com/ibis-project/ibis/commit/a1c46a2a70d3bd920700cc1c4f8e52db075e78a7))
* **ir:** use static connection pooling to prevent dropping temporary state ([6d2ae26](https://github.com/ibis-project/ibis/commit/6d2ae264cbf023a2122f77cbdbb7fd4257f9fbbf))
* **mssql:** set sqlglot to tsql ([1044573](https://github.com/ibis-project/ibis/commit/10445733615abf48fe55085dd217e3ef420d015d))
* **mysql:** remove invalid operations ([8f34a2b](https://github.com/ibis-project/ibis/commit/8f34a2b1ee96a45762699d78d01b7c4b16f104a2))
* **pandas/dask:** handle non numpy scalar results in `wrap_case_result` ([a3b82f7](https://github.com/ibis-project/ibis/commit/a3b82f77379f75d550ede0d15a8a4a4fa5dc28d7))
* **pandas:** don't try to dispatch on arrow dtype if not available ([d22ae7b](https://github.com/ibis-project/ibis/commit/d22ae7b900a7d651b32d4371958490a38ec1bfd9))
* **pandas:** handle casting to arrays with None elements ([382b90f](https://github.com/ibis-project/ibis/commit/382b90fb391b8c8bbf2ae487b8b4d163aefed6bc))
* **pandas:** handle NAs in array conversion ([06bd15d](https://github.com/ibis-project/ibis/commit/06bd15d9919c432cc23b5ded91386ce3bae727d8))
* **polars:** back compat for `concat_str` separator argument ([ced5a61](https://github.com/ibis-project/ibis/commit/ced5a61b0700c963522c15a24ebb018e33d44aba))
* **polars:** back compat for the `reverse`/`descending` argument ([f067d81](https://github.com/ibis-project/ibis/commit/f067d8111c5f2eb7e0ec049cc1cfce24e30d18d4))
* **polars:** polars execute respect limit kwargs ([d962faf](https://github.com/ibis-project/ibis/commit/d962faf93727a72e0542b2439adf4de5a2c24b0f))
* **polars:** properly infer polars categorical dtype ([5a4707a](https://github.com/ibis-project/ibis/commit/5a4707a49402ae949243e980edf07ab29543adbc))
* **polars:** use metric name in aggregate output to dedupe columns ([234d8c1](https://github.com/ibis-project/ibis/commit/234d8c189eb72e080b4f54ff941948b3ab8c9925))
* **pyspark:** fix incorrect `ops.EndsWith` translation rule ([4c0a5a2](https://github.com/ibis-project/ibis/commit/4c0a5a2b4e29b56bfb7a6aea36dbd1b6edbe7bd1))
* **pyspark:** fix isnan and isinf to work on bool ([8dc623a](https://github.com/ibis-project/ibis/commit/8dc623af59a95b82929d756004ba6f21ec55db80))
* **snowflake:** allow loose casting of objects and arrays ([1cf8df0](https://github.com/ibis-project/ibis/commit/1cf8df04bed2037ad7c197cbd2979ce0a0cffe7f))
* **snowflake:** ensure that memtables are translated correctly ([b361e07](https://github.com/ibis-project/ibis/commit/b361e0720edb0b721d5237263b585968b534b5ab))
* **snowflake:** ensure that null comparisons are correct ([9b83699](https://github.com/ibis-project/ibis/commit/9b836993b24fe817629ebc3365b6c051abd3fc97))
* **snowflake:** ensure that quoting matches snowflake behavior, not sqlalchemy ([b6b67f9](https://github.com/ibis-project/ibis/commit/b6b67f963315294a561fcd0060024adc63d2ac30))
* **snowflake:** ensure that we do not try to use a None schema or database ([03e0265](https://github.com/ibis-project/ibis/commit/03e026562656c3f8dcce6ec741fdc47e30892a59))
* **snowflake:** handle the case where pyarrow isn't installed ([b624fa3](https://github.com/ibis-project/ibis/commit/b624fa30d73f3445c41f0ad405ecfdc81e98eaa1))
* **snowflake:** make `array_agg` preserve nulls ([24b95bf](https://github.com/ibis-project/ibis/commit/24b95bf131afd51a6eaf98a6a66c0f91f8451d3c))
* **snowflake:** quote column names on construction of `sa.Column` ([af4db5c](https://github.com/ibis-project/ibis/commit/af4db5c61913dfc883e287d241c84bb1f47c6be0))
* **snowflake:** remove broken pyarrow fetch support ([c440adb](https://github.com/ibis-project/ibis/commit/c440adb4fc747056c67c49854c90159fda1552ab))
* **snowflake:** return `NULL` when trying to call map functions on non-object JSON ([d85fb28](https://github.com/ibis-project/ibis/commit/d85fb28676d0b1a56e07abe192fb4754a93043a6))
* **snowflake:** use `_flatten` to avoid overriding unrelated function in other backends ([8c31594](https://github.com/ibis-project/ibis/commit/8c31594d433e296409848b7fafa76c6d635bd8a1))
* **sqlalchemy:** ensure that isin contains full column expression ([9018eb6](https://github.com/ibis-project/ibis/commit/9018eb60d851dabe8974cf300411cfea5b74571a))
* **sqlalchemy:** get builtin dialects working; mysql/mssql/postgres/sqlite ([d2356bc](https://github.com/ibis-project/ibis/commit/d2356bc09376f21cfa358c381e8f3a4e91e362e5))
* **sqlalchemy:** make `strip` family of functions behave like Python ([dd0a04c](https://github.com/ibis-project/ibis/commit/dd0a04c31d38523f67ff78e09aa418578d239daa))
* **sqlalchemy:** reflect most recent schema when view is replaced ([62c8dea](https://github.com/ibis-project/ibis/commit/62c8deabce1f64e6a828edabd925b9f0356b0424))
* **sqlalchemy:** use `sa.true` instead of Python literal ([8423eba](https://github.com/ibis-project/ibis/commit/8423eba4a46d13cbe00968a12fe3815ce28ecf8c))
* **sqlalchemy:** use indexed group by key references everywhere possible ([9f1ddd8](https://github.com/ibis-project/ibis/commit/9f1ddd8328a9c10155ff934b0fa157d58b63e4fe))
* **sql:** ensure that set operations generate valid sql in the presence of additional constructs such as sort keys ([3e2c364](https://github.com/ibis-project/ibis/commit/3e2c3648aba41b74eec54a31d14cc72e5bef3e31))
* **sqlite:** explicitly disallow array in literal ([de73b37](https://github.com/ibis-project/ibis/commit/de73b37aeeab868fde3612cb993d16725cf3d19a))
* **sqlite:** fix random scalar range ([26d0dde](https://github.com/ibis-project/ibis/commit/26d0ddeaeca096919de966cbf2b7907a54331051))
* support negative string indices ([f84a54d](https://github.com/ibis-project/ibis/commit/f84a54da3629da0b602ecf081db6a0ba07aa6192))
* **trino:** workaround broken dialect ([b502faf](https://github.com/ibis-project/ibis/commit/b502faf19bd0cf3b3b9b00e813f8f6a20d19a4ee))
* **types:** fix argument types of Table.order_by() ([6ed3a97](https://github.com/ibis-project/ibis/commit/6ed3a9716c6a621bf0f55364d80856ae9268dbd2))
* **util:** make convert_unit work with python types ([cb3a90c](https://github.com/ibis-project/ibis/commit/cb3a90c7d7678d96fbc36a21226e683a6a71f790))
* **ux:** give the `value_counts` aggregate column a better name ([abab1d7](https://github.com/ibis-project/ibis/commit/abab1d7d53af9ee300ce3b7ae56d9386cd469b5e))
* **ux:** make string range selectors inclusive ([7071669](https://github.com/ibis-project/ibis/commit/7071669993143a2de570c9dd863f5837eb05cc4d))
* **ux:** make top level set operations work ([f5976b2](https://github.com/ibis-project/ibis/commit/f5976b2b0f53b8c9c2a0d671ece5ca7945d90d70))


### Performance

* **duckdb:** faster `to_parquet`/`to_csv` implementations ([6071bb5](https://github.com/ibis-project/ibis/commit/6071bb5f8238f3ebb159fc197efcd088cdc452a0))
* fix duckdb insert-from-dataframe performance ([cd27b99](https://github.com/ibis-project/ibis/commit/cd27b99206dccbfd704cb720369fbd3bdef1547b))


* **deps:** bump minimum required version of parsy ([22020cb](https://github.com/ibis-project/ibis/commit/22020cb537cc08028b1f17d31197b9c512ccbf65))
* remove spark alias to pyspark and associated cruft ([4b286bd](https://github.com/ibis-project/ibis/commit/4b286bdf52d0e6f899c51dcfff994ea35c6cd96b))


### Refactors

* **analysis:** slightly simplify `find_subqueries()` ([ab3712f](https://github.com/ibis-project/ibis/commit/ab3712fc9747cb1001c088c5244037dc4c9b5d02))
* **backend:** normalize exceptions ([065b66d](https://github.com/ibis-project/ibis/commit/065b66d2f390939d7a517271890e679f01732608))
* **clickhouse:** clean up parsing rules ([6731772](https://github.com/ibis-project/ibis/commit/673177267ee23dc344da62235fb3252eed8b4650))
* **common:** move `frozendict` and `DotDict` to `ibis.common.collections` ([4451375](https://github.com/ibis-project/ibis/commit/4451375b063faec77929853442dc4a79c1787f0f))
* **common:** move the `geospatial` module to the base SQL backend ([3e7bfa3](https://github.com/ibis-project/ibis/commit/3e7bfa358ca929d0bd3dee2a2ec4470af303bfe8))
* **dask:** remove unneeded create_table() ([86885a6](https://github.com/ibis-project/ibis/commit/86885a6f1b3808036dbc310635e76eaaeca5068f))
* **datatype:** clean up parsing rules ([c15fb5f](https://github.com/ibis-project/ibis/commit/c15fb5fc27c13cb679199bd1ed2aeca6ed8c697f))
* **datatype:** remove `Category` type and related APIs ([bb0ee78](https://github.com/ibis-project/ibis/commit/bb0ee786271dc21cb18a8561f523b4cee2549ce0))
* **datatype:** remove `StructType.pairs` property in favor of identical `fields` attribute ([6668122](https://github.com/ibis-project/ibis/commit/66681223a10ad421cfaa5b000dd2dffbf114ba42))
* **datatypes:** move sqlalchemy datatypes to specific backend ([d7b49eb](https://github.com/ibis-project/ibis/commit/d7b49eb2fcce396a40707c9ef4838fba44a4a477))
* **datatypes:** remove `String` parent type from `JSON` type ([34f3898](https://github.com/ibis-project/ibis/commit/34f3898253c35ad81876cd222888cd3c3c8e1b0a))
* **datatype:** use a dictionary to store `StructType` fields rather than `names` and `types` tuples ([84455ac](https://github.com/ibis-project/ibis/commit/84455ac1c0c0ca6941f871a5d49b39864975febc))
* **datatype:** use lazy dispatch when inferring pandas Timedelta objects ([e5280ea](https://github.com/ibis-project/ibis/commit/e5280ea14b100afb8ba04d7dcb449a6fc4ccf0ab))
* drop `limit` kwarg from `to_parquet`/`to_csv` ([a54460c](https://github.com/ibis-project/ibis/commit/a54460c1a5f229e26b84f11fae70c91906ee895f))
* **duckdb:** clean up parsing rules ([30da8f9](https://github.com/ibis-project/ibis/commit/30da8f965fdde83e59e408d52c219dd07407e66e))
* **duckdb:** handle parsing timestamp scale ([16c1443](https://github.com/ibis-project/ibis/commit/16c1443c8d5797ae70e6f1d3f500a766777790c9))
* **duckdb:** remove unused `list<...>` parsing rule ([f040b86](https://github.com/ibis-project/ibis/commit/f040b86e175e3c8dcddc78b0b410feec026e0c77))
* **duckdb:** use a proper sqlalchemy construct for structs and reduce casting ([8daa4a1](https://github.com/ibis-project/ibis/commit/8daa4a1a6b979dad8ae8e0fea56c68bb78b1b550))
* **ir/api:** introduce window frame operation and revamp the window API ([2bc5e5e](https://github.com/ibis-project/ibis/commit/2bc5e5e40684019f7b16d11ad21010a7fa7540a8))
* **ir/backends:** remove various deprecated functions and methods ([a8d3007](https://github.com/ibis-project/ibis/commit/a8d3007d1092a0e7e821b8e3097aa95677c24d5e))
* **ir:** reorganize the `scope` and `timecontext` utilities ([80bd494](https://github.com/ibis-project/ibis/commit/80bd494ee713455551a7d6c2eaf528bef3ee5ec9))
* **ir:** update `ArrayMap` to use the new `callable_with` validation rule ([560474e](https://github.com/ibis-project/ibis/commit/560474eb6d683be7281563f38bc200d644266eb7))
* move pretty repr tests back to their own file ([4a75988](https://github.com/ibis-project/ibis/commit/4a759888f43db382f5a73d6e35479ae3074383bb))
* **nix:** clean up marker argument construction ([12eb916](https://github.com/ibis-project/ibis/commit/12eb916e644e5ee1601639704e2874d4c6298b89))
* **postgres:** clean up datatype parsing ([1f61661](https://github.com/ibis-project/ibis/commit/1f6166114a2d1d31d73deabf9fa45d79a4884cde))
* **postgres:** clean up literal arrays ([21b122d](https://github.com/ibis-project/ibis/commit/21b122dab25c5f93b9922bb66f59733a6c52d37e))
* **pyspark:** remove another private function ([c5081cf](https://github.com/ibis-project/ibis/commit/c5081cf24aa923449e5f6d478ab4e34f0a782eac))
* remove unnecessary top-level rich console ([8083a6b](https://github.com/ibis-project/ibis/commit/8083a6b401b2e1c7d5c67351215f0c97a9cba765))
* **rules:** remove unused `non_negative_integer` and `pair` rules ([e00920a](https://github.com/ibis-project/ibis/commit/e00920abeec9dd920ca9ee112e61b960af5174e5))
* **schema:** remove deprecated `Schema.from_dict()`, `.delete()` and `.append()` methods ([8912b24](https://github.com/ibis-project/ibis/commit/8912b2485922804c98966d33bf11d7984019acb0))
* **snowflake:** remove the need for `parsy` ([c53403a](https://github.com/ibis-project/ibis/commit/c53403a79b9634eb87b8798124b9ada237ff945d))
* **sqlalchemy:** set session parameters once per connection ([ed4b476](https://github.com/ibis-project/ibis/commit/ed4b47610cd2a493065caa3431724c29d8d29cf6))
* **sqlalchemy:** use backend-specific startswith/endswith implementations ([6101de2](https://github.com/ibis-project/ibis/commit/6101de23edec04077cc149099056a6e01e6e4a1d))
* **test_sqlalchemy.py:** move to snapshot testing ([96998f0](https://github.com/ibis-project/ibis/commit/96998f0562788606bdd215fc276ed62cca8ddfa9))
* **tests:** reorganize `rules` test file to the `ibis.expr` subpackage ([47f0909](https://github.com/ibis-project/ibis/commit/47f090953e6f369952612b759bef320cd70f3892))
* **tests:** reorganize `schema` test file to the `ibis.expr` subpackage ([40033e1](https://github.com/ibis-project/ibis/commit/40033e10a176d1b1542d93bd283ed04a37e8bba4))
* **tests:** reorganize datatype test files to the datatypes subpackage ([16199c6](https://github.com/ibis-project/ibis/commit/16199c6269c7cb4c49afb42323e44211185b6c50))
* **trino:** clean up datatype parsing ([84c0e35](https://github.com/ibis-project/ibis/commit/84c0e356404f52998242aee29bd694c3429a2847))
* **ux:** return expression from `Table.info` ([71cc0e0](https://github.com/ibis-project/ibis/commit/71cc0e092695435fe4be79e2b12002388af86a97))


### Deprecations

* **api:** deprecate `summary` API ([e449c07](https://github.com/ibis-project/ibis/commit/e449c076f16156a9868a98adcce0d159137c9583))
* **api:** mark `ibis.sequence()` for removal ([3589f80](https://github.com/ibis-project/ibis/commit/3589f80af4cdb433876c5ac83d37f5c7aac5cf6e))


### Documentation

* add a bunch of string expression examples ([18d3112](https://github.com/ibis-project/ibis/commit/18d3112b984bc9cd7b295e6c40ce8ee2232bcf57))
* add Apache Druid to backend matrix ([764d9c3](https://github.com/ibis-project/ibis/commit/764d9c3ffbf786163e09cbd6cbc4f7b46e9e1217))
* add CNAME file to mkdocs source ([6d19111](https://github.com/ibis-project/ibis/commit/6d191110c804fa06baaa622c86ab20384b6d1496))
* add druid to the backends index docs page ([ad0b6a3](https://github.com/ibis-project/ibis/commit/ad0b6a3c123dfca21065e54aa0e6127201d585ef))
* add missing DataFusion entry to the backends in the README ([8ce025a](https://github.com/ibis-project/ibis/commit/8ce025a4c524e2e87c6d4ef0dde6f0efada80c0d))
* add redirects for common old pages ([c9087f2](https://github.com/ibis-project/ibis/commit/c9087f2172fb0fe21575b3c1b600eb035437c753))
* **api:** document deferred API and its pitfalls ([8493604](https://github.com/ibis-project/ibis/commit/8493604013b0874124f029ad3c3b83e764d047b8))
* **api:** improve `collect` method API documentation ([b4fcef1](https://github.com/ibis-project/ibis/commit/b4fcef12aa3993cfaa3c9a385070a7e4faa9d39a))
* array expression examples ([6812c17](https://github.com/ibis-project/ibis/commit/6812c1713687a9e321f75c87be870d57a018c817))
* **backends:** document default backend configuration ([6d917d3](https://github.com/ibis-project/ibis/commit/6d917d3728a1d7242e77f814a36014dbd5fd7fb8))
* **backends:** link to configuration from the backends list ([144044d](https://github.com/ibis-project/ibis/commit/144044d8067cf5aa3e7beeb3649061a3195b0a14))
* **blob:** blog on ibis + substrait + duckdb ([5dc7a0a](https://github.com/ibis-project/ibis/commit/5dc7a0a05c8066e400dad6b501a519bd910fad7f))
* **blog:** adds examples sneak peek blog + assets folder ([fcbb3d5](https://github.com/ibis-project/ibis/commit/fcbb3d5c75b0baaad362fb9deee89b134d8df42b))
* **blog:** adds to file sneak peek blog ([128194f](https://github.com/ibis-project/ibis/commit/128194f85bd11f7735df2db396467c0cf6ab37f6))
* **blog:** specify parsy 2.0 in substrait blog article ([c264477](https://github.com/ibis-project/ibis/commit/c264477ea80f9008fca27a97c8694be07d98abc6))
* bump query engine count in README and use project-preferred names ([11169f7](https://github.com/ibis-project/ibis/commit/11169f731cfba9689fbe914aad94b1c8e5f83768))
* don't sort backends by coverage percentage by default ([68f73b1](https://github.com/ibis-project/ibis/commit/68f73b1505cc8d60c74a6053e5a577e6f899f10e))
* drop docs versioning ([d7140e7](https://github.com/ibis-project/ibis/commit/d7140e7283c0ca60a69722c031d8ee5bd5a55e29))
* **duckdb:** fix broken docstring examples ([51084ad](https://github.com/ibis-project/ibis/commit/51084adf66071dc04257aab067f4163620a6b1a5))
* enable light/dark mode toggle in docs ([b9e812a](https://github.com/ibis-project/ibis/commit/b9e812a2c21ceff6994549c1b9b8a0b56dc2c246))
* fill out table API with working examples ([16fc8be](https://github.com/ibis-project/ibis/commit/16fc8befaf639bbce1660fce7809c28122696bf2))
* fix notebook logging example ([04b75ef](https://github.com/ibis-project/ibis/commit/04b75ef8046c4d7ee25e0f0ac9360b893dbdaa1b))
* **how-to:** fix sessionize.md to use ibis.read_parquet ([ff9cbf7](https://github.com/ibis-project/ibis/commit/ff9cbf78a4b5a75e60b6b4da6286641108449900))
* improve Expr.substitute() docstring ([b954edd](https://github.com/ibis-project/ibis/commit/b954edd0749cc0e321d0fd89ab71f1c03071762c))
* improve/update pandas walkthrough ([80b05d8](https://github.com/ibis-project/ibis/commit/80b05d8f46450ce3d1f8e5df0b1fac86d91a5f8e))
* **io:** doc/ux improvements for read_parquet and friends ([2541556](https://github.com/ibis-project/ibis/commit/2541556d34452b664d1b9eff6028fc04d7055920)), closes [#5420](https://github.com/ibis-project/ibis/issues/5420)
* **io:** update README.md to recommend installing duckdb as default backend ([0a72ec0](https://github.com/ibis-project/ibis/commit/0a72ec031b1c1a0126218456517265576503e20b)), closes [#5423](https://github.com/ibis-project/ibis/issues/5423) [#5420](https://github.com/ibis-project/ibis/issues/5420)
* move tutorial from docs to external ibis-examples repo ([11b0237](https://github.com/ibis-project/ibis/commit/11b023721c4bc24aba16ad9587c6da07bdac978f))
* **parquet:** add docstring examples for to_parquet incl. partitioning ([8040164](https://github.com/ibis-project/ibis/commit/8040164e303c1a8fff5b827f25adabadafe24263))
* point to `ibis-examples` repo in the README ([1205636](https://github.com/ibis-project/ibis/commit/1205636836d8251565d19ce038af67d911a5ea0e))
* **README.md:** clean up readme, fix typos, alter the example ([383a3d3](https://github.com/ibis-project/ibis/commit/383a3d34669c0e11ded75ab5a6e5955d7f34f16a))
* remove duplicate "or" ([b6ef3cc](https://github.com/ibis-project/ibis/commit/b6ef3ccdc1c4d27e9c7ea010e8e545090b89e5b6))
* remove duplicate spark backend in install docs ([5954618](https://github.com/ibis-project/ibis/commit/5954618e9d1cbea29ee883f02d2155bf09f786c9))
* render `__dunder__` method API documentation ([b532c63](https://github.com/ibis-project/ibis/commit/b532c63bd6ce1d8f7acc3e7b0f764513505a8e75))
* rerender ci-analysis notebook with new table header colors ([50507b6](https://github.com/ibis-project/ibis/commit/50507b63a9b3a40e5c2902f740b200dfabe18e75))
* **streamlit:** fix url for support matrix ([594199b](https://github.com/ibis-project/ibis/commit/594199b67b81d71848bf38c160f3b8e9fbc69094))
* **tutorial:** remove impala from sql tutorial ([7627c13](https://github.com/ibis-project/ibis/commit/7627c137b3af400139a2e019e6fb3184bc2fe876))
* use teal for primary & accent colors ([24be961](https://github.com/ibis-project/ibis/commit/24be96158f67850f528597195d20c00c7ff98cfb))

## [4.1.0](https://github.com/ibis-project/ibis/compare/4.0.0...4.1.0) (2023-01-25)


### Features

* add `ibis.get_backend` function ([2d27df8](https://github.com/ibis-project/ibis/commit/2d27df806049ae9d3cca572e79daff466f09531f))
* add py.typed to allow mypy to type check packages that use ibis ([765d42e](https://github.com/ibis-project/ibis/commit/765d42e6afdcaa7dfc12981092366789d330029b))
* **api:** add `ibis.set_backend` function ([e7fabaf](https://github.com/ibis-project/ibis/commit/e7fabaf9cb4497d1e1f586a96be8ff7b1f63e2fb))
* **api:** add selectors for easier selection of columns ([306bc88](https://github.com/ibis-project/ibis/commit/306bc8818bdf835bee65c8bf7aeec59100eee4d4))
* **bigquery:** add JS UDF support ([e74328b](https://github.com/ibis-project/ibis/commit/e74328b25eb2ee3333d7b54cf0eb0a1a89617caa))
* **bigquery:** add SQL UDF support ([db24173](https://github.com/ibis-project/ibis/commit/db241738052b137f714716dcf7d619cd4c1b2f78))
* **bigquery:** add to_pyarrow method ([30157c5](https://github.com/ibis-project/ibis/commit/30157c52bad41ae09e526c49961dd209eae3224b))
* **bigquery:** implement bitwise operations ([55b69b1](https://github.com/ibis-project/ibis/commit/55b69b1a2e55f2b6655a60dd068870947a5b47e3))
* **bigquery:** implement ops.Typeof ([b219919](https://github.com/ibis-project/ibis/commit/b219919cd7622a27337060efd05255b6dfa00d4d))
* **bigquery:** implement ops.ZeroIfNull ([f4c5607](https://github.com/ibis-project/ibis/commit/f4c560739288210477fd61f7adecb2020f7e24ed))
* **bigquery:** implement struct literal ([c5f2a1d](https://github.com/ibis-project/ibis/commit/c5f2a1d464a56535ecd96749d19bebd28f781706))
* **clickhouse:** properly support native boolean types ([31cc7ba](https://github.com/ibis-project/ibis/commit/31cc7ba3a3e542cc626804c4c3eb6d67fa357ba8))
* **common:** add support for annotating with coercible types ([ae4a415](https://github.com/ibis-project/ibis/commit/ae4a415a2fb7549621b67173519034641659add7))
* **common:** make frozendict truly immutable ([1c25213](https://github.com/ibis-project/ibis/commit/1c2521316176322a90e57090fcbd62f337b3eaa0))
* **common:** support annotations with typing.Literal ([6f89f0b](https://github.com/ibis-project/ibis/commit/6f89f0bfca20c120ebd4ceeb57d7263e9413e63e))
* **common:** support generic mapping and sequence type annotations ([ddc6603](https://github.com/ibis-project/ibis/commit/ddc66033b4da39a2e71bc454fdf2e3fb1fefef1c))
* **dask:** support `connect()` with no arguments ([67eed42](https://github.com/ibis-project/ibis/commit/67eed42ac74bbec3cc9a3eca50b1fa6e30dc1abe))
* **datatype:** add optional timestamp scale parameter ([a38115a](https://github.com/ibis-project/ibis/commit/a38115a65657a6d7bd661d237c34860db3089853))
* **datatypes:** add `as_struct` method to convert schemas to structs ([64be7b1](https://github.com/ibis-project/ibis/commit/64be7b1db7a0c3c2c8631b7fbe025ba4e13247a4))
* **duckdb:** add `read_json` function for consuming newline-delimited JSON files ([65e65c1](https://github.com/ibis-project/ibis/commit/65e65c1a87f544e73a0f3e09f5b7fb1227170624))
* **mssql:** add a bunch of missing types ([c698d35](https://github.com/ibis-project/ibis/commit/c698d355423465670061d7d398003f4da3764689))
* **mssql:** implement inference for `DATETIME2` and `DATETIMEOFFSET` ([aa9f151](https://github.com/ibis-project/ibis/commit/aa9f15111797971415592351b39f1a84db0f7578))
* nicer repr for Backend.tables ([0d319ca](https://github.com/ibis-project/ibis/commit/0d319ca7bec5d6e81c6727318630c242a8cb0a4f))
* **pandas:** support `connect()` with no arguments ([78cbbdd](https://github.com/ibis-project/ibis/commit/78cbbdd015d83acd535d4f1b819a2dec81e3d7e2))
* **polars:** allow ibis.polars.connect() to function without any arguments ([d653a07](https://github.com/ibis-project/ibis/commit/d653a071460e120e87b101d7f1748d74f1e3f93f))
* **polars:** handle casting to scaled timestamps ([099d1ec](https://github.com/ibis-project/ibis/commit/099d1ec0502a0ec1ef201b8b66933e46c8d8b988))
* **postgres:** add `Map(string, string)` support via the built-in `HSTORE` extension ([f968f8f](https://github.com/ibis-project/ibis/commit/f968f8f900818424729b39e237f551005d3a5156))
* **pyarrow:** support conversion to pyarrow map and struct types ([54a4557](https://github.com/ibis-project/ibis/commit/54a4557c8652a4c1cff36ffff897d41c871e4b22))
* **snowflake:** add more array operations ([8d8bb70](https://github.com/ibis-project/ibis/commit/8d8bb707be7b75485024cc0bf1a77780af8673cd))
* **snowflake:** add more map operations ([7ae6e25](https://github.com/ibis-project/ibis/commit/7ae6e25dcfde2becf82f4992c6fdbeec9ce810b1))
* **snowflake:** any/all/notany/notall reductions ([ba1af5e](https://github.com/ibis-project/ibis/commit/ba1af5ea8f920e59217578878871d82296b075be))
* **snowflake:** bitwise reductions ([5aba997](https://github.com/ibis-project/ibis/commit/5aba997526a478ff2f2fa95912d6af1c94581ea7))
* **snowflake:** date from ymd ([035f856](https://github.com/ibis-project/ibis/commit/035f856f76235cc1059d16d37b8ec889fc2975bd))
* **snowflake:** fix array slicing ([bd7af2a](https://github.com/ibis-project/ibis/commit/bd7af2a1b32586855b564c8e3a4190a87405df24))
* **snowflake:** implement `ArrayCollect` ([c425f68](https://github.com/ibis-project/ibis/commit/c425f6861ed0ee19444aced4687a8b8b4a7901c4))
* **snowflake:** implement `NthValue` ([0dca57c](https://github.com/ibis-project/ibis/commit/0dca57cd34747623834f9ed551cda2e4f5c80fb1))
* **snowflake:** implement `ops.Arbitrary` ([45f4f05](https://github.com/ibis-project/ibis/commit/45f4f05455cc499fe0d311167a8b9d7b14f0943a))
* **snowflake:** implement `ops.StructColumn` ([41698ed](https://github.com/ibis-project/ibis/commit/41698ed09f18c2ec13c29e571586f17d3cf0edbf))
* **snowflake:** implement `StringSplit` ([e6acc09](https://github.com/ibis-project/ibis/commit/e6acc09fe7ea11a47fc29e8dc2e637c15425d2dd))
* **snowflake:** implement `StructField` and struct literals ([286a5c3](https://github.com/ibis-project/ibis/commit/286a5c30506836f98758811e4199323f50494c3c))
* **snowflake:** implement `TimestampFromUNIX` ([314637d](https://github.com/ibis-project/ibis/commit/314637decd913003767119defc8ce95909189fdc))
* **snowflake:** implement `TimestampFromYMDHMS` ([1eba8be](https://github.com/ibis-project/ibis/commit/1eba8bef192e93cc97ed8b49b8c60f2bc8e118f4))
* **snowflake:** implement `typeof` operation ([029499c](https://github.com/ibis-project/ibis/commit/029499ca44181256eea1fbb009734287007c7621))
* **snowflake:** implement exists/not exists ([7c8363b](https://github.com/ibis-project/ibis/commit/7c8363b244c51ededb8c8ef0d38b765a9706645a))
* **snowflake:** implement extract millisecond ([3292e91](https://github.com/ibis-project/ibis/commit/3292e91d43ee7b785ec28af9fcaa9d16e137d4f7))
* **snowflake:** make literal maps and params work ([dd759d3](https://github.com/ibis-project/ibis/commit/dd759d381ebf23bed148e40d9f68c62e41315052))
* **snowflake:** regex extract, search and replace ([9c82179](https://github.com/ibis-project/ibis/commit/9c8217908cafd7d65c5702f8da6c198b07816fa9))
* **snowflake:** string to timestamp ([095ded6](https://github.com/ibis-project/ibis/commit/095ded6bcbdb1b1d6c5279fb8934e569e86b57a1))
* **sqlite:** implement `_get_schema_using_query` in SQLite backend ([7ff84c8](https://github.com/ibis-project/ibis/commit/7ff84c864d88b6e1da9c996e1b53a50feb916f82))
* **trino:** compile timestamp types with scale ([67683d3](https://github.com/ibis-project/ibis/commit/67683d32a4aaa1302813ff46afe697695ae53e9f))
* **trino:** enable `ops.ExistsSubquery` and `ops.NotExistsSubquery` ([9b9b315](https://github.com/ibis-project/ibis/commit/9b9b31582aa68bb8482788b51ca5e60350d05da1))
* **trino:** map parameters ([53bd910](https://github.com/ibis-project/ibis/commit/53bd9102ca97924bc727f750627823018dead3f4))
* **ux:** improve error message when column is not found ([b527506](https://github.com/ibis-project/ibis/commit/b5275066766af15ecb3d3c62e63f5bdb23d01f0a))


### Bug Fixes

* **backend:** read the default backend setting in `_default_backend` ([11252af](https://github.com/ibis-project/ibis/commit/11252af1b1eb72dda562184128a9d8280ab9cc1a))
* **bigquery:** move connection logic to do_connect ([42f2106](https://github.com/ibis-project/ibis/commit/42f21067aba79cee3d9d0c0ea3484801045c32f6))
* **bigquery:** remove invalid operations from registry ([911a080](https://github.com/ibis-project/ibis/commit/911a0803c132971109aba17ff5cd6ef16619084e))
* **bigquery:** resolve deprecation warnings for `StructType` and `Schema` ([c9e7078](https://github.com/ibis-project/ibis/commit/c9e70782ed7c1e2ca95dff4a31e212de7e03161a))
* **clickhouse:** fix position call ([702de5d](https://github.com/ibis-project/ibis/commit/702de5ddfa3055baa4f94fde7db24063c1336376))
* correctly visualize array type ([26b0b3f](https://github.com/ibis-project/ibis/commit/26b0b3fb7b41572f994b5d433ccd8cd2be3a0a8b))
* **deps:** make sure pyarrow is not an implicit dependency ([10373f4](https://github.com/ibis-project/ibis/commit/10373f40beddfc3fb56a847b6487877791d00456))
* **duckdb:** make `read_csv` on URLs work ([9e61816](https://github.com/ibis-project/ibis/commit/9e618165a0229739ba9e399daf487f13553a8736))
* **duckdb:** only try to load extensions when necessary for csv ([c77bde7](https://github.com/ibis-project/ibis/commit/c77bde7353de003eeb5a15570b8f833a69555e74))
* **duckdb:** remove invalid operations from registry ([ba2ec59](https://github.com/ibis-project/ibis/commit/ba2ec59aa47a38220c3e480507e90bf1170fef61))
* fallback to default backend with `to_pyarrow`/`to_pyarrow_batches` ([a1a6902](https://github.com/ibis-project/ibis/commit/a1a690234e389ec68626cc498d5f6a09f2c8d47a))
* **impala:** remove broken alias elision ([32b120f](https://github.com/ibis-project/ibis/commit/32b120fef6bc9ecf5a3163ab71ef75b76ceee44d))
* **ir:** error for `order_by` on nonexistent column ([57b1dd8](https://github.com/ibis-project/ibis/commit/57b1dd859e4693cf199b362c16f9c679df33c273))
* **ir:** ops.Where output shape should consider all arguments ([6f87064](https://github.com/ibis-project/ibis/commit/6f87064615a050c60af964f13d76af5c5885a964))
* **mssql:** infer bit as boolean everywhere ([24f9d7c](https://github.com/ibis-project/ibis/commit/24f9d7ceb00440fe0c2699a34ae4b475e73c9acb))
* **mssql:** pull nullability from column information ([490f8b4](https://github.com/ibis-project/ibis/commit/490f8b48fe2e14816f82176eb230661bb99fdb8b))
* **mysql:** fix mysql query schema inference ([12f6438](https://github.com/ibis-project/ibis/commit/12f6438097a2de1e5d973b629e22ca1f6dae73b5))
* **polars:** remove non-working Binary and Decimal literal inference ([0482d15](https://github.com/ibis-project/ibis/commit/0482d1504a7faa310b5bf6b784c248cf84dfecb5))
* **postgres:** use permanent views to avoid connection pool defeat ([49a4991](https://github.com/ibis-project/ibis/commit/49a4991de3dc98826504df3e8b0bfb54195c4996))
* **pyspark:** fix substring constant translation ([40d2072](https://github.com/ibis-project/ibis/commit/40d2072b1743705f301df61cb7dafc7ad45f23b1))
* **set ops:** raise if no tables passed to set operations ([bf4bdde](https://github.com/ibis-project/ibis/commit/bf4bdde437dce043072c763178255131f1e59c53))
* **snowflake:** bring back bitwise operations ([260facd](https://github.com/ibis-project/ibis/commit/260facdca91a8aeabbb1ea39e94a51a2a261684b))
* **snowflake:** don't always insert a cast ([ee8817b](https://github.com/ibis-project/ibis/commit/ee8817b3662f48502fef5902d36c5744221affb1))
* **snowflake:** implement working `TimestampNow` ([42d95b0](https://github.com/ibis-project/ibis/commit/42d95b0f741ae88996e342e4ee8a7002237563e1))
* **snowflake:** make sqlalchemy 2.0 compatible ([8071255](https://github.com/ibis-project/ibis/commit/80712554646c932334e79b2fa50e63bde24a56b4))
* **snowflake:** re-enable `ops.TableArrayView` ([a1ad2b7](https://github.com/ibis-project/ibis/commit/a1ad2b77c6d6d90f3c1e176a7a173e0c8cf7f5c9))
* **snowflake:** remove invalid operations from registry ([2831559](https://github.com/ibis-project/ibis/commit/283155948c42e844c0de11d0ea185cc71c40bbf5))
* **sql:** add `typeof` test and bring back implementations ([7dc5356](https://github.com/ibis-project/ibis/commit/7dc53561cb903146156d600312b4053145658da9))
* **sqlalchemy:** 2.0 compatibility ([837a736](https://github.com/ibis-project/ibis/commit/837a736df9700365562e8684a52639c612fb8240))
* **sqlalchemy:** fix view creation with select stmts that have bind parameters ([d760e69](https://github.com/ibis-project/ibis/commit/d760e6980e19b2ebe5cc1fcf9cf95824563aa4c5))
* **sqlalchemy:** handle correlated exists sanely ([efa42bd](https://github.com/ibis-project/ibis/commit/efa42bd2baa05233db6915b9ba7c658b1c7ef445))
* **sqlalchemy:** handle generic geography/geometry by name instead of geotype ([23c35e1](https://github.com/ibis-project/ibis/commit/23c35e1e6ba3da42d8a1bed9bc516089394997cb))
* **sqlalchemy:** use `exec_driver_sql` in view teardown ([2599c9b](https://github.com/ibis-project/ibis/commit/2599c9b16bd81e704847c90202247f0a40344f04))
* **sqlalchemy:** use the backend's compiler instead of `AlchemyCompiler` ([9f4ff54](https://github.com/ibis-project/ibis/commit/9f4ff54e4e762831e00eea3d25e05bf65f9e7e99))
* **sql:** fix broken call to `ibis.map` ([045edc7](https://github.com/ibis-project/ibis/commit/045edc74ce2ef8fa31eaf8ea60b7006021a5d6b6))
* **sqlite:** interpolate `pathlib.Path` correctly in `attach` ([0415bd3](https://github.com/ibis-project/ibis/commit/0415bd3db3777a2c29793703a6fc3764e097255a))
* **trino:** ensure connecting works with trino 0.321 ([07cee38](https://github.com/ibis-project/ibis/commit/07cee3880d36fdc25cd6b3996889a3b36dfc648c))
* **trino:** remove invalid operations from registry ([665265c](https://github.com/ibis-project/ibis/commit/665265c2395dadd135414249a159f251b89518a7))
* **ux:** remove extra trailing newline in expression repr ([ee6d58a](https://github.com/ibis-project/ibis/commit/ee6d58a5a9ac10dff19c87afe6b482bed83d5b41))


### Documentation

* add BigQuery backend docs ([09d8995](https://github.com/ibis-project/ibis/commit/09d8995cc89a3fd097e41829c2f231b28fd86958))
* add streamlit app for showing the backend operation matrix ([3228f64](https://github.com/ibis-project/ibis/commit/3228f641fe227449943d4297fb84f59f1e4aa37c))
* allow deselecting geospatial ops in backend support matrix ([012da8c](https://github.com/ibis-project/ibis/commit/012da8c1b7da9ec916ce147f31c23cd709bbf6b4))
* **api:** document more public expression APIs ([337018f](https://github.com/ibis-project/ibis/commit/337018f108b6e602b0aa18d540b1e0b78ea396e0))
* **backend-info:** prevent app from trying install duckdb extensions ([3d94082](https://github.com/ibis-project/ibis/commit/3d94082082230e52c392bb201628c6ea18878b0f))
* clean up gen_matrix.py after adding streamlit app ([deb80f2](https://github.com/ibis-project/ibis/commit/deb80f2bdb6e00ad653744060a76cb892fbe49de))
* **duckdb:** add `to_pyarrow_batches` documentation ([ec1ffce](https://github.com/ibis-project/ibis/commit/ec1ffcee52b7d2079a2f02ca4568bb6e0ef5548e))
* embed streamlit operation matrix app to docs ([469a50d](https://github.com/ibis-project/ibis/commit/469a50d9a435cfa8baab3fd990da706bcbfd554c))
* make firefox render the proper iframe height ([ff1d4dc](https://github.com/ibis-project/ibis/commit/ff1d4dc434628cddfc0894e0858c027f124aba3a))
* publish raw data for operation matrix ([62e68da](https://github.com/ibis-project/ibis/commit/62e68dad4e93a04cd05e92dc6513d03e2d20dad4))
* re-order when to download test data ([8ce8c16](https://github.com/ibis-project/ibis/commit/8ce8c16bf39418a2f271a0fdafc5f4b6b1395a7a))
* **release:** update breaking changes in the release notes for 4.0.0 ([4e91401](https://github.com/ibis-project/ibis/commit/4e91401b290dfbbcfd342eb19e589bc22ae0ce18))
* remove trailing parenthesis ([4294397](https://github.com/ibis-project/ibis/commit/4294397e377fef7d4f85432d30dacf8564fe8387))
* update ibis-version-4.0.0-release.md ([f6701df](https://github.com/ibis-project/ibis/commit/f6701df7acb8ec2c0dc82a5e858d7356cae53f3c))
* update links to contributing guides ([da615e4](https://github.com/ibis-project/ibis/commit/da615e4122389cd769b85c165d9b3e4e54f2df60))


### Refactors

* **bigquery:** explicitly disallow INT64 in JS UDF ([fb33bf9](https://github.com/ibis-project/ibis/commit/fb33bf9b1d33f3f38010f983552330f084152b8b))
* **datatype:** add custom sqlalchemy nested types for backend differentiation ([dec70f5](https://github.com/ibis-project/ibis/commit/dec70f53ebe45103f1392be5a5c738f163700c69))
* **datatype:** introduce to_sqla_type dispatching on dialect ([a8bbc00](https://github.com/ibis-project/ibis/commit/a8bbc0011f4c4a7519adaf841ff8cd5bf8de7a80))
* **datatypes:** remove Geography and Geometry types in favor of GeoSpatial ([d44978c](https://github.com/ibis-project/ibis/commit/d44978ca53d4261871173f277289e2e479073fcf))
* **datatype:** use a mapping to store `StructType` fields rather than `names` and `types` tuples ([ff34c7b](https://github.com/ibis-project/ibis/commit/ff34c7be985615a307d0b3ff491930b2230c5014))
* **dtypes:** expose nbytes property for integer and floating point datatypes ([ccf80fd](https://github.com/ibis-project/ibis/commit/ccf80fdf6021d821bb45a3c453686733459f3fd2))
* **duckdb:** remove `.raw_sql` call ([abc939e](https://github.com/ibis-project/ibis/commit/abc939ed9c537266509fe8762a3b70f241d454ae))
* **duckdb:** use sqlalchemy-views to reduce string hacking ([c162750](https://github.com/ibis-project/ibis/commit/c1627506b6dd72730bed1972f88357cee5686204))
* **ir:** remove UnnamedMarker ([dd352b1](https://github.com/ibis-project/ibis/commit/dd352b1b2dded6b6f21ae3f69dca0289c376e7a2))
* **postgres:** use a bindparam for metadata queries ([b6b4669](https://github.com/ibis-project/ibis/commit/b6b466985bdaa4bcf656ec5f0917055b2da39eab))
* remove empty unused file ([9d63fd6](https://github.com/ibis-project/ibis/commit/9d63fd6abaee0746f02727f26eed21f54e6bc40d))
* **schema:** use a mapping to store `Schema` fields rather than `names` and `types` tuples ([318179a](https://github.com/ibis-project/ibis/commit/318179add952a3b2fc80a1b6bf82191dc84b5157))
* simplify `_find_backend` implementation ([60f1a1b](https://github.com/ibis-project/ibis/commit/60f1a1b54ada6c75d687f4d974f98dd5ed74f68f))
* **snowflake:** remove unnecessary `parse_json` call in `ops.StructField` impl ([9e80231](https://github.com/ibis-project/ibis/commit/9e80231539aa307e607e2b82b35df9e09ede8385))
* **snowflake:** remove unnecessary casting ([271554c](https://github.com/ibis-project/ibis/commit/271554c7fd50c9d21efc1741da623a65f8193922))
* **snowflake:** use `unary` instead of `fixed_arity(..., 1)` ([4a1c7c9](https://github.com/ibis-project/ibis/commit/4a1c7c9a989257e2b92ed5090dad6faafb5813b9))
* **sqlalchemy:** clean up quoting implementation ([506ce01](https://github.com/ibis-project/ibis/commit/506ce01df0ca460258b2fc0a28af6c5552f51ce4))
* **sqlalchemy:** generalize handling of failed type inference ([b0f4e4c](https://github.com/ibis-project/ibis/commit/b0f4e4ca77d8168f377b3a5c81a90749e462fc9b))
* **sqlalchemy:** move `_get_schema_using_query` to base class ([296cd7d](https://github.com/ibis-project/ibis/commit/296cd7d9518686d6c5d81747900d28b060480bc4))
* **sqlalchemy:** remove the need for deferred columns ([e4011aa](https://github.com/ibis-project/ibis/commit/e4011aaa23209fde72aca085ab97d8e21f5a9a36))
* **sqlalchemy:** remove use of deprecated `isnot` ([4ec53a4](https://github.com/ibis-project/ibis/commit/4ec53a49e706f7b6a051aa32d82271da0b24e0fb))
* **sqlalchemy:** use `exec_driver_sql` everywhere ([e8f96b6](https://github.com/ibis-project/ibis/commit/e8f96b6c039ec8ca147f0d24a81bae5ab18d00d7))
* **sql:** finally remove `_CorrelatedRefCheck` ([f49e429](https://github.com/ibis-project/ibis/commit/f49e429ece0314899be6a9749d2d75e6ba12071d))


### Deprecations

* **api:** deprecate `.to_projection` in favor of `.as_table` ([7706a86](https://github.com/ibis-project/ibis/commit/7706a86f6cc4b24d79be8f8ece0c960e36d0fd22))
* **api:** deprecate `get_column`/`s` in favor of `__getitem__`/`__getattr__` syntax ([e6372e2](https://github.com/ibis-project/ibis/commit/e6372e2f9b77c81d2c234bb36bff9caff988cefa))
* **ir:** schedule DatabaseTable.change_name for removal ([e4bae26](https://github.com/ibis-project/ibis/commit/e4bae26b6671a16ceeab9beff4e21d66a6e2fbb9))
* **schema:** schedule `Schema.delete()` and `Schema.append()` for removal ([45ac9a9](https://github.com/ibis-project/ibis/commit/45ac9a9a07987aa27318a471ebb481d5387d7de5))

## [4.0.0](https://github.com/ibis-project/ibis/compare/3.2.0...4.0.0) (2023-01-09)


### ⚠ BREAKING CHANGES

* functions, methods and classes marked as deprecated are removed now
* **ir:** replace `HLLCardinality` with `ApproxCountDistinct` and `CMSMedian` with `ApproxMedian` operations.
* **backends:** the datatype of returned execution results now more closely matches that of the ibis expression's type. Downstream code may need to be adjusted.
* **ir:** the `JSONB` type is replaced by the `JSON` type.
* **dev-deps:** expression types have been removed from `ibis.expr.api`. Use `import ibis.expr.types as ir` to access these types.
* **common:** removed `@immutable_property` decorator, use `@attribute.default` instead
* **timestamps:** the `timezone` argument to `to_timestamp` is gone. This was only supported in the BigQuery backend. Append `%Z` to the format string and the desired time zone to the input column if necessary.
* **deps:** ibis now supports at minimum duckdb 0.3.3. Please upgrade your duckdb install as needed.
* **api:** previously `ibis.connect` would return a `Table` object when calling `connect` on a parquet/csv file. This now returns a backend containing a single table created from that file. When possible users may use `ibis.read` instead to read files into ibis tables.
* **api:** `histogram()`'s `closed` argument no longer exists because it never had any effect. Remove it from your `histogram` method calls.
* **pandas/dask:** the pandas and Dask backends now interpret casting ints to/from timestamps as seconds since the unix epoch, matching other backends.
* **datafusion:** `register_csv` and `register_parquet` are removed. Pass filename to `register` method instead.
* **ir:** `ops.NodeList` and `ir.List` are removed. Use tuples to represent sequence of expressions instead.
* **api:** `re_extract` now follows `re.match` behavior. In particular, the `0`th group is now the entire string if there's a match, otherwise the groups are 1-based.
* **datatypes:** enums are now strings. Likely no action needed since no functionality existed.
* **ir:** Replace `t[t.x.topk(...)]` with `t.semi_join(t.x.topk(...), "x")`.
* **ir:** `ir.Analytic.type()` and `ir.TopK.type()` methods are removed.
* **api:** the default limit for table/column expressions is now `None` (meaning no limit).
* **ir**: join changes: previously all column names that collided between `left` and `right` tables were renamed with an appended suffix. Now for the case of inner joins with only equality predicates, colliding columns that are known to be equal due to the join predicates aren't renamed.
* **impala:** kerberos support is no longer installed by default for the `impala` backend. To add support you'll need to install the `kerberos` package separately.
* **ir:** `ops.DeferredSortKey` is removed. Use `ops.SortKey` directly instead.
* **ir:** `ibis.common.grounds.Annotable` is mutable by default now
* **ir:** `node.has_resolved_name()` is removed, use `isinstance(node, ops.Named)` instead; `node.resolve_name()` is removed use `node.name` instead
* **ir:** removed `ops.Node.flat_args()`, directly use `node.args` property instead
* **ir:** removed `ops.Node.inputs` property, use the multipledispatched `get_node_arguments()` function in the pandas backend
* **ir:** `Node.blocks()` method has been removed.
* **ir:** `HasSchema` mixin class is no longer available, directly subclass `ops.TableNode` and implement schema property instead
* **ir:** Removed `Node.output_type` property in favor of abstractmethod `Node.to_expr()` which now must be explicitly implemented
* **ir:** `Expr(Op(Expr(Op(Expr(Op)))))` is now represented as `Expr(Op(Op(Op)))`, so code using ibis internals must be migrated
* **pandas:** Use timezone conversion functions to compute the original machine localized value
* **common:** use `ibis.common.validators.{Parameter, Signature}` instead
* **ir:** `ibis.expr.lineage.lineage()` is now removed
* **ir:** removed `ir.DestructValue`, `ir.DestructScalar` and `ir.DestructColumn`, use `table.unpack()` instead
* **ir:** removed `Node.root_tables()` method, use `ibis.expr.analysis.find_immediate_parent_tables()` instead
* **impala:** use other methods for pinging the database

### Features

* add experimental decorator ([791335f](https://github.com/ibis-project/ibis/commit/791335f18e4d08f9c86098318fee7dfe9b8b1118))
* add to_pyarrow and to_pyarrow_batches ([a059cf9](https://github.com/ibis-project/ibis/commit/a059cf95bf556c82276a141c97ac493ebe57d322))
* add unbind method to expressions ([4b91b0b](https://github.com/ibis-project/ibis/commit/4b91b0b02aa5edfdc5b48324b9743c7165a2c16a)), closes [#4536](https://github.com/ibis-project/ibis/issues/4536)
* add way to specify sqlglot dialect on backend ([f1c0608](https://github.com/ibis-project/ibis/commit/f1c0608f0ef2f3986e4983b35c161ad7049c9702))
* **alchemy:** implement json getitem for sqlalchemy backends ([7384087](https://github.com/ibis-project/ibis/commit/7384087898d2f4c82196e42373d60915531d38c2))
* **api:** add `agg` alias for `aggregate` ([907583f](https://github.com/ibis-project/ibis/commit/907583f8cdbf74a72d8f30cf991bca824fd89c06))
* **api:** add `agg` alias to `group_by` ([6b6367c](https://github.com/ibis-project/ibis/commit/6b6367c6b55c619d71862b47776f68eead30c0be))
* **api:** add `ibis.read` top level API function ([e67132c](https://github.com/ibis-project/ibis/commit/e67132c53a06a1e6ff8ef8eacaafcb68857751b6))
* **api:** add JSON `__getitem__` operation ([3e2efb4](https://github.com/ibis-project/ibis/commit/3e2efb433d41209fe99a4db9e19df93f8b55211a))
* **api:** implement `__array__` ([1402347](https://github.com/ibis-project/ibis/commit/140234723446af8257ca9ea08e5d7c74195b2f64))
* **api:** make `drop` variadic ([1d69702](https://github.com/ibis-project/ibis/commit/1d697025b88bd7259f0ea53722a4c769feb16a61))
* **api:** return object from `to_sql` to support notebook syntax highlighting ([87c9833](https://github.com/ibis-project/ibis/commit/87c98339f8b9ca6a08c5297fedde513f47b6ebe3))
* **api:** use `rich` for interactive `__repr__` ([04758b8](https://github.com/ibis-project/ibis/commit/04758b863b7c0550904f2bcacec3104fdc06ba0a))
* **backend:** make `ArrayCollect` filterable ([1e1a5cf](https://github.com/ibis-project/ibis/commit/1e1a5cfca27f16409161d94b3c1f0425e4a1a4a4))
* **backends/mssql:** add backend support for Microsoft Sql Server ([fc39323](https://github.com/ibis-project/ibis/commit/fc393238222d1e55bfb814a67eb116506a906fdb))
* **bigquery:** add ops.DateFromYMD, ops.TimeFromHMS, ops.TimestampFromYMDHMS ([a4a7936](https://github.com/ibis-project/ibis/commit/a4a793697159ae215f87bef5b959957e6234e8cb))
* **bigquery:** add ops.ExtractDayOfYear ([30c547a](https://github.com/ibis-project/ibis/commit/30c547ab351940c13eddd00372c611c2cae3f465))
* **bigquery:** add support for correlation ([4df9f8b](https://github.com/ibis-project/ibis/commit/4df9f8b8ca4e0e1fab660d17acc5180bb8b9d2e4))
* **bigquery:** implement `argmin` and `argmax` ([40c5f0d](https://github.com/ibis-project/ibis/commit/40c5f0d31784e0d942d8e10f75acbce10090c4a9))
* **bigquery:** implement `pi` and `e` ([b91370a](https://github.com/ibis-project/ibis/commit/b91370a62591f584d0c22c2db357f9e4582d935a))
* **bigquery:** implement array repeat ([09d1e2f](https://github.com/ibis-project/ibis/commit/09d1e2f768a935d29aa607f3536fc93e12fc82dc))
* **bigquery:** implement JSON getitem functionality ([9c0e775](https://github.com/ibis-project/ibis/commit/9c0e7755ea0157aa3473e8699a04277b795c8a9c))
* **bigquery:** implement ops.ArraySlice ([49414ef](https://github.com/ibis-project/ibis/commit/49414efabfe2200f658282d807fd175275013a9e))
* **bigquery:** implement ops.Capitalize ([5757bb0](https://github.com/ibis-project/ibis/commit/5757bb04d767cc3d6e1f050bc3ac418d37efd7ea))
* **bigquery:** implement ops.Clip ([5495d6d](https://github.com/ibis-project/ibis/commit/5495d6d7ced1eb00d60595b17268018f3c1ea98e))
* **bigquery:** implement ops.Degrees, ops.Radians ([5119b93](https://github.com/ibis-project/ibis/commit/5119b93609f34035ced2ccc784190b0cb4a15a39))
* **bigquery:** implement ops.ExtractWeekOfYear ([477d287](https://github.com/ibis-project/ibis/commit/477d287602a299bd479fc869ea1306be54bbf110))
* **bigquery:** implement ops.RandomScalar ([5dc8482](https://github.com/ibis-project/ibis/commit/5dc848262d22d32035f50daf818afdf3b5cc3e11))
* **bigquery:** implement ops.StructColumn, ops.ArrayColumn ([2bbf73c](https://github.com/ibis-project/ibis/commit/2bbf73cb7ab462af7be46e399ab29e55b6d0a7d0))
* **bigquery:** implement ops.Translate ([77a4b3e](https://github.com/ibis-project/ibis/commit/77a4b3ee23ae57af735fbdab6fae3469293dbc7b))
* **bigquery:** implementt ops.NthValue ([b43ba28](https://github.com/ibis-project/ibis/commit/b43ba28528ac28542f5448dd7eccb34233e72d3e))
* **bigquery:** move bigquery backend back into the main repo ([cd5e881](https://github.com/ibis-project/ibis/commit/cd5e881c975f33cf3dc8910645996ce6ac94be81))
* **clickhouse:** handle more options in `parse_url` implementation ([874c5c0](https://github.com/ibis-project/ibis/commit/874c5c06a00c0f243c538032273bcf5e969dcb77))
* **clickhouse:** implement `INTERSECT ALL`/`EXCEPT ALL` ([f65fbc3](https://github.com/ibis-project/ibis/commit/f65fbc3fa9d5bd52a5f99024cad78f01bc4f42d3))
* **clickhouse:** implement quantile/multiquantile ([96d7d1b](https://github.com/ibis-project/ibis/commit/96d7d1bd35148a1fcd4251e1868cb5679ee42f75))
* **common:** support function annotations with both typehints and rules ([7e23f3e](https://github.com/ibis-project/ibis/commit/7e23f3eae8142b7621b9fd38b3df02c08f6a8a2d))
* **dask:** implement `mode` aggregation ([017f07a](https://github.com/ibis-project/ibis/commit/017f07aad0d711714c133a1cedb0b73aac4b356f))
* **dask:** implement json getitem ([381d805](https://github.com/ibis-project/ibis/commit/381d805055d62c6b993f542903e06050cbca7793))
* **datafusion:** convert column expressions to pyarrow ([0a888de](https://github.com/ibis-project/ibis/commit/0a888de96db91fc0a139dfd52de890790276459d))
* **datafusion:** enable `topk` ([d44903f](https://github.com/ibis-project/ibis/commit/d44903f3b8701d40437077a13a787e37799c8ef4))
* **datafusion:** implement `Limit` ([1ddc876](https://github.com/ibis-project/ibis/commit/1ddc876942dd66f17e016c936ffefae04e0b6938))
* **datafusion:** implement `ops.StringConcat` ([6bb5b4f](https://github.com/ibis-project/ibis/commit/6bb5b4f8f74e3130b6783fa98e5ef5d6c2cf9841))
* **decompile:** support rendering ibis expression as python code ([7eebc67](https://github.com/ibis-project/ibis/commit/7eebc670b5ff01c3802bf47011fe1db72dc38754))
* **deps:** support shapely 2.0 ([68dff10](https://github.com/ibis-project/ibis/commit/68dff10ec64fb0843f7ddc7203a44b541369547d))
* display qualified named in deprecation warnings ([a6e2a49](https://github.com/ibis-project/ibis/commit/a6e2a498e71dc2f91897b00e34674ec249617cab))
* **docs:** first draft of Ibis for pandas users ([7f7c9b5](https://github.com/ibis-project/ibis/commit/7f7c9b51b937db1079783d89bcca3d06d87ccb4a))
* **duckdb:** enable registration of parquet files from s3 ([fced465](https://github.com/ibis-project/ibis/commit/fced465867fee0f51d04865cfd5e5a3abc916028))
* **duckdb:** implement `mode` aggregation ([36fd152](https://github.com/ibis-project/ibis/commit/36fd1523a54fd58aa4657109ac1fe579c2c8259d))
* **duckdb:** implement `to_timestamp` ([26ca1e4](https://github.com/ibis-project/ibis/commit/26ca1e4f3160de594c8bb3047a3588e607385ad1))
* **duckdb:** implement quantile/multiquantile ([fac9705](https://github.com/ibis-project/ibis/commit/fac9705941d8e8aa11da378cca5b6269b1414e93))
* **duckdb:** overwrite views when calling `register` ([ae07438](https://github.com/ibis-project/ibis/commit/ae07438573d1958d491a903aa290e66cbd694639))
* **duckdb:** pass through kwargs to file loaders ([14fa2aa](https://github.com/ibis-project/ibis/commit/14fa2aa4213a1da5f1cb5ecad309c2856238268d))
* **duckdb:** support out of core execution for in-memory connections ([a4d4ba2](https://github.com/ibis-project/ibis/commit/a4d4ba26ef583be1104d2a40f9a0a7805ac9b3f0))
* **duckdb:** support registering external postgres tables with duckdb ([8633e6b](https://github.com/ibis-project/ibis/commit/8633e6b0c8b6cc92d8e8109c8d56a335f184c68a))
* **expr:** split ParseURL operation into multiple URL extract operations ([1f0fcea](https://github.com/ibis-project/ibis/commit/1f0fcea99fbb1dc97bd546812bad42feb7fa245e))
* **impala:** implement `strftime` ([d3ede8d](https://github.com/ibis-project/ibis/commit/d3ede8d9b6893b8135edb2010ceb04fbb3b9ca2b))
* **impala:** support date literals ([cd334c4](https://github.com/ibis-project/ibis/commit/cd334c42fd5c6076883461521976504454780c06))
* **insert:** add support for list+dict to sqlalchemy backends ([15d399e](https://github.com/ibis-project/ibis/commit/15d399ee31898569c3fba0fdc7638646d0f578a7))
* **ir/pandas/dask/clickhouse:** revamp Map type support ([62b6f2d](https://github.com/ibis-project/ibis/commit/62b6f2df9db29cacab26e335fe959d7ae3d4725a))
* **ir:** add `is_*` methods to `DataType`s ([79f5c2b](https://github.com/ibis-project/ibis/commit/79f5c2b4303e4887c3817f0074821310432813e2))
* **ir:** prototype for parsing SQL into an ibis expression ([1301183](https://github.com/ibis-project/ibis/commit/130118386b4c00b8367a993fe3af0701fdb8bf1c))
* **ir:** support python 3.10 pattern matching on Annotable nodes ([eca93eb](https://github.com/ibis-project/ibis/commit/eca93ebee29e34063cf20a0e29709fd2dc5e48ab))
* **mssql:** add window function support ([ef1be45](https://github.com/ibis-project/ibis/commit/ef1be45debf94c7298edb50a93803ca4398f0c90))
* **mssql:** detect schema from SQL ([ff79928](https://github.com/ibis-project/ibis/commit/ff79928048b6522a329721cc0b662af5e84fcd8d))
* **mssql:** extract quarter ([7d04266](https://github.com/ibis-project/ibis/commit/7d042666b3fda75ca08fdb725f1acf577dc4dd8e))
* **mssql:** implement ops.DayOfWeekIndex ([4125593](https://github.com/ibis-project/ibis/commit/412559301699c241ce9269011ee6fe186a96ee1d))
* **mssql:** implement ops.ExtractDayOfYear ([ae026d5](https://github.com/ibis-project/ibis/commit/ae026d56820f46ecfa59e94d1eefb461f3fa9603))
* **mssql:** implement ops.ExtractEpochSeconds ([4f49b5b](https://github.com/ibis-project/ibis/commit/4f49b5ba5ac62f2d96593dfd77986aaba835db7f))
* **mssql:** implement ops.ExtractWeekOfYear ([f1394bc](https://github.com/ibis-project/ibis/commit/f1394bcb75627075e94a2d2dd125d9547bceace1))
* **mssql:** implement ops.Ln, ops.Log, ops.Log2, ops.Log10 ([f8ee1d8](https://github.com/ibis-project/ibis/commit/f8ee1d821ae6d797149dc09f1fe522e6001aac47))
* **mssql:** implement ops.RandomScalar ([4149450](https://github.com/ibis-project/ibis/commit/414945013db3814ef12adff1dc7a1f40439d61f7))
* **mssql:** implement ops.TimestampTruncate, ops.DateTruncate ([738e496](https://github.com/ibis-project/ibis/commit/738e496c7b0a23ea6a4e57272f8fdbf30fcbd4f7))
* **mssql:** implementt ops.DateFromYMD, ops.TimestampFromYMDHMS, ops.TimeFromHMS ([e84f2ce](https://github.com/ibis-project/ibis/commit/e84f2cebea9f06675f371144891c2dc89d5561d4))
* open `*.db` files with sqlite in `ibis.connect` ([37baf05](https://github.com/ibis-project/ibis/commit/37baf055a73eeba42cdb73507b7379c7c4040db4))
* **pandas:** implement `mode` aggregation ([fc023b5](https://github.com/ibis-project/ibis/commit/fc023b584239c946ca31781ec63133d365349351))
* **pandas:** implement `RegexReplace` for `str` ([23713cc](https://github.com/ibis-project/ibis/commit/23713cc3d5a1fa0fbac124df2858a5bcc8c09f79))
* **pandas:** implement json getitem ([8fa1190](https://github.com/ibis-project/ibis/commit/8fa1190865c46786709c79be84f29013f79e0a83))
* **pandas:** implement quantile/multiquantile ([cd4dcaa](https://github.com/ibis-project/ibis/commit/cd4dcaa86b21c8b50d897df1471787575d2bc3a9))
* **pandas:** support `histogram` API ([5bfc0fe](https://github.com/ibis-project/ibis/commit/5bfc0fe493e3fcc7523237ed3e187a10207ebc8f))
* **polars:** enable `topk` ([8bfb16a](https://github.com/ibis-project/ibis/commit/8bfb16a13b88a3f344e686b936163645efabb7a9))
* **polars:** implement `mode` aggregation ([7982ba2](https://github.com/ibis-project/ibis/commit/7982ba203475abf56664b149fcc63352babd543f))
* **polars:** initial support for polars backend ([afecb0a](https://github.com/ibis-project/ibis/commit/afecb0ae925639113a6cfdc2073c3a5fbc1021e0))
* **postgres:** implement `mode` aggregation ([b2f1c2d](https://github.com/ibis-project/ibis/commit/b2f1c2db1209806339f25e522b2cf69615f54c53))
* **postgres:** implement quantile and multiquantile ([82ed4f5](https://github.com/ibis-project/ibis/commit/82ed4f51f79887a626ae8a6b8d1a0cfb667d9ec6))
* **postgres:** prettify array literals ([cdc60d5](https://github.com/ibis-project/ibis/commit/cdc60d5b43fc88e6837465cfbebd24787670e9f9))
* **pyspark:** add support for struct operations ([ce05987](https://github.com/ibis-project/ibis/commit/ce059870d76e68c1241ad331ae468bfeabbf213f))
* **pyspark:** enable `topk` ([0f748e0](https://github.com/ibis-project/ibis/commit/0f748e08d6af1ef760b0191e5cdd0ae0170fff64))
* **pyspark:** implement `pi` and `e` ([fea81c6](https://github.com/ibis-project/ibis/commit/fea81c6f2b7744da93e2adc94944015e2024fb65))
* **pyspark:** implement json getitem ([9bfb748](https://github.com/ibis-project/ibis/commit/9bfb748da751bfa441cfeb8129b27e1591a874ef))
* **pyspark:** implement quantile and multiquantile ([743f411](https://github.com/ibis-project/ibis/commit/743f4115a9c7d1a6e70a61a4f208572313e31f16))
* **pyspark:** support `histogram` API ([8f4808c](https://github.com/ibis-project/ibis/commit/8f4808c517d3a2ad2e3df4050809cbfdc8aa7f81))
* **snowflake:** enable day-of-week column expression ([6fd9c33](https://github.com/ibis-project/ibis/commit/6fd9c33da156517665e8998f6328384c57086a84))
* **snowflake:** handle date and timestamp literals ([ec2392d](https://github.com/ibis-project/ibis/commit/ec2392d02e8fcbba876add9abd5cabfb6f6c8a60))
* **snowflake:** implement `mode` aggregation ([f35915e](https://github.com/ibis-project/ibis/commit/f35915e938763976b690f742807c54e1455e1aea))
* **snowflake:** implement `parse_url` ([a9746e3](https://github.com/ibis-project/ibis/commit/a9746e32ef9dce222c83efc1c25a5d80edee5f53))
* **snowflake:** implement `rowid` scalar ([7e1425a](https://github.com/ibis-project/ibis/commit/7e1425ac866f2de965d22d891a2c14e162c699e7))
* **snowflake:** implement `time` literal ([068fc50](https://github.com/ibis-project/ibis/commit/068fc50144aee04e0655a0889824a85d912320f8))
* **snowflake:** implement scalar ([cc07d91](https://github.com/ibis-project/ibis/commit/cc07d91e3f0972f5e5a0eec7d0af6b1a2604ec13))
* **snowflake:** initial commit for snowflake backend ([a8687dd](https://github.com/ibis-project/ibis/commit/a8687dd85f9e3f8b29f5ebabf181cbbcb77185b7))
* **snowflake:** support reductions in window functions via automatic ordering ([0234e5c](https://github.com/ibis-project/ibis/commit/0234e5ca1a274ff64206a6683321375a6eaf9c42))
* **sql:** add ops.StringSQLILike ([7dc4924](https://github.com/ibis-project/ibis/commit/7dc49246bdad1f9dd6992a7a650f4109199e0c2f))
* **sqlalchemy:** implement `ops.Where` using `IF`/`IFF` functions ([4cc9c15](https://github.com/ibis-project/ibis/commit/4cc9c153b9f2cdc06c29678d8281feb00fa64c1e))
* **sqlalchemy:** in-memory tables have name in generated SQL ([01b4c60](https://github.com/ibis-project/ibis/commit/01b4c6050638b10e29ba85fb5bb188c0647a67fc))
* **sql:** improve error message in fixed_arity helper ([891a1ad](https://github.com/ibis-project/ibis/commit/891a1ad579b6df68025c25d1f8996546500eb131))
* **sqlite:** add `type_map` arg to override type inference ([1961bad](https://github.com/ibis-project/ibis/commit/1961bad2606624ca50102eeee49d1db73e4ccc8f))
* **sqlite:** fix impl for missing `pi` and `e` functions ([24b6d2f](https://github.com/ibis-project/ibis/commit/24b6d2f111afd8426a57ffdaf63de7cf65e99248))
* **sqlite:** support `con.sql` with explicit schema specified ([7ca82f3](https://github.com/ibis-project/ibis/commit/7ca82f3bc7b2d096859bab9b5d821bb8ed52aaf2))
* **sqlite:** support wider range of datetime formats ([f65093a](https://github.com/ibis-project/ibis/commit/f65093a7f08b4768d99fab167beb789151bfde13))
* support both `postgresql://` and `postgres://` in `ibis.connect` ([2f7a7b4](https://github.com/ibis-project/ibis/commit/2f7a7b404f29abc5a22d11e6dd2138b0b1ed5a59))
* support deferred predicates in join ([b51a64b](https://github.com/ibis-project/ibis/commit/b51a64b40a6feb274a541fd677c39dab8d59b55f))
* support more operations with unsigned integers ([9992953](https://github.com/ibis-project/ibis/commit/999295367480d8584bdd89bc3e53adc5b26be0ad))
* support passing callable to relabel ([0bceefd](https://github.com/ibis-project/ibis/commit/0bceefd4ca17f9ca27957bc3a9f0f437ce0d1a2c))
* support tab completion for getitem access of table columns ([732dba4](https://github.com/ibis-project/ibis/commit/732dba4926ab01b2363adaf3306ea838cb52d83f))
* support Table.fillna for SQL backends ([26d4cac](https://github.com/ibis-project/ibis/commit/26d4cacf93f41b66f1407e313e86ec9f0a48aa29))
* **trino:** add `bit_xor` aggregation ([830acf4](https://github.com/ibis-project/ibis/commit/830acf4d896a5d3bfedd2549aada3aa2f274e59b))
* **trino:** add `EXTRACT`-based functionality ([6549657](https://github.com/ibis-project/ibis/commit/654965705d437ec7f90c0658d3f436a355414b71))
* **trino:** add millisecond scale to *_trunc function ([3065248](https://github.com/ibis-project/ibis/commit/3065248be55a8d691d6e552ce0eb3c9634232c95))
* **trino:** add some basic aggregation ops ([7ecf7ab](https://github.com/ibis-project/ibis/commit/7ecf7ab939b2f85615a7694240ba98782d678e5f))
* **trino:** extract milliseconds ([09517a5](https://github.com/ibis-project/ibis/commit/09517a5b8b21cfbad6b65c8a0f5c798a36ff212c))
* **trino:** implement `approx_median` ([1cba8bd](https://github.com/ibis-project/ibis/commit/1cba8bd9f43fc5756a32341c6bcf18f45a1c8b1d))
* **trino:** implement `parse_url` ([2bc87fc](https://github.com/ibis-project/ibis/commit/2bc87fc40307eca2329d862849629587a82eb5f6))
* **trino:** implement `round`, `cot`, `pi`, and `e` ([c0e8736](https://github.com/ibis-project/ibis/commit/c0e8736a9a69f0860b99d271a2c47069ab91bb19))
* **trino:** implement arbitrary first support ([0c7d3b3](https://github.com/ibis-project/ibis/commit/0c7d3b36cc46cf5eb8b0f19988dfb9b4db1f9905))
* **trino:** implement array collect support ([dfeb600](https://github.com/ibis-project/ibis/commit/dfeb600fad89eed6c7a23434cc1f096e46cac7eb))
* **trino:** implement array column support ([dadf9a8](https://github.com/ibis-project/ibis/commit/dadf9a865973b3609ac0bb87bdbf635c61653486))
* **trino:** implement array concat ([240c55d](https://github.com/ibis-project/ibis/commit/240c55dea684d8f1c808758e32e614ae8d0ad739))
* **trino:** implement array index ([c5f3a96](https://github.com/ibis-project/ibis/commit/c5f3a96fc992f85f52b49acbc624c95241635e59))
* **trino:** implement array length support ([2d7cc65](https://github.com/ibis-project/ibis/commit/2d7cc659b112e3c4c740a11a34351088d05a7265))
* **trino:** implement array literal support ([2182177](https://github.com/ibis-project/ibis/commit/218217791ee1ce48ebb0e5422edb4a6762ebf2be))
* **trino:** implement array repeat ([2ee3d10](https://github.com/ibis-project/ibis/commit/2ee3d1092b1993dcb4f550d85ef728bae29bbdd1))
* **trino:** implement array slicing ([643792e](https://github.com/ibis-project/ibis/commit/643792e137791934d7b2c794a792814f4b4e54ea))
* **trino:** implement basic struct operations ([cc3c937](https://github.com/ibis-project/ibis/commit/cc3c937d59271e01491d4f72417c666efcbc5af3))
* **trino:** implement bitwise agg support ([5288b35](https://github.com/ibis-project/ibis/commit/5288b351b41c459fab7af4d687c29a83bfbc1d23))
* **trino:** implement bitwise scalar/column ops ([ac4876c](https://github.com/ibis-project/ibis/commit/ac4876c3bd7af271474d0e91435bd0d96c6273b3))
* **trino:** implement default precision and scale ([37f8a47](https://github.com/ibis-project/ibis/commit/37f8a47364807bb501c12a63e888d68635f91377))
* **trino:** implement group concat support ([5c41439](https://github.com/ibis-project/ibis/commit/5c41439d17622819c34f2367308e07ead3013c66))
* **trino:** implement json getitem support ([7c41566](https://github.com/ibis-project/ibis/commit/7c41566f871322f08c93a689463bf4be780df6d6))
* **trino:** implement map operations ([4efc5ce](https://github.com/ibis-project/ibis/commit/4efc5ceb3660a9cbab8809790642cf5a5eea616e))
* **trino:** implement more generic and numeric ops ([63b45c8](https://github.com/ibis-project/ibis/commit/63b45c8183cc15a1cefde220caea3c90d7a6bd35))
* **trino:** implement ops.Capitalize ([dff14fc](https://github.com/ibis-project/ibis/commit/dff14fca8bc340f2b9823e793c53c0a40e25b276))
* **trino:** implement ops.DateFromYMD ([edd2994](https://github.com/ibis-project/ibis/commit/edd29947285ef029217e1db05ec3e13eb84d396b))
* **trino:** implement ops.DateTruncate, ops.TimestampTruncate ([32f4862](https://github.com/ibis-project/ibis/commit/32f486281cb5548213e7d924492478e9b4cf395e))
* **trino:** implement ops.DayOfWeekIndex, ops.DayOfWeekName ([a316d6d](https://github.com/ibis-project/ibis/commit/a316d6d3566feb959e2431d3ba07f630f5ecd7a2))
* **trino:** implement ops.ExtractDayOfYear ([b0a3465](https://github.com/ibis-project/ibis/commit/b0a3465acfe61a37653a9c9c9e8c12e25747874d))
* **trino:** implement ops.ExtractEpochSeconds ([10b82f1](https://github.com/ibis-project/ibis/commit/10b82f1116d5ade5029be5d53bb579d05e56e9a9))
* **trino:** implement ops.ExtractWeekOfYear ([cf719b8](https://github.com/ibis-project/ibis/commit/cf719b86c0b7046fb62da1c1e18b86ccc7677886))
* **trino:** implement ops.Repeat ([e9f6851](https://github.com/ibis-project/ibis/commit/e9f6851586a0bd7e00d3b16265ff9fe9ece5d1db))
* **trino:** implement ops.Strftime ([a436823](https://github.com/ibis-project/ibis/commit/a436823b232562fdd25a5b3b2a9a80e0a9fd510e))
* **trino:** implement ops.StringAscii ([93fd32d](https://github.com/ibis-project/ibis/commit/93fd32dc3dae9abbc52e8d2dadf1747a74daa7d6))
* **trino:** implement ops.StringContains ([d5cb2ec](https://github.com/ibis-project/ibis/commit/d5cb2ec4b654f9c7bf9a9f6ca22d847b77018f17))
* **trino:** implement ops.StringSplit ([62d79a6](https://github.com/ibis-project/ibis/commit/62d79a6bc08be599aa4a42c1623f1a135b11a62b))
* **trino:** implement ops.StringToTimestamp ([b766f62](https://github.com/ibis-project/ibis/commit/b766f6204f95a75082586b5756434271e427185f))
* **trino:** implement ops.StrRight ([691b39c](https://github.com/ibis-project/ibis/commit/691b39c0e13078fd9d07bf1d2beec19486268cef))
* **trino:** implement ops.TimeFromHMS ([e5cacc2](https://github.com/ibis-project/ibis/commit/e5cacc29c2d337aec1b4d55fab3a34e6ce906182))
* **trino:** implement ops.TimestampFromUNIX ([ce5d726](https://github.com/ibis-project/ibis/commit/ce5d72664cb0d808e257d4212171dbbb2f6cb38b))
* **trino:** implement ops.TimestampFromYMDHMS ([9fa7304](https://github.com/ibis-project/ibis/commit/9fa730453ffca951bf0bf72db90fa108c99f1a4a))
* **trino:** implement ops.TimestampNow ([c832e4c](https://github.com/ibis-project/ibis/commit/c832e4cce3323d30e2304f481d26099c0541e523))
* **trino:** implement ops.Translate ([410ae1e](https://github.com/ibis-project/ibis/commit/410ae1ef4689e8fb9b345bfe9ae5a99b9261d75c))
* **trino:** implement quantile/multiquantile ([bc7fdab](https://github.com/ibis-project/ibis/commit/bc7fdaba7095303ae62cc57f4406865c341516f9))
* **trino:** implement regex functions ([9e493c5](https://github.com/ibis-project/ibis/commit/9e493c5524538a6dfde81041fa7867c9ceec741e))
* **trino:** implement window function support ([5b6cc45](https://github.com/ibis-project/ibis/commit/5b6cc455edc977ca1c17afce8f66726e20293b4a))
* **trino:** initial trino backend ([c367865](https://github.com/ibis-project/ibis/commit/c36786596e15a6d6161b26b4e64cb3a741fb26d2))
* **trino:** support string date scalar parameter ([9092530](https://github.com/ibis-project/ibis/commit/9092530ba2bde5988bb7247d07db6ae409dfea47))
* **trino:** use proper `approx_distinct` function ([3766fff](https://github.com/ibis-project/ibis/commit/3766fff1ccdd63c358bc37d10cd86c002d638719))


### Bug Fixes

* `ibis.connect` always returns a backend ([2d5b155](https://github.com/ibis-project/ibis/commit/2d5b1550643bbba6a7d5082a06f6e684a247f43f))
* allow inserting memtable with alchemy backends ([c02fcc3](https://github.com/ibis-project/ibis/commit/c02fcc3f96686ca485c7a30d88666bc02842753d))
* always display at least one column in the table repr ([5ea9e5a](https://github.com/ibis-project/ibis/commit/5ea9e5a9570cb21c3873012ef23e648300dd4f2b))
* **analysis:** only lower sort keys that are in an agg's output ([6bb4f66](https://github.com/ibis-project/ibis/commit/6bb4f663f644641d6dd3e61d221ea8fe39d029d5))
* **api:** allow arbitrary sort keys ([a980b34](https://github.com/ibis-project/ibis/commit/a980b3405209c482e7000955e7ceb74da6bedbd9))
* **api:** allow boolean scalars in predicate APIs ([2a2636b](https://github.com/ibis-project/ibis/commit/2a2636bde79bb4416eb3455035e5690d1c296e29))
* **api:** allow deferred instances as input to `ibis.desc` and `ibis.asc` ([6861347](https://github.com/ibis-project/ibis/commit/6861347638a8a8d4c0de077156c532338194817a))
* **api:** ensure that window functions are propagated ([4fb1106](https://github.com/ibis-project/ibis/commit/4fb110651c93c7ff38ef2898998e9e280d70dd93))
* **api:** make `re_extract` conform to semantics of Python's `re.match` ([5981227](https://github.com/ibis-project/ibis/commit/598122736a8fc44eb3b909c3100d8271d156d9b3))
* auto-register csv and parquet with duckdb using `ibis.connect` ([67c4f87](https://github.com/ibis-project/ibis/commit/67c4f87401cf1be9b2abb32bb1f589f81257f347))
* avoid renaming known equal columns for inner joins with equality predicates ([5d4b0ed](https://github.com/ibis-project/ibis/commit/5d4b0ed28aa681e899d8c41a8e767cd05d294bd2))
* **backends:** fix casting and execution result types in many backends ([46c21dc](https://github.com/ibis-project/ibis/commit/46c21dca4f125f9a337c5a5489b8ba86f78a3425))
* **bigquery:** don't try to parse database when name is already fully qualified ([ae3c113](https://github.com/ibis-project/ibis/commit/ae3c1139b35a1b18fac27533bee7442fb2726fb8))
* **bigquery:** fix integer to timestamp casting ([f5bacad](https://github.com/ibis-project/ibis/commit/f5bacade0c99d25b66ba9bc5c82118f907bc34ee))
* **bigquery:** normalize W frequency in *_trunc ([893cd49](https://github.com/ibis-project/ibis/commit/893cd496b0f11079a1b2a8a20462c00825155e9a))
* catch `TypeError` instead of more specific error ([6db19d8](https://github.com/ibis-project/ibis/commit/6db19d8269ce397137de8c23ea02b6ee89d0a0a1))
* change default limit to None ([8d1526a](https://github.com/ibis-project/ibis/commit/8d1526a8d46939d9e9f23697527c630737d0633d))
* clarify and normalize behavior of `Table.rowid` ([92b03d6](https://github.com/ibis-project/ibis/commit/92b03d660e65fa87f44fd297c0dcfa7500a9a860))
* **clickhouse:** ensure that correlated subqueries' columns can be referenced ([708d682](https://github.com/ibis-project/ibis/commit/708d682a8e9207f88578590adb1d423c0072f550))
* **clickhouse:** fix list_tables to use database name ([edc3511](https://github.com/ibis-project/ibis/commit/edc3511cc497424619f0cbaacfb8fb6058abe9c9))
* **clickhouse:** make `any`/`all` filterable and reduce code size ([99b10e2](https://github.com/ibis-project/ibis/commit/99b10e29e5f031da97d51291fe455d1c26f432d9))
* **clickhouse:** use clickhouse's dbapi ([bd0da12](https://github.com/ibis-project/ibis/commit/bd0da127e1bc918b6ac01c50309dcd108767fe00))
* **common:** support copying variadic annotable instances ([ee0d9ad](https://github.com/ibis-project/ibis/commit/ee0d9ad52aa88e296f1965dabb821bf22ff530ac))
* **dask:** make filterable reductions work ([0f759fc](https://github.com/ibis-project/ibis/commit/0f759fc71f0206a0ae9bf16afdcee4e8486515d0))
* **dask:** raise TypeError with informative message in ibis.dask.connect ([4e67f7a](https://github.com/ibis-project/ibis/commit/4e67f7a0d0b5ffa7667cf402639e2cbc89d9a37c))
* define `to_pandas`/`to_pyarrow` on DataType/Schema classes directly ([22f3b4d](https://github.com/ibis-project/ibis/commit/22f3b4d7fbc778302b4ed2b76348a423412070a9))
* **deps:** bound shapely to a version that doesn't segfault ([be5a779](https://github.com/ibis-project/ibis/commit/be5a7798229227a2f37eada3e2c79d4f50ea5656))
* **deps:** update dependency datafusion to >=0.6,<0.8 ([4c73870](https://github.com/ibis-project/ibis/commit/4c73870eca9d16e931ba2de1b3803adc7ebc3a7c))
* **deps:** update dependency geopandas to >=0.6,<0.13 ([58a32dc](https://github.com/ibis-project/ibis/commit/58a32dc1204b75ba8556f0da234c7a939e83d52e))
* **deps:** update dependency packaging to v22 ([e0b6177](https://github.com/ibis-project/ibis/commit/e0b61771bb17b224a06bc843444f5f3cfc1fdb25))
* **deps:** update dependency rich to v13 ([4f313dd](https://github.com/ibis-project/ibis/commit/4f313dd22537a9e827230a454422f8fef6b3a66c))
* **deps:** update dependency sqlglot to v10 ([db19d43](https://github.com/ibis-project/ibis/commit/db19d43f3e2e4c0054294e15a46b9df360340d04))
* **deps:** update dependency sqlglot to v9 ([cf330ac](https://github.com/ibis-project/ibis/commit/cf330acc3fdc04712dc24081caecbae99bf2c7da))
* **docs:** make sure data can be downloaded when building notebooks ([fa7da17](https://github.com/ibis-project/ibis/commit/fa7da1718ce1835b1f154fbe3ba854afeb0dd0aa))
* don't fuse filters & selections that contain window functions ([d757069](https://github.com/ibis-project/ibis/commit/d7570692a5521cac36bd78cc2f6cf636535defc3))
* drop snowflake support for RowID ([dd378f1](https://github.com/ibis-project/ibis/commit/dd378f1b6db69697536ea90e5f7bb410c4bebc4b))
* **duckdb:** drop incorrect `translate` implementation ([8690151](https://github.com/ibis-project/ibis/commit/86901511a4cee5f143b552fc060111d116e0b38b))
* **duckdb:** fix bug in json getitem for duckdb ([49ce739](https://github.com/ibis-project/ibis/commit/49ce739ba0cc9a735efcda2cdbd04b6de67ccd04))
* **duckdb:** keep `ibis.now()` type semantics ([eca4a2c](https://github.com/ibis-project/ibis/commit/eca4a2cdc969ee45c6ed449d7a102dc39e24f0a9))
* **duckdb:** make array repeat actually work ([021f4de](https://github.com/ibis-project/ibis/commit/021f4de5adf854410e7c8ec8117e3490532587d3))
* **duckdb:** replace all in `re_replace` ([c138f0f](https://github.com/ibis-project/ibis/commit/c138f0fd37fb4d3ccbebd99c9b53f85eecd5d241))
* **duckdb:** rereflect sqla table on re-registration ([613b311](https://github.com/ibis-project/ibis/commit/613b3119a6d5ab5a377f111e9adf3f8afca835a1)), closes [#4729](https://github.com/ibis-project/ibis/issues/4729)
* **duckdb:** s3 priority ([a2d03d1](https://github.com/ibis-project/ibis/commit/a2d03d1e3859c4b1db91a57603405a56d9330cf4))
* **duckdb:** silence duckdb-engine warnings ([359adc3](https://github.com/ibis-project/ibis/commit/359adc3356b6616c24829a3ae654fef31b670da9))
* ensure numpy ops dont accidentally cast ibis types ([a7ca6c8](https://github.com/ibis-project/ibis/commit/a7ca6c83d6544e3180c478e217422dc0168e768c))
* exclude geospatial ops from pandas/dask/polars `has_operation` ([6f1d265](https://github.com/ibis-project/ibis/commit/6f1d265f1a5df20d30f20b5ac3c2dec07174f416))
* fix `table.mutate` with deferred named expressions ([5877d0b](https://github.com/ibis-project/ibis/commit/5877d0b7a92044bb82637225e31d27543ddaadc7))
* fix bug when disabling `show_types` in interactive repr ([2402506](https://github.com/ibis-project/ibis/commit/2402506b92a4e74ae0809e8ee2291694114228a8))
* fix expression repr for table -> value operations ([dbf92f5](https://github.com/ibis-project/ibis/commit/dbf92f54bdc566db555fadd150892fdc3f4ee36e))
* handle dimensionality of empty outputs ([3a88170](https://github.com/ibis-project/ibis/commit/3a88170128aed4f3987656b67759d86817153689))
* improve rich repr support ([522db9c](https://github.com/ibis-project/ibis/commit/522db9c646850f6245ae770400d14ac731bbcee2))
* **ir:** normalize `date` types ([39056b5](https://github.com/ibis-project/ibis/commit/39056b5db91bc0a1e2d7885228811ea3cb0addcb))
* **ir:** normalize timestamps to `datetime.datetime` values ([157efde](https://github.com/ibis-project/ibis/commit/157efdeb341193bf6ab8123e64c5d79ebff81c42))
* make `col.day_of_week` not an expr ([96e1580](https://github.com/ibis-project/ibis/commit/96e1580aa581111d8de4eb19aa0e526663b1f6e7))
* **mssql:** fix integer to timestamp casting ([9122eef](https://github.com/ibis-project/ibis/commit/9122eef7cd2533338d5866b878e6fec8781f7108))
* **mssql:** fix ops.TimeFromHMS ([d2188e1](https://github.com/ibis-project/ibis/commit/d2188e11e5d5e56c579a01d14b2bb5249c9c636c))
* **mssql:** fix ops.TimestampFromUNIX ([ec28add](https://github.com/ibis-project/ibis/commit/ec28add159c4a116fb62dc5e593d15eb9fda579e))
* **mssql:** fix round without argument ([52a60ce](https://github.com/ibis-project/ibis/commit/52a60ce8bcee61e491a43509cee4dfaa3163a474))
* **mssql:** use double-dollar sign to prevent from interpolating a value ([b82da5d](https://github.com/ibis-project/ibis/commit/b82da5d2e6434ba4dc31787eecb2d002259d9bfd))
* **mysql:** fix mysql `startswith`/`endswith` to be case sensitive ([d7469cc](https://github.com/ibis-project/ibis/commit/d7469ccabca6c3a587626b8925dbca66d4aed67f))
* **mysql:** handle out of bounds timestamps and fix milliseconds calculation ([1f7649a](https://github.com/ibis-project/ibis/commit/1f7649aff0b72a90229b4d34f8efa100a1d496be))
* **mysql:** upcast bool agg args ([8c5f9a5](https://github.com/ibis-project/ibis/commit/8c5f9a555f1407a05e903b4e8ee0d2de4f0911e0))
* pandas/dask now cast int<->timestamp as seconds since epoch ([bbfe998](https://github.com/ibis-project/ibis/commit/bbfe998e3be698a1d9bb3d00ad0fb949362bff45))
* **pandas:** drop `RowID` implementation ([05f5016](https://github.com/ibis-project/ibis/commit/05f5016b45111e74beea0bb67b29bc22278e3594))
* **pandas:** make quantile/multiquantile with filter work ([6b5abd6](https://github.com/ibis-project/ibis/commit/6b5abd6e263b48fffd2a42eeb9d547eafc6f14ce))
* **pandas:** support `substr` with no `length` ([b2c2922](https://github.com/ibis-project/ibis/commit/b2c29225365ff3a70a4284637f09936b76bc29ef))
* **pandas:** use localized UTC time for `now` operation ([f6d7327](https://github.com/ibis-project/ibis/commit/f6d73272b41861fefab650df7139c2733ccb24a3))
* **pandas:** use the correct context when aggregating over a window ([e7fa5c0](https://github.com/ibis-project/ibis/commit/e7fa5c0fbc2b30720ac660353aa3391f06c451b7))
* **polars:** fix polars `startswith` to call the right method ([9e6f397](https://github.com/ibis-project/ibis/commit/9e6f397acfbcd2da4f9d457c6ad6bc59ccfb2c2e))
* **polars:** workaround passing `pl.Null` to the null type ([fd9633b](https://github.com/ibis-project/ibis/commit/fd9633b37660b7bfe24a6fe93e1c38c8fcfd4eed))
* **postgres/duckdb:** fix negative slicing by copying the trino impl ([39e3962](https://github.com/ibis-project/ibis/commit/39e39621c680bf7663f946d5953b8a9df7457b2b))
* **postgres:** fix array repeat to work with literals ([3c46eb1](https://github.com/ibis-project/ibis/commit/3c46eb1a6f8c65cdf0fe93dd1f7f31463f2230f7))
* **postgres:** fix array_index operation ([63ef892](https://github.com/ibis-project/ibis/commit/63ef89211f0c3a695482c6be35d6f106104ec709))
* **postgres:** make any/all translation rules use `reduction` helper ([78bfd1d](https://github.com/ibis-project/ibis/commit/78bfd1df6a8e93c029b57dd02b3b8bf8c990ab85))
* **pyspark:** handle `datetime.datetime` literals ([4f94abe](https://github.com/ibis-project/ibis/commit/4f94abeb66d91020be653fde204779f72a04bc58))
* remove kerberos extra for impala dialect ([6ed3e5f](https://github.com/ibis-project/ibis/commit/6ed3e5f2c65817991ce2746f87aa96de29c465f2))
* **repr:** don't repeat value in repr for literals ([974eeb6](https://github.com/ibis-project/ibis/commit/974eeb63d42783e5d7739f89f8644383169b5b0c))
* **repr:** fix off by one in repr ([322c8dc](https://github.com/ibis-project/ibis/commit/322c8dc6bfa841532eb7084c4a0ea07ba0b23dd8))
* **s3:** fix quoting and autonaming for s3 ([ce09266](https://github.com/ibis-project/ibis/commit/ce092662fbacba6fa769cb2c93311ceeb1540e6e))
* **select:** raise error on attempt to select no columns in projection ([94ac10e](https://github.com/ibis-project/ibis/commit/94ac10e6ea7670b1ef7223de9709bd10f794a099))
* **snowflake:** fix extracting query parameter by ([75af240](https://github.com/ibis-project/ibis/commit/75af2404fc59f814f458b7ab0a8a18a231ec58e1))
* **snowflake:** fix failing snowflake url extraction functions ([2eee50b](https://github.com/ibis-project/ibis/commit/2eee50bc4541adaa5735145a108d8ff933bc4043))
* **snowflake:** fix snowflake list_databases ([680cd24](https://github.com/ibis-project/ibis/commit/680cd244b680d320cdd52834a0b7c131aa491a34))
* **snowflake:** handle schema when getting table ([f6fff5b](https://github.com/ibis-project/ibis/commit/f6fff5b8a78b69c9a44ad046650557cbd37b23cd))
* **snowflake:** snowflake now likes Tuesdays ([1bf9d7c](https://github.com/ibis-project/ibis/commit/1bf9d7cc530a22e466dd5e29afa7a32a42824d00))
* **sqlalchemy:** allow passing pd.DataFrame to create ([1a083f6](https://github.com/ibis-project/ibis/commit/1a083f6d2f5c30d644a61fee8b01399e22b9ec15))
* **sqlalchemy:** ensure that arbitrary expressions are valid sort keys ([cb1a013](https://github.com/ibis-project/ibis/commit/cb1a013f14732c5ccc560833275acec18ff169d1))
* **sql:** avoid generating cartesian products yet again ([fdc52a2](https://github.com/ibis-project/ibis/commit/fdc52a26f9c3a821c284297fe6dc23a87e0cedc4))
* **sqlite:** fix sqlite `startswith`/`endswith` to be case sensitive ([fd4a88d](https://github.com/ibis-project/ibis/commit/fd4a88ddd04fc47d1e1ad93f61eaadfff3a02753))
* standardize list_tables signature everywhere ([abafe1b](https://github.com/ibis-project/ibis/commit/abafe1b8659ab4bfb179863cf8a57958a2195811)), closes [#2877](https://github.com/ibis-project/ibis/issues/2877)
* support `arbitrary` with no arguments ([45156f5](https://github.com/ibis-project/ibis/commit/45156f5ba03d8ff69e8a002364a44b15d4302bc7))
* support dtype in `__array__` methods ([1294b76](https://github.com/ibis-project/ibis/commit/1294b76769888c8e30f8afd8585009d24ab849ad))
* **test:** ensure that file-based url tests don't expect data to exist ([c2b635a](https://github.com/ibis-project/ibis/commit/c2b635a22974573078d53ad91d92cf6be60efc3a))
* **trino:** fix integer to timestamp casting ([49321a6](https://github.com/ibis-project/ibis/commit/49321a68fa4a9e80f5d276f43d43a2c85dd03506))
* **trino:** make filterable any/all reductions work ([992bd18](https://github.com/ibis-project/ibis/commit/992bd18561ae4f34f170f5acb60e0a10fde224da))
* truncate columns in repr for wide tables ([aadcba1](https://github.com/ibis-project/ibis/commit/aadcba138600937c82d97805ad51b6283adc5ed1))
* **typo:** in StringValue helpstr ([b2e2093](https://github.com/ibis-project/ibis/commit/b2e2093891e4d3a495abea5bc8c5af681dd86c13))
* **ux:** improve error messages for rlz.comparable failures ([5ca41d2](https://github.com/ibis-project/ibis/commit/5ca41d26b29c8756bdd9446508e39352bd990de3))
* **ux:** prevent infinite looping when formatting a floating column of all nans ([b6afe98](https://github.com/ibis-project/ibis/commit/b6afe98f4738e50a256bc326f4fe80faeae9c73d))
* visualize(label_edges=True) works for NodeList ops ([a91ceae](https://github.com/ibis-project/ibis/commit/a91ceaefcba343f1f9ccccd10d7773e33df57e50))
* **visualize:** dedup nodes and edges and add `verbose` argument for debugging ([521e188](https://github.com/ibis-project/ibis/commit/521e188bf3942913ebaae1d5e317b0294c2f331a))
* **visualize:** handle join predicates in visualize ([d63cb57](https://github.com/ibis-project/ibis/commit/d63cb5720f7afe0c7aaf4ca79f2290abe9e98613))
* **window:** allow window range tuples in preceding or following ([77172b3](https://github.com/ibis-project/ibis/commit/77172b30b4319f4bbbe1af5b438322c21741d693))


### Deprecations

* deprecate `Table.groupby` alias in favor of `Table.group_by` ([39cea3b](https://github.com/ibis-project/ibis/commit/39cea3bb90dd42ef78fd41d46664fa40865c2b9e))
* deprecate `Table.sort_by` in favor of `Table.order_by` ([7ac7103](https://github.com/ibis-project/ibis/commit/7ac7103692260f62dc975e45a8526bdfa9fe1fa2))


### Performance

* add benchmark for known-slow table expression ([e9617f0](https://github.com/ibis-project/ibis/commit/e9617f0854030e70365eb264bcb3b58078e79e9e))
* **expr:** traverse nodes only once during compilation ([69019ed](https://github.com/ibis-project/ibis/commit/69019edef23f7e461365ee0de1cbe8e5534a74d5))
* fix join performance by avoiding Projection construction ([ed532bf](https://github.com/ibis-project/ibis/commit/ed532bfbc3c0915b06089f9f5aebc48b6724b072))
* **node:** give `Node`s the default Python repr ([eb26b11](https://github.com/ibis-project/ibis/commit/eb26b1102eb814ba38ad90c35c1a526fabedc0b9))
* **ux:** remove pandas import overhead from `import ibis` ([ea452fc](https://github.com/ibis-project/ibis/commit/ea452fc32690aeb40f0a38722b18adb5f38869ac))


* **deps:** bump duckdb lower bound ([4539683](https://github.com/ibis-project/ibis/commit/4539683037623caf49811d9fa6c2541f9adab07b))
* **dev-deps:** replace flake8 et al with `ruff` and fix lints ([9c1b282](https://github.com/ibis-project/ibis/commit/9c1b2821a0051f0c1d8992f3bd1579d407884f7b))


### Refactors

* add `lazy_singledispatch` utility ([180ecff](https://github.com/ibis-project/ibis/commit/180ecff7f10486ded0fd6d9f93905660548dd5ef))
* add `rlz.lazy_instance_of` ([4e30480](https://github.com/ibis-project/ibis/commit/4e30480f3ba9809d2fc31d04d67971a6ae4615c4))
* add `Temporal` base class for temporal data types ([694eec4](https://github.com/ibis-project/ibis/commit/694eec4f08b6d305dfea183b64d098cc39b69de4))
* **api:** add deprecated Node.op() [#4519](https://github.com/ibis-project/ibis/issues/4519) ([2b0826b](https://github.com/ibis-project/ibis/commit/2b0826b1a5b4294dd10520945fb11eca555669fd))
* avoid roundtripping to expression for `IFF` ([3068ae2](https://github.com/ibis-project/ibis/commit/3068ae20f284de19b08efb5127afbecc370ff401))
* clean up `cot` implementations to have one less function call ([0f304e5](https://github.com/ibis-project/ibis/commit/0f304e503f9f3d514695cb95719aed5150796608))
* clean up timezone support in ops.TimestampFromYMDHMS ([2e183a9](https://github.com/ibis-project/ibis/commit/2e183a909f232bceae96f19bfdf3f663592c10a1))
* cleanup str method docstrings ([36bd36c](https://github.com/ibis-project/ibis/commit/36bd36c7ba565f37f15588fdb6fbbdf27492847f))
* **clickhouse:** implement sqlglot-based compiler ([5cc5d4b](https://github.com/ibis-project/ibis/commit/5cc5d4b8efeebb4ad5215469c203253db296ea18))
* **clickhouse:** simplify Quantile and MultiQuantile implementation ([9e16e9e](https://github.com/ibis-project/ibis/commit/9e16e9e96d642acd6df58a37cf13783b20109ece))
* **common:** allow traversal and substitution of tuple and dictionary arguments ([60f4806](https://github.com/ibis-project/ibis/commit/60f4806474041fd78b54aa4b9c233f1918b50468))
* **common:** enforce slots definitions for Base subclasses ([6c3df91](https://github.com/ibis-project/ibis/commit/6c3df912f234e57082f38dde560fe3684ecb1ecf))
* **common:** move Parameter and Signature to validators.py ([da20537](https://github.com/ibis-project/ibis/commit/da205373427599a8200f273c627675178eac6f40))
* **common:** reduce implementation complexity of annotations ([27cee71](https://github.com/ibis-project/ibis/commit/27cee710e7554b5d881bf0d66a56c429d5c51648))
* **datafusion:** align register API across backends ([08046aa](https://github.com/ibis-project/ibis/commit/08046aae0126bb4125b7d96953ec7bda0a99756a))
* **datafusion:** get name from expr ([fea3e5b](https://github.com/ibis-project/ibis/commit/fea3e5b4a1c0affa011086bf98d8a33a3ae3af2b))
* **datatypes:** remove Enum ([145e706](https://github.com/ibis-project/ibis/commit/145e706b7bf0e13fafa75dbc69452b25468ad0e6))
* **dev-deps:** remove unnecessary `poetry2nix` overrides ([5ed95bc](https://github.com/ibis-project/ibis/commit/5ed95bcbe15f9a6789be27ff36341bb0708394c4))
* don't sort new columns in mutate ([72ec96a](https://github.com/ibis-project/ibis/commit/72ec96a9fb5bb102d3a11bdb9a23135ae4edfc7f))
* **duckdb:** use lambda to define backend operations ([5d14de6](https://github.com/ibis-project/ibis/commit/5d14de6722eb34c6604a124f6f11cb711f16bd44))
* **impala:** move impala SQL tests to snapshots ([927bf65](https://github.com/ibis-project/ibis/commit/927bf65e0df199d774bd07f9fb31ad11449fc179))
* **impala:** replace custom pooling with sqlalchemy QueuePool ([626cdca](https://github.com/ibis-project/ibis/commit/626cdcaedd140f868c77731147608e9a7801c45c))
* **ir:** `ops.List` -> `ops.NodeList` ([6765bd2](https://github.com/ibis-project/ibis/commit/6765bd2996ab8c3b88890a8a429717e6679820e3))
* **ir:** better encapsulate graph traversal logic, schema and datatype objects are not traversable anymore ([1a07725](https://github.com/ibis-project/ibis/commit/1a07725fdb761a1519e97ee67b4d001b8820224b))
* **ir:** generalize handling and traversal of node sequences ([e8bcd0f](https://github.com/ibis-project/ibis/commit/e8bcd0f681ddd889ea4bdebf75ae8662cd87f689))
* **ir:** make all value operations 'Named' for more consistent naming semantics ([f1eb4d2](https://github.com/ibis-project/ibis/commit/f1eb4d2fc5e6e49b9920cb2d3d225f46add2f96b))
* **ir:** move random() to api.py ([e136f1b](https://github.com/ibis-project/ibis/commit/e136f1b4620cfc3e07a15b3285f1fc77e98bf712))
* **ir:** remove `ops.DeferredSortKey` ([e629633](https://github.com/ibis-project/ibis/commit/e629633130716038822106726a4e79526e08daa9))
* **ir:** remove `ops.TopKNode` and `ir.TopK` ([d4dc544](https://github.com/ibis-project/ibis/commit/d4dc54435836e80ee58c423f7fda626842bc95a0))
* **ir:** remove Analytic expression's unused type() method ([1864bc1](https://github.com/ibis-project/ibis/commit/1864bc140f68ef2bca59df032ccb2be2fe582064))
* **ir:** remove DecimalValue.precision(), DecimalValue.scale() method ([be975bc](https://github.com/ibis-project/ibis/commit/be975bcb1ee82fe010d98d8f06b7d0c48d99c2cc))
* **ir:** remove DestructValue expressions ([762d384](https://github.com/ibis-project/ibis/commit/762d3849e375daabb069c46850e87dd7a56cf899))
* **ir:** remove duplicated literal creation code ([7dfb56f](https://github.com/ibis-project/ibis/commit/7dfb56fa0f2395b928c7108e8f4dec493a2daa32))
* **ir:** remove intermediate expressions ([c6fb0c0](https://github.com/ibis-project/ibis/commit/c6fb0c040987f92d2b79f96556f0e7ecc70e87ec))
* **ir:** remove lin.lineage() since it's not used anywhere ([120b1d7](https://github.com/ibis-project/ibis/commit/120b1d784b3a1c3a741d1473845d4b859db26f38))
* **ir:** remove node.blocks() in favor of more explicit type handling ([37d8ce4](https://github.com/ibis-project/ibis/commit/37d8ce4c67bc39453fe9e37fa2ab03d5051a30ab))
* **ir:** remove Node.inputs since it is an implementation detail of the pandas backend ([6d2c49c](https://github.com/ibis-project/ibis/commit/6d2c49cccc82ba2e15a9a4b3bcec53ffdd1de638))
* **ir:** remove node.root_tables() and unify parent table handling ([fbb07c1](https://github.com/ibis-project/ibis/commit/fbb07c19cc7853131083477debffb35d0dc12b5f))
* **ir:** remove ops.AggregateSelection in favor of an.simplify_aggregation ([ecf6ed3](https://github.com/ibis-project/ibis/commit/ecf6ed33cbf110a48f659081d5c7f25be2a982b5))
* **ir:** remove ops.NodeList and ir.List in favor of builtin tuples ([a90ce35](https://github.com/ibis-project/ibis/commit/a90ce35d8a1e65154036e9a73dc5e74a58611ae6))
* **ir:** remove pydantic dependency and make grounds more composable ([9da0f41](https://github.com/ibis-project/ibis/commit/9da0f4195b6f567e390862788b76159c01effaa4))
* **ir:** remove sch.HasSchema and introduce ops.Projection base class for ops.Selection ([c3b0139](https://github.com/ibis-project/ibis/commit/c3b01398df5e41f3a2afb919a9ede34ec6cb8e73))
* **ir:** remove unnecessary complexity introduced by variadic annotation ([698314b](https://github.com/ibis-project/ibis/commit/698314b73ea23c247cd5559acf16d6d5118cdbf0))
* **ir:** resolve circular imports so operations can be globally imported for types ([d2a3919](https://github.com/ibis-project/ibis/commit/d2a3919da1907867f3f14033a3c46a55ebd7de99))
* **ir:** simplify analysis.substitute_unbound() ([a6c7406](https://github.com/ibis-project/ibis/commit/a6c740631c2be6222ac65a0338e609e80cf4b7af))
* **ir:** simplify SortKey construction using rules ([4d63280](https://github.com/ibis-project/ibis/commit/4d632803ef71af80226abd6592a2a44d9e07e577))
* **ir:** simplify switch-case builders ([9acf717](https://github.com/ibis-project/ibis/commit/9acf7178b08e7a31acddbe37422c5dcc56cc3fa7))
* **ir:** split datatypes package into multiple submodules ([cce6535](https://github.com/ibis-project/ibis/commit/cce6535a20a407622031f889f046ca1fddf31750))
* **ir:** split out table count into `CountStar` operation ([e812e6e](https://github.com/ibis-project/ibis/commit/e812e6e6fb98832854aab90d3051aae8e207633e))
* **ir:** support replacing nodes in the tree ([6a0df5a](https://github.com/ibis-project/ibis/commit/6a0df5a242f9c1e28f6b45df6c4e71941213a8ba))
* **ir:** support variadic annotable arguments and add generic graph traversal routines ([5d6a289](https://github.com/ibis-project/ibis/commit/5d6a289b5f7dd9f632af0d27de567cbd7ee59ff8))
* **ir:** unify aggregation construction to use AggregateSelection ([c7d6a6f](https://github.com/ibis-project/ibis/commit/c7d6a6ff9bf02d064330cfb7fc67ff0134e9c0d3))
* make `quantile`, `any`, and `all` reductions filterable ([1bafc9e](https://github.com/ibis-project/ibis/commit/1bafc9ed9ffec030689d2ebef4ccd04b0a1fa5dc))
* make sure `value_counts` always has a projection ([a70a302](https://github.com/ibis-project/ibis/commit/a70a302dc02c6cd133b492e2726a770579cdf61e))
* **mssql:** use lambda to define backend operations ([1437cfb](https://github.com/ibis-project/ibis/commit/1437cfba00226077f7686bd5f7d542944f0b6e81))
* **mysql:** dedup extract code ([d551944](https://github.com/ibis-project/ibis/commit/d551944fc8ccf95bf3a08e44645108957c971aa6))
* **mysql:** use lambda to define backend operations ([d10bff8](https://github.com/ibis-project/ibis/commit/d10bff826b3aeef15da9bd29e30e5a31172e0dbf))
* **polars:** match duckdb registration api ([ac59dac](https://github.com/ibis-project/ibis/commit/ac59dacb6963052ad9dfa2fa6631798be845b6d3))
* **postgres:** use lambda to define backend operations ([4c85d7b](https://github.com/ibis-project/ibis/commit/4c85d7bef84b4c7ad67aae29182cfb8df9db1bcd))
* remove dead `compat.py` module ([eda0fdb](https://github.com/ibis-project/ibis/commit/eda0fdb5c23ea67e599decba4c975f618672edf0))
* remove deprecated approximate aggregation classes ([53fc6cb](https://github.com/ibis-project/ibis/commit/53fc6cbffed821300273f2c616f35447d567efa9))
* remove deprecated functions and classes ([be1cdda](https://github.com/ibis-project/ibis/commit/be1cdda3051cfff8364bd4104ba296a7135f8040))
* remove duplicate `_random_identifier` calls ([26e7942](https://github.com/ibis-project/ibis/commit/26e7942a3de6463ef05340eb63f2199ab703a93a))
* remove setup.py and related infrastructure ([adfcce1](https://github.com/ibis-project/ibis/commit/adfcce127c2f6a91e14936ccd75992a6f082d629))
* remove the `JSONB` type ([c4fc0ec](https://github.com/ibis-project/ibis/commit/c4fc0ec3dbb5b2352df6cac2cfc8c5b98e174723))
* rename some infer methods for consistency ([a8f5579](https://github.com/ibis-project/ibis/commit/a8f557958e90ada1d6595b4ef6673339d068e957))
* replace isinstance dtype checking with `is_*` methods ([386adc2](https://github.com/ibis-project/ibis/commit/386adc248e118c194433596c8896e6004d7906b4))
* rework registration / file loading ([c60e30d](https://github.com/ibis-project/ibis/commit/c60e30d42466cef36e10e6bccb0b2131d6052e43))
* **rules:** generalize field referencing using rlz.ref() ([0afb8b9](https://github.com/ibis-project/ibis/commit/0afb8b91cd00a54e01eb40657da7b370415a1b57))
* simplify `ops.ArrayColumn` in postgres backend ([f9677cc](https://github.com/ibis-project/ibis/commit/f9677cc8e9c3fac29a2204acfd5d46e81b6e1684))
* simplify histogram implementation by using window functions ([41cbc29](https://github.com/ibis-project/ibis/commit/41cbc29ca74a7534fc1b2716b22f1c0e96591248))
* simplify ops.ArrayColumn in alchemy backend ([28ff4a8](https://github.com/ibis-project/ibis/commit/28ff4a8f38aff99cbf92f296bd77e882dc960d0f))
* **snowflake:** use lambda to define backend operations ([cb33fce](https://github.com/ibis-project/ibis/commit/cb33fce2929efae0ccaed63c6c3aa95eac8360d2))
* split up custom nix code; remove unused derivations ([57dff10](https://github.com/ibis-project/ibis/commit/57dff1073c8ef060a831df6176b78879a85b512e))
* **sqlite:** use lambda to define backend operations ([b937391](https://github.com/ibis-project/ibis/commit/b937391b0c9d2f929da5f54445679bffc7d9090e))
* **test:** make clickhouse tests use `pytest-snapshot` ([413dbd2](https://github.com/ibis-project/ibis/commit/413dbd29fe4c5dc95ca73d1fe6f634cbd50e0497))
* **tests:** move sql output to golden dir ([6a6a453](https://github.com/ibis-project/ibis/commit/6a6a45320ed4afdd54e99d1b397087accd421afb))
* **test:** sort regex test cases by name instead of posix-ness ([0dfb0e7](https://github.com/ibis-project/ibis/commit/0dfb0e752dfba417abf8abce74af96b3e8c950d6))
* **tests:** replace `sqlgolden` with `pytest-snapshot` ([5700eb0](https://github.com/ibis-project/ibis/commit/5700eb01320d48d73126f3c16d2ab6fe408326cd))
* **timestamps:** remove `timezone` argument to `to_timestamp` API ([eb4762e](https://github.com/ibis-project/ibis/commit/eb4762edc50a0df6718d6281f85fd0cc48107cbb))
* **trino:** use lambda to define backend operations ([dbd61a5](https://github.com/ibis-project/ibis/commit/dbd61a549db3b4be588245b30ffc0c14eb9fe224))
* uncouple `MultiQuantile` class from `Quantile` ([9c48f8c](https://github.com/ibis-project/ibis/commit/9c48f8c9ba34f910b83920277d09204d4968064d))
* use `rlz.lazy_instance_of` to delay shapely import ([d14badc](https://github.com/ibis-project/ibis/commit/d14badc9822174e2e38712b85b6e11583b0250fa))
* use lazy dispatch for `dt.infer` ([2e56540](https://github.com/ibis-project/ibis/commit/2e56540fa35ef1df04560140a5de1626cad5399c))


### Documentation

* add `backend_sensitive` decorator ([836f237](https://github.com/ibis-project/ibis/commit/836f23702bbb51a0efcc427fbf5d8df0910be3ad))
* add `pip install poetry` dev env setup step ([69940b1](https://github.com/ibis-project/ibis/commit/69940b17fa430acacc8ab3b12df59c990f287f3f))
* add bigquery ci data analysis notebook ([2b1d4e5](https://github.com/ibis-project/ibis/commit/2b1d4e52b1317098156d58a8ed994cfa040cee20))
* add how to sessionize guide ([18989dd](https://github.com/ibis-project/ibis/commit/18989dda75d4387ebe6a31ac38b3d78190947534))
* add issue templates ([4480c18](https://github.com/ibis-project/ibis/commit/4480c18ee162b8ee23c80435b8b5313216400e53))
* add missing argument descriptions ([ea757fa](https://github.com/ibis-project/ibis/commit/ea757fa9bf9d095127609c7a392dd8b758a8dc59))
* add mssql backend page ([63c0f19](https://github.com/ibis-project/ibis/commit/63c0f190f8110fa8434608a28f87bfc987e665a3))
* added 4.0 release blog post ([bcc0eca](https://github.com/ibis-project/ibis/commit/bcc0ecadd08702d3e48691e7706cb5f1ac33016f))
* added memtable howto guide ([5dde9bd](https://github.com/ibis-project/ibis/commit/5dde9bda663783eeca26d142076c492b433e6c27))
* **backends:** add duckdb and mssql to the backend index page ([7b13218](https://github.com/ibis-project/ibis/commit/7b13218ba21dd3de0e8168c898e827d33c9901c7))
* bring back git revision localized date plugin ([e4fc2c9](https://github.com/ibis-project/ibis/commit/e4fc2c99d2f5f514b9e18936c1eedd42e40302db))
* created how to guide for deferred expressions ([2a9f6ab](https://github.com/ibis-project/ibis/commit/2a9f6abdb14c7a20f8dd755ca7a0aa03229de815))
* **dev:** python-duckdb now available for windows with conda ([7f76b09](https://github.com/ibis-project/ibis/commit/7f76b098bf1fc14c98449fcd8799233c70e68c4c))
* document how to create a table from a pandas dataframe using ibis.memtable ([c6521ec](https://github.com/ibis-project/ibis/commit/c6521ec852b1d1a1055a01a4a7c8704170f1540f))
* fix backends label in feature request issue form ([cf852d3](https://github.com/ibis-project/ibis/commit/cf852d36136acf32747a529e89dee94c3812334f))
* fix broken docstrings; reduce docstring noise; workaround griffe ([bd1c637](https://github.com/ibis-project/ibis/commit/bd1c6371b4181ef487e3973632cf6401155219c9))
* fix docs for building docs ([23af567](https://github.com/ibis-project/ibis/commit/23af56782540c29a323ca62324005b113d1ba960))
* fix feature-request issue template ([6fb62f5](https://github.com/ibis-project/ibis/commit/6fb62f50ef352aa1cc16c2a3c13532fc17e47381))
* fix installation section for conda ([7af6ac1](https://github.com/ibis-project/ibis/commit/7af6ac11fd097284bcf07e654b6d0d4f664bca03))
* fix landing page links ([1879362](https://github.com/ibis-project/ibis/commit/1879362386afd7aae84e686a1e89bd52d955b368))
* fix links to make docs work locally and remotely ([13c7810](https://github.com/ibis-project/ibis/commit/13c7810e336bd11ae10b191ce16f667f79067571))
* fix pyarrow batches docstring ([dba9594](https://github.com/ibis-project/ibis/commit/dba95949083f595c5c55241d3511509eba1213a1))
* fix single line docstring summaries ([8028201](https://github.com/ibis-project/ibis/commit/80282017123552eca8cf05ea5515913aa3bd2040))
* fix snowflake doc link in readme.md ([9aff68e](https://github.com/ibis-project/ibis/commit/9aff68e3ae78a9979adfc4bd08566b480bbc0615))
* fix the inline example for ibis.dask.do_connect ([6a533f0](https://github.com/ibis-project/ibis/commit/6a533f06e8f863cad0cdf8c77ac5ca79792f1dc2))
* fix tutorial link on install page ([b34811a](https://github.com/ibis-project/ibis/commit/b34811af64b73b7f13346f9e8def4d78b668613f))
* fix typo in first example of the homepage ([9a8a25a](https://github.com/ibis-project/ibis/commit/9a8a25a7aa7b8ae8f429a6c410be7f12ea85b8cd))
* formatting and syntax highlighting fixes ([50864da](https://github.com/ibis-project/ibis/commit/50864da061e1cfe9edb0a3b9ebf875893730e4d0))
* front page rework ([24b795a](https://github.com/ibis-project/ibis/commit/24b795a8963f119b69348641e6bbd250c2e41ae2))
* **how-to:** use parquet data source for sessionization, fix typos, more deferred usage ([974be37](https://github.com/ibis-project/ibis/commit/974be371fcd70aac9d88f60970378392e9faa83f))
* improve the docstring of the generic connect method ([ee87802](https://github.com/ibis-project/ibis/commit/ee87802f6e546157c231d25fe9a86de1d7b40691))
* issue template cleanups ([fed37da](https://github.com/ibis-project/ibis/commit/fed37dac20171f989fa6cad4eecd67533d038c71))
* list ([e331247](https://github.com/ibis-project/ibis/commit/e33124702c41131cb15b29096792aa54bb19861a))
* **polars:** add backend docs page ([e303b68](https://github.com/ibis-project/ibis/commit/e303b6887431cd6c63f3c63b2aa8a494ecfb6108))
* remove hrs ([4c30de4](https://github.com/ibis-project/ibis/commit/4c30de41d01a4434e8165d0ba5ec2ffd94af9fad))
* renamed how to guides to be more consistent ([1bdc5bd](https://github.com/ibis-project/ibis/commit/1bdc5bd1f7a935d6c1c72c28fd4f872e3db83276))
* sentence structure in the Notes section ([ac20232](https://github.com/ibis-project/ibis/commit/ac202322e964d81daacf7526e116014993ae7061))
* show interactive prompt for python ([5d7d913](https://github.com/ibis-project/ibis/commit/5d7d9130e784a6d077c623e3ab6ada32c5069499))
* split out geospatial operations in the support matrix docs ([0075c28](https://github.com/ibis-project/ibis/commit/0075c28452f9e6cda28d74fdfe4cbfd97669945c))
* **trino:** add backend docs ([2f262cd](https://github.com/ibis-project/ibis/commit/2f262cd316e68c3d64f18dfefe6844b939d2d276))
* typo ([6bac645](https://github.com/ibis-project/ibis/commit/6bac64524133ad01ea00c5b4ca8784185249af79))
* typos headers and formatting ([9566cbb](https://github.com/ibis-project/ibis/commit/9566cbbf688ad028b1d8156a0205af016dfd8ef0))
* **udf:** examples in pandas have the incorrect import path ([49028b8](https://github.com/ibis-project/ibis/commit/49028b86d83447bf37290a6b00b677646beae356))
* update filename ([658a296](https://github.com/ibis-project/ibis/commit/658a2960b4f3ef1d6b6ca39336f06d77e5d25e39))
* update line ([4edfce0](https://github.com/ibis-project/ibis/commit/4edfce0d56827a6d2d48fe2f27f03dba8dda92cb))
* update readme ([19a3f3c](https://github.com/ibis-project/ibis/commit/19a3f3c84c04ed7a44e09f82cdc5ab62c6b106b8))
* use buf/feat prefix only ([2561a29](https://github.com/ibis-project/ibis/commit/2561a29dc689989c987e494518c456e458650580))
* use components instead of pieces ([179ca1e](https://github.com/ibis-project/ibis/commit/179ca1e05258fda1424da4f271fed60230c85b3a))
* use heading instead of bulleted bold ([99b044e](https://github.com/ibis-project/ibis/commit/99b044e8ec9ad5e131faa9ed69eaee3a6b909c91))
* use library instead of project ([fd2d915](https://github.com/ibis-project/ibis/commit/fd2d915e2fc358ba3b42ad9a37523b5e080c3586))
* use present tense for use cases and "why" section ([6cc7416](https://github.com/ibis-project/ibis/commit/6cc7416d8c1623400b399469192a28d64642bdd7))
* **www:** fix frontpage example ([7db39e8](https://github.com/ibis-project/ibis/commit/7db39e81d1f0ee613c9694f1bdac18f1ba9a6acf))

## [3.2.0](https://github.com/ibis-project/ibis/compare/3.1.0...3.2.0) (2022-09-15)


### Features

* add api to get backend entry points ([0152f5e](https://github.com/ibis-project/ibis/commit/0152f5e4a608c531de146a6aa6df8087d8d4c182))
* **api:** add `and_` and `or_` helpers ([94bd4df](https://github.com/ibis-project/ibis/commit/94bd4df81d73f8c894eb5b72e747b7a79cdf14f6))
* **api:** add `argmax` and `argmin` column methods ([b52216a](https://github.com/ibis-project/ibis/commit/b52216ac96167781878a99cf285bbd1afb106fb3))
* **api:** add `distinct` to `Intersection` and `Difference` operations ([cd9a34c](https://github.com/ibis-project/ibis/commit/cd9a34ce7643fcdaf1e8f8166b968fd8a175308a))
* **api:** add `ibis.memtable` API for constructing in-memory table expressions ([0cc6948](https://github.com/ibis-project/ibis/commit/0cc694862154d0ae62b92151b8e08f2a0a8e34b9))
* **api:** add `ibis.sql` to easily get a formatted SQL string ([d971cc3](https://github.com/ibis-project/ibis/commit/d971cc397b10e9e3b2d3388cb4827400a29bc615))
* **api:** add `Table.unpack()` and `StructValue.lift()` APIs for projecting struct fields ([ced5f53](https://github.com/ibis-project/ibis/commit/ced5f539ef00192dbcdcd495e5546c523aa62846))
* **api:** allow transmute-style select method ([d5fc364](https://github.com/ibis-project/ibis/commit/d5fc3643e26950ecfeaf46768aca357dfd0223c3))
* **api:** implement all bitwise operators ([7fc5073](https://github.com/ibis-project/ibis/commit/7fc507348f5edc7bc665560e191522e2979c0c18))
* **api:** promote `psql` to a `show_sql` public API ([877a05d](https://github.com/ibis-project/ibis/commit/877a05d89347bf61439ef2122d62da7ddaf67312))
* **clickhouse:** add dataframe external table support for memtables ([bc86aa7](https://github.com/ibis-project/ibis/commit/bc86aa7bd703c85c7bf8cfa1d51efc9e90d1f9ea))
* **clickhouse:** add enum, ipaddr, json, lowcardinality to type parser ([8f0287f](https://github.com/ibis-project/ibis/commit/8f0287f4462d697b295456c643ecba5092c19332))
* **clickhouse:** enable support for working window functions ([310a5a8](https://github.com/ibis-project/ibis/commit/310a5a8a430faccb347883fab7dc43019af9bce4))
* **clickhouse:** implement `argmin` and `argmax` ([ee7c878](https://github.com/ibis-project/ibis/commit/ee7c87830364b00995fce82bd320280476237375))
* **clickhouse:** implement bitwise operations ([348cd08](https://github.com/ibis-project/ibis/commit/348cd0892b987508505e0180423ff9c0fb29f502))
* **clickhouse:** implement struct scalars ([1f3efe9](https://github.com/ibis-project/ibis/commit/1f3efe90de976e569152699bb156f948e54e21b6))
* **dask:** implement `StringReplace` execution ([1389f4b](https://github.com/ibis-project/ibis/commit/1389f4bc31ac3b497ea40d276f3e2d19b32974df))
* **dask:** implement ungrouped `argmin` and `argmax` ([854aea7](https://github.com/ibis-project/ibis/commit/854aea78cd474801e543f98942a110a8571e56b8))
* **deps:** support duckdb 0.5.0 ([47165b2](https://github.com/ibis-project/ibis/commit/47165b233b1b2764009bf39e30ca7b59b1266b70))
* **duckdb:** handle query parameters in `ibis.connect` ([fbde95d](https://github.com/ibis-project/ibis/commit/fbde95dcab7b67af9cc9796326df8367d5b3e778))
* **duckdb:** implement `argmin` and `argmax` ([abf03f1](https://github.com/ibis-project/ibis/commit/abf03f108570d1f3e06373dcaaa50fa29abae209))
* **duckdb:** implement bitwise xor ([ca3abed](https://github.com/ibis-project/ibis/commit/ca3abedccbcb69ab682b1f27932f06ab465f11cc))
* **duckdb:** register tables from pandas/pyarrow objects ([36e48cc](https://github.com/ibis-project/ibis/commit/36e48cc7c4107123d0bca9f0f663e17bc02d91b8))
* **duckdb:** support unsigned integer types ([2e67918](https://github.com/ibis-project/ibis/commit/2e6791898ecfec88c3874e9c431d8d0b01395dd2))
* **impala:** implement bitwise operations ([c5302ab](https://github.com/ibis-project/ibis/commit/c5302ab8eef5a29c541dbb08c780f8c09f41102f))
* implement dropna for SQL backends ([8a747fb](https://github.com/ibis-project/ibis/commit/8a747fb79b71fba44f6f524c3d32f06ca1d66152))
* **log:** make BaseSQLBackend._log print by default ([12de5bb](https://github.com/ibis-project/ibis/commit/12de5bbee7d823e0a0f47a3bdc153eb2ed3d0b45))
* **mysql:** register BLOB types ([1e4fb92](https://github.com/ibis-project/ibis/commit/1e4fb92b09471e221fc7387af16526008afd855a))
* **pandas:** implement `argmin` and `argmax` ([bf9b948](https://github.com/ibis-project/ibis/commit/bf9b94846ee3a0b5de386f164b22ce9712944985))
* **pandas:** implement `NotContains` on grouped data ([976dce7](https://github.com/ibis-project/ibis/commit/976dce73a02b7bc8dc0222ffef52532aaafb8744))
* **pandas:** implement `StringReplace` execution ([578795f](https://github.com/ibis-project/ibis/commit/578795ff3871a47c46d52876091b58c844485d58))
* **pandas:** implement Contains with a group by ([c534848](https://github.com/ibis-project/ibis/commit/c53484894ff631e5929b065498c84222b785db4d))
* **postgres:** implement bitwise xor ([9b1ebf5](https://github.com/ibis-project/ibis/commit/9b1ebf50ba329223460839c8bc1daf08a5a02452))
* **pyspark:** add option to treat nan as null in aggregations ([bf47250](https://github.com/ibis-project/ibis/commit/bf472502ce08328124520d7e386f8d60bd8b04c9))
* **pyspark:** implement `ibis.connect` for pyspark ([a191744](https://github.com/ibis-project/ibis/commit/a191744047bcde4e4479ede3f63995d5925ba6ed))
* **pyspark:** implement `Intersection` and `Difference` ([9845a3c](https://github.com/ibis-project/ibis/commit/9845a3c115138aafeb324dfa72efaf8ea88c1535))
* **pyspark:** implement bitwise operators ([33cadb1](https://github.com/ibis-project/ibis/commit/33cadb1275bc3561de1bcb54bc7a628b341b4978))
* **sqlalchemy:** implement bitwise operator translation ([bd9f64c](https://github.com/ibis-project/ibis/commit/bd9f64cc34f5b8e27c17b6b992bf8acc2486e09c))
* **sqlalchemy:** make `ibis.connect` with sqlalchemy backends ([b6cefb9](https://github.com/ibis-project/ibis/commit/b6cefb9a2a47c6865d7ed1b3e98442f3b1be7f94))
* **sqlalchemy:** properly implement `Intersection` and `Difference` ([2bc0b69](https://github.com/ibis-project/ibis/commit/2bc0b697e611dfd13fa6c9e8dae891bce72bdeef))
* **sql:** implement `StringReplace` translation ([29daa32](https://github.com/ibis-project/ibis/commit/29daa32ad8c9709e44fc22e3bce580c41aff9547))
* **sqlite:** implement bitwise xor and bitwise not ([58c42f9](https://github.com/ibis-project/ibis/commit/58c42f99cc9c99aa3bed88f6d0840fa4be8755b5))
* support `table.sort_by(ibis.random())` ([693005d](https://github.com/ibis-project/ibis/commit/693005d5046eaace37b8c6d6d7d09104de99fe9b))
* **type-system:** infer pandas' string dtype ([5f0eb5d](https://github.com/ibis-project/ibis/commit/5f0eb5d09e6debfe04bb30b2ffea75f8cbd42e7a))
* **ux:** add duckdb as the default backend ([8ccb81d](https://github.com/ibis-project/ibis/commit/8ccb81d49e252b57310bdb3a97eeb77ef1d28bac))
* **ux:** use `rich` to format `Table.info()` output ([67234c3](https://github.com/ibis-project/ibis/commit/67234c3a25927e4e457191e81010c2a8f8a3c7e5))
* **ux:** use `sqlglot` for pretty printing SQL ([a3c81c5](https://github.com/ibis-project/ibis/commit/a3c81c5bc95c5c87c168551ef4eb3efe084c8cf3))
* variadic union, intersect, & difference functions ([05aca5a](https://github.com/ibis-project/ibis/commit/05aca5a699c725fa167ff0fa020752ecd4b7547b))


### Bug Fixes

* **api:** make sure column names that are already inferred are not overwritten ([6f1cb16](https://github.com/ibis-project/ibis/commit/6f1cb1602a237e8cdd3c98572aa59bb10e872b39))
* **api:** support deferred objects in existing API functions ([241ce6a](https://github.com/ibis-project/ibis/commit/241ce6aeab639c73b147e4744800bf19f3e884e3))
* **backend:** ensure that chained limits respect prior limits ([02a04f5](https://github.com/ibis-project/ibis/commit/02a04f5f36bfad3ea525db727bcaaf4a49b0e243))
* **backends:** ensure select after filter works ([e58ca73](https://github.com/ibis-project/ibis/commit/e58ca7323a4ee57d9e9cfcc2ebce12c5502ba749))
* **backends:** only recommend installing ibis-foo when foo is a known backend ([ac6974a](https://github.com/ibis-project/ibis/commit/ac6974a66b9492960981db4a9471b8ffb24786f1))
* **base-sql:** fix String-generating backend string concat implementation ([3cf78c1](https://github.com/ibis-project/ibis/commit/3cf78c190990c289e61712eb65856990f52ba003))
* **clickhouse:** add IPv4/IPv6 literal inference ([0a2f315](https://github.com/ibis-project/ibis/commit/0a2f3150723c31e2b8df120ecdbd8fe92db48ab0))
* **clickhouse:** cast repeat `times` argument to `UInt64` ([b643544](https://github.com/ibis-project/ibis/commit/b6435444832b0d5cc57b4473ba3bbbb99537b5da))
* **clickhouse:** fix listing tables from databases with no tables ([08900c3](https://github.com/ibis-project/ibis/commit/08900c33df9045dc7c6602626a776f89860342ed))
* **compilers:** make sure memtable rows have names in the SQL string compilers ([18e7f95](https://github.com/ibis-project/ibis/commit/18e7f9555c3bf8c602ebcaef0188e03d7b0bd7ee))
* **compiler:** use `repr` for SQL string `VALUES` data ([75af658](https://github.com/ibis-project/ibis/commit/75af658df785529e65bf2fdeca472d3a71d1ff09))
* **dask:** ensure predicates are computed before projections ([5cd70e1](https://github.com/ibis-project/ibis/commit/5cd70e12f10eb6847a3553c8e31263ae31fd47f8))
* **dask:** implement timestamp-date binary comparisons ([48d5058](https://github.com/ibis-project/ibis/commit/48d5058cc83d006ed381b9256363fcd62606ea49))
* **dask:** set dask upper bound due to large scale test breakage ([796c645](https://github.com/ibis-project/ibis/commit/796c645b74a67be9f4b40b32ef0ab9b0dfc27556)), closes [#9221](https://github.com/ibis-project/ibis/issues/9221)
* **decimal:** add decimal type inference ([3fe3fd8](https://github.com/ibis-project/ibis/commit/3fe3fd8927da565d7e85958566e8f87b600fc748))
* **deps:** update dependency duckdb-engine to >=0.1.8,<0.4.0 ([113dc8f](https://github.com/ibis-project/ibis/commit/113dc8f667ccdf3ac1897ec2a1eef41ebd41463f))
* **deps:** update dependency duckdb-engine to >=0.1.8,<0.5.0 ([ef97c9d](https://github.com/ibis-project/ibis/commit/ef97c9d948b1c80fb3cd9cb49745bc9d5fef9da2))
* **deps:** update dependency parsy to v2 ([9a06131](https://github.com/ibis-project/ibis/commit/9a061310873c9aa3773b38d93ef724276daa5705))
* **deps:** update dependency shapely to >=1.6,<1.8.4 ([0c787d2](https://github.com/ibis-project/ibis/commit/0c787d25e7f1c302fcc83b137ae1875c2578f4f2))
* **deps:** update dependency shapely to >=1.6,<1.8.5 ([d08c737](https://github.com/ibis-project/ibis/commit/d08c737e90b20fe76b2abc0f46940b8868f29071))
* **deps:** update dependency sqlglot to v5 ([f210bb8](https://github.com/ibis-project/ibis/commit/f210bb8843214da7d790afb03817382cc88c8c19))
* **deps:** update dependency sqlglot to v6 ([5ca4533](https://github.com/ibis-project/ibis/commit/5ca4533b5fe63eb76a470d89bdb82af73f83135e))
* **duckdb:** add missing types ([59bad07](https://github.com/ibis-project/ibis/commit/59bad077d61ac3266b9b6695cc9b3fb559858592))
* **duckdb:** ensure that in-memory connections remain in their creating thread ([39bc537](https://github.com/ibis-project/ibis/commit/39bc537d6d5684f02e77f246e6a751392663e039))
* **duckdb:** use `fetch_arrow_table()` to be able to handle big timestamps ([85a76eb](https://github.com/ibis-project/ibis/commit/85a76eb558ad76a83a407c0e0b1bcede0bbffa8b))
* fix bug in pandas & dask `difference` implementation ([88a78fa](https://github.com/ibis-project/ibis/commit/88a78faffe8db2fc04cd9dd74bf699e0cd1c4aac))
* fix dask `where` implementation ([49f8845](https://github.com/ibis-project/ibis/commit/49f88452e45e8b0159a49c32d5ddeea5ab78da3c))
* **impala:** add date column dtype to impala to ibis type dict ([c59e94e](https://github.com/ibis-project/ibis/commit/c59e94e2ae95a9c61830c1208be015f74408e667)), closes [#4449](https://github.com/ibis-project/ibis/issues/4449)
* pandas where supports scalar for `left` ([48f6c1e](https://github.com/ibis-project/ibis/commit/48f6c1eff2d8d1d5440925f7ca5736618c3e4522))
* **pandas:** fix anti-joins ([10a659d](https://github.com/ibis-project/ibis/commit/10a659d66145403c872ec176b8a739613f0061e7))
* **pandas:** implement timestamp-date binary comparisons ([4fc666d](https://github.com/ibis-project/ibis/commit/4fc666d18b384d0ebda0120e9a099ed164f3b0bb))
* **pandas:** properly handle empty groups when aggregating with `GroupConcat` ([6545f4d](https://github.com/ibis-project/ibis/commit/6545f4dbf41ba5f99666b365bcc5dfe445ece8e7))
* **pyspark:** fix broken `StringReplace` implementation ([22cb297](https://github.com/ibis-project/ibis/commit/22cb297df9d43b2859d8f59783da6c3e84749d58))
* **pyspark:** make sure `ibis.connect` works with pyspark ([a7ab107](https://github.com/ibis-project/ibis/commit/a7ab107200effd7a1f71fd2ae0af33ae2845232f))
* **pyspark:** translate predicates before projections ([b3d1c80](https://github.com/ibis-project/ibis/commit/b3d1c80c4ff23ebde21b009a9d744f775d61aba7))
* **sqlalchemy:** fix float64 type mapping ([8782773](https://github.com/ibis-project/ibis/commit/87827735ca26fd0c843246218a4ff11db8745429))
* **sqlalchemy:** handle reductions with multiple arguments ([5b2039b](https://github.com/ibis-project/ibis/commit/5b2039b17ee28c36714ae5958cc33261a78debeb))
* **sqlalchemy:** implement `SQLQueryResult` translation ([786a50f](https://github.com/ibis-project/ibis/commit/786a50f84b4a0ca3f07fb650f42fbb7cfdeb7c51))
* **sql:** fix sql compilation after making `InMemoryTable` a subclass of `PhysicalTable` ([aac9524](https://github.com/ibis-project/ibis/commit/aac9524b691c74d2681da5c0e72ad9913c13e4a0))
* squash several bugs in `sort_by` asc/desc handling ([222b2ba](https://github.com/ibis-project/ibis/commit/222b2bad2d1bb4ad87c68844a6d1bb5a2f91293a))
* support chained set operations in SQL backends ([227aed3](https://github.com/ibis-project/ibis/commit/227aed33188c21148e0ae70dfa3c15e3bb8a6e25))
* support filters on InMemoryTable exprs ([abfaf1f](https://github.com/ibis-project/ibis/commit/abfaf1fd7b65bcdc1fd4e2bd2abcd3a5a8ec09da))
* **typo:** in BaseSQLBackend.compile docstring ([0561b13](https://github.com/ibis-project/ibis/commit/0561b130b2851cb6b8fc493e129d1992c9831504))


### Deprecations

* `right` kwarg in `union`/`intersect`/`difference` ([719a5a1](https://github.com/ibis-project/ibis/commit/719a5a1689ad94b8a0514cc6bb11ed81387e51a9))
* **duckdb:** deprecate `path` argument in favor of `database` ([fcacc20](https://github.com/ibis-project/ibis/commit/fcacc203e066297ce57d2cb2c7e6f791094c513a))
* **sqlite:** deprecate `path` argument in favor of `database` ([0f85919](https://github.com/ibis-project/ibis/commit/0f8591966dd0c98018e218e93faeefb16f5f927f))


### Performance

* **pandas:** remove reexecution of alias children ([64efa53](https://github.com/ibis-project/ibis/commit/64efa5376b9b96e23666242dad618512d886a124))
* **pyspark:** ensure that pyspark DDL doesn't use `VALUES` ([422c98d](https://github.com/ibis-project/ibis/commit/422c98db7584bffda407fd393037a2c921437ccf))
* **sqlalchemy:** register DataFrames cheaply where possible ([ee9f1be](https://github.com/ibis-project/ibis/commit/ee9f1be10073bc7789c2519314a5229e7163c88d))


### Documentation

* add `to_sql` ([e2821a5](https://github.com/ibis-project/ibis/commit/e2821a56c7d867b8b591f1777019843a2ffca797))
* add back constraints for transitive doc dependencies and fix docs ([350fd43](https://github.com/ibis-project/ibis/commit/350fd43e8777636230901cd928c76793110635be))
* add coc reporting information ([c2355ba](https://github.com/ibis-project/ibis/commit/c2355ba1c0532ff0a5eddd1267107d3554fa31d3))
* add community guidelines documentation ([fd0893f](https://github.com/ibis-project/ibis/commit/fd0893f3895345d754b5f324d42e320995ceba83))
* add HeavyAI to the readme ([4c5ca80](https://github.com/ibis-project/ibis/commit/4c5ca80a91b218bb1f9e61bef58f5cf120ece0aa))
* add how-to bfill and ffill ([ff84027](https://github.com/ibis-project/ibis/commit/ff840274572bc825caf3905b404c2397a67b3d10))
* add how-to for ibis+duckdb register ([73a726e](https://github.com/ibis-project/ibis/commit/73a726e0d5380951a1c3a0d5fb6b1119a392c307))
* add how-to section to docs ([33c4b93](https://github.com/ibis-project/ibis/commit/33c4b9393c432b2603309bb79f43316175a27b63))
* **duckdb:** add installation note for duckdb >= 0.5.0 ([608b1fb](https://github.com/ibis-project/ibis/commit/608b1fb7e6b63330a46136aca162037e34e9b521))
* fix `memtable` docstrings ([72bc0f5](https://github.com/ibis-project/ibis/commit/72bc0f5172c0a3d17bde29cfc00db4c60d2fee3a))
* fix flake8 line length issues ([fb7af75](https://github.com/ibis-project/ibis/commit/fb7af7587492e13d24ab52b9f29581456aaed966))
* fix markdown ([4ab6b95](https://github.com/ibis-project/ibis/commit/4ab6b950adf703db6db4c7772f70280bc6f626f4))
* fix relative links in tutorial ([2bd075f](https://github.com/ibis-project/ibis/commit/2bd075fdd20729d6e6e1d117b38a24ec23fa7d0f)), closes [#4064](https://github.com/ibis-project/ibis/issues/4064) [#4201](https://github.com/ibis-project/ibis/issues/4201)
* make attribution style uniform across the blog ([05561e0](https://github.com/ibis-project/ibis/commit/05561e06454f503fcbd7945421f5c1269f7d1815))
* move the blog out to the top level sidebar for visibility ([417ba64](https://github.com/ibis-project/ibis/commit/417ba64074402436fc479b5a67761d74bb46357d))
* remove underspecified UDF doc page ([0eb0ac0](https://github.com/ibis-project/ibis/commit/0eb0ac09f0ea9af973a80de4bc46f28a8b04e5db))

## [3.1.0](https://github.com/ibis-project/ibis/compare/3.0.2...3.1.0) (2022-07-26)


### Features

* add `__getattr__` support to `StructValue` ([75bded1](https://github.com/ibis-project/ibis/commit/75bded1897fa4f905ee31334b434a0f088cb7ebd))
* allow selection subclasses to define new node args ([2a7dc41](https://github.com/ibis-project/ibis/commit/2a7dc4106f9956b92aa8db301577e453e44c0ada))
* **api:** accept `Schema` objects in public `ibis.schema` ([0daac6c](https://github.com/ibis-project/ibis/commit/0daac6c73baf94541128be8413dc5f7e5309159b))
* **api:** add `.tables` accessor to `BaseBackend` ([7ad27f0](https://github.com/ibis-project/ibis/commit/7ad27f0a89e873554187d0a194f16c71c0429012))
* **api:** add `e` function to public API ([3a07e70](https://github.com/ibis-project/ibis/commit/3a07e7080c4946012f62f7c7dfb5e627bbd4f369))
* **api:** add `ops.StructColumn` operation ([020bfdb](https://github.com/ibis-project/ibis/commit/020bfdb50bebcb2aae4cfae75972d5208ac11d7d))
* **api:** add cume_dist operation ([6b6b185](https://github.com/ibis-project/ibis/commit/6b6b1852e1c56217f55c1b5ce91075bb22a54b14))
* **api:** add toplevel ibis.connect() ([e13946b](https://github.com/ibis-project/ibis/commit/e13946b620775a095519038a89206cd1834925c7))
* **api:** handle literal timestamps with timezone embedded in string ([1ae976b](https://github.com/ibis-project/ibis/commit/1ae976b3028ed4ea09b16c94e4f5942a54e902ea))
* **api:** ibis.connect() default to duckdb for parquet/csv extensions ([ff2f088](https://github.com/ibis-project/ibis/commit/ff2f08899f38aad74ecad522813e72e53fcd4099))
* **api:** make struct metadata more convenient to access ([3fd9bd8](https://github.com/ibis-project/ibis/commit/3fd9bd8b925602b9213af9ec5add5cda94ee0635))
* **api:** support tab completion for backends ([eb75fc5](https://github.com/ibis-project/ibis/commit/eb75fc57d614e5204841d0e3763c385204f95f5d))
* **api:** underscore convenience api ([81716da](https://github.com/ibis-project/ibis/commit/81716dacc8cf856fbffa2fa633955d964551c7d0))
* **api:** unnest ([98ecb09](https://github.com/ibis-project/ibis/commit/98ecb09ef90b596903a1db0ffff22229ce3a3776))
* **backends:** allow column expressions from non-foreign tables on the right side of `isin`/`notin` ([e1374a4](https://github.com/ibis-project/ibis/commit/e1374a4fc98ea8a4b7e0df6e6599bcde94c197b1))
* **base-sql:** implement trig and math functions ([addb2c1](https://github.com/ibis-project/ibis/commit/addb2c116b44727bb3c1ba07e91be1f82a4d4e80))
* **clickhouse:** add ability to pass arbitrary kwargs to Clickhouse do_connect ([583f599](https://github.com/ibis-project/ibis/commit/583f59965a84c6c00090385b5243d017bcf339f4))
* **clickhouse:** implement `ops.StructColumn` operation ([0063007](https://github.com/ibis-project/ibis/commit/0063007ee48066e07a264e397d41e004701dae06))
* **clickhouse:** implement array collect ([8b2577d](https://github.com/ibis-project/ibis/commit/8b2577dfed432eaa658f7961af6f85714e64d595))
* **clickhouse:** implement ArrayColumn ([1301f18](https://github.com/ibis-project/ibis/commit/1301f18dd87ffe36701f6daabca7a6f0d66c97bf))
* **clickhouse:** implement bit aggs ([f94a5d2](https://github.com/ibis-project/ibis/commit/f94a5d2c23797f5c0cb71c1edd08f1e2931eb0f4))
* **clickhouse:** implement clip ([12dfe50](https://github.com/ibis-project/ibis/commit/12dfe50730de560d79709a4ef2def74dfa134c83))
* **clickhouse:** implement covariance and correlation ([a37c155](https://github.com/ibis-project/ibis/commit/a37c155f83e0a3398ca979f35d6a8993e0f6b5c6))
* **clickhouse:** implement degrees ([7946c0f](https://github.com/ibis-project/ibis/commit/7946c0f23e6e1266ab95b96ffd0e90f4b0e5c788))
* **clickhouse:** implement proper type serialization ([80f4ab9](https://github.com/ibis-project/ibis/commit/80f4ab9908bba36b6ddca10809ef498c967c7401))
* **clickhouse:** implement radians ([c7b7f08](https://github.com/ibis-project/ibis/commit/c7b7f085f4319cc8051fd7b5a5f87a1b0b1e8c9d))
* **clickhouse:** implement strftime ([222f2b5](https://github.com/ibis-project/ibis/commit/222f2b55b85ceb1856159b56285b900db867f00d))
* **clickhouse:** implement struct field access ([fff69f3](https://github.com/ibis-project/ibis/commit/fff69f32276c30280b7fd17d7457940514293c00))
* **clickhouse:** implement trig and math functions ([c56440a](https://github.com/ibis-project/ibis/commit/c56440a6fab3b1f80e1a2488cd50058fc5fe3612))
* **clickhouse:** support subsecond timestamp literals ([e8698a6](https://github.com/ibis-project/ibis/commit/e8698a6d39451cf21aee4272f9c4ab79b85af4a9))
* **compiler:** restore `intersect_class` and `difference_class` overrides in base SQL backend ([2c46a15](https://github.com/ibis-project/ibis/commit/2c46a158e35d8b0e5d8aed372b3bb8328efad1b3))
* **dask:** implement trig functions ([e4086bb](https://github.com/ibis-project/ibis/commit/e4086bbe474f64dbb3427f1a4f32e3283661abe2))
* **dask:** implement zeroifnull ([38487db](https://github.com/ibis-project/ibis/commit/38487db2587d0decce42793ca46f53801c907eb7))
* **datafusion:** implement negate ([69dd64d](https://github.com/ibis-project/ibis/commit/69dd64d5c6d700e8f59715cfaafe997845bb6ff8))
* **datafusion:** implement trig functions ([16803e1](https://github.com/ibis-project/ibis/commit/16803e10cacfac3b2d2667731d50b040f92ad90a))
* **duckdb:** add register method to duckdb backend to load parquet and csv files ([4ccc6fc](https://github.com/ibis-project/ibis/commit/4ccc6fc9107a3745c358fc7944b801635a49a3e1))
* **duckdb:** enable find_in_set test ([377023d](https://github.com/ibis-project/ibis/commit/377023d390b96492508405fcc0fcb172680015c6))
* **duckdb:** enable group_concat test ([4b9ad6c](https://github.com/ibis-project/ibis/commit/4b9ad6c69ee9fd8bda0a455285895417a9c8aaaf))
* **duckdb:** implement `ops.StructColumn` operation ([211bfab](https://github.com/ibis-project/ibis/commit/211bfab697d6945812a36474baee6187440c5a48))
* **duckdb:** implement approx_count_distinct ([03c89ad](https://github.com/ibis-project/ibis/commit/03c89adaeee1743cfd17bb8dd65f4418cf8fa8b0))
* **duckdb:** implement approx_median ([894ce90](https://github.com/ibis-project/ibis/commit/894ce90d01df72c2fdb8c24af5b2b9c4bad8412a))
* **duckdb:** implement arbitrary first and last aggregation ([8a500bc](https://github.com/ibis-project/ibis/commit/8a500bc61c7259de6101715d5b328deda077a543))
* **duckdb:** implement NthValue ([1bf2842](https://github.com/ibis-project/ibis/commit/1bf2842dca1d595d80f6cd9f56d84cc3872c8e17))
* **duckdb:** implement strftime ([aebc252](https://github.com/ibis-project/ibis/commit/aebc2522da46978f2546e392eb6875a3fd0ee887))
* **duckdb:** return the `ir.Table` instance from DuckDB's `register` API ([0d05d41](https://github.com/ibis-project/ibis/commit/0d05d411d9da7e70f99203874f4f99252f7a381b))
* **mysql:** implement FindInSet ([e55bbbf](https://github.com/ibis-project/ibis/commit/e55bbbf6292babf460bd88cc5ca160acfada9002))
* **mysql:** implement StringToTimestamp ([169250f](https://github.com/ibis-project/ibis/commit/169250f670c74ac531c1df05a9e79c4625cae169))
* **pandas:** implement bitwise aggregations ([37ff328](https://github.com/ibis-project/ibis/commit/37ff328e30e38d69f1f2da6847b8c8b3b3c583e0))
* **pandas:** implement degrees ([25b4f69](https://github.com/ibis-project/ibis/commit/25b4f69883099f2ad161faefa861a40ca7460cd4))
* **pandas:** implement radians ([6816b75](https://github.com/ibis-project/ibis/commit/6816b75cf065668d0a8b30be475ad1bf07a6081b))
* **pandas:** implement trig functions ([1fd52d2](https://github.com/ibis-project/ibis/commit/1fd52d2ec472b80c1d836d319ed1a4c09d64d0da))
* **pandas:** implement zeroifnull ([48e8ed1](https://github.com/ibis-project/ibis/commit/48e8ed10c7062bedb351585785991980406b2eda))
* **postgres/duckdb:** implement covariance and correlation ([464d3ef](https://github.com/ibis-project/ibis/commit/464d3efde078d84b036cc3dfb2d2d0263a402e53))
* **postgres:** implement ArrayColumn ([7b0a506](https://github.com/ibis-project/ibis/commit/7b0a506b86d319ed433cdffaba42d3e487a15e46))
* **pyspark:** implement approx_count_distinct ([1fe1d75](https://github.com/ibis-project/ibis/commit/1fe1d75e9e56161c69c5df998424e05549dd7fbc))
* **pyspark:** implement approx_median ([07571a9](https://github.com/ibis-project/ibis/commit/07571a9ee2e5885bd32e8a08ba67a040eea71f6d))
* **pyspark:** implement covariance and correlation ([ae818fb](https://github.com/ibis-project/ibis/commit/ae818fb71b8f25685d79bb284e1709d65e739191))
* **pyspark:** implement degrees ([f478c7c](https://github.com/ibis-project/ibis/commit/f478c7caa6ddb5069d188cc391d42498ad7e7b95))
* **pyspark:** implement nth_value ([abb559d](https://github.com/ibis-project/ibis/commit/abb559d349293bf8e9f153bc1bc5de0e93cd8659))
* **pyspark:** implement nullifzero ([640234b](https://github.com/ibis-project/ibis/commit/640234b95d0cd43d69508f2c968253474f00f118))
* **pyspark:** implement radians ([18843c0](https://github.com/ibis-project/ibis/commit/18843c029dc4256a3db382184da4287644448a1b))
* **pyspark:** implement trig functions ([fd7621a](https://github.com/ibis-project/ibis/commit/fd7621aa16d1ffd428ff7d7fd99ce103f728ec33))
* **pyspark:** implement Where ([32b9abb](https://github.com/ibis-project/ibis/commit/32b9abb3f4abba1d93ecfd41ab09658d4a347d61))
* **pyspark:** implement xor ([550b35b](https://github.com/ibis-project/ibis/commit/550b35bc91c5e4c77f91d8e6f929055f20de13f4))
* **pyspark:** implement zeroifnull ([db13241](https://github.com/ibis-project/ibis/commit/db132418171f66e2902ac921e78c4bc1dfa99668))
* **pyspark:** topk support ([9344591](https://github.com/ibis-project/ibis/commit/9344591b2611950eea3e8151c03a12a11d38285d))
* **sqlalchemy:** add degrees and radians ([8b7415f](https://github.com/ibis-project/ibis/commit/8b7415f3d9b709b426b058b6231c2e22e62b0305))
* **sqlalchemy:** add xor translation rule ([2921664](https://github.com/ibis-project/ibis/commit/2921664370178781d3783f6c2839426c25ee3914))
* **sqlalchemy:** allow non-primitive arrays ([4e02918](https://github.com/ibis-project/ibis/commit/4e0291848d7239bfa92460c41b889be26734d5e5))
* **sqlalchemy:** implement approx_count_distinct as count distinct ([4e8bcab](https://github.com/ibis-project/ibis/commit/4e8bcabebbff735b5ebeb54331aff2d6d20ff0da))
* **sqlalchemy:** implement clip ([8c02639](https://github.com/ibis-project/ibis/commit/8c026390abe7a22742c538d6011a1db8bc2d62c9))
* **sqlalchemy:** implement trig functions ([34c1514](https://github.com/ibis-project/ibis/commit/34c151433c9d21ae320a66fb3f00044350062994))
* **sqlalchemy:** implement Where ([7424704](https://github.com/ibis-project/ibis/commit/742470483e2c22484f6d8a8cc4f6f5bc7f81fd5b))
* **sqlalchemy:** implement zeroifnull ([4735e9a](https://github.com/ibis-project/ibis/commit/4735e9a390fba12776d1795889923f67d6ca7c15))
* **sqlite:** implement BitAnd, BitOr and BitXor ([e478479](https://github.com/ibis-project/ibis/commit/e4784791997335010b5d78c8bf97f302566c973d))
* **sqlite:** implement cotangent ([01e7ce7](https://github.com/ibis-project/ibis/commit/01e7ce763c1728544c2329887d8a1e29c87235e9))
* **sqlite:** implement degrees and radians ([2cf9c5e](https://github.com/ibis-project/ibis/commit/2cf9c5e6d86a067bca1b322c5a8abe4cd216d5b4))


### Bug Fixes

* **api:** bring back null datatype parsing ([fc131a1](https://github.com/ibis-project/ibis/commit/fc131a1d87fe32125e5a7aa51d929312e0ae4814))
* **api:** compute the type from both branches of `Where` expressions ([b8f4120](https://github.com/ibis-project/ibis/commit/b8f4120a96b7b94344b038439f2863c0ad185866))
* **api:** ensure that `Deferred` objects work in aggregations ([bbb376c](https://github.com/ibis-project/ibis/commit/bbb376c24ac38536c88184559e46723732508221))
* **api:** ensure that nulls can be cast to any type to allow caller promotion ([fab4393](https://github.com/ibis-project/ibis/commit/fab4393a6ed93f60931781748a5ea76acdf5bbd2))
* **api:** make ExistSubquery and NotExistsSubquery pure boolean operations ([dd70024](https://github.com/ibis-project/ibis/commit/dd7002455e482cfec633db052be4536e67978ac0))
* **backends:** make execution transactional where possible ([d1ea269](https://github.com/ibis-project/ibis/commit/d1ea2693df8c77d86ac302749f0fa17b2c5f2401))
* **clickhouse:** cast empty result dataframe ([27ae68a](https://github.com/ibis-project/ibis/commit/27ae68ae2da28e5c46ba6c7e5ecaed25b6982fac))
* **clickhouse:** handle empty IN and NOT IN expressions ([2c892eb](https://github.com/ibis-project/ibis/commit/2c892ebd0bceb9562ee80cb078df5180590da36f))
* **clickhouse:** return null instead of empty string for group_concat when values are filtered out ([b826b40](https://github.com/ibis-project/ibis/commit/b826b4047229e0983eb8a1bae3132700ea596a15))
* **compiler:** fix bool bool comparisons ([1ac9a9e](https://github.com/ibis-project/ibis/commit/1ac9a9e67aff1804c1cd9e44ab8acf8f39545c55))
* **dask/pandas:** allow limit to be `None` ([9f91d6b](https://github.com/ibis-project/ibis/commit/9f91d6bd3bda4563fb3595a49aa0d043bc3b2c5d))
* **dask:** aggregation with multi-key groupby fails on dask backend ([4f8bc70](https://github.com/ibis-project/ibis/commit/4f8bc70d21ce3a4565232e3850821fc01bf200e9))
* **datafusion:** handle predicates in aggregates ([4725571](https://github.com/ibis-project/ibis/commit/4725571bc3e29ab4312ecf51888294b0bcd12795))
* **deps:** update dependency datafusion to >=0.4,<0.7 ([f5b244e](https://github.com/ibis-project/ibis/commit/f5b244e06396a8861b5cc9d74bd06b9c5fc935cb))
* **deps:** update dependency duckdb to >=0.3.2,<0.5.0 ([57ee818](https://github.com/ibis-project/ibis/commit/57ee8180c58fd29d5e8eba6886913921a77149d9))
* **deps:** update dependency duckdb-engine to >=0.1.8,<0.3.0 ([3e379a0](https://github.com/ibis-project/ibis/commit/3e379a030b908eb8da6f9f97fddf1f715fcf321f))
* **deps:** update dependency geoalchemy2 to >=0.6.3,<0.13 ([c04a533](https://github.com/ibis-project/ibis/commit/c04a533983c6a5d3eb2f22e87111d6a287ebd93f))
* **deps:** update dependency geopandas to >=0.6,<0.12 ([b899c37](https://github.com/ibis-project/ibis/commit/b899c376f8eefae9097428c19808acba54685a92))
* **deps:** update dependency Shapely to >=1.6,<1.8.3 ([87a49ad](https://github.com/ibis-project/ibis/commit/87a49adba2ae260f8de2027b6e53c954ba0931e2))
* **deps:** update dependency toolz to >=0.11,<0.13 ([258a641](https://github.com/ibis-project/ibis/commit/258a641711e1e5441357e0af0ad8753a4e0d3d0d))
* don't mask udf module in __init__.py ([3e567ba](https://github.com/ibis-project/ibis/commit/3e567ba7a147c08222e792650c531731749fe400))
* **duckdb:** ensure that paths with non-extension `.` chars are parsed correctly ([9448fd3](https://github.com/ibis-project/ibis/commit/9448fd38b69d45c04f1023de044ad311de3adf69))
* **duckdb:** fix struct datatype parsing ([5124763](https://github.com/ibis-project/ibis/commit/51247636f7ba90ef89cf13ddb1e0e277bcf53919))
* **duckdb:** force string_agg separator to be a constant ([21cdf2f](https://github.com/ibis-project/ibis/commit/21cdf2f05550a514cc3aa7fb211baba5b3bf7e28))
* **duckdb:** handle multiple dotted extensions; quote names; consolidate implementations ([1494246](https://github.com/ibis-project/ibis/commit/1494246d7dc42b36dbd6799d5386a1615ab79710))
* **duckdb:** remove timezone function invocation ([33d38fc](https://github.com/ibis-project/ibis/commit/33d38fc7399c2febe783a2967124ed423c15d1ea))
* **geospatial:** ensure that later versions of numpy are compatible with geospatial code ([33f0afb](https://github.com/ibis-project/ibis/commit/33f0afb3b7e80136a7f03d50e0280b6a7d6bdb7c))
* **impala:** a delimited table explicitly declare stored as textfile ([04086a4](https://github.com/ibis-project/ibis/commit/04086a47536989ef25be33d3e9bc942911ea353e)), closes [#4260](https://github.com/ibis-project/ibis/issues/4260)
* **impala:** remove broken nth_value implementation ([dbc9cc2](https://github.com/ibis-project/ibis/commit/dbc9cc21ee468f58970e9ba7509cb2e10822c76d))
* **ir:** don't attempt fusion when projections aren't exactly equivalent ([3482ba2](https://github.com/ibis-project/ibis/commit/3482ba2b6f2da6d37f8678ab5cd94717ada37d44))
* **mysql:** cast mysql timestamp literals to ensure correct return type ([8116e04](https://github.com/ibis-project/ibis/commit/8116e04673bd65ef39d6d09a0aefc90a57006987))
* **mysql:** implement integer to timestamp using `from_unixtime` ([1b43004](https://github.com/ibis-project/ibis/commit/1b43004e41f171491dfc6101c6d674cf75a2749a))
* **pandas/dask:** look at pre_execute for has_operation reporting ([cb44efc](https://github.com/ibis-project/ibis/commit/cb44efcfb609f48e6965ba9a546c466bd97d7dbd))
* **pandas:** execute negate on bool as `not` ([330ab4f](https://github.com/ibis-project/ibis/commit/330ab4f0bcf22f970f9c3e17cbe5ef8cc6fdf96b))
* **pandas:** fix struct inference from dict in the pandas backend ([5886a9a](https://github.com/ibis-project/ibis/commit/5886a9ad10e360a0c582c8883e282436d9077b27))
* **pandas:** force backend options registration on trace.enable() calls ([8818fe6](https://github.com/ibis-project/ibis/commit/8818fe69f24e87caf90e996a7ddab49d4519d3a7))
* **pandas:** handle empty boolean column casting in Series conversion ([f697e3e](https://github.com/ibis-project/ibis/commit/f697e3e61b4c6525375ab5bf78a653bd3f2124ca))
* **pandas:** handle struct columns with NA elements ([9a7c510](https://github.com/ibis-project/ibis/commit/9a7c51054f52ce845408b99680e8b169a38e5089))
* **pandas:** handle the case of selection from a join when remapping overlapping column names ([031c4c6](https://github.com/ibis-project/ibis/commit/031c4c63471fd8ca3169e7f4e7f6a30ff331c9d8))
* **pandas:** perform correct equality comparison ([d62e7b9](https://github.com/ibis-project/ibis/commit/d62e7b9e689514b81b8ec08288244c67c488926a))
* **postgres/duckdb:** cast after milliseconds computation instead of after extraction ([bdd1d65](https://github.com/ibis-project/ibis/commit/bdd1d65938935c2a0a0f5ba89ec9d63ac4c922d0))
* **pyspark:** handle predicates in Aggregation ([842c307](https://github.com/ibis-project/ibis/commit/842c307796491e6dc7f1209e1ab1936e1cfe83c9))
* **pyspark:** prevent spark from trying to convert timezone of naive timestamps ([dfb4127](https://github.com/ibis-project/ibis/commit/dfb412705df3d250b3547083760f99769c924612))
* **pyspark:** remove xpassing test for [#2453](https://github.com/ibis-project/ibis/issues/2453) ([c051e28](https://github.com/ibis-project/ibis/commit/c051e287eb3139104b4772d48785432c3f143c32))
* **pyspark:** specialize implementation of `has_operation` ([5082346](https://github.com/ibis-project/ibis/commit/508234663cc5f633bf7ee9de4c523f06c23d0b2c))
* **pyspark:** use empty check for collect_list in GroupConcat rule ([df66acb](https://github.com/ibis-project/ibis/commit/df66acb2c918f97b848798adfb11defcf3aed1da))
* **repr:** allow DestructValue selections to be formatted by fmt ([4b45d87](https://github.com/ibis-project/ibis/commit/4b45d873267f476197c4a1fe45261715a8fd7a9a))
* **repr:** when formatting DestructValue selections, use struct field names as column names ([d01fe42](https://github.com/ibis-project/ibis/commit/d01fe42b4b8055b29cca8dc5048477616405c176))
* **sqlalchemy:** fix parsing and construction of nested array types ([e20bcc0](https://github.com/ibis-project/ibis/commit/e20bcc0941ac90a38b1263018f32ba8af5e5c267))
* **sqlalchemy:** remove unused second argument when creating temporary views ([8766b40](https://github.com/ibis-project/ibis/commit/8766b40ec8a5a853402bf7ab51629b6fb0ab252e))
* **sqlite:** register conversion to isoformat for `pandas.Timestamp` ([fe95dca](https://github.com/ibis-project/ibis/commit/fe95dca312511b1c43b915d640434c5e3104d79c))
* **sqlite:** test case with whitespace at the end of the line ([7623ae9](https://github.com/ibis-project/ibis/commit/7623ae9597e8b82fe12ca3b5ccac5e6e8540c6fb))
* **sql:** use isoformat for timestamp literals ([70d0ba6](https://github.com/ibis-project/ibis/commit/70d0ba625fe6008ee34e233c40f6b489a751bfa5))
* **type-system:** infer null datatype for empty sequence of expressions ([f67d5f9](https://github.com/ibis-project/ibis/commit/f67d5f911fe8e1791584c051cd0bd8a007f5b8f7))
* use bounded precision for decimal aggregations ([596acfb](https://github.com/ibis-project/ibis/commit/596acfb7665bc11d8753ea36d7df8b6f37995a15))


### Performance Improvements

* **analysis:** add `_projection` as cached_property to avoid reconstruction of projections ([98510c8](https://github.com/ibis-project/ibis/commit/98510c8d4073996e7e6c3b25a758bc15661caf17))
* **lineage:** ensure that expressions are not traversed multiple times in most cases ([ff9708c](https://github.com/ibis-project/ibis/commit/ff9708c64745114b961da2d8c782eac5315ea211))


### Reverts

* ci: install sqlite3 on ubuntu ([1f2705f](https://github.com/ibis-project/ibis/commit/1f2705f137f6925eb8ef7e894192be81ffddc5f9))

### [3.0.2](https://github.com/ibis-project/ibis/compare/3.0.1...3.0.2) (2022-04-28)


### Bug Fixes

* **docs:** fix tempdir location for docs build ([dcd1b22](https://github.com/ibis-project/ibis/commit/dcd1b226903db9d589a40ccd987280de0c8362e3))

### [3.0.1](https://github.com/ibis-project/ibis/compare/3.0.0...3.0.1) (2022-04-28)


### Bug Fixes

* **build:** replace version before exec plugin runs ([573139c](https://github.com/ibis-project/ibis/commit/573139c3569aa6c6a197910f3582c6e24593688e))

## [3.0.0](https://github.com/ibis-project/ibis/compare/2.1.1...3.0.0) (2022-04-25)


### ⚠ BREAKING CHANGES

* **ir:** The following are breaking changes due to simplifying expression internals
  - `ibis.expr.datatypes.DataType.scalar_type` and `DataType.column_type` factory
    methods have been removed, `DataType.scalar` and `DataType.column` class
    fields can be used to directly construct a corresponding expression instance
    (though prefer to use `operation.to_expr()`)
  - `ibis.expr.types.ValueExpr._name` and `ValueExpr._dtype`` fields are not
    accassible anymore. While these were not supposed to used directly now
    `ValueExpr.has_name()`, `ValueExpr.get_name()` and `ValueExpr.type()` methods
    are the only way to retrieve the expression's name and datatype.
  - `ibis.expr.operations.Node.output_type` is a property now not a method,
    decorate those methods with `@property`
  - `ibis.expr.operations.Value` subclasses must define `output_shape` and
    `output_dtype` properties from now on (note the datatype abbreviation `dtype`
    in the property name)
  - `ibis.expr.rules.cast()`, `scalar_like()` and `array_like()` rules have been
    removed
* **api:** Replace `t["a"].distinct()` with `t[["a"]].distinct()`.
* **deps:** The sqlalchemy lower bound is now 1.4
* **ir:** Schema.names and Schema.types attributes now have tuple type rather than list
* **expr:** Columns that were added or used in an aggregation or
mutation would be alphabetically sorted in compiled SQL outputs.  This
was a vestige from when Python dicts didn't preserve insertion order.
Now columns will appear in the order in which they were passed to
`aggregate` or `mutate`
* **api:** `dt.float` is now `dt.float64`; use `dt.float32` for the previous behavior.
* **ir:** Relation-based `execute_node` dispatch rules must now accept tuples of expressions.
* **ir:** removed ibis.expr.lineage.{roots,find_nodes} functions
* **config:** Use `ibis.options.graphviz_repr = True` to enable
* **hdfs:** Use `fsspec` instead of HDFS from ibis
* **udf:** Vectorized UDF coercion functions are no longer a public API.
* The minimum supported Python version is now Python 3.8
* **config:** `register_option` is no longer supported, please submit option requests upstream
* **backends:** Read tables with pandas.read_hdf and use the pandas backend
* The CSV backend is removed. Use Datafusion for CSV execution.
* **backends:** Use the datafusion backend to read parquet files
* `Expr() -> Expr.pipe()`
* coercion functions previously in expr/schema.py are now in udf/vectorized.py
* **api:** `materialize` is removed. Joins with overlapping columns now have suffixes.
* **kudu:** use impala instead: https://kudu.apache.org/docs/kudu_impala_integration.html
* Any code that was relying implicitly on string-y
behavior from UUID datatypes will need to add an explicit cast first.

### Features

* add _repr_html_ for expressions to print as tables in ipython ([cd6fa4e](https://github.com/ibis-project/ibis/commit/cd6fa4e245cbf6e7ce8df41634c307689f1fd60a))
* add duckdb backend ([667f2d5](https://github.com/ibis-project/ibis/commit/667f2d5ae5e0ffcf3ffb56d68e4f02f37a4a2b4b))
* allow construction of decimal literals ([3d9e865](https://github.com/ibis-project/ibis/commit/3d9e865ab3badd092d8155302641a3e91c72c028))
* **api:** add `ibis.asc` expression ([efe177e](https://github.com/ibis-project/ibis/commit/efe177eea0e2cd676d2db104e8611c1c47b7f1a1)), closes [#1454](https://github.com/ibis-project/ibis/issues/1454)
* **api:** add has_operation API to the backend ([4fab014](https://github.com/ibis-project/ibis/commit/4fab0143741a1baf016dd6f880103093c3418685))
* **api:** implement type for SortExpr ([ab19bd6](https://github.com/ibis-project/ibis/commit/ab19bd64f82dd49f967118e22f3ae5042fbf9e0b))
* **clickhouse:** implement string concat for clickhouse ([1767205](https://github.com/ibis-project/ibis/commit/1767205eca7e017b752f8f25629aad6870077777))
* **clickhouse:** implement StrRight operation ([67749a0](https://github.com/ibis-project/ibis/commit/67749a0363666af272b00d2945527d07b6436951))
* **clickhouse:** implement table union ([e0008d7](https://github.com/ibis-project/ibis/commit/e0008d7954cda050d6d39f8f2262ff9d3fcadca9))
* **clickhouse:** implement trim, pad and string predicates ([a5b7293](https://github.com/ibis-project/ibis/commit/a5b72934c38d3a6422274737d5b766d7b4dd9766))
* **datafusion:** implement Count operation ([4797a86](https://github.com/ibis-project/ibis/commit/4797a8680b6e084ca59da3307c48378846643486))
* **datatypes:** unbounded decimal type ([f7e6f65](https://github.com/ibis-project/ibis/commit/f7e6f651c4ea4ed7fe394cf3322e43bebae5e26c))
* **date:** add ibis.date(y,m,d) functionality ([26892b6](https://github.com/ibis-project/ibis/commit/26892b6a11a8b0fd8f31d5b692daf8e20a723ac1)), closes [#386](https://github.com/ibis-project/ibis/issues/386)
* **duckdb/postgres/mysql/pyspark:** implement `.sql` on tables for mixing sql and expressions ([00e8087](https://github.com/ibis-project/ibis/commit/00e80871bfc80fb060dbf54efa477cb5c894a6ad))
* **duckdb:** add functionality needed to pass integer to interval test ([e2119e8](https://github.com/ibis-project/ibis/commit/e2119e81c2e9f2f6c641ee14a69ddba887b97b61))
* **duckdb:** implement _get_schema_using_query ([93cd730](https://github.com/ibis-project/ibis/commit/93cd73021e92c6d7b4bed90babec3a2a237324fc))
* **duckdb:** implement now() function ([6924f50](https://github.com/ibis-project/ibis/commit/6924f50204973e2f0d093cbf7c9e717d0501adb4))
* **duckdb:** implement regexp replace and extract ([18d16a7](https://github.com/ibis-project/ibis/commit/18d16a74ed96d427fa31a67e3786b75d66629081))
* implement `force` argument in sqlalchemy backend base class ([9df7f1b](https://github.com/ibis-project/ibis/commit/9df7f1b1625879278954035e0521861fded5e07d))
* implement coalesce for the pyspark backend ([8183efe](https://github.com/ibis-project/ibis/commit/8183efeff8591a5b1a846d5e31906a44b8ba73dc))
* implement semi/anti join for the pandas backend ([cb36fc5](https://github.com/ibis-project/ibis/commit/cb36fc531d9826b4e5a8e794d7a2b43415540129))
* implement semi/anti join for the pyspark backend ([3e1ba9c](https://github.com/ibis-project/ibis/commit/3e1ba9c1e29673982e69bf50200ec3ded6777740))
* implement the remaining clickhouse joins ([b3aa1f0](https://github.com/ibis-project/ibis/commit/b3aa1f0d77820fcc5644adc659df568301036746))
* **ir:** rewrite and speed up expression repr ([45ce9b2](https://github.com/ibis-project/ibis/commit/45ce9b2c2aa5fb3919a59ede95f60a6b813be730))
* **mysql:** implement _get_schema_from_query ([456cd44](https://github.com/ibis-project/ibis/commit/456cd44879c32bac7f8a798cb8e7e5851e94b4ec))
* **mysql:** move string join impl up to alchemy for mysql ([77a8eb9](https://github.com/ibis-project/ibis/commit/77a8eb9baa2a9061d3b91c10a176648610e53b29))
* **postgres:** implement _get_schema_using_query ([f2459eb](https://github.com/ibis-project/ibis/commit/f2459ebf3995c968c59cd16a7a684297d08ce4f2))
* **pyspark:** implement Distinct for pyspark ([4306ad9](https://github.com/ibis-project/ibis/commit/4306ad9c2537f23a3f24e7424751fcef5b50bd52))
* **pyspark:** implement log base b for pyspark ([527af3c](https://github.com/ibis-project/ibis/commit/527af3c5869a70a16cfcfa4ab197e1b64ef4f5ce))
* **pyspark:** implement percent_rank and enable testing ([c051617](https://github.com/ibis-project/ibis/commit/c051617875a2d9fcdc612ff5ceeddabd73aeb3e9))
* **repr:** add interval info to interval repr ([df26231](https://github.com/ibis-project/ibis/commit/df2623133e66eea7e3ac7a924992ae3e089668b3))
* **sqlalchemy:** implement ilike ([43996c0](https://github.com/ibis-project/ibis/commit/43996c080b8e2c9c53e838b90743878be9764ae7))
* **sqlite:** implement date_truncate ([3ce4f2a](https://github.com/ibis-project/ibis/commit/3ce4f2a87d2eab244fd1550042bc17c3fe3de3e2))
* **sqlite:** implement ISO week of year ([714ff7b](https://github.com/ibis-project/ibis/commit/714ff7be469e705c39dab79f571333bfe9fcea00))
* **sqlite:** implement string join and concat ([6f5f353](https://github.com/ibis-project/ibis/commit/6f5f3538bcddaf51fd8e689a3abcb23fc33f4ecd))
* support of arrays and tuples for clickhouse ([db512a8](https://github.com/ibis-project/ibis/commit/db512a89c14edc7e216558f2c051c4a0e905b543))
* **ver:** dynamic version identifiers ([408f862](https://github.com/ibis-project/ibis/commit/408f862e158868a57fa769b24e6312f3d7fa3e6f))


### Bug Fixes

* added wheel to pyproject toml for venv users ([b0b8e5c](https://github.com/ibis-project/ibis/commit/b0b8e5c612a38c4cde9543a83620a0979e3907cf))
* allow major version changes in CalVer dependencies ([9c3fbe5](https://github.com/ibis-project/ibis/commit/9c3fbe5ee8d26d01a7bc8ccac2bed2a988bc909c))
* **annotable:** allow optional arguments at any position ([778995f](https://github.com/ibis-project/ibis/commit/778995f35951751d475827770e13d591690b3821)), closes [#3730](https://github.com/ibis-project/ibis/issues/3730)
* **api:** add ibis.map and .struct ([327b342](https://github.com/ibis-project/ibis/commit/327b34254918eecca28f3066152d9e53445997d0)), closes [#3118](https://github.com/ibis-project/ibis/issues/3118)
* **api:** map string multiplication with integer to repeat method ([b205922](https://github.com/ibis-project/ibis/commit/b2059227234824fdd42cecddff259d695eef5c1c))
* **api:** thread suffixes parameter to individual join methods ([31a9aff](https://github.com/ibis-project/ibis/commit/31a9aff63711b28192adb35eadf07ef93c0a7313))
* change TimestampType to Timestamp ([e0750be](https://github.com/ibis-project/ibis/commit/e0750be5ee37bb13221ae2a25f4d45edee2106f5))
* **clickhouse:** disconnect from clickhouse when computing version ([11cbf08](https://github.com/ibis-project/ibis/commit/11cbf08adbbea9464ea569ccabda3758ca17c23f))
* **clickhouse:** use a context manager for execution ([a471225](https://github.com/ibis-project/ibis/commit/a471225001964d51e5526f1b625ee7fb3b89cead))
* combine windows during windowization ([7fdd851](https://github.com/ibis-project/ibis/commit/7fdd851c19414a5e9f5625989dcf11e07104b7c4))
* conform epoch_seconds impls to expression return type ([18a70f1](https://github.com/ibis-project/ibis/commit/18a70f111f21d712df880259a4c3a48e70844ac2))
* **context-adjustment:** pass scope when calling adjust_context in pyspark backend ([33aad7b](https://github.com/ibis-project/ibis/commit/33aad7b8b419d3b2cf4edac11317c4610b7a16d3)), closes [#3108](https://github.com/ibis-project/ibis/issues/3108)
* **dask:** fix asof joins for newer version of dask ([50711cc](https://github.com/ibis-project/ibis/commit/50711cc8baba499250934e1d12fd24e65534f31e))
* **dask:** workaround dask bug ([a0f3bd9](https://github.com/ibis-project/ibis/commit/a0f3bd96b2112454ba5431df4f5073de4a7954b0))
* **deps:** update dependency atpublic to v3 ([3fe8f0d](https://github.com/ibis-project/ibis/commit/3fe8f0d5c726cef20650008a4e2140d12652aae9))
* **deps:** update dependency datafusion to >=0.4,<0.6 ([3fb2194](https://github.com/ibis-project/ibis/commit/3fb2194a9594d790d2f25080078e56d99ce72ece))
* **deps:** update dependency geoalchemy2 to >=0.6.3,<0.12 ([dc3c361](https://github.com/ibis-project/ibis/commit/dc3c3610d7a0c7e5f5d3f872d7e3ee06bfce7ef6))
* **deps:** update dependency graphviz to >=0.16,<0.21 ([3014445](https://github.com/ibis-project/ibis/commit/301444553b617cf02fdf46cefb8e594f75a0dc27))
* **duckdb:** add casts to literals to fix binding errors ([1977a55](https://github.com/ibis-project/ibis/commit/1977a559b4345a1ab3414aad93b3bb6fa3d0b007)), closes [#3629](https://github.com/ibis-project/ibis/issues/3629)
* **duckdb:** fix array column type discovery on leaf tables and add tests ([15e5412](https://github.com/ibis-project/ibis/commit/15e5412f9526036f13fa2152b39d52d8a0d69eec))
* **duckdb:** fix log with base b impl ([4920097](https://github.com/ibis-project/ibis/commit/492009792f88f05cf8e173d04d3c9a49ca4b8cc5))
* **duckdb:** support both 0.3.2 and 0.3.3 ([a73ccce](https://github.com/ibis-project/ibis/commit/a73ccce127c50aef502b0ba9ac6fb45d1eef4700))
* enforce the schema's column names in `apply_to` ([b0f334d](https://github.com/ibis-project/ibis/commit/b0f334d8fca2ce41e274ef12903620097b38524e))
* expose ops.IfNull for mysql backend ([156c2bd](https://github.com/ibis-project/ibis/commit/156c2bd325ce2a2049e45aa57f9a2ca929024577))
* **expr:** add more binary operators to char list and implement fallback ([b88184c](https://github.com/ibis-project/ibis/commit/b88184c4ccdba857df85ba21b43c1ae3fa517b82))
* **expr:** fix formatting of table info using tabulate ([b110636](https://github.com/ibis-project/ibis/commit/b110636f09b11df108cf91ac8d20fd8db7ee28d3))
* fix float vs real data type detection in sqlalchemy ([24e6774](https://github.com/ibis-project/ibis/commit/24e677480f830caf367283c6815c6f759ac33d7a))
* fix list_schemas argument ([69c1abf](https://github.com/ibis-project/ibis/commit/69c1abf21fff25d877a71c791f78b0e3ece552f0))
* fix postgres udfs and re-enable ci tests ([7d480d2](https://github.com/ibis-project/ibis/commit/7d480d225d713274f8068af07cb7fcffac438691))
* fix tablecolumn execution for filter following join ([064595b](https://github.com/ibis-project/ibis/commit/064595b9c2a85f6532b93b7b8b5343fabe2dbe29))
* **format:** remove some newlines from formatted expr repr ([ed4fa78](https://github.com/ibis-project/ibis/commit/ed4fa78a484f1b6a08531fa406558c471dd5762f))
* **histogram:** cross_join needs onclause=True ([5d36a58](https://github.com/ibis-project/ibis/commit/5d36a58d2df83b045487e9701e309978c3dd777d)), closes [#622](https://github.com/ibis-project/ibis/issues/622)
* ibis.expr.signature.Parameter is not pickleable ([828fd54](https://github.com/ibis-project/ibis/commit/828fd545de3e33c364a9eec86f4d07b391ea028b))
* implement coalesce properly in the pandas backend ([aca5312](https://github.com/ibis-project/ibis/commit/aca53124fa8044a4a5cfa577ed9ece5cba4c05c8))
* implement count on tables for pyspark ([7fe5573](https://github.com/ibis-project/ibis/commit/7fe557333b49551b86c0ad121386687763420723)), closes [#2879](https://github.com/ibis-project/ibis/issues/2879)
* infer coalesce types when a non-null expression occurs after the first argument ([c5f2906](https://github.com/ibis-project/ibis/commit/c5f2906cb3dd623e9717a73574a67c193db5f246))
* **mutate:** do not lift table column that results from mutate ([ba4e5e5](https://github.com/ibis-project/ibis/commit/ba4e5e56f7e0ccd367a270d6d5090f340d610dae))
* **pandas:** disable range windows with order by ([e016664](https://github.com/ibis-project/ibis/commit/e0166644a9df84a783d04c96e27710999598a897))
* **pandas:** don't reassign the same column to silence SettingWithCopyWarning warning ([75dc616](https://github.com/ibis-project/ibis/commit/75dc6165ff9c5e39823e734367fb395a8013c6f0))
* **pandas:** implement percent_rank correctly ([d8b83e7](https://github.com/ibis-project/ibis/commit/d8b83e7129e0b26128933d26d90169a6750c3de2))
* prevent unintentional cross joins in mutate + filter ([83eef99](https://github.com/ibis-project/ibis/commit/83eef9904fb48c6a9e333572590109b8717c7acd))
* **pyspark:** fix range windows ([a6f2aa8](https://github.com/ibis-project/ibis/commit/a6f2aa896e978310a9e5ca3d87829597dd548566))
* regression in Selection.sort_by with resolved_keys ([c7a69cd](https://github.com/ibis-project/ibis/commit/c7a69cd352af975d6cda587fb2d1282da37718e1))
* regression in sort_by with resolved_keys ([63f1382](https://github.com/ibis-project/ibis/commit/63f138267cf8b4fce80345ffbe137a6c1b3c8fc3)), closes [#3619](https://github.com/ibis-project/ibis/issues/3619)
* remove broken csv pre_execute ([93b662a](https://github.com/ibis-project/ibis/commit/93b662a93203801b9cca343871ee6c274051628d))
* remove importorskip call for backend tests ([2f0bcd8](https://github.com/ibis-project/ibis/commit/2f0bcd8e0d87d210f2ab303cc03e21f0b8fb6abe))
* remove incorrect fix for pandas regression ([339f544](https://github.com/ibis-project/ibis/commit/339f5447d4b84f4e5e5344b49de713f8c027445b))
* remove passing schema into register_parquet ([bdcbb08](https://github.com/ibis-project/ibis/commit/bdcbb083a112d3cc81bb98ee63a26674b5397563))
* **repr:** add ops.TimeAdd to repr binop lookup table ([fd94275](https://github.com/ibis-project/ibis/commit/fd94275e945137be3b95b4aa4c0b1cdb16b7a41d))
* **repr:** allow ops.TableNode in fmt_value ([6f57003](https://github.com/ibis-project/ibis/commit/6f57003620d21e07d31ca5d2013302ba2899fdb0))
* reverse the predicate pushdown substitution ([f3cd358](https://github.com/ibis-project/ibis/commit/f3cd3581b078a7f297f303866aeb30c8f826b19d))
* sort_index to satisfy pandas 1.4.x ([6bac0fc](https://github.com/ibis-project/ibis/commit/6bac0fc2bec2434f5d2eb8b1c2b0328a0e5a80a3))
* **sqlalchemy:** ensure correlated subqueries FROM clauses are rendered ([3175321](https://github.com/ibis-project/ibis/commit/3175321844897ad4fa88547c9474724736685209))
* **sqlalchemy:** use corresponding_column to prevent spurious cross joins ([fdada21](https://github.com/ibis-project/ibis/commit/fdada217afbf0e8b07d421fd2da0092a20c578c7))
* **sqlalchemy:** use replace selectables to prevent semi/anti join cross join ([e8a1a71](https://github.com/ibis-project/ibis/commit/e8a1a715801a48811297ff941b0ee67b8aabc088))
* **sql:** retain column names for named ColumnExprs ([f1b4b6e](https://github.com/ibis-project/ibis/commit/f1b4b6e86b89322c36e5d0eeeb8e356929229843)), closes [#3754](https://github.com/ibis-project/ibis/issues/3754)
* **sql:** walk right join trees and substitute joins with right-side joins with views ([0231592](https://github.com/ibis-project/ibis/commit/02315927b93762ab045a11629d3144fbff8545c1))
* store schema on the pandas backend to allow correct inference ([35070be](https://github.com/ibis-project/ibis/commit/35070be3b5a6311afb20525324030a0634f588fe))


### Performance Improvements

* **datatypes:** speed up __str__ and __hash__ ([262d3d7](https://github.com/ibis-project/ibis/commit/262d3d78642fadd7572033a6c82dcbe1fc472f61))
* fast path for simple column selection ([d178498](https://github.com/ibis-project/ibis/commit/d1784988fdae4e3146b20fcee9811bb1e1316cc8))
* **ir:** global equality cache ([13c2bb2](https://github.com/ibis-project/ibis/commit/13c2bb260c6229e9b5ebfd15ef991846587b8ca5))
* **ir:** introduce CachedEqMixin to speed up equality checks ([b633925](https://github.com/ibis-project/ibis/commit/b633925cc221caede0194091685d0589795297a4))
* **repr:** remove full tree repr from rule validator error message ([65885ab](https://github.com/ibis-project/ibis/commit/65885ab559518908601d943ce1ea188503354a29))
* speed up attribute access ([89d1c05](https://github.com/ibis-project/ibis/commit/89d1c05ac45c5905867391ffb685e8e34c7c2535))
* use assign instead of concat in projections when possible ([985c242](https://github.com/ibis-project/ibis/commit/985c2423708d9855a227f0eaf5ade1d326064eab))


### Miscellaneous Chores

* **deps:** increase sqlalchemy lower bound to 1.4 ([560854a](https://github.com/ibis-project/ibis/commit/560854a1a4c326ee76b96c6fc7793a05a475fa43))
* drop support for Python 3.7 ([0afd138](https://github.com/ibis-project/ibis/commit/0afd138122f73da3addbb73635d67ce77f4d2960))


### Code Refactoring

* **api:** make primitive types more cohesive ([71da8f7](https://github.com/ibis-project/ibis/commit/71da8f756ec9ca5c547c8f9209599c005c0f5c66))
* **api:** remove distinct ColumnExpr API ([3f48cb8](https://github.com/ibis-project/ibis/commit/3f48cb8b71ee30e8e8cdc3b13a50939dc51bfaa0))
* **api:** remove materialize ([24285c1](https://github.com/ibis-project/ibis/commit/24285c178a5d3c1bb875f4eca75736dacebb4ec9))
* **backends:** remove the hdf5 backend ([ff34f3e](https://github.com/ibis-project/ibis/commit/ff34f3e05930da463eae5b98c6ae42b78985415c))
* **backends:** remove the parquet backend ([b510473](https://github.com/ibis-project/ibis/commit/b5104735df7a4329a01a79e8d7f74282caf8aebd))
* **config:** disable graphviz-repr-in-notebook by default ([214ad4e](https://github.com/ibis-project/ibis/commit/214ad4ed4d54d4786477e0b4719ecbc728c3fe51))
* **config:** remove old config code and port to pydantic ([4bb96d1](https://github.com/ibis-project/ibis/commit/4bb96d1cc97bae855b6662e3c7e349bd5baab554))
* dt.UUID inherits from DataType, not String ([2ba540d](https://github.com/ibis-project/ibis/commit/2ba540d2ad237dbba8386f7c6efe1bf159bf29ad))
* **expr:** preserve column ordering in aggregations/mutations ([668be0f](https://github.com/ibis-project/ibis/commit/668be0ff1b120f1217b33e33e8ef8a35d9f00c47))
* **hdfs:** replace HDFS with `fsspec` ([cc6eddb](https://github.com/ibis-project/ibis/commit/cc6eddba1ffe8cdc9e0005381aa3bc1f77bb3fd9))
* **ir:** make Annotable immutable ([1f2b3fa](https://github.com/ibis-project/ibis/commit/1f2b3fae8f8259da64004e1efc3849c59939081a))
* **ir:** make schema annotable ([b980903](https://github.com/ibis-project/ibis/commit/b9809036a39f02e44dea688b7ed06d6374502c7d))
* **ir:** remove unused lineage `roots` and `find_nodes` functions ([d630a77](https://github.com/ibis-project/ibis/commit/d630a77602204ec81faf970d5050e1e77ce26405))
* **ir:** simplify expressions by not storing dtype and name ([e929f85](https://github.com/ibis-project/ibis/commit/e929f85b4ffadad432e6d8ee7267c58aea062454))
* **kudu:** remove support for use of kudu through kudu-python ([36bd97f](https://github.com/ibis-project/ibis/commit/36bd97f03909a8f43eb90fe9506a87e93c5d31a7))
* move coercion functions from schema.py to udf ([58eea56](https://github.com/ibis-project/ibis/commit/58eea56a8476438c3857f5da7a7088bf2020d38c)), closes [#3033](https://github.com/ibis-project/ibis/issues/3033)
* remove blanket __call__ for Expr ([3a71116](https://github.com/ibis-project/ibis/commit/3a711169fbfff7c3a1b94a0daecf69802e78f8b4)), closes [#2258](https://github.com/ibis-project/ibis/issues/2258)
* remove the csv backend ([0e3e02e](https://github.com/ibis-project/ibis/commit/0e3e02e00000ff85d3450af721c3cf1dfd54d5f7))
* **udf:** make coerce functions in ibis.udf.vectorized private ([9ba4392](https://github.com/ibis-project/ibis/commit/9ba439220ea82ed82f84f3bd799943bd75f54518))

## [2.1.1](https://github.com/ibis-project/ibis/compare/2.1.0...2.1.1) (2022-01-12)


### Bug Fixes

* **setup.py:** set the correct version number for 2.1.0 ([f3d267b](https://github.com/ibis-project/ibis/commit/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf))

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
* Support summarization of empty data in pandas backend ([#2908](https://github.com/ibis-project/ibis/issues/2908))
* Unify implementation of fillna and isna in Pyspark backend ([#2882](https://github.com/ibis-project/ibis/issues/2882))
* Support binary operation with Timedelta in Pyspark backend ([#2873](https://github.com/ibis-project/ibis/issues/2873))
* Add `group_concat` operation for Clickhouse backend ([#2839](https://github.com/ibis-project/ibis/issues/2839))
* Support comparison of ColumnExpr to timestamp literal ([#2808](https://github.com/ibis-project/ibis/issues/2808))
* Make op schema a cached property ([#2805](https://github.com/ibis-project/ibis/issues/2805))
* Implement `.insert()` for SQLAlchemy backends ([#2613](https://github.com/ibis-project/ibis/issues/2613), [#2613](https://github.com/ibis-project/ibis/issues/2778))
* Infer categorical and decimal Series to more specific Ibis types in pandas backend ([#2792](https://github.com/ibis-project/ibis/issues/2792))
* Add `startswith` and `endswith` operations ([#2790](https://github.com/ibis-project/ibis/issues/2790))
* Allow more flexible return type for UDFs ([#2776](https://github.com/ibis-project/ibis/issues/2776), [#2797](https://github.com/ibis-project/ibis/issues/2776))
* Implement Clip in the Pyspark backend ([#2779](https://github.com/ibis-project/ibis/issues/2779))
* Use `ndarray` as array representation in pandas backend ([#2753](https://github.com/ibis-project/ibis/issues/2753))
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
* Fix `.cast()` to array outputting list instead of np.array in pandas backend ([#2821](https://github.com/ibis-project/ibis/issues/2821))
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
* Support reduction UDF without groupby to return multiple columns for pandas backend ([#2511](https://github.com/ibis-project/ibis/issues/2511))
* Support analytic and reduction UDF to return multiple columns for pandas backend ([#2487](https://github.com/ibis-project/ibis/issues/2487))
* Support elementwise UDF to return multiple columns for pandas and PySpark backend ([#2473](https://github.com/ibis-project/ibis/issues/2473))
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
* Create ExtractQuarter operation and add its support to Clickhouse, CSV, Impala, MySQL, OmniSciDB, pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark ([#2175](https://github.com/ibis-project/ibis/issues/2175))
* Add translation rules for isnull() and notnull() for pyspark backend ([#2126](https://github.com/ibis-project/ibis/issues/2126))
* Add window operations support to SQLite ([#2232](https://github.com/ibis-project/ibis/issues/2232))
* Implement read_csv for omniscidb backend ([#2062](https://github.com/ibis-project/ibis/issues/2062))
* [OmniSciDB] Add support to week extraction ([#2171](https://github.com/ibis-project/ibis/issues/2171))
* Date, DateDiff and TimestampDiff implementations for OmniSciDB ([#2097](https://github.com/ibis-project/ibis/issues/2097))
* Create ExtractWeekOfYear operation and add its support to Clickhouse, CSV, MySQL, pandas, Parquet, PostgreSQL, PySpark and Spark ([#2177](https://github.com/ibis-project/ibis/issues/2177))
* Add initial support for ibis.random function ([#2060](https://github.com/ibis-project/ibis/issues/2060))
* Added epoch_seconds extraction operation to Clickhouse, CSV, Impala, MySQL, OmniSciDB, pandas, Parquet, PostgreSQL, PySpark, SQLite, Spark and BigQuery :issue:`2273` ([#2178](https://github.com/ibis-project/ibis/issues/2178))
* [OmniSciDB] Add "method" parameter to load_data ([#2165](https://github.com/ibis-project/ibis/issues/2165))
* Add non-nullable info to schema output ([#2117](https://github.com/ibis-project/ibis/issues/2117))
* fillna and nullif implementations for OmnisciDB ([#2083](https://github.com/ibis-project/ibis/issues/2083))
* Add load_data to sqlalchemy's backends and fix database parameter for load/create/drop when database parameter is the same than the current database ([#1981](https://github.com/ibis-project/ibis/issues/1981))
* [OmniSciDB] Add support for within, d_fully_within and point ([#2125](https://github.com/ibis-project/ibis/issues/2125))
* OmniSciDB - Refactor DDL and Client; Add temporary parameter to create_table and "force" parameter to drop_view ([#2086](https://github.com/ibis-project/ibis/issues/2086))
* Create ExtractDayOfYear operation and add its support to Clickhouse, CSV, MySQL, OmniSciDB, pandas, Parquet, PostgreSQL, PySpark, SQLite and Spark ([#2173](https://github.com/ibis-project/ibis/issues/2173))
* Implementations of Log Log2 Log10 for OmniSciDB backend ([#2095](https://github.com/ibis-project/ibis/issues/2095))

## Bugs

* Table expressions do not recognize inet datatype (Postgres backend) ([#2462](https://github.com/ibis-project/ibis/issues/2462))
* Table expressions do not recognize macaddr datatype (Postgres backend) ([#2461](https://github.com/ibis-project/ibis/issues/2461))
* Fix ``aggcontext.Summarize`` not always producing scalar (pandas backend) ([#2410](https://github.com/ibis-project/ibis/issues/2410))
* Fix same window op with different window size on table lead to incorrect results for pyspark backend ([#2414](https://github.com/ibis-project/ibis/issues/2414))
* Fix same column with multiple aliases not showing properly in repr ([#2229](https://github.com/ibis-project/ibis/issues/2229))
* Fix reduction UDFs over ungrouped, bounded windows on pandas backend ([#2395](https://github.com/ibis-project/ibis/issues/2395))
* FEAT: Support rolling window UDF with non numeric inputs for pandas backend. ([#2386](https://github.com/ibis-project/ibis/issues/2386))
* Fix scope get to use hashmap lookup instead of list lookup ([#2386](https://github.com/ibis-project/ibis/issues/2386))
* Fix equality behavior for Literal ops ([#2387](https://github.com/ibis-project/ibis/issues/2387))
* Fix analytic ops over ungrouped and unordered windows on pandas backend ([#2376](https://github.com/ibis-project/ibis/issues/2376))
* Fix the covariance operator in the BigQuery backend. ([#2367](https://github.com/ibis-project/ibis/issues/2367))
* Update impala kerberos dependencies ([#2342](https://github.com/ibis-project/ibis/issues/2342))
* Added verbose logging to SQL backends ([#1320](https://github.com/ibis-project/ibis/issues/1320))
* Fix issue with sql_validate call to OmnisciDB. ([#2256](https://github.com/ibis-project/ibis/issues/2256))
* Add missing float types to pandas backend ([#2237](https://github.com/ibis-project/ibis/issues/2237))
* Allow group_by and order_by as window operation input in pandas backend ([#2252](https://github.com/ibis-project/ibis/issues/2252))
* Fix PySpark compiler error when elementwise UDF output_type is Decimal or Timestamp ([#2223](https://github.com/ibis-project/ibis/issues/2223))
* Fix interactive mode returning a expression instead of the value when used in Jupyter ([#2157](https://github.com/ibis-project/ibis/issues/2157))
* Fix PySpark error when doing alias after selection ([#2127](https://github.com/ibis-project/ibis/issues/2127))
* Fix millisecond issue for OmniSciDB :issue:`2167`, MySQL :issue:`2169`, PostgreSQL :issue:`2166`, pandas :issue:`2168`, BigQuery :issue:`2273` backends ([#2170](https://github.com/ibis-project/ibis/issues/2170))
* [OmniSciDB] Fix TopK when used as filter ([#2134](https://github.com/ibis-project/ibis/issues/2134))

## Support

* Move `ibis.HDFS`, `ibis.WebHDFS` and `ibis.hdfs_connect` to `ibis.impala.*` ([#2497](https://github.com/ibis-project/ibis/issues/2497))
* Drop support to Python 3.6 ([#2288](https://github.com/ibis-project/ibis/issues/2288))
* Simplifying tests directories structure ([#2351](https://github.com/ibis-project/ibis/issues/2351))
* Update ``google-cloud-bigquery`` dependency minimum version to 1.12.0 ([#2304](https://github.com/ibis-project/ibis/issues/2304))
* Remove "experimental" mentions for OmniSciDB and pandas backends ([#2234](https://github.com/ibis-project/ibis/issues/2234))
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
-   All recent versions of SQLAlchemy are now supported
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
-   pandas 0.23 compatibility ([#1458](https://github.com/ibis-project/ibis/issues/1458))
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
against files, MySQL, pandas user defined scalar and aggregations along
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
-   Fix subquery extraction, which wasn't happening in topological
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

This release brings initial pandas backend support along with a number
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

-   Apache Kudu (incubating) integration for Impala users. Will
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

First public release. See https://ibis-project.org for more.

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
