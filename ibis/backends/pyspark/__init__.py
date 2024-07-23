from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyspark
import pyspark.sql.functions as F
import sqlglot as sg
import sqlglot.expressions as sge
from packaging.version import parse as vparse
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, DoubleType, LongType, StringType

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, CanListCatalog
from ibis.backends.pyspark.converter import PySparkPandasData
from ibis.backends.pyspark.datatypes import PySparkSchema, PySparkType
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers import PySparkCompiler
from ibis.expr.operations.udf import InputType
from ibis.legacy.udf.vectorized import _coerce_to_series
from ibis.util import deprecated

try:
    from pyspark.errors import ParseException as PySparkParseException
except ImportError:
    from pyspark.sql.utils import ParseException as PySparkParseException

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from pyspark.sql.streaming import StreamingQuery

    from ibis.expr.api import Watermark

PYSPARK_LT_34 = vparse(pyspark.__version__) < vparse("3.4")

ConnectionMode = Literal["streaming", "batch"]


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


@F.pandas_udf(returnType=DoubleType(), functionType=F.PandasUDFType.SCALAR)
def unwrap_json_float(s: pd.Series) -> pd.Series:
    import json

    import pandas as pd

    def nullify_type_mismatched_value(raw):
        if pd.isna(raw):
            return None

        value = json.loads(raw)
        # exact type check because we want to distinguish between integer
        # and booleans and bool is a subclass of int
        return value if type(value) in (float, int) else None

    return s.map(nullify_type_mismatched_value)


def unwrap_json(typ):
    import json

    import pandas as pd

    type_mapping = {str: StringType(), int: LongType(), bool: BooleanType()}

    @F.pandas_udf(returnType=type_mapping[typ], functionType=F.PandasUDFType.SCALAR)
    def unwrap(s: pd.Series) -> pd.Series:
        def nullify_type_mismatched_value(raw):
            if pd.isna(raw):
                return None

            value = json.loads(raw)
            # exact type check because we want to distinguish between integer
            # and booleans and bool is a subclass of int
            return value if type(value) is typ else None

        return s.map(nullify_type_mismatched_value)

    return unwrap


def _interval_to_string(interval):
    return f"{interval.op().value} {interval.op().dtype.unit.name.lower()}"


class Backend(SQLBackend, CanListCatalog, CanCreateDatabase):
    name = "pyspark"
    compiler = PySparkCompiler()

    class Options(ibis.config.Config):
        """PySpark options.

        Attributes
        ----------
        treat_nan_as_null : bool
            Treat NaNs in floating point expressions as NULL.

        """

        treat_nan_as_null: bool = False

    def _from_url(self, url: ParseResult, **kwargs) -> Backend:
        """Construct a PySpark backend from a URL `url`."""
        conf = SparkConf().setAll(kwargs.items())

        if database := url.path[1:]:
            conf = conf.set("spark.sql.warehouse.dir", str(Path(database).absolute()))

        builder = SparkSession.builder.config(conf=conf)
        session = builder.getOrCreate()
        return self.connect(session, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_dataframes = {}

    def do_connect(
        self,
        session: SparkSession | None = None,
        mode: ConnectionMode = "batch",
        **kwargs,
    ) -> None:
        """Create a PySpark `Backend` for use with Ibis.

        Parameters
        ----------
        session
            A `SparkSession` instance.
        mode
            Can be either "batch" or "streaming". If "batch", every source, sink, and
            query executed within this connection will be interpreted as a batch
            workload. If "streaming", every source, sink, and query executed within
            this connection will be interpreted as a streaming workload.
        kwargs
            Additional keyword arguments used to configure the SparkSession.

        Examples
        --------
        >>> import ibis
        >>> from pyspark.sql import SparkSession
        >>> session = SparkSession.builder.getOrCreate()
        >>> ibis.pyspark.connect(session)
        <ibis.backends.pyspark.Backend at 0x...>

        """
        if session is None:
            from pyspark.sql import SparkSession

            session = SparkSession.builder.getOrCreate()

        if mode not in ("batch", "streaming"):
            raise com.IbisInputError(
                f"Invalid connection mode: {mode}, must be `streaming` or `batch`"
            )
        self._mode = mode

        self._session = session

        # Spark internally stores timestamps as UTC values, and timestamp data
        # that is brought in without a specified time zone is converted as
        # local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
        self._session.conf.set("spark.sql.session.timeZone", "UTC")
        self._session.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")

        for key, value in kwargs.items():
            self._session.conf.set(key, value)

    @util.experimental
    @classmethod
    def from_connection(
        cls, session: SparkSession, mode: ConnectionMode = "batch", **kwargs
    ) -> Backend:
        """Create a PySpark `Backend` from an existing `SparkSession` instance.

        Parameters
        ----------
        session
            A `SparkSession` instance.
        mode
            Can be either "batch" or "streaming". If "batch", every source, sink, and
            query executed within this connection will be interpreted as a batch
            workload. If "streaming", every source, sink, and query executed within
            this connection will be interpreted as a streaming workload.
        kwargs
            Additional keyword arguments used to configure the SparkSession.
        """
        return ibis.pyspark.connect(session, mode, **kwargs)

    def disconnect(self) -> None:
        self._session.stop()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        df = self.raw_sql(query)
        struct_dtype = PySparkType.to_ibis(df.schema)
        return sch.Schema(struct_dtype)

    @property
    def mode(self) -> ConnectionMode:
        return self._mode

    @property
    def version(self):
        return pyspark.__version__

    @property
    def current_database(self) -> str:
        [(db,)] = self._session.sql("SELECT CURRENT_DATABASE()").collect()
        return db

    @property
    def current_catalog(self) -> str:
        [(catalog,)] = self._session.sql("SELECT CURRENT_CATALOG()").collect()
        return catalog

    @contextlib.contextmanager
    def _active_catalog_database(self, catalog: str | None, db: str | None):
        if catalog is None and db is None:
            yield
            return
        if catalog is not None and PYSPARK_LT_34:
            raise com.UnsupportedArgumentError(
                "Catalogs are not supported in pyspark < 3.4"
            )
        current_catalog = self.current_catalog
        current_db = self.current_database

        # This little horrible bit of work is to avoid trying to set
        # the `CurrentDatabase` inside of a catalog where we don't have permission
        # to do so.  We can't have the catalog and database context managers work
        # separately because we need to:
        # 1. set catalog
        # 2. set database
        # 3. set catalog to previous
        # 4. set database to previous
        #
        # Unity catalog has special handling for "USE CATALOG" and "USE DATABASE"
        # and also has weird permissioning around using `setCurrentCatalog` and
        # `setCurrentDatabase`.
        #
        # We attempt to use the Unity-specific Spark SQL to set CATALOG and DATABASE
        # and if that causes a parser exception we fall back to using the catalog API.
        try:
            if catalog is not None:
                try:
                    catalog_sql = sg.to_identifier(catalog).sql(self.dialect)
                    self.raw_sql(f"USE CATALOG {catalog_sql}")
                except PySparkParseException:
                    self._session.catalog.setCurrentCatalog(catalog)
            try:
                db_sql = sg.to_identifier(db).sql(self.dialect)
                self.raw_sql(f"USE DATABASE {db_sql}")
            except PySparkParseException:
                self._session.catalog.setCurrentDatabase(db)
            yield
        finally:
            if catalog is not None:
                try:
                    catalog_sql = sg.to_identifier(current_catalog).sql(self.dialect)
                    self.raw_sql(f"USE CATALOG {catalog_sql}")
                except PySparkParseException:
                    self._session.catalog.setCurrentCatalog(current_catalog)
            try:
                db_sql = sg.to_identifier(current_db).sql(self.dialect)
                self.raw_sql(f"USE DATABASE {db_sql}")
            except PySparkParseException:
                self._session.catalog.setCurrentDatabase(current_db)

    @contextlib.contextmanager
    def _active_catalog(self, name: str | None):
        if name is None or PYSPARK_LT_34:
            yield
            return
        prev_catalog = self.current_catalog
        prev_database = self.current_database
        try:
            try:
                catalog_sql = sg.to_identifier(name).sql(self.dialect)
                self.raw_sql(f"USE CATALOG {catalog_sql};")
            except PySparkParseException:
                self._session.catalog.setCurrentCatalog(name)
            yield
        finally:
            try:
                catalog_sql = sg.to_identifier(prev_catalog).sql(self.dialect)
                db_sql = sg.to_identifier(prev_database).sql(self.dialect)
                self.raw_sql(f"USE CATALOG {catalog_sql};")
                self.raw_sql(f"USE DATABASE {db_sql};")
            except PySparkParseException:
                self._session.catalog.setCurrentCatalog(prev_catalog)
                self._session.catalog.setCurrentDatabase(prev_database)

    def list_catalogs(self, like: str | None = None) -> list[str]:
        catalogs = [res.catalog for res in self._session.sql("SHOW CATALOGS").collect()]
        return self._filter_with_like(catalogs, like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        with self._active_catalog(catalog):
            databases = [
                db.namespace for db in self._session.sql("SHOW DATABASES").collect()
            ]
        return self._filter_with_like(databases, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current catalog and database.

            To specify a table in a separate catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)
        with self._active_catalog(catalog):
            tables = [
                row.tableName
                for row in self._session.sql(
                    f"SHOW TABLES IN {db or self.current_database}"
                ).collect()
            ]
        return self._filter_with_like(tables, like)

    def _wrap_udf_to_return_pandas(self, func, output_dtype):
        def wrapper(*args):
            return _coerce_to_series(func(*args), output_dtype)

        return wrapper

    def _register_udfs(self, expr: ir.Expr) -> None:
        node = expr.op()
        for udf in node.find(ops.ScalarUDF):
            if udf.__input_type__ not in (InputType.PANDAS, InputType.BUILTIN):
                raise NotImplementedError(
                    "Only Builtin UDFs and Pandas UDFs are supported in the PySpark backend"
                )
            # register pandas UDFs
            if udf.__input_type__ == InputType.PANDAS:
                udf_name = self.compiler.__sql_name__(udf)
                udf_func = self._wrap_udf_to_return_pandas(udf.__func__, udf.dtype)
                udf_return = PySparkType.from_ibis(udf.dtype)
                spark_udf = F.pandas_udf(udf_func, udf_return, F.PandasUDFType.SCALAR)
                self._session.udf.register(udf_name, spark_udf)

        for udf in node.find(ops.ElementWiseVectorizedUDF):
            udf_name = self.compiler.__sql_name__(udf)
            udf_func = self._wrap_udf_to_return_pandas(udf.func, udf.return_type)
            udf_return = PySparkType.from_ibis(udf.return_type)
            spark_udf = F.pandas_udf(udf_func, udf_return, F.PandasUDFType.SCALAR)
            self._session.udf.register(udf_name, spark_udf)

        for udf in node.find(ops.ReductionVectorizedUDF):
            udf_name = self.compiler.__sql_name__(udf)
            udf_func = self._wrap_udf_to_return_pandas(udf.func, udf.return_type)
            udf_func = udf.func
            udf_return = PySparkType.from_ibis(udf.return_type)
            spark_udf = F.pandas_udf(udf_func, udf_return, F.PandasUDFType.GROUPED_AGG)
            self._session.udf.register(udf_name, spark_udf)

        for typ in (str, int, bool):
            self._session.udf.register(f"unwrap_json_{typ.__name__}", unwrap_json(typ))
        self._session.udf.register("unwrap_json_float", unwrap_json_float)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = PySparkSchema.from_ibis(op.schema)
        df = self._session.createDataFrame(data=op.data.to_frame(), schema=schema)
        df.createOrReplaceTempView(op.name)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str) -> Any:
        yield self.raw_sql(query)

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)
        return self._session.sql(query, **kwargs)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, params=params, limit=limit, **kwargs)

        schema = table.schema()

        with self._safe_raw_sql(sql) as query:
            df = query.toPandas()  # blocks until finished
            result = PySparkPandasData.convert_table(df, schema)
        return expr.__pandas_result__(result)

    def create_database(
        self,
        name: str,
        *,
        catalog: str | None = None,
        path: str | Path | None = None,
        force: bool = False,
    ) -> Any:
        """Create a new Spark database.

        Parameters
        ----------
        name
            Database name
        catalog
            Catalog to create database in (defaults to `current_catalog`)
        path
            Path where to store the database data; otherwise uses Spark default
        force
            Whether to append `IF NOT EXISTS` to the database creation SQL

        """
        if path is not None:
            properties = sge.Properties(
                expressions=[sge.LocationProperty(this=sge.convert(str(path)))]
            )
        else:
            properties = None

        sql = sge.Create(
            kind="DATABASE",
            exist=force,
            this=sg.to_identifier(name),
            properties=properties,
        )
        with self._active_catalog(catalog):
            with self._safe_raw_sql(sql):
                pass

    def drop_database(
        self, name: str, *, catalog: str | None = None, force: bool = False
    ) -> Any:
        """Drop a Spark database.

        Parameters
        ----------
        name
            Database name
        catalog
            Catalog containing database to drop (defaults to `current_catalog`)
        force
            If False, Spark throws exception if database is not empty or
            database does not exist

        """
        sql = sge.Drop(
            kind="DATABASE", exist=force, this=sg.to_identifier(name), cascade=force
        )
        with self._active_catalog(catalog):
            with self._safe_raw_sql(sql):
                pass

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name. May be fully qualified
        catalog
            Catalog to use
        database
            Database to use to get the active database.

        Returns
        -------
        Schema
            An ibis schema

        """

        table_loc = self._to_sqlglot_table((catalog, database))
        catalog, db = self._to_catalog_db_tuple(table_loc)
        with self._active_catalog_database(catalog, db):
            df = self._session.table(table_name)
            struct = PySparkType.to_ibis(df.schema)

        return sch.Schema(struct)

    def create_table(
        self,
        name: str,
        obj: (
            ir.Table | pd.DataFrame | pa.Table | pl.DataFrame | pl.LazyFrame | None
        ) = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
        format: str = "parquet",
    ) -> ir.Table:
        """Create a new table in Spark.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            If passed, creates table from `SELECT` statement results
        schema
            Mutually exclusive with `obj`, creates an empty table with a schema
        database
            Database name

            To specify a table in a separate catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.
        temp
            Whether the new table is temporary (unsupported)
        overwrite
            If `True`, overwrite existing data
        format
            Format of the table on disk

        Returns
        -------
        Table
            The newly created table.

        Examples
        --------
        >>> con.create_table("new_table_name", table_expr)  # quartodoc: +SKIP # doctest: +SKIP

        """
        if temp is True:
            raise NotImplementedError(
                "PySpark backend does not yet support temporary tables"
            )

        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        temp_memtable_view = None
        if obj is not None:
            if isinstance(obj, ir.Expr):
                table = obj
            else:
                table = ibis.memtable(obj)
                temp_memtable_view = table.op().name
            query = self.compile(table)
            mode = "overwrite" if overwrite else "error"
            with self._active_catalog_database(catalog, db):
                self._run_pre_execute_hooks(table)
                df = self._session.sql(query)
                df.write.saveAsTable(name, format=format, mode=mode)
        elif schema is not None:
            schema = PySparkSchema.from_ibis(schema)
            with self._active_catalog_database(catalog, db):
                self._session.catalog.createTable(name, schema=schema, format=format)
        else:
            raise com.IbisError("The schema or obj parameter is required")

        # Clean up temporary memtable if we've created one
        # for in-memory reads
        if temp_memtable_view is not None:
            self.drop_table(temp_memtable_view)

        return self.table(name, database=(catalog, db))

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a temporary Spark view from a table expression.

        Parameters
        ----------
        name
            View name
        obj
            Expression to use for the view
        database
            Database name
        overwrite
            Replace an existing view of the same name if it exists

        Returns
        -------
        Table
            The created view

        """
        src = sge.Create(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="TEMPORARY VIEW",
            replace=overwrite,
            expression=self.compile(obj),
        )
        self._register_in_memory_tables(obj)
        with self._safe_raw_sql(src):
            pass
        return self.table(name, database=database)

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.

        """
        old = sg.table(old_name, quoted=True)
        new = sg.table(new_name, quoted=True)
        query = sge.AlterTable(
            this=old,
            exists=False,
            actions=[sge.RenameTable(this=new, exists=True)],
        )
        with self._safe_raw_sql(query):
            pass

    def compute_stats(
        self,
        name: str,
        database: str | None = None,
        noscan: bool = False,
    ) -> Any:
        """Issue a `COMPUTE STATISTICS` command for a given table.

        Parameters
        ----------
        name
            Table name
        database
            Database name
        noscan
            If `True`, collect only basic statistics for the table (number of
            rows, size in bytes).

        """
        maybe_noscan = " NOSCAN" * noscan
        table = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            dialect=self.dialect
        )
        return self.raw_sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS{maybe_noscan}")

    def _load_into_cache(self, name, expr):
        query = self.compile(expr)
        t = self._session.sql(query).cache()
        assert t.is_cached
        t.createOrReplaceTempView(name)
        # store the underlying spark dataframe so we can release memory when
        # asked to, instead of when the session ends
        self._cached_dataframes[name] = t

    def _clean_up_cached_table(self, name):
        self._session.catalog.dropTempView(name)
        t = self._cached_dataframes.pop(name)
        assert t.is_cached
        t.unpersist()
        assert not t.is_cached

    def read_delta(
        self,
        path: str | Path,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        path
            The path to the Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        kwargs
            Additional keyword arguments passed to PySpark.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.load.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        if self.mode == "streaming":
            raise NotImplementedError(
                "Reading a Delta Lake table in streaming mode is not supported"
            )
        path = util.normalize_filename(path)
        spark_df = self._session.read.format("delta").load(path, **kwargs)
        table_name = table_name or util.gen_name("read_delta")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source. May be a path to a file or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        kwargs
            Additional keyword arguments passed to PySpark.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.parquet.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        if self.mode == "streaming":
            raise NotImplementedError(
                "Pyspark in streaming mode does not support direction registration of parquet files. "
                "Please use `read_parquet_dir` instead."
            )
        path = util.normalize_filename(path)
        spark_df = self._session.read.parquet(path, **kwargs)
        table_name = table_name or util.gen_name("read_parquet")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    def read_csv(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        source_list
            The data source(s). May be a path to a file or directory of CSV files, or an
            iterable of CSV files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.csv.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        if self.mode == "streaming":
            raise NotImplementedError(
                "Pyspark in streaming mode does not support direction registration of CSV files. "
                "Please use `read_csv_dir` instead."
            )
        inferSchema = kwargs.pop("inferSchema", True)
        header = kwargs.pop("header", True)
        source_list = normalize_filenames(source_list)
        spark_df = self._session.read.csv(
            source_list, inferSchema=inferSchema, header=header, **kwargs
        )
        table_name = table_name or util.gen_name("read_csv")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    def read_json(
        self,
        source_list: str | Sequence[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a JSON file as a table in the current database.

        Parameters
        ----------
        source_list
            The data source(s). May be a path to a file or directory of JSON files, or an
            iterable of JSON files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.json.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        if self.mode == "streaming":
            raise NotImplementedError(
                "Pyspark in streaming mode does not support direction registration of JSON files. "
                "Please use `read_json_dir` instead."
            )
        source_list = normalize_filenames(source_list)
        spark_df = self._session.read.json(source_list, **kwargs)
        table_name = table_name or util.gen_name("read_json")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    @deprecated(
        as_of="9.1",
        instead="use the explicit `read_*` method for the filetype you are trying to read, e.g., read_parquet, read_csv, etc.",
    )
    def register(
        self,
        source: str | Path | Any,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a data source as a table in the current database.

        Parameters
        ----------
        source
            The data source(s). May be a path to a file or directory of
            parquet/csv files, or an iterable of CSV files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        **kwargs
            Additional keyword arguments passed to PySpark loading functions for
            CSV or parquet.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, (list, tuple)):
            first = source[0]
        else:
            self._register_failure()

        if first.startswith(("parquet://", "parq://")) or first.endswith(
            ("parq", "parquet")
        ):
            return self.read_parquet(source, table_name=table_name, **kwargs)
        elif first.startswith(
            ("csv://", "csv.gz://", "txt://", "txt.gz://")
        ) or first.endswith(("csv", "csv.gz", "tsv", "tsv.gz", "txt", "txt.gz")):
            return self.read_csv(source, table_name=table_name, **kwargs)
        else:
            self._register_failure()  # noqa: RET503

    def _register_failure(self):
        import inspect

        msg = ", ".join(
            name for name, _ in inspect.getmembers(self) if name.startswith("read_")
        )
        raise ValueError(
            f"Cannot infer appropriate read function for input, "
            f"please call one of {msg} directly"
        )

    @util.experimental
    def to_delta(
        self,
        expr: ir.Table,
        path: str | Path,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a Delta Lake table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to a Delta Lake table.
        path
            The data source. A string or Path to the Delta Lake table.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        **kwargs
            Additional keyword arguments passed to
            [pyspark.sql.DataFrameWriter](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameWriter.html).

        """
        if self.mode == "streaming":
            raise NotImplementedError(
                "Writing to a Delta Lake table in streaming mode is not supported"
            )
        self._run_pre_execute_hooks(expr)
        df = self._session.sql(self.compile(expr, params=params, limit=limit))
        df.write.format("delta").save(os.fspath(path), **kwargs)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        if self.mode == "streaming":
            raise NotImplementedError(
                "PySpark in streaming mode does not support to_pyarrow"
            )
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        from ibis.formats.pyarrow import PyArrowData

        table_expr = expr.as_table()
        output = pa.Table.from_pandas(
            self.execute(table_expr, params=params, limit=limit, **kwargs),
            preserve_index=False,
        )
        table = PyArrowData.convert_table(output, table_expr.schema())
        return expr.__pyarrow_result__(table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1000000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        if self.mode == "streaming":
            raise NotImplementedError(
                "PySpark in streaming mode does not support to_pyarrow_batches"
            )
        pa = self._import_pyarrow()
        pa_table = self.to_pyarrow(
            expr.as_table(), params=params, limit=limit, **kwargs
        )
        return pa.ipc.RecordBatchReader.from_batches(
            pa_table.schema, pa_table.to_batches(max_chunksize=chunk_size)
        )

    @util.experimental
    def read_kafka(
        self,
        table_name: str | None = None,
        *,
        watermark: Watermark | None = None,
        auto_parse: bool = False,
        schema: sch.Schema | None = None,
        options: Mapping[str, str] | None = None,
    ) -> ir.Table:
        """Register a Kafka topic as a table.

        Parameters
        ----------
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        watermark
            Watermark strategy for the table.
        auto_parse
            Whether to parse Kafka messages automatically. If `False`, the source is read
            as binary keys and values. If `True`, the key is discarded and the value is
            parsed using the provided schema.
        schema
            Schema of the value of the Kafka messages.
        options
            Additional arguments passed to PySpark as .option("key", "value").
            https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html

        Returns
        -------
        ir.Table
            The just-registered table
        """
        if self.mode == "batch":
            raise NotImplementedError(
                "Reading from Kafka in batch mode is not supported"
            )
        spark_df = self._session.readStream.format("kafka")
        for k, v in (options or {}).items():
            spark_df = spark_df.option(k, v)
        spark_df = spark_df.load()

        # parse the values of the Kafka messages using the provided schema
        if auto_parse:
            if schema is None:
                raise com.IbisError(
                    "When auto_parse is True, a schema must be provided to parse the messages"
                )
            schema = PySparkSchema.from_ibis(schema)
            spark_df = spark_df.select(
                F.from_json(F.col("value").cast("string"), schema).alias("parsed_value")
            ).select("parsed_value.*")

        if watermark is not None:
            spark_df = spark_df.withWatermark(
                watermark.time_col,
                _interval_to_string(watermark.allowed_delay),
            )

        table_name = table_name or util.gen_name("read_kafka")
        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    @util.experimental
    def to_kafka(
        self,
        expr: ir.Expr,
        *,
        auto_format: bool = False,
        options: Mapping[str, str] | None = None,
        params: Mapping | None = None,
        limit: str | None = "default",
    ) -> StreamingQuery:
        """Write the results of executing the given expression to a Kafka topic.

        This method does not return outputs. Streaming queries are run continuously in
        the background.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to a Kafka topic.
        auto_format
            Whether to format the Kafka messages before writing. If `False`, the output is
            written as-is. If `True`, the output is converted into JSON and written as the
            value of the Kafka messages.
        options
            PySpark Kafka write arguments.
            https://spark.apache.org/docs/latest/structured-streaming-kafka-integration.html
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.

        Returns
        -------
        StreamingQuery
            A Pyspark StreamingQuery object
        """
        if self.mode == "batch":
            raise NotImplementedError("Writing to Kafka in batch mode is not supported")
        df = self._session.sql(self.compile(expr, params=params, limit=limit))
        if auto_format:
            df = df.select(
                F.to_json(F.struct([F.col(c).alias(c) for c in df.columns])).alias(
                    "value"
                )
            )
        sq = df.writeStream.format("kafka")
        for k, v in (options or {}).items():
            sq = sq.option(k, v)
        sq.start()
        return sq

    @util.experimental
    def read_csv_dir(
        self,
        path: str | Path,
        table_name: str | None = None,
        watermark: Watermark | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV directory as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        watermark
            Watermark strategy for the table.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.ss/api/pyspark.sql.streaming.DataStreamReader.csv.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        inferSchema = kwargs.pop("inferSchema", True)
        header = kwargs.pop("header", True)
        path = util.normalize_filename(path)
        if self.mode == "batch":
            spark_df = self._session.read.csv(
                path, inferSchema=inferSchema, header=header, **kwargs
            )
            if watermark is not None:
                raise com.IbisInputError("Watermark is not supported in batch mode")
        elif self.mode == "streaming":
            spark_df = self._session.readStream.csv(
                path, inferSchema=inferSchema, header=header, **kwargs
            )
            if watermark is not None:
                spark_df = spark_df.withWatermark(
                    watermark.time_col,
                    _interval_to_string(watermark.allowed_delay),
                )
        table_name = table_name or util.gen_name("read_csv_dir")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    @util.experimental
    def read_parquet_dir(
        self,
        path: str | Path,
        table_name: str | None = None,
        watermark: Watermark | None = None,
        schema: sch.Schema | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source. A directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        watermark
            Watermark strategy for the table.
        schema
            Schema of the parquet source.
        kwargs
            Additional keyword arguments passed to PySpark.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.ss/api/pyspark.sql.streaming.DataStreamReader.parquet.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        path = util.normalize_filename(path)
        if self.mode == "batch":
            spark_df = self._session.read
            if schema is not None:
                spark_df = spark_df.schema(PySparkSchema.from_ibis(schema))
            spark_df = spark_df.parquet(path, **kwargs)
            if watermark is not None:
                raise com.IbisInputError("Watermark is not supported in batch mode")
        elif self.mode == "streaming":
            spark_df = self._session.readStream
            if schema is not None:
                spark_df = spark_df.schema(PySparkSchema.from_ibis(schema))
            spark_df = spark_df.parquet(path, **kwargs)
            if watermark is not None:
                spark_df = spark_df.withWatermark(
                    watermark.time_col,
                    _interval_to_string(watermark.allowed_delay),
                )
        table_name = table_name or util.gen_name("read_parquet_dir")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    @util.experimental
    def read_json_dir(
        self,
        path: str | Path,
        table_name: str | None = None,
        watermark: Watermark | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a JSON file as a table in the current database.

        Parameters
        ----------
        path
            The data source. A directory of JSON files.
        table_name
            An optional name to use for the created table. This defaults to
            a random generated name.
        watermark
            Watermark strategy for the table.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.ss/api/pyspark.sql.streaming.DataStreamReader.json.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        path = util.normalize_filename(path)
        if self.mode == "batch":
            spark_df = self._session.read.json(path, **kwargs)
            if watermark is not None:
                raise com.IbisInputError("Watermark is not supported in batch mode")
        elif self.mode == "streaming":
            spark_df = self._session.readStream.json(path, **kwargs)
            if watermark is not None:
                spark_df = spark_df.withWatermark(
                    watermark.time_col,
                    _interval_to_string(watermark.allowed_delay),
                )
        table_name = table_name or util.gen_name("read_json_dir")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    def _to_filesystem_output(
        self,
        expr: ir.Expr,
        format: str,
        path: str | Path,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        options: Mapping[str, str] | None = None,
    ) -> StreamingQuery | None:
        df = self._session.sql(self.compile(expr, params=params, limit=limit))
        if self.mode == "batch":
            df = df.write.format(format)
            for k, v in (options or {}).items():
                df = df.option(k, v)
            df.save(path)
            return None
        sq = df.writeStream.format(format)
        sq = sq.option("path", os.fspath(path))
        for k, v in (options or {}).items():
            sq = sq.option(k, v)
        sq.start()
        return sq

    @util.experimental
    def to_parquet_dir(
        self,
        expr: ir.Expr,
        path: str | Path,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        options: Mapping[str, str] | None = None,
    ) -> StreamingQuery | None:
        """Write the results of executing the given expression to a parquet directory.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to parquet.
        path
            The data source. A string or Path to the parquet directory.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        options
            Additional keyword arguments passed to pyspark.sql.streaming.DataStreamWriter

        Returns
        -------
        StreamingQuery | None
            Returns a Pyspark StreamingQuery object if in streaming mode, otherwise None
        """
        self._run_pre_execute_hooks(expr)
        return self._to_filesystem_output(expr, "parquet", path, params, limit, options)

    @util.experimental
    def to_csv_dir(
        self,
        expr: ir.Expr,
        path: str | Path,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        options: Mapping[str, str] | None = None,
    ) -> StreamingQuery | None:
        """Write the results of executing the given expression to a CSV directory.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to CSV.
        path
            The data source. A string or Path to the CSV directory.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        options
            Additional keyword arguments passed to pyspark.sql.streaming.DataStreamWriter

        Returns
        -------
        StreamingQuery | None
            Returns a Pyspark StreamingQuery object if in streaming mode, otherwise None
        """
        self._run_pre_execute_hooks(expr)
        return self._to_filesystem_output(expr, "csv", path, params, limit, options)
