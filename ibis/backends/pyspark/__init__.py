from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyspark
import sqlglot as sg
import sqlglot.expressions as sge
from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import PandasUDFType, pandas_udf

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateDatabase
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.pyspark.compiler import PySparkCompiler
from ibis.backends.pyspark.converter import PySparkPandasData
from ibis.backends.pyspark.datatypes import PySparkSchema, PySparkType
from ibis.expr.operations.udf import InputType
from ibis.legacy.udf.vectorized import _coerce_to_series

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    import pyarrow as pa


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


class _PySparkCursor:
    """Spark cursor.

    This allows the Spark client to reuse machinery in
    `ibis/backends/base/sql/client.py`.
    """

    def __init__(self, query: DataFrame) -> None:
        """Construct a cursor with query `query`.

        Parameters
        ----------
        query
            PySpark query

        """
        self.query = query

    def fetchall(self):
        """Fetch all rows."""
        result = self.query.collect()  # blocks until finished
        return result

    def fetchmany(self, nrows: int):
        raise NotImplementedError()

    @property
    def columns(self):
        """Return the columns of the result set."""
        return self.query.columns

    @property
    def description(self):
        """Get the fields of the result set's schema."""
        return self.query.schema

    def __enter__(self):
        # For compatibility when constructed from Query.execute()
        """No-op for compatibility."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """No-op for compatibility."""


class Backend(SQLGlotBackend, CanCreateDatabase):
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

    def _from_url(self, url: str, **kwargs) -> Backend:
        """Construct a PySpark backend from a URL `url`."""
        from urllib.parse import parse_qs, urlparse

        url = urlparse(url)
        query_params = parse_qs(url.query)
        params = query_params.copy()

        for name, value in query_params.items():
            if len(value) > 1:
                params[name] = value
            elif len(value) == 1:
                params[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        conf = SparkConf().setAll(params.items())

        if database := url.path[1:]:
            conf = conf.set("spark.sql.warehouse.dir", str(Path(database).absolute()))

        builder = SparkSession.builder.config(conf=conf)
        session = builder.getOrCreate()
        return self.connect(session, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_dataframes = {}

    def do_connect(self, session: SparkSession) -> None:
        """Create a PySpark `Backend` for use with Ibis.

        Parameters
        ----------
        session
            A SparkSession instance

        Examples
        --------
        >>> import ibis
        >>> from pyspark.sql import SparkSession
        >>> session = SparkSession.builder.getOrCreate()
        >>> ibis.pyspark.connect(session)
        <ibis.backends.pyspark.Backend at 0x...>

        """
        self._context = session.sparkContext
        self._session = session
        self._catalog = session.catalog

        # Spark internally stores timestamps as UTC values, and timestamp data
        # that is brought in without a specified time zone is converted as
        # local time to UTC with microsecond resolution.
        # https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#timestamp-with-time-zone-semantics
        self._session.conf.set("spark.sql.session.timeZone", "UTC")
        self._session.conf.set("spark.sql.mapKeyDedupPolicy", "LAST_WIN")

    def disconnect(self) -> None:
        self._session.stop()

    def _metadata(self, query: str):
        cursor = self.raw_sql(query)
        struct_dtype = PySparkType.to_ibis(cursor.query.schema)
        return struct_dtype.items()

    @property
    def version(self):
        return pyspark.__version__

    @property
    def current_database(self) -> str:
        return self._catalog.currentDatabase()

    @contextlib.contextmanager
    def _active_database(self, name: str | None) -> None:
        if name is None:
            yield
            return
        current = self.current_database
        try:
            self._catalog.setCurrentDatabase(name)
            yield
        finally:
            self._catalog.setCurrentDatabase(current)

    def list_databases(self, like: str | None = None) -> list[str]:
        databases = [db.name for db in self._catalog.listDatabases()]
        return self._filter_with_like(databases, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        tables = [
            t.name
            for t in self._catalog.listTables(dbName=database or self.current_database)
        ]
        return self._filter_with_like(tables, like)

    def _wrap_udf_to_return_pandas(self, func, output_dtype):
        def wrapper(*args):
            return _coerce_to_series(func(*args), output_dtype)

        return wrapper

    def _register_udfs(self, expr: ir.Expr) -> None:
        node = expr.op()
        for udf in node.find(ops.ScalarUDF):
            udf_name = self.compiler.__sql_name__(udf)
            udf_func = self._wrap_udf_to_return_pandas(udf.__func__, udf.dtype)
            udf_return = PySparkType.from_ibis(udf.dtype)
            if udf.__input_type__ != InputType.PANDAS:
                raise NotImplementedError(
                    "Only Pandas UDFs are support in the PySpark backend"
                )
            spark_udf = pandas_udf(udf_func, udf_return, PandasUDFType.SCALAR)
            self._session.udf.register(udf_name, spark_udf)

        for udf in node.find(ops.ElementWiseVectorizedUDF):
            udf_name = self.compiler.__sql_name__(udf)
            udf_func = self._wrap_udf_to_return_pandas(udf.func, udf.return_type)
            udf_return = PySparkType.from_ibis(udf.return_type)
            spark_udf = pandas_udf(udf_func, udf_return, PandasUDFType.SCALAR)
            self._session.udf.register(udf_name, spark_udf)

        for udf in node.find(ops.ReductionVectorizedUDF):
            udf_name = self.compiler.__sql_name__(udf)
            udf_func = self._wrap_udf_to_return_pandas(udf.func, udf.return_type)
            udf_func = udf.func
            udf_return = PySparkType.from_ibis(udf.return_type)
            spark_udf = pandas_udf(udf_func, udf_return, PandasUDFType.GROUPED_AGG)
            self._session.udf.register(udf_name, spark_udf)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = PySparkSchema.from_ibis(op.schema)
        df = self._session.createDataFrame(data=op.data.to_frame(), schema=schema)
        df.createOrReplaceTempView(op.name)

    def _fetch_from_cursor(self, cursor, schema):
        df = cursor.query.toPandas()  # blocks until finished
        return PySparkPandasData.convert_table(df, schema)

    def _safe_raw_sql(self, query: str) -> _PySparkCursor:
        return self.raw_sql(query)

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> _PySparkCursor:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)
        query = self._session.sql(query)
        return _PySparkCursor(query)

    def create_database(
        self,
        name: str,
        path: str | Path | None = None,
        force: bool = False,
    ) -> Any:
        """Create a new Spark database.

        Parameters
        ----------
        name
            Database name
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
        with self._safe_raw_sql(sql):
            pass

    def drop_database(self, name: str, force: bool = False) -> Any:
        """Drop a Spark database.

        Parameters
        ----------
        name
            Database name
        force
            If False, Spark throws exception if database is not empty or
            database does not exist

        """
        sql = sge.Drop(kind="DATABASE", exist=force, this=sg.to_identifier(name))
        with self._safe_raw_sql(sql):
            pass

    def get_schema(
        self,
        table_name: str,
        schema: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name. May be fully qualified
        schema
            Spark does not have a schema argument for its table() method,
            so this must be None
        database
            Spark does not have a database argument for its table() method,
            so this must be None

        Returns
        -------
        Schema
            An ibis schema

        """
        if schema is not None:
            raise com.UnsupportedArgumentError(
                "Spark does not support the `schema` argument for `get_schema`"
            )

        with self._active_database(database):
            df = self._session.table(table_name)
            struct = PySparkType.to_ibis(df.schema)

        return sch.Schema(struct)

    def create_table(
        self,
        name: str,
        obj: ir.Table | pd.DataFrame | pa.Table | None = None,
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
        temp
            Whether the new table is temporary
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

        if obj is not None:
            table = obj if isinstance(obj, ir.Expr) else ibis.memtable(obj)
            query = self.compile(table)
            mode = "overwrite" if overwrite else "error"
            with self._active_database(database):
                self._run_pre_execute_hooks(table)
                df = self._session.sql(query)
                df.write.saveAsTable(name, format=format, mode=mode)
        elif schema is not None:
            schema = PySparkSchema.from_ibis(schema)
            with self._active_database(database):
                self._catalog.createTable(name, schema=schema, format=format)
        else:
            raise com.IbisError("The schema or obj parameter is required")

        return self.table(name, database=database)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        schema: str | None = None,
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
        schema
            Schema name
        overwrite
            Replace an existing view of the same name if it exists

        Returns
        -------
        Table
            The created view

        """
        src = sge.Create(
            this=sg.table(
                name, db=schema, catalog=database, quoted=self.compiler.quoted
            ),
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

    def _clean_up_cached_table(self, op):
        name = op.name
        self._session.catalog.dropTempView(name)
        t = self._cached_dataframes.pop(name)
        assert t.is_cached
        t.unpersist()
        assert not t.is_cached

    def read_delta(
        self,
        source: str | Path,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        source
            The path to the Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        kwargs
            Additional keyword arguments passed to PySpark.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.load.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source = util.normalize_filename(source)
        spark_df = self._session.read.format("delta").load(source, **kwargs)
        table_name = table_name or util.gen_name("read_delta")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

    def read_parquet(
        self,
        source: str | Path,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        source
            The data source. May be a path to a file or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        kwargs
            Additional keyword arguments passed to PySpark.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.parquet.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source = util.normalize_filename(source)
        spark_df = self._session.read.parquet(source, **kwargs)
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
            a sequentially generated name.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.csv.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
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
            a sequentially generated name.
        kwargs
            Additional keyword arguments passed to PySpark loading function.
            https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.json.html

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source_list = normalize_filenames(source_list)
        spark_df = self._session.read.json(source_list, **kwargs)
        table_name = table_name or util.gen_name("read_json")

        spark_df.createOrReplaceTempView(table_name)
        return self.table(table_name)

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
            a sequentially generated name.
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

        **kwargs
            PySpark Delta Lake table write arguments. https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrameWriter.save.html

        """
        expr.compile().write.format("delta").save(os.fspath(path), **kwargs)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
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
        pa = self._import_pyarrow()
        pa_table = self.to_pyarrow(
            expr.as_table(), params=params, limit=limit, **kwargs
        )
        return pa.RecordBatchReader.from_batches(
            pa_table.schema, pa_table.to_batches(max_chunksize=chunk_size)
        )
