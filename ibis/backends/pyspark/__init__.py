from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyspark
import sqlalchemy as sa
import sqlglot as sg
from pyspark import SparkConf
from pyspark.sql import DataFrame, SparkSession

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateDatabase
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import canonicalize_context, localize_context
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.compiler import Compiler, TableSetFormatter
from ibis.backends.base.sql.ddl import (
    CreateDatabase,
    DropTable,
    TruncateTable,
    is_fully_qualified,
)
from ibis.backends.pyspark import ddl
from ibis.backends.pyspark.client import PySparkTable
from ibis.backends.pyspark.compiler import PySparkExprTranslator
from ibis.backends.pyspark.datatypes import PySparkType

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


class PySparkTableSetFormatter(TableSetFormatter):
    def _format_in_memory_table(self, op):
        # we don't need to compile the table to a VALUES statement because the
        # table has been registered already by createOrReplaceTempView.
        #
        # The only place where the SQL API is currently used is DDL operations
        return op.name


class PySparkCompiler(Compiler):
    cheap_in_memory_tables = True
    table_set_formatter_class = PySparkTableSetFormatter


class Backend(BaseSQLBackend, CanCreateDatabase):
    compiler = PySparkCompiler
    name = "pyspark"
    _sqlglot_dialect = "spark"

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
        url = sa.engine.make_url(url)

        conf = SparkConf().setAll(url.query.items())

        if database := url.database:
            conf = conf.set(
                "spark.sql.warehouse.dir",
                str(Path(database).absolute()),
            )

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

    @property
    def version(self):
        return pyspark.__version__

    @property
    def current_database(self) -> str:
        return self._catalog.currentDatabase()

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

    def compile(self, expr, timecontext=None, params=None, *args, **kwargs):
        """Compile an ibis expression to a PySpark DataFrame object."""

        if timecontext is not None:
            session_timezone = self._session.conf.get("spark.sql.session.timeZone")
            # Since spark use session timezone for tz-naive timestamps
            # we localize tz-naive context here to match that behavior
            timecontext = localize_context(
                canonicalize_context(timecontext), session_timezone
            )

        # Insert params in scope
        scope = Scope(
            {
                param.op(): raw_value
                for param, raw_value in ({} if params is None else params).items()
            },
            timecontext,
        )
        return PySparkExprTranslator().translate(
            expr.op(),
            scope=scope,
            timecontext=timecontext,
            session=getattr(self, "_session", None),
        )

    def execute(self, expr: ir.Expr, **kwargs: Any) -> Any:
        """Execute an expression."""
        table_expr = expr.as_table()
        df = self.compile(table_expr, **kwargs).toPandas()

        # TODO: remove the extra conversion
        return expr.__pandas_result__(table_expr.__pandas_result__(df))

    def _fully_qualified_name(self, name, database):
        if is_fully_qualified(name):
            return name

        return sg.table(name, db=database, quoted=True).sql(dialect="spark")

    def close(self):
        """Close Spark connection and drop any temporary objects."""
        self._context.stop()

    def fetch_from_cursor(self, cursor, schema):
        df = cursor.query.toPandas()  # blocks until finished
        return schema.apply_to(df)

    def raw_sql(self, query: str) -> _PySparkCursor:
        query = self._session.sql(query)
        return _PySparkCursor(query)

    def _get_schema_using_query(self, query):
        cursor = self.raw_sql(f"SELECT * FROM ({query}) t0 LIMIT 0")
        struct = PySparkType.to_ibis(cursor.query.schema)
        return sch.Schema(struct)

    def _get_jtable(self, name, database=None):
        get_table = self._catalog._jcatalog.getTable
        try:
            jtable = get_table(self._fully_qualified_name(name, database))
        except pyspark.sql.utils.AnalysisException as e1:
            try:
                jtable = get_table(self._fully_qualified_name(name, database=None))
            except pyspark.sql.utils.AnalysisException as e2:
                raise com.IbisInputError(str(e2)) from e1
        return jtable

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Return a table expression from a table or view in the database.

        Parameters
        ----------
        name
            Table name
        database
            Database in which the table resides

        Returns
        -------
        Table
            Table named `name` from `database`
        """
        jtable = self._get_jtable(name, database)
        name, database = jtable.name(), jtable.database()

        qualified_name = self._fully_qualified_name(name, database)

        schema = self.get_schema(qualified_name)
        node = ops.DatabaseTable(
            name, schema, self, namespace=ops.Namespace(database=database)
        )
        return PySparkTable(node)

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
        statement = CreateDatabase(name, path=path, can_exist=force)
        return self.raw_sql(statement.compile())

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
        statement = ddl.DropDatabase(name, must_exist=not force, cascade=force)
        return self.raw_sql(statement.compile())

    def get_schema(
        self,
        table_name: str,
        database: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name. May be fully qualified
        database
            Spark does not have a database argument for its table() method,
            so this must be None

        Returns
        -------
        Schema
            An ibis schema
        """
        if database is not None:
            raise com.UnsupportedArgumentError(
                "Spark does not support the `database` argument for `get_schema`"
            )

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
        >>> con.create_table(
        ...     "new_table_name", table_expr
        ... )  # quartodoc: +SKIP # doctest: +SKIP
        """
        import pandas as pd
        import pyarrow as pa

        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")
        if temp is True:
            raise NotImplementedError(
                "PySpark backend does not yet support temporary tables"
            )
        if obj is not None:
            if isinstance(obj, pa.Table):
                obj = obj.to_pandas()
            if isinstance(obj, pd.DataFrame):
                spark_df = self._session.createDataFrame(obj)
                mode = "overwrite" if overwrite else "error"
                spark_df.write.saveAsTable(name, format=format, mode=mode)
                return None
            else:
                self._register_in_memory_tables(obj)

            ast = self.compiler.to_ast(obj)
            select = ast.queries[0]

            statement = ddl.CTAS(
                name,
                select,
                database=database,
                can_exist=overwrite,
                format=format,
            )
        else:
            statement = ddl.CreateTableWithSchema(
                name,
                schema,
                database=database,
                format=format,
                can_exist=overwrite,
            )

        self.raw_sql(statement.compile())
        return self.table(name, database=database)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self.compile(op.to_expr()).createOrReplaceTempView(op.name)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a Spark view from a table expression.

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
        ast = self.compiler.to_ast(obj)
        select = ast.queries[0]
        statement = ddl.CreateView(
            name, select, database=database, can_exist=overwrite, temporary=True
        )
        self.raw_sql(statement.compile())
        return self.table(name, database=database)

    def drop_table(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table."""
        self.drop_table_or_view(name, database=database, force=force)

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ):
        """Drop a view."""
        self.drop_table_or_view(name, database=database, force=force)

    def drop_table_or_view(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a Spark table or view.

        Parameters
        ----------
        name
            Table or view name
        database
            Database name
        force
            Database may throw exception if table does not exist

        Examples
        --------
        >>> table = "my_table"
        >>> db = "operations"
        >>> con.drop_table_or_view(
        ...     table, db, force=True
        ... )  # quartodoc: +SKIP # doctest: +SKIP
        """
        statement = DropTable(name, database=database, must_exist=not force)
        self.raw_sql(statement.compile())

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.
        """
        statement = ddl.RenameTable(old_name, new_name)
        self.raw_sql(statement.compile())

    def truncate_table(self, name: str, database: str | None = None) -> None:
        """Delete all rows from an existing table.

        Parameters
        ----------
        name
            Table name
        database
            Database name
        """
        statement = TruncateTable(name, database=database)
        self.raw_sql(statement.compile())

    def insert(
        self,
        table_name: str,
        obj: ir.Table | pd.DataFrame | None = None,
        database: str | None = None,
        overwrite: bool = False,
        values: Any | None = None,
        validate: bool = True,
    ) -> Any:
        """Insert data into an existing table.

        Examples
        --------
        >>> table = "my_table"
        >>> con.insert(table, table_expr)  # quartodoc: +SKIP # doctest: +SKIP

        # Completely overwrite contents
        >>> con.insert(
        ...     table, table_expr, overwrite=True
        ... )  # quartodoc: +SKIP # doctest: +SKIP
        """
        table = self.table(table_name, database=database)
        return table.insert(
            obj=obj, overwrite=overwrite, values=values, validate=validate
        )

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
        name = self._fully_qualified_name(name, database)
        return self.raw_sql(f"ANALYZE TABLE {name} COMPUTE STATISTICS{maybe_noscan}")

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return operation in PySparkExprTranslator._registry

    def _load_into_cache(self, name, expr):
        t = expr.compile().cache()
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

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        raise NotImplementedError(f"Backend '{self.name}' backend doesn't support SQL")

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
