from __future__ import annotations

import contextlib
import inspect
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datafusion as df
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateCatalog,
    CanCreateDatabase,
    DirectPyArrowExampleLoader,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoUrl,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import C
from ibis.common.dispatch import lazy_singledispatch
from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowSchema, PyArrowType
from ibis.util import gen_name, normalize_filename, normalize_filenames, warn_deprecated

try:
    from datafusion import ExecutionContext as SessionContext
except ImportError:
    from datafusion import SessionContext

try:
    from datafusion import SessionConfig
except ImportError:
    SessionConfig = None

try:
    from datafusion import RuntimeConfig
except ImportError:
    RuntimeConfig = None

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def as_nullable(dtype: dt.DataType) -> dt.DataType:
    """Recursively convert a possibly non-nullable datatype to a nullable one."""
    if dtype.is_struct():
        return dtype.copy(
            fields={name: as_nullable(typ) for name, typ in dtype.items()},
            nullable=True,
        )
    elif dtype.is_array():
        return dtype.copy(value_type=as_nullable(dtype.value_type), nullable=True)
    elif dtype.is_map():
        return dtype.copy(
            key_type=as_nullable(dtype.key_type),
            value_type=as_nullable(dtype.value_type),
            nullable=True,
        )
    else:
        return dtype.copy(nullable=True)


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoUrl,
    DirectPyArrowExampleLoader,
):
    name = "datafusion"
    supports_arrays = True
    compiler = sc.datafusion.compiler

    @property
    def version(self):
        import importlib.metadata

        return importlib.metadata.version("datafusion")

    def do_connect(
        self, config: Mapping[str, str | Path] | SessionContext | None = None
    ) -> None:
        """Create a DataFusion `Backend` for use with Ibis.

        Parameters
        ----------
        config
            Mapping of table names to files (deprecated in 10.0) or a `SessionContext`
            instance.

        Examples
        --------
        >>> from datafusion import SessionContext
        >>> ctx = SessionContext()
        >>> _ = ctx.from_pydict({"a": [1, 2, 3]}, "mytable")
        >>> import ibis
        >>> con = ibis.datafusion.connect(ctx)
        >>> con.list_tables()
        ['mytable']
        """
        if isinstance(config, SessionContext):
            (self.con, config) = (config, None)
        else:
            if config is not None and not isinstance(config, Mapping):
                raise TypeError("Input to ibis.datafusion.connect must be a mapping")
            elif config is not None and config:  # warn if dict is not empty
                warn_deprecated(
                    "Passing a mapping of tables names to files",
                    as_of="10.0",
                    instead="Please use the explicit `read_*` methods for the files you would like to load instead.",
                )
            if SessionConfig is not None:
                df_config = SessionConfig(
                    {"datafusion.sql_parser.dialect": "PostgreSQL"}
                ).with_information_schema(True)
            else:
                df_config = None
            if RuntimeConfig is None:
                self.con = SessionContext(df_config)
            else:
                # datafusion 40.1.0 has a bug where SessionContext requires
                # both SessionConfig and RuntimeConfig be provided.
                self.con = SessionContext(df_config, RuntimeConfig())

        self._register_builtin_udfs()

        if not config:
            config = {}

        for name, path in config.items():
            self._register(path, table_name=name)

    @util.experimental
    @classmethod
    def from_connection(cls, con: SessionContext, /) -> Backend:
        """Create a DataFusion `Backend` from an existing `SessionContext` instance.

        Parameters
        ----------
        con
            A `SessionContext` instance.
        """
        return ibis.datafusion.connect(con)

    def disconnect(self) -> None:
        pass

    @contextlib.contextmanager
    def _safe_raw_sql(self, sql: sge.Statement) -> Any:
        yield self.raw_sql(sql).collect()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        name = gen_name("datafusion_metadata_view")
        table = sg.table(name, quoted=self.compiler.quoted)
        src = sge.Create(
            this=table,
            kind="VIEW",
            expression=sg.parse_one(query, read=self.dialect),
        )

        with self._safe_raw_sql(src):
            pass

        try:
            df = self.con.table(name)
        finally:
            self.drop_view(name)

        return PyArrowSchema.to_ibis(df.schema())

    def _register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        import pandas as pd
        import pyarrow.dataset as ds

        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, pa.Table):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [source.to_batches()])
            return self.table(table_name)
        elif isinstance(source, pa.RecordBatch):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [[source]])
            return self.table(table_name)
        elif isinstance(source, ds.Dataset):
            self.con.deregister_table(table_name)
            self.con.register_dataset(table_name, source)
            return self.table(table_name)
        elif isinstance(source, pd.DataFrame):
            return self.register(pa.Table.from_pandas(source), table_name, **kwargs)
        else:
            raise ValueError("`source` must be either a string or a pathlib.Path")

        if first.startswith(("parquet://", "parq://")) or first.endswith(
            ("parq", "parquet")
        ):
            return self.read_parquet(source, table_name=table_name, **kwargs)
        elif first.startswith(("csv://", "txt://")) or first.endswith(
            ("csv", "tsv", "txt")
        ):
            return self.read_csv(source, table_name=table_name, **kwargs)
        else:
            self._register_failure()
            return None

    def _register_failure(self):
        import inspect

        msg = ", ".join(
            m[0] for m in inspect.getmembers(self) if m[0].startswith("read_")
        )
        raise ValueError(
            f"Cannot infer appropriate read function for input, "
            f"please call one of {msg} directly"
        )

    def _register_builtin_udfs(self):
        from ibis.backends.datafusion import udfs

        for name, func in inspect.getmembers(
            udfs,
            predicate=lambda m: callable(m)
            and not m.__name__.startswith("_")
            and m.__module__ == udfs.__name__,
        ):
            annotations = typing.get_type_hints(func)
            argnames = list(inspect.signature(func).parameters.keys())
            input_types = [
                PyArrowType.from_ibis(dt.dtype(annotations.get(arg_name)))
                for arg_name in argnames
            ]
            return_type = PyArrowType.from_ibis(dt.dtype(annotations["return"]))
            udf = df.udf(
                func,
                input_types=input_types,
                return_type=return_type,
                volatility="immutable",
                name=name,
            )
            self.con.register_udf(udf)

    def _register_udfs(self, expr: ir.Expr) -> None:
        for udf_node in expr.op().find(ops.ScalarUDF):
            if udf_node.__input_type__ == InputType.PYARROW:
                udf = self._compile_pyarrow_udf(udf_node)
                self.con.register_udf(udf)

        for udf_node in expr.op().find(ops.ElementWiseVectorizedUDF):
            udf = self._compile_elementwise_udf(udf_node)
            self.con.register_udf(udf)

    def _compile_pyarrow_udf(self, udf_node):
        return df.udf(
            udf_node.__func__,
            input_types=[PyArrowType.from_ibis(arg.dtype) for arg in udf_node.args],
            return_type=PyArrowType.from_ibis(udf_node.dtype),
            volatility=getattr(udf_node, "__config__", {}).get(
                "volatility", "volatile"
            ),
            name=udf_node.__func_name__,
        )

    def _compile_elementwise_udf(self, udf_node):
        return df.udf(
            udf_node.func,
            input_types=list(map(PyArrowType.from_ibis, udf_node.input_type)),
            return_type=PyArrowType.from_ibis(udf_node.return_type),
            volatility="volatile",
            name=udf_node.func.__name__,
        )

    def raw_sql(self, query: str | sge.Expression) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        kwargs
            Backend specific query arguments

        """
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect, pretty=True)
        self._log(query)
        return self.con.sql(query)

    @property
    def current_catalog(self):
        return str(
            self.sql(
                "select value from information_schema.df_settings where name='datafusion.catalog.default_catalog'"
            )
            .execute()
            .iloc[0, 0]
        )

    @property
    def current_database(self):
        return str(
            self.sql(
                "select value from information_schema.df_settings where name='datafusion.catalog.default_schema'"
            )
            .execute()
            .iloc[0, 0]
        )

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        code = (
            sg.select(C.table_catalog)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
        ).sql()
        result = self.con.sql(code).to_pydict()
        return self._filter_with_like(result["table_catalog"], like)

    def create_catalog(self, name: str, /, *, force: bool = False) -> None:
        with self._safe_raw_sql(
            sge.Create(kind="DATABASE", this=sg.to_identifier(name), exists=force)
        ):
            pass

    def drop_catalog(self, name: str, /, *, force: bool = False) -> None:
        raise com.UnsupportedOperationError(
            "DataFusion does not support dropping databases"
        )

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        if catalog is None:
            catalog = self.current_catalog
        return self._filter_with_like(self.con.catalog(catalog).names(), like=like)

    def create_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        # not actually a table, but this is how sqlglot represents schema names
        db_name = sg.table(name, db=catalog)
        with self._safe_raw_sql(sge.Create(kind="SCHEMA", this=db_name, exists=force)):
            pass

    def drop_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        db_name = sg.table(name, db=catalog)
        with self._safe_raw_sql(sge.Drop(kind="SCHEMA", this=db_name, exists=force)):
            pass

    def list_tables(
        self, *, like: str | None = None, database: str | None = None
    ) -> list[str]:
        if database is None:
            database = self.current_database
        query = (
            sg.select("table_name")
            .from_("information_schema.tables")
            .where(sg.column("table_schema").eq(sge.convert(database)))
            .order_by("table_name")
        )
        return self._filter_with_like(
            self.raw_sql(query).to_pydict()["table_name"], like
        )

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        if catalog is not None:
            catalog = self.con.catalog(catalog)
        else:
            catalog = self.con.catalog()

        if database is not None:
            try:
                database = catalog.schema(database)
            except AttributeError:
                database = catalog.database(database)
        else:
            try:
                database = catalog.schema()
            except AttributeError:
                database = catalog.database()

        if table_name not in database.names():
            raise com.TableNotFound(table_name)

        table = database.table(table_name)
        return sch.schema(table.schema)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        # self.con.register_table is broken, so we do this roundabout thing
        # of constructing a datafusion DataFrame, which has a side effect
        # of registering the table
        self.con.from_arrow(op.data.to_pyarrow(op.schema), op.name)

    def read_csv(
        self,
        paths: str | Path | list[str | Path] | tuple[str | Path],
        /,
        *,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        paths
            The data source. A string or Path to the CSV file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DataFusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filenames(paths)
        table_name = table_name or gen_name("read_csv")
        # Our other backends support overwriting views / tables when re-registering
        self.con.deregister_table(table_name)
        self.con.register_csv(table_name, path, **kwargs)
        return self.table(table_name)

    def read_parquet(
        self, path: str | Path, /, *, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DataFusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_parquet")
        # Our other backends support overwriting views / tables when reregistering
        self.con.deregister_table(table_name)
        self.con.register_parquet(table_name, path, **kwargs)
        return self.table(table_name)

    def read_delta(
        self, path: str | Path, /, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        path
            The data source. Must be a directory containing a Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to deltalake.DeltaTable.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)

        table_name = table_name or gen_name("read_delta")

        # Our other backends support overwriting views / tables when reregistering
        self.con.deregister_table(table_name)

        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError(
                "The deltalake extra is required to use the "
                "read_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[deltalake]'\n"
            )

        delta_table = DeltaTable(path, **kwargs)
        self.con.register_dataset(table_name, delta_table.to_pyarrow_dataset())
        return self.table(table_name)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        pa = self._import_pyarrow()

        self._register_udfs(expr)
        self._register_in_memory_tables(expr)

        table_expr = expr.as_table()
        raw_sql = self.compile(table_expr, **kwargs)

        frame = self.con.sql(raw_sql)

        schema = sch.Schema(
            {name: as_nullable(typ) for name, typ in table_expr.schema().items()}
        )
        names = schema.names

        struct_schema = schema.as_struct().to_pyarrow()

        def make_gen():
            yield from (
                # convert the renamed + casted columns into a record batch
                pa.RecordBatch.from_struct_array(
                    # rename columns to match schema because datafusion lowercases things
                    pa.RecordBatch.from_arrays(batch.to_pyarrow().columns, names=names)
                    # cast the struct array to the desired types to work around
                    # https://github.com/apache/arrow-datafusion-python/issues/534
                    .to_struct_array()
                    .cast(struct_schema, safe=False)
                )
                for batch in frame.execute_stream()
            )

        return pa.ipc.RecordBatchReader.from_batches(schema.to_pyarrow(), make_gen())

    def to_pyarrow(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ):
        batch_reader = self.to_pyarrow_batches(
            expr, params=params, limit=limit, **kwargs
        )
        arrow_table = batch_reader.read_all()
        return expr.__pyarrow_result__(arrow_table)

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        batch_reader = self.to_pyarrow_batches(
            expr, params=params, limit=limit, **kwargs
        )
        return expr.__pandas_result__(
            batch_reader.read_pandas(timestamp_as_object=True)
        )

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pa.RecordBatchReader
        | pa.RecordBatch
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in DataFusion.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        quoted = self.compiler.quoted

        if isinstance(obj, ir.Expr):
            table = obj

            # If it's a memtable, it will get registered in the pre-execute hooks
            self._run_pre_execute_hooks(table)

            compiler = self.compiler
            relname = "_"
            query = sg.select(
                *(
                    compiler.cast(
                        sg.column(col, table=relname, quoted=quoted), dtype
                    ).as_(col, quoted=quoted)
                    for col, dtype in table.schema().items()
                )
            ).from_(
                compiler.to_sqlglot(table).subquery(
                    sg.to_identifier(relname, quoted=quoted)
                )
            )
        elif obj is not None:
            table_ident = sg.table(name, db=database, quoted=quoted).sql(self.dialect)
            _read_in_memory(obj, table_ident, self, overwrite=overwrite)
            return self.table(name, database=database)
        else:
            query = None

        table_ident = sg.table(name, db=database, quoted=quoted)

        if query is None:
            target = sge.Schema(
                this=table_ident,
                expressions=(schema or table.schema()).to_sqlglot_column_defs(
                    self.dialect
                ),
            )
        else:
            target = table_ident

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
            expression=query,
            replace=overwrite,
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, database=database)

    def truncate_table(self, name: str, /, *, database: str | None = None):
        """Delete all rows from a table.

        Parameters
        ----------
        name
            Table name
        database
            Database name
        """
        # datafusion doesn't support `TRUNCATE TABLE` so we use `DELETE FROM`
        #
        # however datafusion as of 34.0.0 doesn't implement DELETE DML yet
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        ident = sg.table(name, db=db, catalog=catalog).sql(self.dialect)
        with self._safe_raw_sql(sge.delete(ident)):
            pass

    def _create_cached_table(self, name: str, expr: ir.Table) -> ir.Table:
        return self.create_table(name, expr, schema=expr.schema())


@contextlib.contextmanager
def _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
    """Workaround inability to overwrite tables in dataframe API.

    DataFusion has helper methods for loading in-memory data, but these methods
    don't allow overwriting tables.
    The SQL interface allows creating tables from existing tables, so we register
    the data as a table using the dataframe API, then run a

    CREATE [OR REPLACE] TABLE table_name AS SELECT * FROM in_memory_thing

    and that allows us to toggle the overwrite flag.
    """
    src = sge.Create(
        this=table_name,
        kind="TABLE",
        expression=sg.select("*").from_(tmp_name),
        replace=overwrite,
    )

    yield

    _conn.raw_sql(src)
    _conn.drop_table(tmp_name)


@lazy_singledispatch
def _read_in_memory(
    source: Any, table_name: str, _conn: Backend, overwrite: bool = False
):
    raise NotImplementedError("No support for source or imports missing")


@_read_in_memory.register(dict)
def _pydict(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pydict")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_pydict(source, name=tmp_name)


@_read_in_memory.register("polars.DataFrame")
def _polars(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("polars")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_polars(source, name=tmp_name)


@_read_in_memory.register("polars.LazyFrame")
def _polars(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("polars")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_polars(source.collect(), name=tmp_name)


@_read_in_memory.register("pyarrow.Table")
def _pyarrow_table(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_arrow(source, name=tmp_name)


@_read_in_memory.register("pyarrow.RecordBatchReader")
def _pyarrow_rbr(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_arrow(source.read_all(), name=tmp_name)


@_read_in_memory.register("pyarrow.RecordBatch")
def _pyarrow_rb(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.register_record_batches(tmp_name, [[source]])


@_read_in_memory.register("pyarrow.dataset.Dataset")
def _pyarrow_rb(source, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pyarrow")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.register_dataset(tmp_name, source)


@_read_in_memory.register("pandas.DataFrame")
def _pandas(source: pd.DataFrame, table_name, _conn, overwrite: bool = False):
    tmp_name = gen_name("pandas")
    with _create_and_drop_memtable(_conn, table_name, tmp_name, overwrite):
        _conn.con.from_pandas(source, name=tmp_name)
