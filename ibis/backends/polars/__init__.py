from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends import BaseBackend, NoUrl
from ibis.backends.polars.compiler import translate
from ibis.backends.polars.rewrites import bind_unbound_table, rewrite_join
from ibis.backends.sql.dialects import Polars
from ibis.common.dispatch import lazy_singledispatch
from ibis.expr.rewrites import lower_stringslice, replace_parameter
from ibis.formats.polars import PolarsSchema
from ibis.util import gen_name, normalize_filename, normalize_filenames

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd
    import pyarrow as pa


class Backend(BaseBackend, NoUrl):
    name = "polars"
    dialect = Polars

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = dict()
        self._context = pl.SQLContext()

    def do_connect(
        self, tables: Mapping[str, pl.LazyFrame | pl.DataFrame] | None = None
    ) -> None:
        """Construct a client from a dictionary of polars `LazyFrame`s and/or `DataFrame`s.

        Parameters
        ----------
        tables
            An optional mapping of string table names to polars LazyFrames.

        Examples
        --------
        >>> import ibis
        >>> import polars as pl
        >>> ibis.options.interactive = True
        >>> lazy_frame = pl.LazyFrame(
        ...     {"name": ["Jimmy", "Keith"], "band": ["Led Zeppelin", "Stones"]}
        ... )
        >>> con = ibis.polars.connect(tables={"band_members": lazy_frame})
        >>> t = con.table("band_members")
        >>> t
        ┏━━━━━━━━┳━━━━━━━━━━━━━━┓
        ┃ name   ┃ band         ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━┩
        │ string │ string       │
        ├────────┼──────────────┤
        │ Jimmy  │ Led Zeppelin │
        │ Keith  │ Stones       │
        └────────┴──────────────┘
        """
        if tables is not None and not isinstance(tables, Mapping):
            raise TypeError("Input to ibis.polars.connect must be a mapping")

        # tables are emphemeral
        self._tables.clear()

        for name, table in (tables or {}).items():
            self._add_table(name, table)

    def disconnect(self) -> None:
        pass

    @property
    def version(self) -> str:
        return pl.__version__

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self._tables.keys()), like)

    def table(self, name: str, database: None = None) -> ir.Table:
        if database is not None:
            raise com.IbisError(
                "Passing `database` to the Polars backend's `table()` method is not "
                "supported: Polars cannot set a database."
            )

        table = self._tables.get(name)
        if table is None:
            raise com.TableNotFound(name)

        schema = sch.infer(table)
        return ops.DatabaseTable(name, schema, self).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self._add_table(op.name, op.data.to_polars(op.schema).lazy())

    def _finalize_memtable(self, name: str) -> None:
        self.drop_table(name, force=True)

    def _add_table(self, name: str, obj: pl.LazyFrame | pl.DataFrame) -> None:
        if isinstance(obj, pl.DataFrame):
            obj = obj.lazy()
        self._tables[name] = obj
        self._context.register(name, obj)

    def sql(
        self, query: str, schema: sch.Schema | None = None, dialect: str | None = None
    ) -> ir.Table:
        query = self._transpile_sql(query, dialect=dialect)
        if schema is None:
            schema = self._get_schema_using_query(query)
        name = ibis.util.gen_name("polars_dot_sql_table")
        self._add_table(name, self._context.execute(query))
        return self.table(name)

    def read_csv(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV file as a table.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading function.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_csv.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source_list = normalize_filenames(path)
        # Flatten the list if there's only one element because Polars
        # can't handle glob strings, or compressed CSVs in a single-element list
        if len(source_list) == 1:
            source_list = source_list[0]
        table_name = table_name or gen_name("read_csv")
        try:
            table = pl.scan_csv(source_list, **kwargs)
            # triggers a schema computation to handle compressed csv inference
            # and raise a compute error
            table.collect_schema()
        except pl.exceptions.ComputeError:
            # handles compressed csvs
            table = pl.read_csv(source_list, **kwargs)

        self._add_table(table_name, table)
        return self.table(table_name)

    def read_json(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a JSON file as a table.

        Parameters
        ----------
        path
            A string or Path to a JSON file; globs are supported
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading function.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_ndjson.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_json")
        try:
            self._add_table(table_name, pl.scan_ndjson(path, **kwargs))
        except pl.exceptions.ComputeError:
            # handles compressed json files
            self._add_table(table_name, pl.read_ndjson(path, **kwargs))
        return self.table(table_name)

    def read_delta(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a Delta Lake as a table in the current database.

        Parameters
        ----------
        path
            The data source(s). Path to a Delta Lake table directory.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading function.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_delta.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        try:
            import deltalake  # noqa: F401
        except ImportError:
            raise ImportError(
                "The deltalake extra is required to use the "
                "read_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[polars,deltalake]'\n"
            )
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_delta")
        self._add_table(table_name, pl.scan_delta(path, **kwargs))
        return self.table(table_name)

    def read_pandas(
        self, source: pd.DataFrame, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a Pandas DataFrame or pyarrow Table a table in the current database.

        Parameters
        ----------
        source
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading function.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.from_pandas.html
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or gen_name("read_in_memory")

        self._add_table(table_name, pl.from_pandas(source, **kwargs).lazy())
        return self.table(table_name)

    def read_parquet(
        self,
        path: str | Path | Iterable[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source(s). May be a path to a file, an iterable of files,
            or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading function.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_parquet.html
            for more information (if loading a single file or glob; when loading
            multiple files polars' `scan_pyarrow_dataset` method is used instead).

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or gen_name("read_parquet")
        if not isinstance(path, (str, Path)) and len(path) == 1:
            path = path[0]

        if not isinstance(path, (str, Path)) and len(path) > 1:
            self._import_pyarrow()
            import pyarrow.dataset as ds

            paths = [normalize_filename(p) for p in path]
            obj = pl.scan_pyarrow_dataset(
                source=ds.dataset(paths, format="parquet"),
                **kwargs,
            )
            self._add_table(table_name, obj)
        else:
            path = normalize_filename(path)
            self._add_table(table_name, pl.scan_parquet(path, **kwargs))

        return self.table(table_name)

    def create_table(
        self,
        name: str,
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
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        if database is not None:
            raise com.IbisError(
                "Passing `database` to the Polars backend's `create_table()` method is "
                "not supported: Polars cannot set a database."
            )

        if temp is False:
            raise com.IbisError(
                "Passing `temp=False` to the Polars backend's `create_table()` method "
                "is not supported: all tables are in memory and temporary."
            )

        if not overwrite and name in self._tables:
            raise com.IntegrityError(
                f"Table {name!r} already exists. Use `overwrite=True` to clobber "
                "existing tables."
            )

        if schema is not None and obj is None:
            obj = pl.LazyFrame([], schema=PolarsSchema.from_ibis(schema))
            self._add_table(name, obj)
        else:
            _read_in_memory(obj, name, self)

        return self.table(name)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        return self.create_table(
            name, obj=obj, temp=None, database=database, overwrite=overwrite
        )

    def drop_table(self, name: str, *, force: bool = False) -> None:
        if name in self._tables:
            del self._tables[name]
            self._context.unregister(name)
        elif not force:
            raise com.IbisError(f"Table {name!r} does not exist")

    def drop_view(self, name: str, *, force: bool = False) -> None:
        self.drop_table(name, force=force)

    def get_schema(self, table_name):
        return self._tables[table_name].schema

    @classmethod
    @lru_cache
    def _get_operations(cls):
        return tuple(op for op in translate.registry if issubclass(op, ops.Value))

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        # Polars doesn't support geospatial ops, but the dispatcher implements
        # a common base class that makes it appear that it does. Explicitly
        # exclude these operations.
        if issubclass(
            operation, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp, ops.GeoUnaryUnion)
        ):
            return False
        op_classes = cls._get_operations()
        return operation in op_classes or issubclass(operation, op_classes)

    def compile(
        self, expr: ir.Expr, params: Mapping[ir.Expr, object] | None = None, **_: Any
    ):
        if params is None:
            params = dict()
        else:
            params = {param.op(): value for param, value in params.items()}

        node = expr.as_table().op()
        node = node.replace(
            rewrite_join | replace_parameter | bind_unbound_table | lower_stringslice,
            context={"params": params, "backend": self},
        )

        return translate(node, ctx=self._context)

    def _get_sql_string_view_schema(
        self, *, name: str, table: ir.Table, query: str
    ) -> sch.Schema:
        from ibis.backends.sql.compilers.postgres import compiler

        sql = compiler.add_query_to_expr(name=name, table=table, query=query)
        return self._get_schema_using_query(sql)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        lazy_frame = self._context.execute(query, eager=False)
        return sch.infer(lazy_frame)

    def _to_dataframe(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        streaming: bool = False,
        engine: Literal["cpu", "gpu"] | pl.GPUEngine = "cpu",
        **kwargs: Any,
    ) -> pl.DataFrame:
        self._run_pre_execute_hooks(expr)
        table_expr = expr.as_table()
        lf = self.compile(table_expr, params=params, **kwargs)
        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            lf = lf.limit(limit)
        df = lf.collect(streaming=streaming, engine=engine)
        # XXX: Polars sometimes returns data with the incorrect column names.
        # For now we catch this case and rename them here if needed.
        expected_cols = tuple(table_expr.columns)
        if tuple(df.columns) != expected_cols:
            df = df.rename(dict(zip(df.columns, expected_cols)))
        return df

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        streaming: bool = False,
        engine: Literal["cpu", "gpu"] | pl.GPUEngine = "cpu",
        **kwargs: Any,
    ):
        df = self._to_dataframe(
            expr,
            params=params,
            limit=limit,
            streaming=streaming,
            engine=engine,
            **kwargs,
        )
        if isinstance(expr, (ir.Table, ir.Scalar)):
            return expr.__pandas_result__(df.to_pandas())
        else:
            assert isinstance(expr, ir.Column), type(expr)

            dtype = expr.type()
            if dtype.is_temporal():
                return expr.__pandas_result__(df.to_pandas())
            else:
                from ibis.formats.pandas import PandasData

                # note: skip frame-construction overhead
                return PandasData.convert_column(df.to_series().to_pandas(), dtype)

    def to_polars(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        streaming: bool = False,
        engine: Literal["cpu", "gpu"] | pl.GPUEngine = "cpu",
        **kwargs: Any,
    ):
        df = self._to_dataframe(
            expr,
            params=params,
            limit=limit,
            streaming=streaming,
            engine=engine,
            **kwargs,
        )
        return expr.__polars_result__(df)

    def _to_pyarrow_table(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        streaming: bool = False,
        engine: Literal["cpu", "gpu"] | pl.GPUEngine = "cpu",
        **kwargs: Any,
    ):
        from ibis.formats.pyarrow import PyArrowData

        df = self._to_dataframe(
            expr,
            params=params,
            limit=limit,
            streaming=streaming,
            engine=engine,
            **kwargs,
        )
        return PyArrowData.convert_table(df.to_arrow(), expr.as_table().schema())

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ):
        result = self._to_pyarrow_table(expr, params=params, limit=limit, **kwargs)
        return expr.__pyarrow_result__(result)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ):
        self._import_pyarrow()
        table = self._to_pyarrow_table(expr, params=params, limit=limit, **kwargs)
        return table.to_reader(chunk_size)

    def _create_cached_table(self, name, expr):
        return self.create_table(name, self.compile(expr).cache())

    def _drop_cached_table(self, name):
        self.drop_table(name, force=True)


@lazy_singledispatch
def _read_in_memory(source: Any, table_name: str, _conn: Backend, **kwargs: Any):
    raise NotImplementedError(
        f"The `{_conn.name}` backend currently does not support "
        f"reading data of {type(source)!r}"
    )


@_read_in_memory.register("ibis.expr.types.Table")
def _table(source, table_name, _conn, **kwargs: Any):
    _conn._add_table(table_name, source.to_polars())


@_read_in_memory.register("polars.DataFrame")
@_read_in_memory.register("polars.LazyFrame")
def _polars(source, table_name, _conn, **kwargs: Any):
    _conn._add_table(table_name, source)


@_read_in_memory.register("pyarrow.Table")
@_read_in_memory.register("pyarrow.RecordBatchReader")
@_read_in_memory.register("pyarrow.RecordBatch")
def _pyarrow(source, table_name, _conn, **kwargs: Any):
    _conn._add_table(table_name, pl.from_arrow(source, **kwargs).lazy())


@_read_in_memory.register("pandas.DataFrame")
def _pandas(source: pd.DataFrame, table_name, _conn, **kwargs: Any):
    _conn._add_table(table_name, pl.from_pandas(source, **kwargs).lazy())
