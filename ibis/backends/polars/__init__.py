from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, Database
from ibis.backends.polars.compiler import translate
from ibis.backends.polars.datatypes import dtype_to_polars, schema_from_polars
from ibis.common.patterns import Replace
from ibis.util import gen_name, normalize_filename

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping

    import pandas as pd
    import pyarrow as pa


class Backend(BaseBackend):
    name = "polars"
    builder = None

    _sqlglot_dialect = "postgres"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = dict()
        self._context = pl.SQLContext()

    def do_connect(
        self, tables: MutableMapping[str, pl.LazyFrame | pl.DataFrame] | None = None
    ) -> None:
        """Construct a client from a dictionary of polars `LazyFrame`s and/or `DataFrame`s.

        Parameters
        ----------
        tables
            An optional mapping of string table names to polars LazyFrames.
        """
        for name, table in (tables or {}).items():
            self._add_table(name, table)

    @property
    def version(self) -> str:
        return pl.__version__

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self._tables.keys()), like)

    def table(self, name: str, _schema: sch.Schema | None = None) -> ir.Table:
        schema = schema_from_polars(self._tables[name].schema)
        return ops.DatabaseTable(name, schema, self).to_expr()

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
            The data source(s). May be a path to a file, a parquet directory, or a pandas
            dataframe.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Polars loading functions for
            CSV or parquet.
            See https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_csv.html
            and https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.scan_parquet.html
            for more information

        Returns
        -------
        ir.Table
            The just-registered table
        """

        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, (list, tuple)):
            first = str(source[0])
        else:
            try:
                return self.read_pandas(source, table_name=table_name, **kwargs)
            except ValueError:
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

    def _add_table(self, name: str, obj: pl.LazyFrame | pl.DataFrame) -> None:
        if isinstance(obj, pl.DataFrame):
            obj = obj.lazy()
        self._tables[name] = obj
        self._context.register(name, obj)

    def _remove_table(self, name: str) -> None:
        del self._tables[name]
        self._context.unregister(name)

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
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
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
        path = normalize_filename(path)
        table_name = table_name or gen_name("read_csv")
        try:
            self._add_table(table_name, pl.scan_csv(path, **kwargs))
        except pl.exceptions.ComputeError:
            # handles compressed csvs
            self._add_table(table_name, pl.read_csv(path, **kwargs))
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

    def database(self, name=None):
        return Database(name, self)

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        if schema is not None and obj is None:
            obj = pl.LazyFrame(
                [],
                schema={name: dtype_to_polars(dtype) for name, dtype in schema.items()},
            )

        if database is not None:
            raise com.IbisError(
                "Passing `database` to the Polars backend create_table method has no "
                "effect: Polars cannot set a database."
            )

        if temp is not None:
            raise com.IbisError(
                "Passing `temp=True` to the Polars backend create_table method has no "
                "effect: all tables are in memory and temporary. "
            )

        if not overwrite and name in self._tables:
            raise com.IntegrityError(
                f"Table {name} already exists. Use overwrite=True to clobber existing tables"
            )

        if isinstance(obj, ir.Table):
            obj = self.to_pyarrow(obj)

        if not isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            obj = pl.LazyFrame(obj)

        self._add_table(name, obj)
        return self.table(name)

    def get_schema(self, table_name, database=None):
        return self._tables[table_name].schema

    @classmethod
    @lru_cache
    def _get_operations(cls):
        return frozenset(op for op in translate.registry if issubclass(op, ops.Value))

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        # Polars doesn't support geospatial ops, but the dispatcher implements
        # a common base class that makes it appear that it does. Explicitly
        # exclude these operations.
        if issubclass(operation, (ops.GeoSpatialUnOp, ops.GeoSpatialBinOp)):
            return False
        op_classes = cls._get_operations()
        return operation in op_classes or any(
            issubclass(operation, op_impl) for op_impl in op_classes
        )

    def compile(
        self, expr: ir.Expr, params: Mapping[ir.Expr, object] | None = None, **_: Any
    ):
        node = expr.op()
        ctx = self._context

        if params:
            params = {param.op(): value for param, value in params.items()}
            rule = Replace(
                ops.ScalarParameter,
                lambda _: ops.Literal(value=params[_], dtype=_.dtype),
            )
            node = node.replace(rule)
            expr = node.to_expr()

        node = expr.as_table().op()
        return translate(node, ctx=ctx)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        return schema_from_polars(
            self._context.execute(f"SELECT * FROM ({query}) LIMIT 0").schema
        )

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ):
        lf = self.compile(expr, params=params, **kwargs)
        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            df = lf.fetch(limit)
        else:
            df = lf.collect()

        if isinstance(expr, (ir.Table, ir.Scalar)):
            return expr.__pandas_result__(df.to_pandas())
        else:
            assert isinstance(expr, ir.Column), type(expr)
            if expr.type().is_temporal():
                return df.to_pandas().iloc[:, 0]
            else:
                # note: skip frame-construction overhead
                return df.to_series().to_pandas()

    def _to_pyarrow_table(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ):
        lf = self.compile(expr, params=params, **kwargs)
        if limit is not None:
            df = lf.fetch(limit)
        else:
            df = lf.collect()

        table = df.to_arrow()
        if isinstance(expr, (ir.Table, ir.Value)):
            schema = expr.as_table().schema().to_pyarrow()
            return table.rename_columns(schema.names).cast(schema)
        else:
            raise com.IbisError(f"Cannot execute expression of type: {type(expr)}")

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

    def _load_into_cache(self, name, expr):
        self.create_table(name, self.compile(expr).cache())

    def _clean_up_cached_table(self, op):
        self._remove_table(op.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)
