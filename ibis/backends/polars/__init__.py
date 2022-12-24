from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import polars as pl

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.polars.compiler import translate
from ibis.common.dispatch import RegexDispatcher

_register_file = RegexDispatcher("_register_file")


def _name_from_path(path: Path) -> str:
    base, *_ = path.name.partition(os.extsep)
    return base.replace("-", "_")


@_register_file.register(r"parquet://(?P<path>.+)", priority=10)
def _parquet(_, path, table_name=None, **kwargs):
    path = Path(path).absolute()
    table_name = table_name or _name_from_path(path)
    return ("scan_parquet", path, table_name)


@_register_file.register(r"csv://(?P<path>.+)", priority=10)
def _csv(_, path, table_name=None, **kwargs):
    path = Path(path).absolute()
    table_name = table_name or _name_from_path(path)
    return ("scan_csv", path, table_name)


@_register_file.register(r"(?:file://)?(?P<path>.+)", priority=9)
def _file(raw, path, table_name=None, **kwargs):
    num_sep_chars = len(os.extsep)
    extension = "".join(Path(path).suffixes)[num_sep_chars:]
    if not extension:
        raise ValueError(
            f"""Unrecognized file type or extension: {raw}

        Valid prefixes are parquet://, csv://, or file://
        Supported file extensions are parquet and csv"""
        )
    return _register_file(f"{extension}://{path}", table_name=table_name, **kwargs)


class Backend(BaseBackend):
    name = "polars"
    builder = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = dict()

    def do_connect(self, tables: MutableMapping[str, pl.LazyFrame]) -> None:
        """Construct a client from a dictionary of `polars.LazyFrame`s.

        Parameters
        ----------
        tables
            Mutable mapping of string table names to polars LazyFrames.
        """
        self._tables.update(tables)

    @property
    def version(self) -> str:
        return pl.__version__

    def current_database(self):
        raise NotImplementedError('polars backend does not support databases')

    def list_databases(self, like=None):
        raise NotImplementedError('polars backend does not support databases')

    def list_tables(self, like=None, database=None):
        return self._filter_with_like(list(self._tables.keys()), like)

    def table(self, name: str, _schema: sch.Schema = None) -> ir.Table:
        schema = sch.infer(self._tables[name])
        return self.table_class(name, schema, self).to_expr()

    def register(
        self,
        source: str | Path | Any,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        if isinstance(source, (str, Path)):
            method, path, name = _register_file(
                str(source), table_name=table_name, **kwargs
            )
            self._tables[name] = getattr(pl, method)(path, **kwargs)
        else:
            if table_name is None:
                raise ValueError(
                    "Must specify table_name for pandas DataFrame registration"
                )
            name = table_name
            self._tables[name] = pl.from_pandas(source, **kwargs).lazy()
        return self.table(name)

    def database(self, name=None):
        return self.database_class(name, self)

    def load_data(self, table_name, obj, **kwargs):
        # kwargs is a catch all for any options required by other backends.
        self._tables[table_name] = obj

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
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        **kwargs: Any,
    ):
        node = expr.op()
        if params:
            node = node.replace({p.op(): v for p, v in params.items()})
            expr = node.to_expr()

        if isinstance(expr, ir.Table):
            return translate(node)
        elif isinstance(expr, ir.Column):
            # expression must be named for the projection
            node = expr.to_projection().op()
            return translate(node)
        elif isinstance(expr, ir.Scalar):
            if an.is_scalar_reduction(node):
                node = an.reduction_to_aggregation(node).op()
                return translate(node)
            else:
                # doesn't have any _tables associated so create projection
                # based off of an empty table
                return pl.DataFrame().lazy().select(translate(node))
        else:
            raise com.IbisError(f"Cannot compile expression of type: {type(expr)}")

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: str = 'default',
        **kwargs: Any,
    ):
        df = self.compile(expr, params=params).collect()
        if isinstance(expr, ir.Table):
            return df.to_pandas()
        elif isinstance(expr, ir.Column):
            return df.to_pandas().iloc[:, 0]
        elif isinstance(expr, ir.Scalar):
            return df.to_pandas().iat[0, 0]
        else:
            raise com.IbisError(f"Cannot execute expression of type: {type(expr)}")

    def _to_pyarrow_table(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: int | None = None,
        **kwargs: Any,
    ):
        lf = self.compile(expr, params=params, **kwargs)
        if limit is not None:
            df = lf.fetch(limit)
        else:
            df = lf.collect()

        table = df.to_arrow()
        if isinstance(expr, ir.Table):
            schema = expr.schema().to_pyarrow()
            return table.cast(schema)
        elif isinstance(expr, ir.Value):
            schema = sch.schema({expr.get_name(): expr.type().to_pyarrow()})
            schema = schema.to_pyarrow()
            return table.cast(schema)
        else:
            raise com.IbisError(f"Cannot execute expression of type: {type(expr)}")

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: int | None = None,
        **kwargs: Any,
    ):
        pa = self._import_pyarrow()
        result = self._to_pyarrow_table(expr, params=params, limit=limit, **kwargs)
        if isinstance(expr, ir.Table):
            return result
        elif isinstance(expr, ir.Column):
            if len(column := result[0]):
                return column.combine_chunks()
            else:
                return pa.array(column)
        elif isinstance(expr, ir.Scalar):
            return result[0][0]
        else:
            raise com.IbisError(f"Cannot execute expression of type: {type(expr)}")

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
