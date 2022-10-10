from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import pandas as pd
import polars as pl

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.polars.compiler import translate


class Backend(BaseBackend):
    name = 'polars'
    builder = None

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self._tables = None

    def do_connect(
        self,
        _tables: MutableMapping[str, pl.LazyFrame],
        *args,
        **kwargs,
    ) -> None:
        """Construct a client from a dictionary of polars LazyFrames.

        Parameters
        ----------
        _tables
            Mutable mapping of string table names to polars LazyFrames.
        """
        if isinstance(_tables, dict):
            self._tables = _tables
        else:
            self._tables = dict()

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

    def register_csv(self, name: str, path: str | Path, **scan_csv_kwargs) -> None:
        """Register a CSV file with with `name` located at `path`.

        Parameters
        ----------
        name
            The name of the table
        path
            The path to the CSV file
        scan_csv_kwargs
            kwargs passed to `pl.scan_csv`
        """
        self._tables[name] = pl.scan_csv(path, **scan_csv_kwargs)

    def register_parquet(
        self, name: str, path: str | Path, **scan_parquet_args
    ) -> None:
        """Register a Parquet file with with `name` located at `path`.

        Parameters
        ----------
        name
            The name of the table
        path
            The path to the Parquet file
        scan_parquet_args
            kwargs passed to `pl.scan_parquet`
        """
        self._tables[name] = pl.scan_parquet(path, **scan_parquet_args)

    def register_pandas(self, name: str, df: pd.DataFrame) -> None:
        """Register a pandas DataFrame with with `name`.

        Parameters
        ----------
        name
            The name of the table
        df
            The pandas DataFrame
        """
        self._tables[name] = pl.from_pandas(df).lazy()

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
