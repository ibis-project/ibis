from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.datafusion.compiler import translate

try:
    from datafusion import ExecutionContext as SessionContext
except ImportError:
    from datafusion import SessionContext


def _to_pyarrow_table(frame):
    batches = frame.collect()
    if batches:
        return pa.Table.from_batches(batches)
    else:
        # TODO(kszucs): file a bug to datafusion because the fields'
        # nullability from frame.schema() is not always consistent
        # with the first record batch's schema
        return pa.Table.from_batches(batches, schema=frame.schema())


class Backend(BaseBackend):
    name = 'datafusion'
    builder = None

    @property
    def version(self):
        import importlib.metadata

        return importlib.metadata.version("datafusion")

    def do_connect(
        self,
        config: Mapping[str, str | Path] | SessionContext,
    ) -> None:
        """Create a Datafusion backend for use with Ibis.

        Parameters
        ----------
        config
            Mapping of table names to files.

        Examples
        --------
        >>> import ibis
        >>> config = {"t": "path/to/file.parquet", "s": "path/to/file.csv"}
        >>> ibis.datafusion.connect(config)
        """
        if isinstance(config, SessionContext):
            self._context = config
        else:
            self._context = SessionContext()

        for name, path in config.items():
            strpath = str(path)
            if strpath.endswith('.csv'):
                self.register_csv(name, path)
            elif strpath.endswith('.parquet'):
                self.register_parquet(name, path)
            else:
                raise ValueError(
                    "Currently the DataFusion backend only supports CSV "
                    "files with the extension .csv and Parquet files with "
                    "the .parquet extension."
                )

    def current_database(self) -> str:
        raise NotImplementedError()

    def list_databases(self, like: str | None = None) -> list[str]:
        raise NotImplementedError()

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
    ) -> list[str]:
        """List the available tables."""
        tables = list(self._context.tables())
        if like is not None:
            pattern = re.compile(like)
            return list(filter(lambda t: pattern.findall(t), tables))
        return tables

    def table(
        self,
        name: str,
        schema: sch.Schema | None = None,
    ) -> ir.Table:
        """Get an ibis expression representing a DataFusion table.

        Parameters
        ----------
        name
            The name of the table to retreive
        schema
            An optional schema for the table

        Returns
        -------
        Table
            A table expression
        """
        catalog = self._context.catalog()
        database = catalog.database('public')
        table = database.table(name)
        schema = sch.infer(table.schema)
        return self.table_class(name, schema, self).to_expr()

    def register_csv(
        self,
        name: str,
        path: str | Path,
        schema: sch.Schema | None = None,
    ) -> None:
        """Register a CSV file with with `name` located at `path`.

        Parameters
        ----------
        name
            The name of the table
        path
            The path to the CSV file
        schema
            An optional schema
        """
        self._context.register_csv(name, str(path), schema=schema)

    def register_parquet(
        self,
        name: str,
        path: str | Path,
    ) -> None:
        """Register a parquet file with with `name` located at `path`.

        Parameters
        ----------
        name
            The name of the table
        path
            The path to the Parquet file
        """
        self._context.register_parquet(name, str(path))

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: str = 'default',
        **kwargs: Any,
    ):
        if isinstance(expr, ir.Table):
            frame = self.compile(expr, params, **kwargs)
            table = _to_pyarrow_table(frame)
            return table.to_pandas()
        elif isinstance(expr, ir.Column):
            # expression must be named for the projection
            expr = expr.name('tmp').to_projection()
            frame = self.compile(expr, params, **kwargs)
            table = _to_pyarrow_table(frame)
            return table['tmp'].to_pandas()
        elif isinstance(expr, ir.Scalar):
            if expr.op().root_tables():
                # there are associated datafusion tables so convert the expr
                # to a selection which we can directly convert to a datafusion
                # plan
                expr = expr.name('tmp').to_projection()
                frame = self.compile(expr, params, **kwargs)
            else:
                # doesn't have any tables associated so create a plan from a
                # dummy datafusion table
                compiled = self.compile(expr, params, **kwargs)
                frame = self._context.empty_table().select(compiled)
            table = _to_pyarrow_table(frame)
            return table[0][0].as_py()
        else:
            raise com.IbisError(
                f"Cannot execute expression of type: {type(expr)}"
            )

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        **kwargs: Any,
    ):
        return translate(expr)

    @classmethod
    @lru_cache
    def _get_operations(cls):
        from ibis.backends.datafusion.compiler import translate

        return frozenset(
            op for op in translate.registry if issubclass(op, ops.Value)
        )

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        op_classes = cls._get_operations()
        return operation in op_classes or any(
            issubclass(operation, op_impl) for op_impl in op_classes
        )
