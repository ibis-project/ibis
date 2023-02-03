from __future__ import annotations

import itertools
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend
from ibis.backends.datafusion.compiler import translate
from ibis.util import normalize_filename

try:
    from datafusion import ExecutionContext as SessionContext
except ImportError:
    from datafusion import SessionContext

import datafusion

# counters for in-memory, parquet, and csv reads
# used if no table name is specified
pa_n = itertools.count(0)
csv_n = itertools.count(0)


class Backend(BaseBackend):
    name = 'datafusion'
    builder = None

    @property
    def version(self):
        import importlib.metadata

        return importlib.metadata.version("datafusion")

    def do_connect(
        self,
        config: Mapping[str, str | Path] | SessionContext | None = None,
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

        config = config or {}

        for name, path in config.items():
            self.register(path, table_name=name)

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

    def table(self, name: str, schema: sch.Schema | None = None) -> ir.Table:
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
        schema = sch.schema(table.schema)
        return self.table_class(name, schema, self).to_expr()

    def register(
        self, source: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a CSV or Parquet file with `table_name` located at `source`.

        Parameters
        ----------
        source
            The path to the file
        table_name
            The name of the table
        kwargs
            Datafusion-specific keyword arguments
        """
        if isinstance(source, (str, Path)):
            first = str(source)
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

    def read_csv(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to Datafusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)
        table_name = table_name or f"ibis_read_csv_{next(csv_n)}"
        # Our other backends support overwriting views / tables when reregistering
        self._context.deregister_table(table_name)
        self._context.register_csv(table_name, path, **kwargs)
        return self.table(table_name)

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
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
            Additional keyword arguments passed to Datafusion loading function.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        path = normalize_filename(path)
        table_name = table_name or f"ibis_read_parquet_{next(pa_n)}"
        # Our other backends support overwriting views / tables when reregistering
        self._context.deregister_table(table_name)
        self._context.register_parquet(table_name, path, **kwargs)
        return self.table(table_name)

    def _get_frame(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> datafusion.DataFrame:
        if isinstance(expr, ir.Table):
            return self.compile(expr, params, **kwargs)
        elif isinstance(expr, ir.Column):
            # expression must be named for the projection
            expr = expr.name('tmp').as_table()
            return self.compile(expr, params, **kwargs)
        elif isinstance(expr, ir.Scalar):
            if an.find_immediate_parent_tables(expr.op()):
                # there are associated datafusion tables so convert the expr
                # to a selection which we can directly convert to a datafusion
                # plan
                expr = expr.name('tmp').as_table()
                frame = self.compile(expr, params, **kwargs)
            else:
                # doesn't have any tables associated so create a plan from a
                # dummy datafusion table
                compiled = self.compile(expr, params, **kwargs)
                frame = self._context.empty_table().select(compiled)
            return frame
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
    ) -> pa.ipc.RecordBatchReader:
        pa = self._import_pyarrow()
        frame = self._get_frame(expr, params, limit, **kwargs)
        return pa.ipc.RecordBatchReader.from_batches(frame.schema(), frame.collect())

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        limit: int | str | None = "default",
        **kwargs: Any,
    ):
        output = self.to_pyarrow(expr, params=params, limit=limit, **kwargs)
        if isinstance(expr, ir.Table):
            return output.to_pandas()
        elif isinstance(expr, ir.Column):
            series = output.to_pandas()
            series.name = expr.get_name()
            return series
        elif isinstance(expr, ir.Scalar):
            return output.as_py()
        else:
            raise com.IbisError(f"Cannot execute expression of type: {type(expr)}")

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] = None,
        **kwargs: Any,
    ):
        return translate(expr.op())

    @classmethod
    @lru_cache
    def _get_operations(cls):
        from ibis.backends.datafusion.compiler import translate

        return frozenset(op for op in translate.registry if issubclass(op, ops.Value))

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        op_classes = cls._get_operations()
        return operation in op_classes or any(
            issubclass(operation, op_impl) for op_impl in op_classes
        )
