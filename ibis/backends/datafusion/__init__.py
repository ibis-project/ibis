from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datafusion
import pyarrow as pa

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, CanCreateDatabase, CanCreateSchema
from ibis.backends.datafusion.compiler import translate
from ibis.util import gen_name, normalize_filename

try:
    from datafusion import ExecutionContext as SessionContext
except ImportError:
    from datafusion import SessionContext

try:
    from datafusion import SessionConfig
except ImportError:
    SessionConfig = None

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


class Backend(BaseBackend, CanCreateDatabase, CanCreateSchema):
    name = "datafusion"
    builder = None
    supports_in_memory_tables = False

    @property
    def version(self):
        import importlib.metadata

        return importlib.metadata.version("datafusion")

    def do_connect(
        self, config: Mapping[str, str | Path] | SessionContext | None = None
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
            if SessionConfig is not None:
                df_config = SessionConfig().with_information_schema(True)
            else:
                df_config = None
            self._context = SessionContext(df_config)

        if not config:
            config = {}

        for name, path in config.items():
            self.register(path, table_name=name)

    @property
    def current_database(self) -> str:
        raise NotImplementedError()

    @property
    def current_schema(self) -> str:
        return NotImplementedError()

    def list_databases(self, like: str | None = None) -> list[str]:
        code = "SELECT DISTINCT table_catalog FROM information_schema.tables"
        if like:
            code += f" WHERE table_catalog LIKE {like!r}"
        result = self._context.sql(code).to_pydict()
        return result["table_catalog"]

    def create_database(self, name: str, force: bool = False) -> None:
        code = "CREATE DATABASE"
        if force:
            code += " IF NOT EXISTS"
        code += f" {name}"
        self._context.sql(code)

    def drop_database(self, name: str, force: bool = False) -> None:
        raise com.UnsupportedOperationError(
            "DataFusion does not support dropping databases"
        )

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(
            self._context.catalog(
                database if database is not None else "datafusion"
            ).names(),
            like=like,
        )

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        create_stmt = "CREATE SCHEMA"
        if force:
            create_stmt += " IF NOT EXISTS"

        create_stmt += " "
        create_stmt += ".".join(filter(None, [database, name]))
        self._context.sql(create_stmt)

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        raise com.UnsupportedOperationError(
            "DataFusion does not support dropping schemas"
        )

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
            The name of the table to retrieve
        schema
            An optional schema for the table

        Returns
        -------
        Table
            A table expression
        """
        catalog = self._context.catalog()
        database = catalog.database()
        table = database.table(name)
        schema = sch.schema(table.schema)
        return ops.DatabaseTable(name, schema, self).to_expr()

    def register(
        self,
        source: str | Path | pa.Table | pa.RecordBatch | pa.Dataset | pd.DataFrame,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a data set with `table_name` located at `source`.

        Parameters
        ----------
        source
            The data source(s). May be a path to a file or directory of
            parquet/csv files, a pandas dataframe, or a pyarrow table, dataset
            or record batch.
        table_name
            The name of the table
        kwargs
            Datafusion-specific keyword arguments

        Examples
        --------
        Register a csv:

        >>> import ibis
        >>> conn = ibis.datafusion.connect(config)
        >>> conn.register("path/to/data.csv", "my_table")
        >>> conn.table("my_table")

        Register a PyArrow table:

        >>> import pyarrow as pa
        >>> tab = pa.table({"x": [1, 2, 3]})
        >>> conn.register(tab, "my_table")
        >>> conn.table("my_table")

        Register a PyArrow dataset:

        >>> import pyarrow.dataset as ds
        >>> dataset = ds.dataset("path/to/table")
        >>> conn.register(dataset, "my_table")
        >>> conn.table("my_table")
        """
        import pandas as pd

        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, pa.Table):
            self._context.deregister_table(table_name)
            self._context.register_record_batches(table_name, [source.to_batches()])
            return self.table(table_name)
        elif isinstance(source, pa.RecordBatch):
            self._context.deregister_table(table_name)
            self._context.register_record_batches(table_name, [[source]])
            return self.table(table_name)
        elif isinstance(source, pa.dataset.Dataset):
            self._context.deregister_table(table_name)
            self._context.register_dataset(table_name, source)
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
        table_name = table_name or gen_name("read_csv")
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
        table_name = table_name or gen_name("read_parquet")
        # Our other backends support overwriting views / tables when reregistering
        self._context.deregister_table(table_name)
        self._context.register_parquet(table_name, path, **kwargs)
        return self.table(table_name)

    def read_delta(
        self, source_table: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        source_table
            The data source. Must be a directory
            containing a Delta Lake table.
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
        source_table = normalize_filename(source_table)

        table_name = table_name or gen_name("read_delta")

        # Our other backends support overwriting views / tables when reregistering
        self._context.deregister_table(table_name)

        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError(
                "The deltalake extra is required to use the "
                "read_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[deltalake]'\n"
            )

        delta_table = DeltaTable(source_table, **kwargs)

        return self.register(delta_table.to_pyarrow_dataset(), table_name=table_name)

    def _get_frame(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> datafusion.DataFrame:
        return self.compile(expr.as_table(), params, **kwargs)

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
        params: Mapping[ir.Expr, object] | None = None,
        limit: int | str | None = "default",
        **kwargs: Any,
    ):
        output = self.to_pyarrow(expr.as_table(), params=params, limit=limit, **kwargs)
        return expr.__pandas_result__(output.to_pandas(timestamp_as_object=True))

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, object] | None = None,
        **kwargs: Any,
    ):
        return translate(expr.op(), ctx=self._context)

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

    def create_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)
