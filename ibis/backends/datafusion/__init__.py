from __future__ import annotations

import inspect
import typing
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datafusion as df
import pyarrow as pa
import pyarrow.dataset as ds
import sqlglot as sg
from sqlglot import exp, transforms
from sqlglot.dialects import Postgres

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, CanCreateDatabase, CanCreateSchema
from ibis.backends.base.sqlglot import STAR, C
from ibis.backends.datafusion.compiler.core import translate
from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowType
from ibis.util import gen_name, log, normalize_filename

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

_exclude_exp = (exp.Pow,)


# the DataFusion dialect was created to skip the power function to operator transformation
# in the future this could be used to optimize sqlglot for datafusion
class DataFusion(Postgres):
    class Generator(Postgres.Generator):
        TRANSFORMS = {
            exp: trans
            for exp, trans in Postgres.Generator.TRANSFORMS.items()
            if exp not in _exclude_exp
        } | {
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_qualify,
                ]
            ),
        }


class Backend(BaseBackend, CanCreateDatabase, CanCreateSchema):
    name = "datafusion"
    dialect = "datafusion"
    builder = None
    supports_in_memory_tables = True

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
            (self.con, config) = (config, None)
        else:
            if SessionConfig is not None:
                df_config = SessionConfig(
                    {"datafusion.sql_parser.dialect": "PostgreSQL"}
                ).with_information_schema(True)
            else:
                df_config = None
            self.con = SessionContext(df_config)

        self._register_builtin_udfs()

        if not config:
            config = {}

        for name, path in config.items():
            self.register(path, table_name=name)

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
            volatility=getattr(udf_node, "config", {}).get("volatility", "volatile"),
            name=udf_node.__full_name__,
        )

    def _compile_elementwise_udf(self, udf_node):
        return df.udf(
            udf_node.func,
            input_types=list(map(PyArrowType.from_ibis, udf_node.input_type)),
            return_type=PyArrowType.from_ibis(udf_node.return_type),
            volatility="volatile",
            name=udf_node.func.__name__,
        )

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        log(sql)

    def raw_sql(self, query: str | sg.exp.Expression) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        kwargs
            Backend specific query arguments
        """
        with suppress(AttributeError):
            query = query.sql(dialect=self.dialect, pretty=True)
        self._log(query)
        return self.con.sql(query)

    @property
    def current_database(self) -> str:
        raise NotImplementedError()

    @property
    def current_schema(self) -> str:
        return NotImplementedError()

    def list_databases(self, like: str | None = None) -> list[str]:
        code = (
            sg.select(C.table_catalog)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
        ).sql()
        result = self.con.sql(code).to_pydict()
        return self._filter_with_like(result["table_catalog"], like)

    def create_database(self, name: str, force: bool = False) -> None:
        self.raw_sql(
            sg.exp.Create(kind="DATABASE", this=sg.to_identifier(name), exists=force)
        )

    def drop_database(self, name: str, force: bool = False) -> None:
        raise com.UnsupportedOperationError(
            "DataFusion does not support dropping databases"
        )

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        return self._filter_with_like(
            self.con.catalog(
                database if database is not None else "datafusion"
            ).names(),
            like=like,
        )

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        # not actually a table, but this is how sqlglot represents schema names
        schema_name = sg.table(name, db=database)
        self.raw_sql(sg.exp.Create(kind="SCHEMA", this=schema_name, exists=force))

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        schema_name = sg.table(name, db=database)
        self.raw_sql(sg.exp.Drop(kind="SCHEMA", this=schema_name, exists=force))

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
    ) -> list[str]:
        """List the available tables."""
        return self._filter_with_like(self.con.tables(), like)

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
        catalog = self.con.catalog()
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
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [source.to_batches()])
            return self.table(table_name)
        elif isinstance(source, pa.RecordBatch):
            self.con.deregister_table(table_name)
            self.con.register_record_batches(table_name, [[source]])
            return self.table(table_name)
        elif isinstance(source, pa.dataset.Dataset):
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

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        name = op.name
        schema = op.schema

        self.con.deregister_table(name)
        if batches := op.data.to_pyarrow(schema).to_batches():
            self.con.register_record_batches(name, [batches])
        else:
            empty_dataset = ds.dataset([], schema=schema.to_pyarrow())
            self.con.register_dataset(name=name, dataset=empty_dataset)

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        if self.supports_in_memory_tables:
            for memtable in expr.op().find(ops.InMemoryTable):
                self._register_in_memory_table(memtable)

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
        self.con.deregister_table(table_name)
        self.con.register_csv(table_name, path, **kwargs)
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
        self.con.deregister_table(table_name)
        self.con.register_parquet(table_name, path, **kwargs)
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
        self.con.deregister_table(table_name)

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

        self._register_in_memory_tables(expr)

        frame = self.con.sql(self.compile(expr.as_table(), params, **kwargs))
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

    def _to_sqlglot(
        self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = translate(table_expr.op(), params=params)
        assert not isinstance(sql, sg.exp.Subquery)

        if isinstance(sql, sg.exp.Table):
            sql = sg.select(STAR).from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a DataFusion SQL string."""
        return self._to_sqlglot(expr, limit=limit, params=params, **kwargs).sql(
            dialect=self.dialect, pretty=True
        )

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        from ibis.backends.datafusion.compiler.values import translate_val

        return translate_val.dispatch(operation) is not translate_val.dispatch(object)

    def create_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def create_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_table(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)

    def drop_view(self, *_, **__) -> ir.Table:
        raise NotImplementedError(self.name)
