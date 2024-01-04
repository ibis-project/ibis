from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import BaseBackend
from ibis.backends.base.sqlglot.compiler import STAR

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    import pyarrow as pa

    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir
    from ibis.backends.base.sqlglot.compiler import SQLGlotCompiler
    from ibis.common.typing import SupportsSchema


class SQLGlotBackend(BaseBackend):
    compiler: ClassVar[SQLGlotCompiler]
    name: ClassVar[str]

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        # singledispatchmethod overrides `__get__` so we can't directly access
        # the dispatcher
        dispatcher = cls.compiler.visit_node.register.__self__.dispatcher
        return dispatcher.dispatch(operation) is not dispatcher.dispatch(object)

    def table(
        self, name: str, schema: str | None = None, database: str | None = None
    ) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        schema
            Schema name
        database
            Database name

        Returns
        -------
        Table
            Table expression
        """
        table_schema = self.get_schema(name, schema=schema, database=database)
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(database=database, schema=schema),
        ).to_expr()

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = self.compiler.translate(table_expr.op(), params=params)
        assert not isinstance(sql, sge.Subquery)

        if isinstance(sql, sge.Table):
            sql = sg.select(STAR).from_(sql)

        assert not isinstance(sql, sge.Subquery)
        return sql

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a SQL string."""
        query = self._to_sqlglot(expr, limit=limit, params=params, **kwargs)
        sql = query.sql(dialect=self.name, pretty=True)
        self._log(sql)
        return sql

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        return self.compile(expr, **kwargs)

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        from ibis import util

        util.log(sql)

    def sql(
        self,
        query: str,
        schema: SupportsSchema | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        query = self._transpile_sql(query, dialect=dialect)
        if schema is None:
            schema = self._get_schema_using_query(query)
        return ops.SQLQueryResult(query, ibis.schema(schema), self).to_expr()

    @abc.abstractmethod
    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        """Return the metadata of a SQL query."""

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        schema: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        src = sge.Create(
            this=sg.table(
                name, db=schema, catalog=database, quoted=self.compiler.quoted
            ),
            kind="VIEW",
            replace=overwrite,
            expression=self.compile(obj),
        )
        self._register_in_memory_tables(obj)
        with self._safe_raw_sql(src):
            pass
        return self.table(name, database=database)

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        for memtable in expr.op().find(ops.InMemoryTable):
            self._register_in_memory_table(memtable)

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        src = sge.Drop(
            this=sg.table(
                name, db=schema, catalog=database, quoted=self.compiler.quoted
            ),
            kind="VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def _get_temp_view_definition(self, name: str, definition: str) -> str:
        return sge.Create(
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            kind="VIEW",
            expression=definition,
            replace=True,
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
        )

    def _create_temp_view(self, table_name, source):
        if table_name not in self._temp_views and table_name in self.list_tables():
            raise ValueError(
                f"{table_name} already exists as a non-temporary table or view"
            )

        with self._safe_raw_sql(self._get_temp_view_definition(table_name, source)):
            pass

        self._temp_views.add(table_name)
        self._register_temp_view_cleanup(table_name)

    def _register_temp_view_cleanup(self, name: str) -> None:
        """Register a clean up function for a temporary view.

        No-op by default.

        Parameters
        ----------
        name
            The temporary view to register for clean up.
        """

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(op.name)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, params=params, limit=limit, **kwargs)

        schema = table.schema()

        with self._safe_raw_sql(sql) as cur:
            result = self._fetch_from_cursor(cur, schema)
        return expr.__pandas_result__(result)

    def drop_table(
        self,
        name: str,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        drop_stmt = sg.exp.Drop(
            kind="TABLE",
            this=sg.table(
                name, db=schema, catalog=database, quoted=self.compiler.quoted
            ),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)

        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            while batch := cursor.fetchmany(chunk_size):
                yield batch

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return an iterator of pyarrow record batches.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.

        Returns
        -------
        RecordBatchReader
            Collection of pyarrow `RecordBatch`s.
        """
        pa = self._import_pyarrow()

        schema = expr.as_table().schema()
        array_type = schema.as_struct().to_pyarrow()
        arrays = (
            pa.array(map(tuple, batch), type=array_type)
            for batch in self._cursor_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size
            )
        )
        batches = map(pa.RecordBatch.from_struct_array, arrays)

        return pa.ipc.RecordBatchReader.from_batches(schema.to_pyarrow(), batches)
