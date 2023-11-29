from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import BaseBackend
from ibis.backends.base.sqlglot.compiler import STAR

if TYPE_CHECKING:
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

        sql = self.compiler.translate(table_expr.op(), params=params)
        assert not isinstance(sql, sge.Subquery)

        if isinstance(sql, sge.Table):
            sql = sg.select(STAR).from_(sql)

        assert not isinstance(sql, sge.Subquery)
        return sql

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a ClickHouse SQL string."""
        return self._to_sqlglot(expr, limit=limit, params=params, **kwargs).sql(
            dialect=self.name, pretty=True
        )

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

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))
