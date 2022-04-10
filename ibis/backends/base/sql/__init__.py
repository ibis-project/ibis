from __future__ import annotations

import abc
from typing import Any, Mapping

import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base import BaseBackend
from ibis.backends.base.sql.compiler import Compiler
from ibis.expr.typing import TimeContext

__all__ = [
    'BaseSQLBackend',
]


class BaseSQLBackend(BaseBackend):
    """Base backend class for backends that compile to SQL."""

    compiler = Compiler
    table_class = ops.DatabaseTable
    table_expr_class = ir.TableExpr

    def table(self, name: str, database: str | None = None) -> ir.TableExpr:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        database
            Database name

        Returns
        -------
        TableExpr
            Table expression
        """
        qualified_name = self._fully_qualified_name(name, database)
        schema = self.get_schema(qualified_name)
        node = self.table_class(qualified_name, schema, self)
        return self.table_expr_class(node)

    def _fully_qualified_name(self, name, database):
        # XXX
        return name

    def sql(self, query: str) -> ir.TableExpr:
        """Convert a SQL query to an Ibis table expression.

        Parameters
        ----------
        query
            SQL string

        Returns
        -------
        TableExpr
            Table expression
        """
        # Get the schema by adding a LIMIT 0 on to the end of the query. If
        # there is already a limit in the query, we find and remove it
        limited_query = f'SELECT * FROM ({query}) t0 LIMIT 0'
        schema = self._get_schema_using_query(limited_query)
        return ops.SQLQueryResult(query, schema, self).to_expr()

    def _get_schema_using_query(self, query):
        raise NotImplementedError(
            f"Backend {self.name} does not support .sql()"
        )

    def raw_sql(self, query: str, results: bool = False) -> Any:
        """Execute a query string.

        Could have unexpected results if the query modifies the behavior of
        the session in a way unknown to Ibis; be careful.

        Parameters
        ----------
        query
            DML or DDL statement

        Returns
        -------
        Any
            Backend cursor
        """
        # TODO results is unused, it can be removed
        # (requires updating Impala tests)
        # TODO `self.con` is assumed to be defined in subclasses, but there
        # is nothing that enforces it. We should find a way to make sure
        # `self.con` is always a DBAPI2 connection, or raise an error
        cursor = self.con.execute(query)  # type: ignore
        if cursor:
            return cursor
        cursor.release()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.ScalarExpr, Any] | None = None,
        limit: str = 'default',
        **kwargs: Any,
    ):
        """Compile and execute an Ibis expression.

        Compile and execute Ibis expression using this backend client
        interface, returning results in-memory in the appropriate object type

        Parameters
        ----------
        expr
            Ibis expression
        limit
            For expressions yielding result sets; retrieve at most this number
            of values/rows. Overrides any limit already set on the expression.
        params
            Named unbound parameters
        kwargs
            Backend specific arguments. For example, the clickhouse backend
            uses this to receive `external_tables` as a dictionary of pandas
            DataFrames.

        Returns
        -------
        DataFrame | Series | Scalar
            * `TableExpr`: pandas.DataFrame
            * `ColumnExpr`: pandas.Series
            * `ScalarExpr`: Python scalar value
        """
        # TODO Reconsider having `kwargs` here. It's needed to support
        # `external_tables` in clickhouse, but better to deprecate that
        # feature than all this magic.
        # we don't want to pass `timecontext` to `raw_sql`
        kwargs.pop('timecontext', None)
        query_ast = self.compiler.to_ast_ensure_limit(
            expr, limit, params=params
        )
        sql = query_ast.compile()
        self._log(sql)
        cursor = self.raw_sql(sql, **kwargs)
        schema = self.ast_schema(query_ast, **kwargs)
        result = self.fetch_from_cursor(cursor, schema)

        if hasattr(getattr(query_ast, 'dml', query_ast), 'result_handler'):
            result = query_ast.dml.result_handler(result)

        return result

    @abc.abstractmethod
    def fetch_from_cursor(self, cursor, schema):
        """Fetch data from cursor."""

    def ast_schema(self, query_ast, **kwargs) -> sch.Schema:
        """Return the schema of the expression.

        Parameters
        ----------
        query_ast
            The AST of the query
        kwargs
            Backend specific parameters

        Returns
        -------
        Schema
            An ibis schema

        Raises
        ------
        ValueError
            if `self.expr` doesn't have a schema.
        """
        dml = getattr(query_ast, 'dml', query_ast)
        expr = getattr(dml, 'parent_expr', getattr(dml, 'table_set', None))

        if isinstance(expr, (ir.TableExpr, sch.HasSchema)):
            return expr.schema()
        elif isinstance(expr, ir.ValueExpr):
            return sch.schema([(expr.get_name(), expr.type())])
        else:
            raise ValueError(
                'Expression with type {} does not have a '
                'schema'.format(type(self.expr))
            )

    def _log(self, sql: str) -> None:
        """Log the SQL, usually to the standard output.

        This method can be implemented by subclasses. The logging happens
        when `ibis.options.verbose` is `True`.
        """

    def compile(
        self,
        expr: ir.Expr,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        timecontext: TimeContext | None = None,
    ) -> Any:
        """Compille an Ibis expression.

        Parameters
        ----------
        expr
            Ibis expression
        limit
            For expressions yielding result sets; retrieve at most this number
            of values/rows. Overrides any limit already set on the expression.
        params
            Named unbound parameters
        timecontext
            Additional information about data source time boundaries

        Returns
        -------
        Any
            The output of compilation. The type of this value depends on the
            backend.
        """
        return self.compiler.to_ast_ensure_limit(
            expr, limit, params=params
        ).compile()

    def explain(
        self,
        expr: ir.Expr | str,
        params: Mapping[ir.Expr, Any] | None = None,
    ) -> str:
        """Explain an expression.

        Return the query plan associated with the indicated expression or SQL
        query.

        Returns
        -------
        str
            Query plan
        """
        if isinstance(expr, ir.Expr):
            context = self.compiler.make_context(params=params)
            query_ast = self.compiler.to_ast(expr, context)
            if len(query_ast.queries) > 1:
                raise Exception('Multi-query expression')

            query = query_ast.queries[0].compile()
        else:
            query = expr

        statement = f'EXPLAIN {query}'

        cur = self.raw_sql(statement)
        result = self._get_list(cur)
        cur.release()

        return '\n'.join(['Query:', util.indent(query, 2), '', *result])

    @classmethod
    def has_operation(cls, operation: type[ops.ValueOp]) -> bool:
        translator = cls.compiler.translator_class
        op_classes = translator._registry.keys() | translator._rewrites.keys()
        return operation in op_classes

    def _create_temp_view(self, view, definition):
        raise NotImplementedError(
            f"The {self.name} backend does not implement temporary view "
            "creation"
        )
