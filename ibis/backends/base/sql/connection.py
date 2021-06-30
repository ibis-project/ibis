import abc
from typing import Any, Callable, Dict, Optional, Union

import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import BaseConnection
from ibis.backends.base.sql import Compiler

# TODO Is there a type for any DBAPI2 cursor?
Cursor = Any


class BaseSQLConnection(BaseConnection):
    @abc.abstractmethod
    @property
    def compiler(self) -> Compiler:
        """Compiler class for the backend.

        Must be a subclass of `ibis.backends.base.sql.Compiler`.
        """

    def compile(
        self, expr: ir.Expr, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compile the expression.
        """
        return self.client.compiler.to_sql(expr, params=params)

    def raw_sql(self, query: str) -> Optional[Cursor]:
        """Execute a given query string.

        Could have unexpected results if the query modifies the behavior of
        the session in a way unknown to Ibis; be careful.

        Parameters
        ----------
        query : string
          DML or DDL statement

        Returns
        -------
        Backend cursor
        """
        cursor = self.con.execute(query)
        if cursor:
            return cursor
        cursor.release()

    def execute(
        self,
        expr: ir.Expr,
        params: Optional[Dict[str, Any]] = None,
        limit: Optional[Union[int, str]] = 'default',
        **kwargs,
    ):
        """Compile and execute the given Ibis expression.

        Compile and execute Ibis expression using this backend client
        interface, returning results in-memory in the appropriate object type

        Parameters
        ----------
        expr : Expr
        limit : int, default None
          For expressions yielding result yets; retrieve at most this number of
          values/rows. Overrides any limit already set on the expression.
        params : not yet implemented
        kwargs : Backends can receive extra params. For example, clickhouse
            uses this to receive external_tables as dataframes.

        Returns
        -------
        output : input type dependent
          Table expressions: pandas.DataFrame
          Array expressions: pandas.Series
          Scalar expressions: Python scalar value
        """
        # TODO `limit` default should be None, and `str` shouldn't be
        # a supported type (requires changing `to_ast_ensure_limit`)
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
    def fetch_from_cursor(self, cursor: Cursor, schema: sch.Schema):
        """Fetch data from cursor."""

    def explain(self, expr: Union[str, ir.Expr], params=None) -> str:
        """Explain expression.

        Query for and return the query plan associated with the indicated
        expression or SQL query.

        Returns
        -------
        plan : string
        """
        if isinstance(expr, ir.Expr):
            context = self.compiler.make_context(params=params)
            query_ast = self.compiler.to_ast(expr, context)
            if len(query_ast.queries) > 1:
                raise Exception('Multi-query expression')

            query = query_ast.queries[0].compile()
        else:
            query = expr

        statement = 'EXPLAIN {0}'.format(query)

        cur = self.raw_sql(statement)
        result = self._get_list(cur)
        cur.release()

        return '\n'.join(['Query:', util.indent(query, 2), '', *result])

    def add_operation(self, operation: Callable) -> Callable:
        """
        Decorator to add a translation function to the backend for a specific
        operation.

        Operations are defined in `ibis.expr.operations`, and a translation
        function receives the translator object and an expression as
        parameters, and returns a value depending on the backend. For example,
        in SQL backends, a NullLiteral operation could be translated simply
        with the string "NULL".

        Examples
        --------
        >>> @ibis.sqlite.add_operation(ibis.expr.operations.NullLiteral)
        ... def _null_literal(translator, expression):
        ...     return 'NULL'
        """

        def decorator(translation_function):
            self.client.compiler.translator_class.add_operation(
                operation, translation_function
            )

        return decorator
