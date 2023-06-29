from __future__ import annotations

import abc
import contextlib
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import toolz

import ibis.common.exceptions as exc
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import BaseBackend
from ibis.backends.base.sql.compiler import Compiler

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

__all__ = ['BaseSQLBackend']


class BaseSQLBackend(BaseBackend):
    """Base backend class for backends that compile to SQL."""

    compiler = Compiler

    @property
    def _sqlglot_dialect(self) -> str:
        return self.name

    def _from_url(self, url: str, **kwargs: Any) -> BaseBackend:
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments passed to the `connect` method.

        Returns
        -------
        BaseBackend
            A backend instance
        """
        import sqlalchemy as sa

        url = sa.engine.make_url(url)
        new_kwargs = kwargs.copy()
        kwargs = {}

        for name in ("host", "port", "database", "password"):
            if value := (
                getattr(url, name, None)
                or os.environ.get(f"{self.name.upper()}_{name.upper()}")
            ):
                kwargs[name] = value
        if username := url.username:
            kwargs["user"] = username

        kwargs.update(url.query)
        new_kwargs = toolz.merge(kwargs, new_kwargs)
        self._convert_kwargs(new_kwargs)
        return self.connect(**new_kwargs)

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        database
            Database name

        Returns
        -------
        Table
            Table expression
        """
        if database is not None and not isinstance(database, str):
            raise exc.IbisTypeError(
                f"`database` must be a string; got {type(database)}"
            )
        qualified_name = self._fully_qualified_name(name, database)
        schema = self.get_schema(qualified_name)
        node = ops.DatabaseTable(name, schema, self, namespace=database)
        return node.to_expr()

    def _fully_qualified_name(self, name, database):
        # XXX
        return name

    def sql(
        self, query: str, schema: sch.Schema | None = None, dialect: str | None = None
    ) -> ir.Table:
        """Convert a SQL query to an Ibis table expression.

        Parameters
        ----------
        query
            SQL string
        schema
            The expected schema for this query. If not provided, will be
            inferred automatically if possible.
        dialect
            Optional string indicating the dialect of `query`. The default
            value of `None` will use the backend's native dialect.

        Returns
        -------
        Table
            Table expression
        """
        query = self._transpile_sql(query, dialect=dialect)
        if schema is None:
            schema = self._get_schema_using_query(query)
        else:
            schema = sch.schema(schema)
        return ops.SQLQueryResult(query, schema, self).to_expr()

    def _get_schema_using_query(self, query):
        raise NotImplementedError(f"Backend {self.name} does not support .sql()")

    def raw_sql(self, query: str):
        """Execute a query string.

        !!! warning "The returned cursor object must be **manually** released if results are returned."

        Parameters
        ----------
        query
            DDL or DML statement
        """
        return self.con.execute(query)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        yield self.raw_sql(*args, **kwargs)

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        with self._safe_raw_sql(sql) as cursor:
            while batch := cursor.fetchmany(chunk_size):
                yield batch

    @util.experimental
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

    def _register_udfs(self, expr: ir.Expr) -> None:
        """Return an iterator of DDL strings, once for each UDFs contained within `expr`."""
        if self.supports_python_udfs:
            raise NotImplementedError(self.name)

    def _define_udf_translation_rules(self, expr: ir.Expr) -> None:
        if self.supports_python_udfs:
            raise NotImplementedError(self.name)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
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
            * `Table`: pandas.DataFrame
            * `Column`: pandas.Series
            * `Scalar`: Python scalar value
        """
        # TODO Reconsider having `kwargs` here. It's needed to support
        # `external_tables` in clickhouse, but better to deprecate that
        # feature than all this magic.
        # we don't want to pass `timecontext` to `raw_sql`
        self._run_pre_execute_hooks(expr)

        kwargs.pop('timecontext', None)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        self._log(sql)

        schema = expr.as_table().schema()

        with self._safe_raw_sql(sql, **kwargs) as cursor:
            result = self.fetch_from_cursor(cursor, schema)

        return expr.__pandas_result__(result)

    def _register_in_memory_table(self, _: ops.InMemoryTable) -> None:
        raise NotImplementedError(self.name)

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        if self.compiler.cheap_in_memory_tables:
            for memtable in an.find_memtables(expr.op()):
                self._register_in_memory_table(memtable)

    @abc.abstractmethod
    def fetch_from_cursor(self, cursor, schema):
        """Fetch data from cursor."""

    def _log(self, sql: str) -> None:
        """Log the SQL, usually to the standard output.

        This method can be implemented by subclasses. The logging
        happens when `ibis.options.verbose` is `True`.
        """
        util.log(sql)

    def compile(
        self,
        expr: ir.Expr,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        timecontext: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> Any:
        """Compile an Ibis expression.

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
        self._define_udf_translation_rules(expr)
        return self.compiler.to_ast_ensure_limit(expr, limit, params=params).compile()

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        return str(self.compile(expr, **kwargs))

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

        with self._safe_raw_sql(statement) as cur:
            result = self._get_list(cur)

        return '\n'.join(['Query:', util.indent(query, 2), '', *result])

    @classmethod
    @lru_cache
    def _get_operations(cls):
        translator = cls.compiler.translator_class
        return translator._registry.keys() | translator._rewrites.keys()

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return operation in cls._get_operations()

    def _create_temp_view(self, view, definition):
        raise NotImplementedError(
            f"The {self.name} backend does not implement temporary view creation"
        )
