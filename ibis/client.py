"""Ibis generic client classes and functions."""
import abc
from typing import List, Optional

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql.compiler import QueryContext, Select
from ibis.config import options
from ibis.expr.typing import TimeContext


class Client:
    """Base class for all clients."""

    def __init__(self, backend):
        self.database_class = backend.database_class
        self.table_class = backend.table_class

        if backend.kind in ('sql', 'sqlalchemy', 'spark'):
            self.context_class = backend.context_class
            if backend.kind in ('sql', 'spark'):
                self.table_expr_class = backend.table_expr_class


Backends = List[Client]


class SQLClient(Client, metaclass=abc.ABCMeta):
    """Generic SQL client."""

    context_class = QueryContext
    table_class = ops.DatabaseTable
    table_expr_class = ir.TableExpr

    def table(self, name, database=None):
        """Create a table expression.

        Create a table expression that references a particular table in the
        database.

        Parameters
        ----------
        name : string
        database : string, optional

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._fully_qualified_name(name, database)
        schema = self.get_schema(qualified_name)
        node = self.table_class(qualified_name, schema, self)
        return self.table_expr_class(node)

    @property
    def current_database(self):
        """Return the current database."""
        return self.con.database

    def database(self, name=None):
        """Create a database object.

        Create a Database object for a given database name that can be used for
        exploring and manipulating the objects (tables, functions, views, etc.)
        inside.

        Parameters
        ----------
        name : string
          Name of database

        Returns
        -------
        database : Database
        """
        # TODO: validate existence of database
        if name is None:
            name = self.current_database
        return self.database_class(name, self)

    def _fully_qualified_name(self, name, database):
        # XXX
        return name

    def sql(self, query):
        """Convert a SQL query to an Ibis table expression.

        Parameters
        ----------
        query : string

        Returns
        -------
        table : TableExpr
        """
        # Get the schema by adding a LIMIT 0 on to the end of the query. If
        # there is already a limit in the query, we find and remove it
        limited_query = 'SELECT * FROM ({}) t0 LIMIT 0'.format(query)
        schema = self._get_schema_using_query(limited_query)
        return ops.SQLQueryResult(query, schema, self).to_expr()

    def raw_sql(self, query: str, results=False):
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
        # TODO results is unused, it can be removed
        # (requires updating Impala tests)
        cursor = self.con.execute(query)
        if cursor:
            return cursor
        cursor.release()

    def execute(self, expr, params=None, limit='default', **kwargs):
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
        # TODO Reconsider having `kwargs` here. It's needed to support
        # `external_tables` in clickhouse, but better to deprecate that
        # feature than all this magic.
        # we don't want to pass `timecontext` to `raw_sql`
        kwargs.pop('timecontext', None)
        query_ast = self._build_ast_ensure_limit(expr, limit, params=params)
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

    def ast_schema(self, query_ast):
        """Return the schema of the expression.

        Returns
        -------
        Schema

        Raises
        ------
        ValueError
            if self.expr doesn't have a schema.
        """
        dml = getattr(query_ast, 'dml', query_ast)
        expr = getattr(dml, 'parent_expr', getattr(dml, 'table_set', None))

        if isinstance(expr, (ir.TableExpr, ir.ExprList, sch.HasSchema)):
            return expr.schema()
        elif isinstance(expr, ir.ValueExpr):
            return sch.schema([(expr.get_name(), expr.type())])
        else:
            raise ValueError(
                'Expression with type {} does not have a '
                'schema'.format(type(self.expr))
            )

    def _log(self, sql):
        """Log the SQL, usually to the standard output.

        This method can be implemented by subclasses. The logging happens
        when `ibis.options.verbose` is `True`.
        """
        pass

    def compile(
        self,
        expr,
        params=None,
        limit=None,
        timecontext: Optional[TimeContext] = None,
    ):
        """Translate expression.

        Translate expression to one or more queries according to
        backend target.

        Returns
        -------
        output : single query or list of queries
        """
        query_ast = self._build_ast_ensure_limit(expr, limit, params=params)
        return query_ast.compile()

    def _build_ast_ensure_limit(self, expr, limit, params=None):
        context = self.context_class(params=params)

        query_ast = self._build_ast(expr, context)
        # note: limit can still be None at this point, if the global
        # default_limit is None
        for query in reversed(query_ast.queries):
            if (
                isinstance(query, Select)
                and not isinstance(expr, ir.ScalarExpr)
                and query.table_set is not None
            ):
                if query.limit is None:
                    if limit == 'default':
                        query_limit = options.sql.default_limit
                    else:
                        query_limit = limit
                    if query_limit:
                        query.limit = {'n': query_limit, 'offset': 0}
                elif limit is not None and limit != 'default':
                    query.limit = {'n': limit, 'offset': query.limit['offset']}
        return query_ast

    def explain(self, expr, params=None):
        """Explain expression.

        Query for and return the query plan associated with the indicated
        expression or SQL query.

        Returns
        -------
        plan : string
        """
        if isinstance(expr, ir.Expr):
            context = self.context_class(params=params)
            query_ast = self._build_ast(expr, context)
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

    def _build_ast(self, expr, context):
        # Implement in clients
        raise NotImplementedError(type(self).__name__)


def validate_backends(backends) -> list:
    """Validate backends.

    Parameters
    ----------
    backends

    Returns
    -------
    list
        A list containing a specified backend or the default backend.

    Raises
    ------
    ibis.common.exceptions.IbisError
        If no backend is specified and
        if there is no default backend specified.
    ValueError
        if there are more than one backend specified.
    """
    if not backends:
        default = options.default_backend
        if default is None:
            raise com.IbisError(
                'Expression depends on no backends, and found no default'
            )
        return [default]

    if len(backends) > 1:
        raise ValueError('Multiple backends found')
    return backends


def execute(
    expr,
    limit: str = 'default',
    params: dict = None,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
):
    """Execute given expression using the backend available.

    Parameters
    ----------
    expr : Expr
    limit : string
    params : dict
    timecontext: Optional[TimeContext]
    kwargs : dict

    Returns
    -------
    output : input type dependent
      Table expressions: pandas.DataFrame
      Array expressions: pandas.Series
      Scalar expressions: Python scalar value
    """
    (backend,) = validate_backends(list(find_backends(expr)))
    return backend.execute(
        expr, limit=limit, params=params, timecontext=timecontext, **kwargs
    )


def compile(
    expr,
    limit: str = None,
    params: dict = None,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
) -> str:
    """Translate given expression.

    Parameters
    ----------
    expr : Expr
    limit : string
    params : dict
    timecontext: Optional[TimeContext]
    kwargs : dict

    Returns
    -------
    expression_translated : string
    """
    (backend,) = validate_backends(list(find_backends(expr)))
    return backend.compile(
        expr, limit=limit, params=params, timecontext=timecontext, **kwargs
    )


def find_backends(expr):
    """Find backends.

    Parameters
    ----------
    expr : Expr

    Returns
    -------
    client : Client
        Backend found.
    """
    seen_backends = set()

    stack = [expr.op()]
    seen = set()

    while stack:
        node = stack.pop()

        if node not in seen:
            seen.add(node)

            for arg in node.flat_args():
                if isinstance(arg, Client):
                    if arg not in seen_backends:
                        yield arg
                        seen_backends.add(arg)
                elif isinstance(arg, ir.Expr):
                    stack.append(arg.op())


class Database:
    """Generic Database class."""

    def __init__(self, name, client):
        """Initialize the new object."""
        self.name = name
        self.client = client

    def __repr__(self) -> str:
        """Return type name and the name of the database."""
        return '{}({!r})'.format(type(self).__name__, self.name)

    def __dir__(self) -> set:
        """Return a set of attributes and tables available for the database.

        Returns
        -------
        set
            A set of the attributes and tables available for the database.
        """
        attrs = dir(type(self))
        unqualified_tables = [self._unqualify(x) for x in self.tables]
        return sorted(frozenset(attrs + unqualified_tables))

    def __contains__(self, key: str) -> bool:
        """
        Check if the given table (key) is available for the current database.

        Parameters
        ----------
        key : string

        Returns
        -------
        bool
            True if the given key (table name) is available for the current
            database.
        """
        return key in self.tables

    @property
    def tables(self) -> list:
        """Return a list with all available tables.

        Returns
        -------
        list
        """
        return self.list_tables()

    def __getitem__(self, key: str) -> ir.TableExpr:
        """Return a TableExpr for the given table name (key).

        Parameters
        ----------
        key : string

        Returns
        -------
        TableExpr
        """
        return self.table(key)

    def __getattr__(self, key: str) -> ir.TableExpr:
        """Return a TableExpr for the given table name (key).

        Parameters
        ----------
        key : string

        Returns
        -------
        TableExpr
        """
        return self.table(key)

    def _qualify(self, value):
        return value

    def _unqualify(self, value):
        return value

    def drop(self, force: bool = False):
        """Drop the database.

        Parameters
        ----------
        force : boolean, default False
          If True, Drop any objects if they exist, and do not fail if the
          databaes does not exist.
        """
        self.client.drop_database(self.name, force=force)

    def namespace(self, ns: str):
        """
        Create a database namespace for accessing objects with common prefix.

        Creates a derived Database instance for collections of objects having a
        common prefix. For example, for tables fooa, foob, and fooc, creating
        the "foo" namespace would enable you to reference those objects as a,
        b, and c, respectively.

        Parameters
        ----------
        ns : string

        Returns
        -------
        ns : DatabaseNamespace
        """
        return DatabaseNamespace(self, ns)

    def table(self, name: str) -> ir.TableExpr:
        """Return a table expression referencing a table in this database.

        Parameters
        ----------
        name : string

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._qualify(name)
        return self.client.table(qualified_name, self.name)

    def list_tables(self, like: str = None) -> list:
        """Return a list of all tables available for the current database.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'.

        Returns
        -------
        list
            A list with all tables available for the current database.
        """
        return self.client.list_tables(
            like=self._qualify_like(like), database=self.name
        )

    def _qualify_like(self, like: str) -> str:
        return like


class DatabaseNamespace(Database):
    """Database Namespace class."""

    def __init__(self, parent, namespace):
        """Initialize the Database Namespace object."""
        self.parent = parent
        self.namespace = namespace

    def __repr__(self) -> str:
        """Return the database typename and database and namespace name."""
        return "{}(database={!r}, namespace={!r})".format(
            type(self).__name__, self.name, self.namespace
        )

    @property
    def client(self):
        """Return the client.

        Returns
        -------
        client : Client
        """
        return self.parent.client

    @property
    def name(self) -> str:
        """Return the name of the database.

        Returns
        -------
        name : string
        """
        return self.parent.name

    def _qualify(self, value):
        return self.namespace + value

    def _unqualify(self, value):
        return value.replace(self.namespace, '', 1)

    def _qualify_like(self, like):
        if like:
            return self.namespace + like
        else:
            return '{0}*'.format(self.namespace)


class DatabaseEntity:
    """Database Entity class."""

    pass


class View(DatabaseEntity):
    """View class."""

    def drop(self):
        """Drop method."""
        pass
