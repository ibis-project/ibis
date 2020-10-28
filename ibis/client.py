"""Ibis generic client classes and functions."""
import abc
from typing import List, Optional

import ibis.backends.base_sqlalchemy.compiler as comp
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
import ibis.util as util
from ibis.config import options
from ibis.expr.typing import TimeContext


class Client:
    """Generic Ibis client."""

    pass


Backends = List[Client]


class Query:
    """Abstraction for DML query execution.

    This class enables queries, progress, and more
    (for backends supporting such functionality).
    """

    def __init__(self, client, sql, **kwargs):
        self.client = client

        dml = getattr(sql, 'dml', sql)
        self.expr = getattr(
            dml, 'parent_expr', getattr(dml, 'table_set', None)
        )

        if not isinstance(sql, str):
            self.compiled_sql = sql.compile()
        else:
            self.compiled_sql = sql

        self.result_wrapper = getattr(dml, 'result_handler', None)
        self.extra_options = kwargs

    def execute(self, **kwargs):
        """Execute a DML expression.

        Returns
        -------
        output : input type dependent
          Table expressions: pandas.DataFrame
          Array expressions: pandas.Series
          Scalar expressions: Python scalar value
        """
        # synchronous by default
        with self.client._execute(
            self.compiled_sql, results=True, **kwargs
        ) as cur:
            result = self._fetch(cur)

        return self._wrap_result(result)

    def _wrap_result(self, result):
        if self.result_wrapper is not None:
            result = self.result_wrapper(result)
        return result

    def _fetch(self, cursor):
        raise NotImplementedError

    def schema(self):
        """Return the schema of the expression.

        Returns
        -------
        Schema

        Raises
        ------
        ValueError
            if self.expr doesn't have a schema.
        """
        if isinstance(self.expr, (ir.TableExpr, ir.ExprList, sch.HasSchema)):
            return self.expr.schema()
        elif isinstance(self.expr, ir.ValueExpr):
            return sch.schema([(self.expr.get_name(), self.expr.type())])
        else:
            raise ValueError(
                'Expression with type {} does not have a '
                'schema'.format(type(self.expr))
            )


class SQLClient(Client, metaclass=abc.ABCMeta):
    """Generic SQL client."""

    dialect = comp.Dialect
    query_class = Query
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
        schema = self._get_table_schema(qualified_name)
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

    def _execute(self, query, results=False, **kwargs):
        cur = self.con.execute(query)
        if results:
            return cur
        else:
            cur.release()

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

    def raw_sql(self, query, results=False):
        """Execute a given query string.

        Could have unexpected results if the query modifies the behavior of
        the session in a way unknown to Ibis; be careful.

        Parameters
        ----------
        query : string
          DML or DDL statement
        results : boolean, default False
          Pass True if the query as a result set

        Returns
        -------
        cur : ImpalaCursor if results=True, None otherwise
          You must call cur.release() after you are finished using the cursor.
        """
        return self._execute(query, results=results)

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

        Returns
        -------
        output : input type dependent
          Table expressions: pandas.DataFrame
          Array expressions: pandas.Series
          Scalar expressions: Python scalar value
        """
        query_ast = self._build_ast_ensure_limit(expr, limit, params=params)
        query = self._get_query(query_ast, **kwargs)
        self._log(query.compiled_sql)
        result = self._execute_query(query, **kwargs)
        return result

    def _get_query(self, dml, **kwargs):
        return self.query_class(self, dml, **kwargs)

    def _execute_query(self, query, **kwargs):
        return query.execute()

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
        context = self.dialect.make_context(params=params)

        query_ast = self._build_ast(expr, context)
        # note: limit can still be None at this point, if the global
        # default_limit is None
        for query in reversed(query_ast.queries):
            if (
                isinstance(query, comp.Select)
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
            context = self.dialect.make_context(params=params)
            query_ast = self._build_ast(expr, context)
            if len(query_ast.queries) > 1:
                raise Exception('Multi-query expression')

            query = query_ast.queries[0].compile()
        else:
            query = expr

        statement = 'EXPLAIN {0}'.format(query)

        with self._execute(statement, results=True) as cur:
            result = self._get_list(cur)

        return 'Query:\n{0}\n\n{1}'.format(
            util.indent(query, 2), '\n'.join(result)
        )

    def _build_ast(self, expr, context):
        # Implement in clients
        raise NotImplementedError(type(self).__name__)


class QueryPipeline:
    """Execute a series of queries, and capture any result sets generated.

    Note: No query pipelines have yet been implemented.
    """

    pass


def validate_backends(backends) -> list:
    """Validate bacckends.

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
