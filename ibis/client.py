import abc

import six

from ibis.config import options

import ibis.util as util
import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.operations as ops
import ibis.sql.compiler as comp


class Client(object):
    pass


class Query(object):

    """Abstraction for DML query execution to enable queries, progress,
    cancellation and more (for backends supporting such functionality).
    """

    def __init__(self, client, sql, **kwargs):
        self.client = client

        dml = getattr(sql, 'dml', sql)
        self.expr = getattr(
            dml, 'parent_expr', getattr(dml, 'table_set', None)
        )

        if not isinstance(sql, six.string_types):
            self.compiled_sql = sql.compile()
        else:
            self.compiled_sql = sql

        self.result_wrapper = getattr(dml, 'result_handler', None)
        self.extra_options = kwargs

    def execute(self):
        # synchronous by default
        with self.client._execute(self.compiled_sql, results=True) as cur:
            result = self._fetch(cur)

        return self._wrap_result(result)

    def _wrap_result(self, result):
        if self.result_wrapper is not None:
            result = self.result_wrapper(result)
        return result

    def _fetch(self, cursor):
        raise NotImplementedError

    def schema(self):

        if isinstance(self.expr, (ir.TableExpr, ir.ExprList, sch.HasSchema)):
            return self.expr.schema()
        elif isinstance(self.expr, ir.ValueExpr):
            return sch.schema([(self.expr.get_name(), self.expr.type())])
        else:
            raise ValueError('Expression with type {} does not have a '
                             'schema'.format(type(self.expr)))


class SQLClient(six.with_metaclass(abc.ABCMeta, Client)):

    dialect = comp.Dialect
    query_class = Query
    table_class = ops.DatabaseTable
    table_expr_class = ir.TableExpr

    def table(self, name, database=None):
        """
        Create a table expression that references a particular table in the
        database

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
        return self.con.database

    def database(self, name=None):
        """
        Create a Database object for a given database name that can be used for
        exploring and manipulating the objects (tables, functions, views, etc.)
        inside

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

    def _execute(self, query, results=False):
        cur = self.con.execute(query)
        if results:
            return cur
        else:
            cur.release()

    def sql(self, query):
        """
        Convert a SQL query to an Ibis table expression

        Parameters
        ----------

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
        """
        Execute a given query string. Could have unexpected results if the
        query modifies the behavior of the session in a way unknown to Ibis; be
        careful.

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
        """
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
        result = self._execute_query(query_ast, **kwargs)
        return result

    def _execute_query(self, dml, **kwargs):
        query = self.query_class(self, dml, **kwargs)
        return query.execute()

    def compile(self, expr, params=None, limit=None):
        """
        Translate expression to one or more queries according to backend target

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
            if (isinstance(query, comp.Select) and
                    not isinstance(expr, ir.ScalarExpr) and
                    query.table_set is not None):
                if query.limit is None:
                    if limit == 'default':
                        query_limit = options.sql.default_limit
                    else:
                        query_limit = limit
                    if query_limit:
                        query.limit = {
                            'n': query_limit,
                            'offset': 0
                        }
                elif limit is not None and limit != 'default':
                    query.limit = {'n': limit,
                                   'offset': query.limit['offset']}
        return query_ast

    def explain(self, expr, params=None):
        """
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

        return 'Query:\n{0}\n\n{1}'.format(util.indent(query, 2),
                                           '\n'.join(result))

    def _build_ast(self, expr, context):
        # Implement in clients
        raise NotImplementedError(type(self).__name__)


class QueryPipeline(object):
    """
    Execute a series of queries, and capture any result sets generated

    Note: No query pipelines have yet been implemented
    """
    pass


def validate_backends(backends):
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


def execute(expr, limit='default', params=None, **kwargs):
    backend, = validate_backends(list(find_backends(expr)))
    return backend.execute(expr, limit=limit, params=params, **kwargs)


def compile(expr, limit=None, params=None, **kwargs):
    backend, = validate_backends(list(find_backends(expr)))
    return backend.compile(expr, limit=limit, params=params, **kwargs)


def find_backends(expr):
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


class Database(object):

    def __init__(self, name, client):
        self.name = name
        self.client = client

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.name)

    def __dir__(self):
        attrs = dir(type(self))
        unqualified_tables = [self._unqualify(x) for x in self.tables]
        return sorted(frozenset(attrs + unqualified_tables))

    def __contains__(self, key):
        return key in self.tables

    @property
    def tables(self):
        return self.list_tables()

    def __getitem__(self, key):
        return self.table(key)

    def __getattr__(self, key):
        return self.table(key)

    def _qualify(self, value):
        return value

    def _unqualify(self, value):
        return value

    def drop(self, force=False):
        """
        Drop the database

        Parameters
        ----------
        drop : boolean, default False
          Drop any objects if they exist, and do not fail if the databaes does
          not exist
        """
        self.client.drop_database(self.name, force=force)

    def namespace(self, ns):
        """
        Creates a derived Database instance for collections of objects having a
        common prefix. For example, for tables fooa, foob, and fooc, creating
        the "foo" namespace would enable you to reference those objects as a,
        b, and c, respectively.

        Returns
        -------
        ns : DatabaseNamespace
        """
        return DatabaseNamespace(self, ns)

    def table(self, name):
        """
        Return a table expression referencing a table in this database

        Returns
        -------
        table : TableExpr
        """
        qualified_name = self._qualify(name)
        return self.client.table(qualified_name, self.name)

    def list_tables(self, like=None):
        return self.client.list_tables(like=self._qualify_like(like),
                                       database=self.name)

    def _qualify_like(self, like):
        return like


class DatabaseNamespace(Database):

    def __init__(self, parent, namespace):
        self.parent = parent
        self.namespace = namespace

    def __repr__(self):
        return "{}(database={!r}, namespace={!r})".format(
            type(self).__name__, self.name, self.namespace
        )

    @property
    def client(self):
        return self.parent.client

    @property
    def name(self):
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


class DatabaseEntity(object):
    pass


class View(DatabaseEntity):

    def drop(self):
        pass
