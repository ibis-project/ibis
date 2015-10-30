# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ibis.compat import zip as czip
from ibis.config import options
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.compiler as comp
import ibis.common as com
import ibis.util as util


class Client(object):

    pass


class Query(object):

    """
    Abstraction for DDL query execution to enable both synchronous and
    asynchronous queries, progress, cancellation and more (for backends
    supporting such functionality).
    """

    def __init__(self, client, ddl):
        self.client = client

        if isinstance(ddl, comp.DDL):
            self.compiled_ddl = ddl.compile()
        else:
            self.compiled_ddl = ddl

        self.result_wrapper = getattr(ddl, 'result_handler', None)

    def execute(self):
        # synchronous by default
        with self.client._execute(self.compiled_ddl, results=True) as cur:
            result = self._fetch(cur)

        return self._wrap_result(result)

    def _wrap_result(self, result):
        if self.result_wrapper is not None:
            result = self.result_wrapper(result)
        return result

    def _fetch(self, cursor):
        import pandas as pd
        rows = cursor.fetchall()
        # TODO(wesm): please evaluate/reimpl to optimize for perf/memory
        dtypes = [self._db_type_to_dtype(x[1]) for x in cursor.description]
        names = [x[0] for x in cursor.description]
        cols = {}
        for (col, name, dtype) in czip(czip(*rows), names, dtypes):
            try:
                cols[name] = pd.Series(col, dtype=dtype)
            except TypeError:
                # coercing to specified dtype failed, e.g. NULL vals in int col
                cols[name] = pd.Series(col)
        return pd.DataFrame(cols, columns=names)

    def _db_type_to_dtype(self, db_type):
        raise NotImplementedError


class AsyncQuery(Query):

    """
    Abstract asynchronous query
    """

    def execute(self):
        raise NotImplementedError

    def is_finished(self):
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError


class SQLClient(Client):

    sync_query = Query
    async_query = Query

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
        node = ops.DatabaseTable(qualified_name, schema, self)
        return self._table_expr_klass(node)

    @property
    def _table_expr_klass(self):
        return ir.TableExpr

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
        limited_query = """\
SELECT *
FROM (
{0}
) t0
LIMIT 0""".format(query)
        schema = self._get_schema_using_query(limited_query)

        node = ops.SQLQueryResult(query, schema, self)
        return ir.TableExpr(node)

    def raw_sql(self, query, results=False):
        """
        Execute a given query string. Could have unexpected results if the
        query modifies the behavior of the session in a way unknown to Ibis; be
        careful.

        Parameters
        ----------
        query : string
          SQL or DDL statement
        results : boolean, default False
          Pass True if the query as a result set

        Returns
        -------
        cur : ImpalaCursor if results=True, None otherwise
          You must call cur.release() after you are finished using the cursor.
        """
        return self._execute(query, results=results)

    def execute(self, expr, params=None, limit='default', async=False):
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
        async : boolean, default False

        Returns
        -------
        output : input type dependent
          Table expressions: pandas.DataFrame
          Array expressions: pandas.Series
          Scalar expressions: Python scalar value
        """
        ast = self._build_ast_ensure_limit(expr, limit)

        if len(ast.queries) > 1:
            raise NotImplementedError
        else:
            return self._execute_query(ast.queries[0], async=async)

    def _execute_query(self, ddl, async=False):
        klass = self.async_query if async else self.sync_query
        return klass(self, ddl).execute()

    def compile(self, expr, params=None, limit=None):
        """
        Translate expression to one or more queries according to backend target

        Returns
        -------
        output : single query or list of queries
        """
        ast = self._build_ast_ensure_limit(expr, limit)
        queries = [query.compile() for query in ast.queries]
        return queries[0] if len(queries) == 1 else queries

    def _build_ast_ensure_limit(self, expr, limit):
        ast = self._build_ast(expr)
        # note: limit can still be None at this point, if the global
        # default_limit is None
        for query in reversed(ast.queries):
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
        return ast

    def explain(self, expr):
        """
        Query for and return the query plan associated with the indicated
        expression or SQL query.

        Returns
        -------
        plan : string
        """
        if isinstance(expr, ir.Expr):
            ast = self._build_ast(expr)
            if len(ast.queries) > 1:
                raise Exception('Multi-query expression')

            query = ast.queries[0].compile()
        else:
            query = expr

        statement = 'EXPLAIN {0}'.format(query)

        with self._execute(statement, results=True) as cur:
            result = self._get_list(cur)

        return 'Query:\n{0}\n\n{1}'.format(util.indent(query, 2),
                                           '\n'.join(result))

    def _build_ast(self, expr):
        # Implement in clients
        raise NotImplementedError


class QueryPipeline(object):
    """
    Execute a series of queries, possibly asynchronously, and capture any
    result sets generated

    Note: No query pipelines have yet been implemented
    """
    pass


def execute(expr, limit='default', async=False):
    backend = find_backend(expr)
    return backend.execute(expr, limit=limit, async=async)


def compile(expr, limit=None):
    backend = find_backend(expr)
    return backend.compile(expr, limit=limit)


def find_backend(expr):
    backends = []

    def walk(expr):
        node = expr.op()
        for arg in node.flat_args():
            if isinstance(arg, Client):
                backends.append(arg)
            elif isinstance(arg, ir.Expr):
                walk(arg)

    walk(expr)
    backends = util.unique_by_key(backends, id)

    if len(backends) > 1:
        raise ValueError('Multiple backends found')
    elif len(backends) == 0:
        default = options.default_backend
        if default is None:
            raise com.IbisError('Expression depends on no backends, '
                                'and found no default')
        return default

    return backends[0]


class Database(object):

    def __init__(self, name, client):
        self.name = name
        self.client = client

    def __repr__(self):
        return "{0}('{1}')".format('Database', self.name)

    def __dir__(self):
        attrs = dir(type(self))
        unqualified_tables = [self._unqualify(x) for x in self.tables]
        return list(sorted(set(attrs + unqualified_tables)))

    def __contains__(self, key):
        return key in self.tables

    @property
    def tables(self):
        return self.list_tables()

    def __getitem__(self, key):
        return self.table(key)

    def __getattr__(self, key):
        special_attrs = ['_ipython_display_', 'trait_names',
                         '_getAttributeNames']

        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in special_attrs:
                raise
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
        return ("{0}(database={1!r}, namespace={2!r})"
                .format('DatabaseNamespace', self.name, self.namespace))

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
