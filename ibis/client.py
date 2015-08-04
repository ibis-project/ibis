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

from itertools import izip
from posixpath import join as pjoin
from six import BytesIO
import Queue
import threading

import hdfs

from impala.error import Error as ImpylaError
import impala.dbapi as impyla_dbapi

from ibis.config import options

from ibis.filesystems import HDFS, WebHDFS

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.compiler as sql
import ibis.sql.ddl as ddl
import ibis.util as util


class Client(object):

    pass


class SQLClient(Client):

    def database(self, name):
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
        return Database(name, self)

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

    def execute(self, expr, params=None, limit=None):
        """

        """
        ast = self._build_ast_ensure_limit(expr, limit)

        # TODO: create some query pipeline executor abstraction
        output = None
        for query in ast.queries:
            sql_string = query.compile()

            with self._execute(sql_string, results=True) as cur:
                result = self._fetch_from_cursor(cur)

            if isinstance(query, ddl.Select):
                if query.result_handler is not None:
                    result = query.result_handler(result)

                output = result

        return output

    def _build_ast_ensure_limit(self, expr, limit):
        ast = sql.build_ast(expr)
        # note: limit can still be None at this point, if the global
        # default_limit is None
        for query in reversed(ast.queries):
            if (isinstance(query, ddl.Select) and
                    not isinstance(expr, ir.ScalarExpr) and
                    query.table_set is not None):
                if query.limit is None:
                    query_limit = limit or options.sql.default_limit
                    if query_limit:
                        query.limit = {
                            'n': query_limit,
                            'offset': 0
                        }
                elif limit is not None:
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
            ast = sql.build_ast(expr)
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

    def _db_type_to_dtype(self, db_type):
        raise NotImplementedError

    def _fetch_from_cursor(self, cursor):
        import pandas as pd
        rows = cursor.fetchall()
        # TODO(wesm): please evaluate/reimpl to optimize for perf/memory
        dtypes = [self._db_type_to_dtype(x[1]) for x in cursor.description]
        names = [x[0] for x in cursor.description]
        cols = {}
        for (col, name, dtype) in zip(izip(*rows), names, dtypes):
            try:
                cols[name] = pd.Series(col, dtype=dtype)
            except TypeError:
                # coercing to specified dtype failed, e.g. NULL vals in int col
                cols[name] = pd.Series(col)
        return pd.DataFrame(cols, columns=names)


class ImpalaConnection(object):

    """
    Database connection wrapper
    """

    def __init__(self, pool_size=8, database='default', **params):
        self.params = params
        self.codegen_disabled = False
        self.database = database

        self.lock = threading.Lock()

        self.connection_pool = Queue.Queue(pool_size)
        self.connection_pool_size = 0
        self.max_pool_size = pool_size

        self.ping()

    def set_database(self, name):
        self.database = name

    def disable_codegen(self, disabled=True):
        self.codegen_disabled = disabled

    def execute(self, query):
        if isinstance(query, ddl.DDLStatement):
            query = query.compile()

        cursor = self._get_cursor()
        self.log(query)

        try:
            cursor.execute(query)
        except:
            cursor.release()
            self.error('Exception caused by {0}'.format(query))
            raise

        return cursor

    def log(self, msg):
        if options.verbose:
            (options.verbose_log or to_stdout)(msg)

    def error(self, msg):
        self.log(msg)

    def fetchall(self, query):
        cursor = self.execute(query)
        results = cursor.fetchall()
        cursor.release()
        return results

    def _get_cursor(self):
        try:
            cur = self.connection_pool.get(False)
            if cur.database != self.database:
                cur = self._new_cursor()
            if cur.codegen_disabled != self.codegen_disabled:
                cur.disable_codegen(self.codegen_disabled)
            return cur
        except Queue.Empty:
            if self.connection_pool_size < self.max_pool_size:
                cursor = self._new_cursor()
                self.connection_pool_size += 1
                return cursor
            else:
                raise com.InternalError('Too many concurrent / hung queries')

    def _new_cursor(self):
        params = self.params.copy()
        con = impyla_dbapi.connect(database=self.database, **params)

        # make sure the connection works
        cursor = con.cursor()
        cursor.ping()

        wrapper = ImpalaCursor(cursor, self, self.database)

        if self.codegen_disabled:
            wrapper.disable_codegen(self.codegen_disabled)

        return wrapper

    def ping(self):
        self._new_cursor()


class ImpalaCursor(object):

    def __init__(self, cursor, con, database, codegen_disabled=False):
        self.cursor = cursor
        self.con = con
        self.database = database
        self.codegen_disabled = codegen_disabled

    def __del__(self):
        self.cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def disable_codegen(self, disabled=True):
        self.codegen_disabled = disabled
        query = ('SET disable_codegen={0}'
                 .format('true' if disabled else 'false'))
        self.cursor.execute(query)

    @property
    def description(self):
        return self.cursor.description

    def release(self):
        self.con.connection_pool.put(self)

    def execute(self, stmt):
        self.cursor.execute(stmt)

    def fetchall(self):
        return self.cursor.fetchall()


class ImpalaClient(SQLClient):

    """
    An Ibis client interface that uses Impala
    """

    _HS2_TTypeId_to_dtype = {
        'BOOLEAN': 'bool',
        'TINYINT': 'int8',
        'SMALLINT': 'int16',
        'INT': 'int32',
        'BIGINT': 'int64',
        'TIMESTAMP': 'datetime64[ns]',
        'FLOAT': 'float32',
        'DOUBLE': 'float64',
        'STRING': 'string',
        'DECIMAL': 'object',
        'BINARY': 'string',
        'VARCHAR': 'string',
        'CHAR': 'string'
    }

    def __init__(self, con, hdfs_client=None, **params):
        self.con = con

        if isinstance(hdfs_client, hdfs.Client):
            hdfs_client = WebHDFS(hdfs_client)
        elif hdfs_client is not None and not isinstance(hdfs_client, HDFS):
            raise TypeError(hdfs_client)

        self._hdfs = hdfs_client

        self._ensure_temp_db_exists()

    @property
    def hdfs(self):
        if self._hdfs is None:
            raise com.IbisError('No HDFS connection; must pass connection '
                                'using the hdfs_client argument to '
                                'ibis.make_client')
        return self._hdfs

    @property
    def _table_expr_klass(self):
        return ImpalaTable

    def disable_codegen(self, disabled=True):
        """
        Turn off or on LLVM codegen in Impala query execution

        Parameters
        ----------
        disabled : boolean, default True
          To disable codegen, pass with no argument or True. To enable codegen,
          pass False
        """
        self.con.disable_codegen(disabled)

    def log(self, msg):
        if options.verbose:
            (options.verbose_log or to_stdout)(msg)

    def _fully_qualified_name(self, name, database):
        if ddl._is_fully_qualified(name):
            return name

        database = database or self.current_database
        return '{0}.`{1}`'.format(database, name)

    def _db_type_to_dtype(self, db_type):
        return self._HS2_TTypeId_to_dtype[db_type]

    def list_tables(self, like=None, database=None):
        """
        List tables in the current (or indicated) database. Like the SHOW
        TABLES command in the impala-shell.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'
        database : string, default None
          If not passed, uses the current/default database

        Returns
        -------
        tables : list of strings
        """
        statement = 'SHOW TABLES'
        if database:
            statement += ' IN {0}'.format(database)
        if like:
            statement += " LIKE '{0}'".format(like)

        with self._execute(statement, results=True) as cur:
            result = self._get_list(cur)

        return result

    def _get_list(self, cur, i=0):
        tuples = cur.fetchall()
        if len(tuples) > 0:
            return list(zip(*tuples)[i])
        else:
            return []

    def set_database(self, name):
        """
        Set the default database scope for client
        """
        self.con.set_database(name)

    def exists_database(self, name):
        """
        Checks if a given database exists

        Parameters
        ----------
        name : string
          Database name

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_databases(like=name)) > 0

    def create_database(self, name, path=None, fail_if_exists=True):
        """
        Create a new Impala database

        Parameters
        ----------
        name : string
          Database name
        path : string, default None
          HDFS path where to store the database data; otherwise uses Impala
          default
        """
        if path:
            # explicit mkdir ensures the user own the dir rather than impala,
            # which is easier for manual cleanup, if necessary
            self._hdfs.mkdir(path, create_parent=True)
        statement = ddl.CreateDatabase(name, path=path,
                                       fail_if_exists=fail_if_exists)
        self._execute(statement)

    def drop_database(self, name, force=False):
        """
        Drop an Impala database

        Parameters
        ----------
        name : string
          Database name
        force : boolean, default False
          If False and there are any tables in this database, raises an
          IntegrityError
        """
        if not force or self.exists_database(name):
            tables = self.list_tables(database=name)
        else:
            tables = []
        if force:
            for table in tables:
                self.log('Dropping {0}'.format('{0}.{1}'.format(name, table)))
                self.drop_table_or_view(table, database=name)
        else:
            if len(tables) > 0:
                raise com.IntegrityError('Database {0} must be empty before '
                                         'being dropped, or set '
                                         'force=True'.format(name))
        statement = ddl.DropDatabase(name, must_exist=not force)
        self._execute(statement)

    def list_databases(self, like=None):
        """
        List databases in the Impala cluster. Like the SHOW DATABASES command
        in the impala-shell.

        Parameters
        ----------
        like : string, default None
          e.g. 'foo*' to match all tables starting with 'foo'

        Returns
        -------
        databases : list of strings
        """
        statement = 'SHOW DATABASES'
        if like:
            statement += " LIKE '{0}'".format(like)

        with self._execute(statement, results=True) as cur:
            results = self._get_list(cur)

        return results

    def get_partition_schema(self, table_name, database=None):
        """
        For partitioned tables, return the schema (names and types) for the
        partition columns

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string, default None

        Returns
        -------
        partition_schema : ibis Schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)

        schema = self.get_schema(table_name, database=database)

        name_to_type = dict(zip(schema.names, schema.types))

        query = 'SHOW PARTITIONS {0}'.format(qualified_name)

        partition_fields = []
        with self._execute(query, results=True) as cur:
            result = self._fetch_from_cursor(cur)

            for x in result.columns:
                if x not in name_to_type:
                    break
                partition_fields.append((x, name_to_type[x]))

        pnames, ptypes = zip(*partition_fields)
        return ir.Schema(pnames, ptypes)

    def get_schema(self, table_name, database=None):
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string, default None

        Returns
        -------
        schema : ibis Schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)
        query = 'DESCRIBE {0}'.format(qualified_name)
        tuples = self.con.fetchall(query)

        names, types, comments = zip(*tuples)

        ibis_types = []
        for t in types:
            t = t.lower()
            t = _impala_to_ibis_type_mapping.get(t, t)
            ibis_types.append(t)

        names = [x.lower() for x in names]

        return ir.Schema(names, ibis_types)

    def exists_table(self, name, database=None):
        """
        Determine if the indicated table or view exists

        Parameters
        ----------
        name : string
        database : string, default None

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_tables(like=name)) > 0

    def create_view(self, name, expr, database=None):
        """
        Create an Impala view from a table expression

        Parameters
        ----------
        name : string
        expr : ibis TableExpr
        database : string, default None
        """
        ast = sql.build_ast(expr)
        select = ast.queries[0]
        statement = ddl.CreateView(name, select, database=database)
        self._execute(statement)

    def drop_view(self, name, database=None, force=False):
        """
        Drop an Impala view

        Parameters
        ----------
        name : string
        database : string, default None
        force : boolean, default False
          Database may throw exception if table does not exist
        """
        statement = ddl.DropView(name, database=database,
                                 must_exist=not force)
        self._execute(statement)

    def create_table(self, table_name, expr=None, schema=None, database=None,
                     format='parquet', overwrite=False, external=False,
                     path=None, partition=None, like_parquet=None):
        """
        Create a new table in Impala using an Ibis table expression

        Parameters
        ----------
        table_name : string
        expr : TableExpr, optional
          If passed, creates table from select statement results
        schema : ibis.Schema, optional
          Mutually exclusive with expr, creates an empty table with a
          particular schema
        database : string, default None (optional)
        format : {'parquet'}
        overwrite : boolean, default False
          Do not create table if table with indicated name already exists
        external : boolean, default False
          Create an external table; Impala will not delete the underlying data
          when the table is dropped
        path : string, default None
          Specify the path where Impala reads and writes files for the table
        partition : list of strings
          Must pass a schema to use this. Cannot partition from an expression
          (create-table-as-select)
        like_parquet : string (HDFS path), optional
          Can specify in lieu of a schema

        Examples
        --------
        con.create_table('new_table_name', table_expr)
        """
        if like_parquet is not None:
            raise NotImplementedError

        if expr is not None:
            ast = sql.build_ast(expr)
            select = ast.queries[0]

            if partition is not None:
                # Fairly certain this is currently the case
                raise ValueError('partition not supported with '
                                 'create-table-as-select')

            statement = ddl.CTAS(table_name, select,
                                 database=database,
                                 overwrite=overwrite,
                                 format=format,
                                 external=external,
                                 path=path)
        elif schema is not None:
            statement = ddl.CreateTableWithSchema(
                table_name, schema, ddl.NoFormat(),
                database=database,
                format=format,
                overwrite=overwrite,
                external=external,
                path=path, partition=partition)
        else:
            raise com.IbisError('Must pass expr or schema')

        self._execute(statement)

    def pandas(self, df, name=None, database=None, persist=False):
        """
        Create a (possibly temp) parquet table from a local pandas DataFrame.
        """
        name, database = self._get_concrete_table_path(name, database,
                                                       persist=persist)
        qualified_name = self._fully_qualified_name(name, database)

        # write df to a temp CSV file on HDFS
        temp_csv_hdfs_dir = pjoin(options.impala.temp_hdfs_path, util.guid())
        buf = BytesIO()
        df.to_csv(buf, header=False, index=False, na_rep='\N')
        self.hdfs.put(pjoin(temp_csv_hdfs_dir, '0.csv'), buf)

        # define a temporary table using delimited data
        schema = util.pandas_to_ibis_schema(df)
        table = self.delimited_file(
            temp_csv_hdfs_dir, schema,
            name='ibis_tmp_pandas_{0}'.format(util.guid()), database=database,
            external=True, persist=False)

        # CTAS into Parquet
        self.create_table(name, expr=table, database=database,
                          format='parquet', overwrite=False)

        # cleanup
        self.hdfs.delete(temp_csv_hdfs_dir, recursive=True)

        return self._wrap_new_table(qualified_name, persist)

    def avro_file(self, hdfs_dir, avro_schema,
                  name=None, database=None,
                  external=True, persist=False):
        """
        Create a (possibly temporary) table to read a collection of Avro data.

        Parameters
        ----------
        hdfs_dir : string
          Absolute HDFS path to directory containing avro files
        avro_schema : dict
          The Avro schema for the data as a Python dict
        name : string, default None
        database : string, default None
        external : boolean, default True
        persist : boolean, default False

        Returns
        -------
        avro_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(name, database,
                                                       persist=persist)

        qualified_name = self._fully_qualified_name(name, database)
        stmt = ddl.CreateTableAvro(name, hdfs_dir, avro_schema,
                                   database=database,
                                   external=external)
        self._execute(stmt)
        return self._wrap_new_table(qualified_name, persist)

    def delimited_file(self, hdfs_dir, schema, name=None, database=None,
                       delimiter=',', escapechar=None, lineterminator=None,
                       external=True, persist=False):
        """
        Interpret delimited text files (CSV / TSV / etc.) as an Ibis table. See
        `parquet_file` for more exposition on what happens under the hood.

        Parameters
        ----------
        hdfs_dir : string
          HDFS directory name containing delimited text files
        schema : ibis Schema
        name : string, default None
          Name for temporary or persistent table; otherwise random one
          generated
        database : string
          Database to create the (possibly temporary) table in
        delimiter : length-1 string, default ','
          Pass None if there is no delimiter
        escapechar : length-1 string
          Character used to escape special characters
        lineterminator : length-1 string
          Character used to delimit lines
        external : boolean, default True
          Create table as EXTERNAL (data will not be deleted on drop). Not that
          if persist=False and external=False, whatever data you reference will
          be deleted
        persist : boolean, default False
          If True, do not delete the table upon garbage collection of ibis
          table object

        Returns
        -------
        delimited_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(name, database,
                                                       persist=persist)

        qualified_name = self._fully_qualified_name(name, database)

        stmt = ddl.CreateTableDelimited(name, hdfs_dir, schema,
                                        database=database,
                                        delimiter=delimiter,
                                        external=external,
                                        lineterminator=lineterminator,
                                        escapechar=escapechar)
        self._execute(stmt)
        return self._wrap_new_table(qualified_name, persist)

    def parquet_file(self, hdfs_dir, schema=None, name=None, database=None,
                     external=True, like_file=None, like_table=None,
                     persist=False):
        """
        Make indicated parquet file in HDFS available as an Ibis table.

        The table created can be optionally named and persisted, otherwise a
        unique name will be generated. Temporarily, for any non-persistent
        external table created by Ibis we will attempt to drop it when the
        underlying object is garbage collected (or the Python interpreter shuts
        down normally).

        Parameters
        ----------
        hdfs_dir : string
          Path in HDFS
        schema : ibis Schema
          If no schema provided, and neither of the like_* argument is passed,
          one will be inferred from one of the parquet files in the directory.
        like_file : string
          Absolute path to Parquet file in HDFS to use for schema
          definitions. An alternative to having to supply an explicit schema
        like_table : string
          Fully scoped and escaped string to an Impala table whose schema we
          will use for the newly created table.
        name : string, optional
          random unique name generated otherwise
        database : string, optional
          Database to create the (possibly temporary) table in
        external : boolean, default True
          If a table is external, the referenced data will not be deleted when
          the table is dropped in Impala. Otherwise (external=False) Impala
          takes ownership of the Parquet file.
        persist : boolean, default False
          Do not drop the table upon Ibis garbage collection / interpreter
          shutdown

        Returns
        -------
        parquet_table : ImpalaTable
        """
        name, database = self._get_concrete_table_path(name, database,
                                                       persist=persist)

        # If no schema provided, need to find some absolute path to a file in
        # the HDFS directory
        if like_file is None and like_table is None and schema is None:
            like_file = self.hdfs.find_any_file(hdfs_dir)

        qualified_name = self._fully_qualified_name(name, database)

        stmt = ddl.CreateTableParquet(name, hdfs_dir,
                                      schema=schema,
                                      database=database,
                                      example_file=like_file,
                                      example_table=like_table,
                                      external=external)
        self._execute(stmt)
        return self._wrap_new_table(qualified_name, persist)

    def _get_concrete_table_path(self, name, database, persist=False):
        if not persist:
            if name is None:
                name = util.guid()

            if database is None:
                self._ensure_temp_db_exists()
                database = options.impala.temp_db
            return name, database
        else:
            if name is None:
                raise com.IbisError('Must pass table name if persist=True')
            return name, database

    def _ensure_temp_db_exists(self):
        # TODO: session memoize to avoid unnecessary `SHOW DATABASES` calls
        name, path = options.impala.temp_db, options.impala.temp_hdfs_path
        if not self.exists_database(name):
            self.create_database(name, path=path, fail_if_exists=True)

    def _wrap_new_table(self, qualified_name, persist):
        if persist:
            t = self.table(qualified_name)
        else:
            schema = self._get_table_schema(qualified_name)
            node = ImpalaTemporaryTable(qualified_name, schema, self)
            t = self._table_expr_klass(node)

        # Compute number of rows in table for better default query planning
        cardinality = t.count().execute()
        set_card = ("alter table {0} set tblproperties('numRows'='{1}', "
                    "'STATS_GENERATED_VIA_STATS_TASK' = 'true')"
                    .format(qualified_name, cardinality))
        self._execute(set_card)

        return t

    def text_file(self, hdfs_path, column_name='value'):
        """
        Interpret text data as a table with a single string column.

        Parameters
        ----------

        Returns
        -------
        text_table : TableExpr
        """
        pass

    def insert(self, table_name, expr, database=None, overwrite=False):
        """
        Insert into existing table

        Parameters
        ----------
        table_name : string
        expr : TableExpr
        database : string, default None
        overwrite : boolean, default False
          If True, will replace existing contents of table

        Examples
        --------
        con.insert('my_table', table_expr)

        # Completely overwrite contents
        con.insert('my_table', table_expr, overwrite=True)
        """
        ast = sql.build_ast(expr)
        select = ast.queries[0]
        statement = ddl.InsertSelect(table_name, select,
                                     database=database,
                                     overwrite=overwrite)
        self._execute(statement)

    def drop_table(self, table_name, database=None, force=False):
        """
        Drop an Impala table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        force : boolean, default False
          Database may throw exception if table does not exist

        Examples
        --------
        con.drop_table('my_table', database='operations', force=True)
        """
        statement = ddl.DropTable(table_name, database=database,
                                  must_exist=not force)
        self._execute(statement)

    def truncate_table(self, table_name, database=None):
        """
        Delete all rows from, but do not drop, an existing table

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        """
        statement = ddl.TruncateTable(table_name, database=database)
        self._execute(statement)

    def drop_table_or_view(self, name, database=None, force=False):
        """
        Attempt to drop a relation that may be a view or table
        """
        try:
            self.drop_table(name, database=database)
        except Exception as e:
            try:
                self.drop_view(name, database=database)
            except:
                raise e

    def cache_table(self, table_name, database=None, pool='default'):
        """
        Caches a table in cluster memory in the given pool.

        Parameters
        ----------
        table_name : string
        database : string default None (optional)
        pool : string, default 'default'
           The name of the pool in which to cache the table

        Examples
        --------
        con.cache_table('my_table', database='operations', pool='op_4GB_pool')
        """
        statement = ddl.CacheTable(table_name, database=database, pool=pool)
        self._execute(statement)

    def _get_table_schema(self, tname):
        query = 'SELECT * FROM {0} LIMIT 0'.format(tname)
        return self._get_schema_using_query(query)

    def _get_schema_using_query(self, query):
        with self._execute(query, results=True) as cur:
            # resets the state of the cursor and closes operation
            cur.fetchall()
            names, ibis_types = self._adapt_types(cur.description)

        # per #321; most Impala tables will be lower case already, but Avro
        # data, depending on the version of Impala, might have field names in
        # the metastore cased according to the explicit case in the declared
        # avro schema. This is very annoying, so it's easier to just conform on
        # all lowercase fields from Impala.
        names = [x.lower() for x in names]

        return ir.Schema(names, ibis_types)

    def _adapt_types(self, descr):
        names = []
        adapted_types = []
        for col in descr:
            names.append(col[0])
            impala_typename = col[1]
            typename = _impala_to_ibis_type_mapping[impala_typename.lower()]

            if typename == 'decimal':
                precision, scale = col[4:6]
                adapted_types.append(ir.DecimalType(precision, scale))
            else:
                adapted_types.append(typename)
        return names, adapted_types


_impala_to_ibis_type_mapping = {
    'boolean': 'boolean',
    'tinyint': 'int8',
    'smallint': 'int16',
    'int': 'int32',
    'bigint': 'int64',
    'float': 'float',
    'double': 'double',
    'string': 'string',
    'timestamp': 'timestamp',
    'decimal': 'decimal'
}


def _set_limit(query, k):
    limited_query = '{0}\nLIMIT {1}'.format(query, k)

    return limited_query


def to_stdout(x):
    print(x)


# ----------------------------------------------------------------------
# ORM-ish usability layer


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
        common prefix. For example, for tables foo_a, foo_b, and foo_c,
        creating the "foo_" namespace would enable you to reference those
        objects as a, b, and c, respectively.

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

    def list_udfs(self, like=None):
        return self.client.list_udfs(like=self._qualify_like(like),
                                     database=self.name)

    def list_udas(self, like=None):
        return self.client.list_udas(like=self._qualify_like(like),
                                     database=self.name)

    def _qualify_like(self, like):
        return like


class DatabaseEntity(object):
    pass


class View(DatabaseEntity):

    def drop(self):
        pass


class ScalarFunction(DatabaseEntity):

    def drop(self):
        pass


class AggregateFunction(DatabaseEntity):

    def drop(self):
        pass


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


class ImpalaTable(ir.TableExpr, DatabaseEntity):

    """
    References a physical table in the Impala-Hive metastore
    """

    @property
    def _table_name(self):
        return self.op().args[0]

    @property
    def _client(self):
        return self.op().args[2]

    def compute_stats(self):
        """
        Invoke Impala COMPUTE STATS command to compute column, table, and
        partition statistics. No return value.
        """
        stmt = 'COMPUTE STATS {0}'.format(self._table_name)
        self._client._execute(stmt)

    def drop(self):
        """
        Drop the table from the database
        """
        self._client.drop_table_or_view(self._table_name)


class ImpalaTemporaryTable(ops.DatabaseTable):

    def __del__(self):
        try:
            self.cleanup()
        except com.IbisError:
            pass

    def cleanup(self):
        try:
            self.source.drop_table(self.name)
        except ImpylaError:
            # database might have been dropped
            pass
