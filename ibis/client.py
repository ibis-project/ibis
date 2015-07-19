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
        return ir.TableExpr(node)

    @property
    def current_database(self):
        return self.con.database

    def _fully_qualified_name(self, name, database):
        # XXX
        return name

    def _execute(self, query):
        return self.con.execute(query)

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
        limited_query = _set_limit(query, 0)
        schema = self._get_schema_using_query(limited_query)

        node = ops.SQLQueryResult(query, schema, self)
        return ir.TableExpr(node)

    def execute(self, expr, params=None, limit=None):
        """

        """
        ast, expr = self._build_ast_ensure_limit(expr, limit)

        # TODO: create some query pipeline executor abstraction
        output = None
        for query in ast.queries:
            sql_string = query.compile()

            cursor = self._execute(sql_string)
            result = self._fetch_from_cursor(cursor)
            if isinstance(query, ddl.Select):
                if query.result_handler is not None:
                    result = query.result_handler(result)

                output = result

        return output

    def _build_ast_ensure_limit(self, expr, limit):
        ast = sql.build_ast(expr)
        if not limit:
            limit = options.sql.default_limit
        # note: limit can still be None at this point, if the global
        # default_limit is None
        if limit is not None and isinstance(expr, ir.TableExpr):
            for query in reversed(ast.queries):
                if isinstance(query, ddl.Select):
                    if query.limit is None:
                        expr = expr.limit(limit)
                        ast = sql.build_ast(expr)
                        break
        return ast, expr

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

    def __init__(self, database='default', **params):
        self.params = params
        self.con = None
        self.cursor = None
        self.codegen_disabled = False
        self.database = database
        self.ensure_connected()

    def __del__(self):
        if self.cursor is not None:
            self.cursor.close()

    def set_database(self, name):
        self.database = name
        self.connect()

    def disable_codegen(self, disabled=True):
        query = 'SET disable_codegen={0}'.format(
            'true' if disabled else 'false')
        self.codegen_disabled = disabled
        self.cursor.execute(query)

    def execute(self, query, retries=3):
        if isinstance(query, ddl.DDLStatement):
            query = query.compile()

        from impala.error import DatabaseError
        try:
            cursor = self.cursor
        except DatabaseError:
            if retries > 0:
                self.ensure_connected()
                self.fetchall(query, retries=retries - 1)
            else:
                raise

        self.log(query)

        try:
            cursor.execute(query)
        except:
            try:
                self.error('Exception caused by {0}'.format(query))
            except:
                pass
            raise

        return cursor

    def log(self, msg):
        if options.verbose:
            (options.verbose_log or to_stdout)(msg)

    def error(self, msg):
        self.log(msg)

    def fetchall(self, query, retries=3):
        cursor = self.execute(query, retries=retries)
        return cursor.fetchall()

    def ensure_connected(self):
        if self.con is None or not self.cursor.ping():
            self.connect()

    def connect(self):
        params = self.params.copy()

        self.con = impyla_dbapi.connect(database=self.database, **params)
        self.cursor = self.con.cursor()
        self.cursor.ping()

        if self.codegen_disabled:
            self.disable_codegen(True)


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

        self.con.ensure_connected()
        self._ensure_temp_db_exists()

    @property
    def hdfs(self):
        if self._hdfs is None:
            raise com.IbisError('No HDFS connection; must pass connection '
                                'using the hdfs_client argument to '
                                'ibis.make_client')
        return self._hdfs

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

        cur = self._execute(statement)
        return self._get_list(cur)

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
        tables = self.list_tables(database=name)
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

        cur = self._execute(statement)
        return self._get_list(cur)

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
            t = _impala_type_mapping.get(t, t)
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
                     path=None):
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

        Examples
        --------
        con.create_table('new_table_name', table_expr)
        """
        if expr is not None:
            ast = sql.build_ast(expr)
            select = ast.queries[0]
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
                path=path)
        else:
            raise com.IbisError('Must pass expr or schema')

        self._execute(statement)

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
        avro_table : TableExpr
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
        delimited_table : TableExpr
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
                     external=True, like_file=None,
                     like_table=None,
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
        parquet_table : TableExpr
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
            return self.table(qualified_name)
        else:
            schema = self._get_table_schema(qualified_name)
            node = ImpalaTemporaryTable(qualified_name, schema, self)
            return ir.TableExpr(node)

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
        cursor = self._execute(query)

        # resets the state of the cursor and closes operation
        cursor.fetchall()

        names, ibis_types = self._adapt_types(cursor.description)

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
            typename = _impala_type_mapping[impala_typename.lower()]

            if typename == 'decimal':
                precision, scale = col[4:6]
                adapted_types.append(ir.DecimalType(precision, scale))
            else:
                adapted_types.append(typename)
        return names, adapted_types


_impala_type_mapping = {
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


def _set_limit(query, k):
    limited_query = '{0}\nLIMIT {1}'.format(query, k)

    return limited_query


def to_stdout(x):
    print(x)
