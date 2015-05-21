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


from ibis.common import IbisError
from ibis.config import options

from ibis.filesystems import HDFS

import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.compiler as sql
import ibis.sql.ddl as ddl
import ibis.sql.identifiers as ident


class Connection(object):

    pass


class SQLConnection(Connection):

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

    def _fully_qualified_name(self, name, database):
        # XXX
        return name

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

    def execute(self, expr, params=None, default_limit=None):
        """

        """
        ast, expr = self._build_ast_ensure_limit(expr, default_limit)

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

    def _build_ast_ensure_limit(self, expr, default_limit):
        ast = sql.build_ast(expr)
        if default_limit is not None and isinstance(expr, ir.TableExpr):
            for query in ast.queries:
                if not isinstance(query, ddl.Select):
                    continue

                if query.limit is None:
                    k = options.sql.default_limit
                    expr = expr.limit(k)
                    ast = sql.build_ast(expr)
        return ast, expr

    def _fetch_from_cursor(self, cursor):
        import pandas as pd
        rows = cursor.fetchall()
        names = [x[0] for x in cursor.description]
        return pd.DataFrame.from_records(rows, columns=names)


class ImpalaConnection(SQLConnection):

    def __init__(self, hdfs_client=None, **params):
        self.hdfs_client = hdfs_client

        self.params = params
        self.con = None

        self._connect()

    def _fully_qualified_name(self, name, database):
        if database is not None:
            return '{}.`{}`'.format(database, name)
        else:
            # TODO: This is not foolproof
            if '.' not in name and name.lower() in ident.impala_identifiers:
                return '`{}`'.format(name)
            else:
                return name

    def _connect(self):
        import impala.dbapi as db
        self.con = db.connect(**self.params)

    def _fetchall(self, query, retries=3):
        cursor = self._execute(query, retries=retries)
        return cursor.fetchall()

    def _execute(self, query, retries=3):
        if isinstance(query, ddl.DDLStatement):
            query = query.compile()

        from impala.error import DatabaseError
        try:
            cursor = self.con.cursor()
        except DatabaseError:
            if retries > 0:
                self._connect()
                self._fetchall(query, retries=retries - 1)
            else:
                raise

        cursor.execute(query)
        return cursor

    def set_database(self, name):
        pass

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
            statement += ' IN {}'.format(database)
        if like:
            statement += " LIKE '{}'".format(like)

        cur = self._execute(statement)
        return self._get_list(cur)

    def _get_list(self, cur, i=0):
        return list(zip(*cur.fetchall())[i])

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
            statement += " LIKE '{}'".format(like)

        cur = self._execute(statement)
        return self._get_list(cur)

    def get_schema(self, table_name, database=None):
        pass

    def create_table(self, table_name, expr, database=None, format='parquet',
                     overwrite=False):
        """
        Create a new table in Impala using an Ibis table expression

        Parameters
        ----------
        table_name : string
        expr : TableExpr
        database : string, default None (optional)
        format : {'parquet'}
        overwrite : boolean, default False
          Do not create table if table with indicated name already exists

        Examples
        --------
        con.create_table('new_table_name', table_expr)
        """
        ast = sql.build_ast(expr)
        select = ast.queries[0]
        statement = ddl.CTAS(table_name, select,
                             database=database,
                             overwrite=overwrite)
        self._execute(statement)

    def avro_file(self, hdfs_dir, schema, avro_schema=None,
                  name=None, database=None,
                  external=True, persist=False):
        """
        Create a (possibly temporary) table to read a collection of Avro data.

        Parameters
        ----------
        hdfs_dir
        schema : ibis Schema
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
        if name is None:
            name = self._random_tmp_table()

        stmt = ddl.CreateTableAvro(name, hdfs_dir, schema, avro_schema,
                                   external=external)
        self._execute(stmt)
        qualified_name = self._fully_qualified_name(name, database)
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
        if name is None:
            name = self._random_tmp_table()

        stmt = ddl.CreateTableDelimited(name, hdfs_dir, schema,
                                        delimiter=delimiter,
                                        external=external)
        self._execute(stmt)
        qualified_name = self._fully_qualified_name(name, database)
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
        if name is None:
            name = self._random_tmp_table()

        # If no schema provided, need to find some absolute path to a file in
        # the HDFS directory
        if like_file is None and like_table is None and schema is None:
            like_file = self._find_any_file(hdfs_dir)

        stmt = ddl.CreateTableParquet(name, hdfs_dir, schema=schema,
                                      example_file=like_file,
                                      example_table=like_table,
                                      external=external)
        self._execute(stmt)

        qualified_name = self._fully_qualified_name(name, database)
        return self._wrap_new_table(qualified_name, persist)

    def _wrap_new_table(self, qualified_name, persist):
        if persist:
            return self.table(qualified_name)
        else:
            schema = self._get_table_schema(qualified_name)
            node = ImpalaTemporaryTable(qualified_name, schema, self)
            return ir.TableExpr(node)

    def _find_any_file(self, hdfs_dir):
        contents = self.hdfs_client.ls(hdfs_dir)
        for filename, meta in contents:
            if meta['type'].lower() == 'file':
                return filename
        raise IbisError('No files found in the passed directory')

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

    def _random_tmp_table(self):
        import uuid
        table_name = 'ibis_tmp_' + uuid.uuid4().get_hex()
        return table_name

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

    def drop_table(self, table_name, database=None, must_exist=False):
        """

        Parameters
        ----------
        table_name : string
        database : string, default None (optional)
        must_exist : boolean, default False
          Database may throw exception if table does not exist

        Examples
        --------
        con.drop_table('my_table', database='operations', must_exist=True)
        """
        statement = ddl.DropTable(table_name, database=database,
                                  must_exist=must_exist)
        self._execute(statement)

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
        query = 'SELECT * FROM {} LIMIT 0'.format(tname)
        return self._get_schema_using_query(query)

    def _get_schema_using_query(self, query):
        cursor = self._execute(query)

        # resets the state of the cursor and closes operation
        cursor.fetchall()

        names, ibis_types = self._adapt_types(cursor.description)
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
        except IbisError:
            pass

    def cleanup(self):
        self.source.drop_table(self.name)


def _set_limit(query, k):
    limited_query = '{}\nLIMIT {}'.format(query, k)

    return limited_query


WEBHDFS_DEFAULT_PORT = 50070


def impala_connect(host='localhost', port=21050, protocol='hiveserver2',
                   database=None, timeout=45, use_ssl=False, ca_cert=None,
                   use_ldap=False, ldap_user=None, ldap_password=None,
                   use_kerberos=False, kerberos_service_name='impala',
                   hdfs_config=None, hdfs_client=None):
    """
    Create an Impala connection for use with Ibis

    Parameters
    ----------
    host : host name
    port : int, default 21050 (HiveServer 2)
    protocol : {'hiveserver2', 'beeswax'}
    database :
    timeout :
    use_ssl :
    ca_cert :
    use_ldap : boolean, default False
    ldap_user :
    ldap_password :
    use_kerberos : boolean, default False
    kerberos_service_name : string, default 'impala'
    hdfs_config : dict, with below keys
      host :
      webhdfs_port
      params : dict
        Parameter dict to pass to HDFS constructor
    hdfs_client : HDFS instance (using hdfs library)
      If you created an HDFS client instance elsewhere

    Returns
    -------
    con : ImpalaConnection
    """
    if hdfs_config is not None:
        hdfs_client = HDFS(hdfs_config['host'],
                           hdfs_config.get('webhdfs_port',
                                           WEBHDFS_DEFAULT_PORT),
                           params=hdfs_config.get('params'))
    else:
        hdfs_client = None

    params = {
        'host': host,
        'port': port,
        'protocol': protocol,
        'database': database,
        'timeout': timeout,
        'use_ssl': use_ssl,
        'ca_cert': ca_cert,
        'use_ldap': use_ldap,
        'ldap_user': ldap_user,
        'ldap_password': ldap_password,
        'use_kerberos': use_kerberos,
        'kerberos_service_name': kerberos_service_name
    }
    return ImpalaConnection(hdfs_client=hdfs_client, **params)
