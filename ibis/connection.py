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
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.compiler as sql
import ibis.sql.ddl as ddl


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
        if database is not None:
            raise NotImplementedError

        schema = self._get_table_schema(name)
        node = ops.DatabaseTable(name, schema, self)
        return ir.TableExpr(node)

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

    def __init__(self, **params):
        self.params = params
        self.con = None
        self._connect()

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

    def avro_file(self, hdfs_path):
        pass

    def delimited_file(self, hdfs_path, delimiter=',', escapechar=None,
                       lineterminator=None):
        pass

    def parquet_file(self, hdfs_path, name=None, database=None, external=True,
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
        hdfs_path : string
          Path in HDFS
        name : string, optional
          random unique name generated otherwise
        database : string, optional
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

        stmt = ddl.CreateTableParquet(name, hdfs_path, external=external)
        self._execute(stmt)
        print 'Created {}'.format(name)
        return self._wrap_new_table(name, database, persist)

    def _wrap_new_table(self, name, database, persist):
        if persist:
            return self.table(name, database=database)
        else:
            schema = self._get_table_schema(name, database=database)
            node = ImpalaTemporaryTable(name, schema, self)
            return ir.TableExpr(node)

    def text_file(self, hdfs_path):
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

    def _get_table_schema(self, name):
        query = 'SELECT * FROM {} LIMIT 0'.format(name)
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


def impala_connect(host='localhost', port=21050, protocol='hiveserver2',
                   database=None, timeout=45, use_ssl=False, ca_cert=None,
                   use_ldap=False, ldap_user=None, ldap_password=None,
                   use_kerberos=False, kerberos_service_name='impala'):
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
    return ImpalaConnection(**params)
