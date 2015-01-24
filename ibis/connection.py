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


import ibis.expr.base as ir


class Connection(object):

    pass


class SQLConnection(object):

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
        node = ir.DatabaseTable(name, schema, self)
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

        node = ir.SQLQueryResult(query, schema, self)
        return ir.TableExpr(node)

    def execute(expr, params=None):
        """

        """
        from ibis.sql.compiler import build_ast
        ast = build_ast(expr)

        pass


class ImpalaConnection(SQLConnection):

    def __init__(self, **params):
        self.params = params
        self.con = None
        self._connect()

    def _connect(self):
        pass

    def _fetchall(self, query, retries=3):
        cursor = self._execute(query, retries=retries)
        return cursor.fetchall()

    def _execute(self, query, retries=3):
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

    def _get_table_schema(self, name):
        query = 'SELECT * FROM {} LIMIT 0'.format(name)
        return self._get_schema_from_query(query)

    def _get_schema_using_query(self, query):
        cursor = self._execute(query)
        # Logic cribbed from impyla
        schema = [tup[:2] for tup in cursor.description]

        # resets the state of the cursor and closes operation
        cursor.fetchall()

        names, types = zip(*schema)
        return ir.Schema(names, types)


def _set_limit(query, k):
    pass


def impala_connect(host='localhost', port=21050, protocol='hiveserver2',
                   database=None, timeout=45, use_ssl=False, ca_cert=None,
                   use_ldap=False, ldap_user=None, ldap_password=None,
                   use_kerberos=False, kerberos_service_name='impala'):
    params = {
        host: host,
        port: port,
        protocol: protocol,
        database: database,
        timeout: timeout,
        use_ssl: use_ssl,
        ca_cert: ca_cert,
        use_ldap: use_ldap,
        ldap_user: ldap_user,
        ldap_password: ldap_password,
        use_kerberos: use_kerberos,
        kerberos_service_name: kerberos_service_name
    }
    return ImpalaConnection(params)
