# Copyright 2015 Cloudera Inc.
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

import sqlalchemy as sa

from ibis.client import Database
from .compiler import PostgreSQLDialect
import ibis.expr.types as ir
import ibis.sql.alchemy as alch


class PostgreSQLTable(alch.AlchemyTable):
    pass


class PostgreSQLDatabase(Database):
    pass


class PostgreSQLClient(alch.AlchemyClient):

    """
    The Ibis PostgreSQL client class
    """

    dialect = PostgreSQLDialect
    database_class = PostgreSQLDatabase

    def __init__(self, host=None, user=None, password=None, port=None,
                 database=None, url=None, driver=None):
        if url is None:
            if user is not None:
                if password is None:
                    userpass = user
                else:
                    userpass = '{0}:{1}'.format(user, password)

                address = '{0}@{1}'.format(userpass, host)
            else:
                address = host

            if port is not None:
                address = '{0}:{1}'.format(address, port)

            if database is not None:
                address = '{0}/{1}'.format(address, database)

            if driver is not None and driver != 'psycopg2':
                raise NotImplementedError(driver)

            url = 'postgresql://{0}'.format(address)

        url = sa.engine.url.make_url(url)
        self.name = url.database
        self.database_name = 'public'
        self.con = sa.create_engine(url)
        self.meta = sa.MetaData(bind=self.con, reflect=True)

    @property
    def current_database(self):
        return self.database_name

    def list_databases(self):
        raise NotImplementedError

    def set_database(self):
        raise NotImplementedError

    @property
    def client(self):
        return self

    def table(self, name, database=None):
        """
        Create a table expression that references a particular table in the
        PostgreSQL database

        Parameters
        ----------
        name : string

        Returns
        -------
        table : TableExpr
        """
        alch_table = self._get_sqla_table(name)
        node = PostgreSQLTable(alch_table, self)
        return self._table_expr_klass(node)

    def drop_table(self):
        pass

    def create_table(self, name, expr=None):
        pass

    @property
    def _table_expr_klass(self):
        return ir.TableExpr
