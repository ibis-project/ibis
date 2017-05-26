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

import os

import sqlalchemy as sa

from ibis.client import Database
from .compiler import SQLiteDialect
import ibis.sql.alchemy as alch
import ibis.common as com


class SQLiteTable(alch.AlchemyTable):
    pass


class SQLiteDatabase(Database):
    pass


class SQLiteClient(alch.AlchemyClient):

    """
    The Ibis SQLite client class
    """

    dialect = SQLiteDialect
    database_class = SQLiteDatabase

    def __init__(self, path=None, create=False):
        super(SQLiteClient, self).__init__(sa.create_engine('sqlite://'))
        self.name = path
        self.database_name = 'default'

        if path:
            self.attach(self.database_name, path, create=create)

    @property
    def current_database(self):
        return self.database_name

    def list_databases(self):
        raise NotImplementedError(
            'Listing databases in SQLite is not implemented'
        )

    def set_database(self, name):
        raise NotImplementedError('set_database is not implemented for SQLite')

    def attach(self, name, path, create=False):
        """
        Connect another SQLite database file

        Parameters
        ----------
        name : string
          Database name within SQLite
        path : string
          Path to sqlite3 file
        create : boolean, default False
          If file does not exist, create file if True otherwise raise Exception
        """
        if not os.path.exists(path) and not create:
            raise com.IbisError('File {0} does not exist'.format(path))

        self.con.execute("ATTACH DATABASE '{0}' AS '{1}'".format(path, name))

    @property
    def client(self):
        return self

    def table(self, name, database=None):
        """
        Create a table expression that references a particular table in the
        SQLite database

        Parameters
        ----------
        name : string
        database : string, optional
          name of the attached database that the table is located in.

        Returns
        -------
        table : TableExpr
        """
        alch_table = self._get_sqla_table(name, schema=database)
        node = SQLiteTable(alch_table, self)
        return self._table_expr_klass(node)

    def list_tables(self, like=None, database=None, schema=None):
        if database is None:
            database = self.database_name
        return super(SQLiteClient, self).list_tables(like, schema=database)
