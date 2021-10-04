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
import errno
import os
from typing import Optional

import sqlalchemy

from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from . import udf
from .compiler import SQLiteCompiler


class Backend(BaseAlchemyBackend):
    name = 'sqlite'
    # TODO check if there is a reason to not use the parent AlchemyDatabase, or
    # if there is technical debt that makes this required
    database_class = Database
    compiler = SQLiteCompiler

    def connect(self, path=None, create=False):

        """
        Create an Ibis client connected to a SQLite database.

        Multiple database files can be created using the attach() method

        Parameters
        ----------
        path : string, default None
            File path to the SQLite database file. If None, creates an
            in-memory transient database and you can use attach() to add more
            files
        create : boolean, default False
            If file does not exist, create it
        """
        new_backend = super().connect(sqlalchemy.create_engine("sqlite://"))
        new_backend.database_name = "base"

        if path is not None:
            new_backend.attach(new_backend.database_name, path, create=create)

        udf.register_all(new_backend.con)

        return new_backend

    def list_tables(self, like=None, database=None):
        if database is None:
            database = self.current_database
        return super().list_tables(like, database=database)

    def attach(self, name, path, create: bool = False) -> None:
        """Connect another SQLite database file

        Parameters
        ----------
        name : string
            Database name within SQLite
        path : string
            Path to sqlite3 file
        create : boolean, optional
            If file does not exist, create file if True otherwise raise an
            Exception

        """
        if not os.path.exists(path) and not create:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), path
            )

        quoted_name = self.con.dialect.identifier_preparer.quote(name)
        self.raw_sql(
            "ATTACH DATABASE {path!r} AS {name}".format(
                path=path, name=quoted_name
            )
        )
        self.has_attachment = True

    def _get_sqla_table(self, name, schema=None, autoload=True):
        return sqlalchemy.Table(
            name,
            self.meta,
            schema=schema or self.current_database,
            autoload=autoload,
        )

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
        TableExpr

        """
        alch_table = self._get_sqla_table(name, schema=database)
        node = self.table_class(alch_table, self)
        return self.table_expr_class(node)

    def _table_from_schema(
        self, name, schema, database: Optional[str] = None
    ) -> sqlalchemy.Table:
        columns = self._columns_from_schema(name, schema)
        return sqlalchemy.Table(name, self.meta, schema=database, *columns)
