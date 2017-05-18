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

import getpass

import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

from ibis.client import Database
from ibis.sql.postgres.compiler import PostgreSQLDialect
import ibis.expr.types as ir
import ibis.sql.alchemy as alch


class PostgreSQLTable(alch.AlchemyTable):
    pass


class PostgreSQLDatabase(alch.AlchemyDatabase):

    def schema(self, name):
        return PostgreSQLSchema(name, self)


class PostgreSQLSchema(alch.AlchemyDatabaseSchema):
    pass


class PostgreSQLClient(alch.AlchemyClient):

    """The Ibis PostgreSQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    dialect = PostgreSQLDialect
    database_class = PostgreSQLDatabase
    default_database_name = 'public'
    schema_class = PostgreSQLSchema

    def __init__(
        self,
        host=None,
        user=None,
        password=None,
        port=None,
        database=None,
        url=None,
        driver=None
    ):
        if url is None:
            if driver is not None and driver != 'psycopg2':
                raise NotImplementedError(
                    'psycopg2 is currently the only supported driver'
                )
            url = sa.engine.url.URL(
                'postgresql+psycopg2',
                username=user or getpass.getuser(),
                password=password,
                host=host or 'localhost',
                port=port,
                database=database or self.__class__.default_database_name,
            )
        else:
            url = sa.engine.url.make_url(url)
        self.name = url.database
        self.database_name = self.__class__.default_database_name
        self.con = sa.create_engine(url)
        self.inspector = Inspector.from_engine(self.con)
        self.meta = sa.MetaData(bind=self.con)
        self.schema_name = None

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return the
            database named ``self.current_database``.

        Returns
        -------
        db : Database
            An :class:`ibis.client.Database` instance.

        Notes
        -----
        This creates a new connection if `name` is both not ``None`` and not
        equal to the current database.
        """
        if name == self.current_database or (
            name is None and name != self.current_database
        ):
            return self.database_class(self.current_database, self)
        else:
            url = self.con.url
            client_class = type(self)
            new_client = client_class(
                host=url.host,
                user=url.username,
                port=url.port,
                password=url.password,
                database=name,
            )
            return self.database_class(name, new_client)

    def schema(self, name):
        return self.database().schema(name)

    @property
    def current_database(self):
        """The name of the current database this client is connected to."""
        return self.database_name

    @property
    def current_schema(self):
        return self.current_schema

    def list_databases(self):
        # http://dba.stackexchange.com/a/1304/58517

        return [
            row.datname for row in self.con.execute(
                'SELECT datname FROM pg_database WHERE NOT datistemplate'
            )
        ]

    def list_schemas(self):
        """list databases here means list schemas"""
        return self.inspector.get_schema_names()

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with PostgreSQL client. To use a different'
            ' database, use client.database({!r})'.format(name)
        )

    def set_schema(self, name):
        self.schema_name = name

    @property
    def client(self):
        return self

    def table(self, name, database=None, schema=None):
        """Create a table expression that references a particular a table
        called `name` in a PostgreSQL database called `database`.

        Parameters
        ----------
        name : str
            The name of the table to retrieve.
        database : str, optional
            The database in which the table referred to by `name` resides. If
            ``None`` then the ``current_database`` is used.
        schema : str, optional
            The schema to in which the table referred to resides.  If 
            ``None`` then the ``current_schema`` is used.

        Returns
        -------
        table : TableExpr
            A table expression.
        """
        schema = schema or self.current_schema
        if database is not None and database != self.current_database:
            return self.database(name=database).table(name=name, schema=schema)
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = PostgreSQLTable(alch_table, self)
            return self._table_expr_klass(node)

    def list_tables(self, like=None, database=None, schema=None):
        schema = schema or self.current_schema
        if database is not None and database != self.current_database:
            return self.database(name=database).list_tables(like=like, schema=schema)
        else:
            return super(PostgreSQLClient, self).list_tables(like=like, schema=schema)

    @property
    def _table_expr_klass(self):
        return ir.TableExpr
