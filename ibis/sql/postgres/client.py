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
import psycopg2  # NOQA fail early if the driver is missing
import contextlib
import sqlalchemy as sa

import ibis.sql.alchemy as alch
import ibis.expr.datatypes as dt

from ibis.sql.postgres.compiler import PostgreSQLDialect


@dt.dtype.register(sa.dialects.postgresql.DOUBLE_PRECISION)
def sa_postgres_double(satype, nullable=True):
    return dt.Double(nullable=nullable)


class PostgreSQLTable(alch.AlchemyTable):
    pass


class PostgreSQLSchema(alch.AlchemyDatabaseSchema):
    pass


class PostgreSQLDatabase(alch.AlchemyDatabase):
    schema_class = PostgreSQLSchema


class PostgreSQLClient(alch.AlchemyClient):

    """The Ibis PostgreSQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    dialect = PostgreSQLDialect
    database_class = PostgreSQLDatabase

    def __init__(self, host='localhost', user=None, password=None, port=5432,
                 database='public', url=None, driver='psycopg2'):
        if url is None:
            if driver != 'psycopg2':
                raise NotImplementedError(
                    'psycopg2 is currently the only supported driver'
                )
            user = user or getpass.getuser()
            url = sa.engine.url.URL('postgresql+psycopg2', host=host,
                                    port=port, username=user,
                                    password=password, database=database)
        else:
            url = sa.engine.url.make_url(url)

        super(PostgreSQLClient, self).__init__(sa.create_engine(url))
        self.database_name = url.database

    @contextlib.contextmanager
    def begin(self):
        with super(PostgreSQLClient, self).begin() as bind:
            previous_timezone = bind.execute('SHOW TIMEZONE').scalar()
            bind.execute('SET TIMEZONE = UTC')
            try:
                yield bind
            finally:
                bind.execute("SET TIMEZONE = '{}'".format(previous_timezone))

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return
            the database named ``self.current_database``.

        Returns
        -------
        db : PostgreSQLDatabase
            An :class:`ibis.sql.postgres.client.PostgreSQLDatabase` instance.

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
        """Get a schema object from the current database for the schema named `name`.

        Parameters
        ----------
        name : str

        Returns
        -------
        schema : PostgreSQLSchema
            An :class:`ibis.sql.postgres.client.PostgreSQLSchema` instance.

        """
        return self.database().schema(name)

    @property
    def current_database(self):
        """The name of the current database this client is connected to."""
        return self.database_name

    def list_databases(self):
        # http://dba.stackexchange.com/a/1304/58517
        return [
            row.datname for row in self.con.execute(
                'SELECT datname FROM pg_database WHERE NOT datistemplate'
            )
        ]

    def list_schemas(self):
        """List all the schemas in the current database."""
        return self.inspector.get_schema_names()

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with PostgreSQL client. To use a different'
            ' database, use client.database({!r})'.format(name)
        )

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
            The schema in which the table resides.  If ``None`` then the
            `public` schema is assumed.

        Returns
        -------
        table : TableExpr
            A table expression.
        """
        if database is not None and database != self.current_database:
            return (
                self.database(name=database)
                    .table(name=name, schema=schema)
            )
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = PostgreSQLTable(alch_table, self, self._schemas.get(name))
            return self._table_expr_klass(node)

    def list_tables(self, like=None, database=None, schema=None):
        if database is not None and database != self.current_database:
            return (
                self.database(name=database)
                    .list_tables(like=like, schema=schema)
            )
        else:
            parent = super(PostgreSQLClient, self)
            return parent.list_tables(like=like, schema=schema)
