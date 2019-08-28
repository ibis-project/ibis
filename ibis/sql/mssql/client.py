import contextlib
import getpass
import warnings

import pyodbc # NOQA fail early if the driver is missing
import sqlalchemy as sa
from sqlalchemy.dialects.mssql.pyodbc import MSDialect_pyodbc

import ibis.expr.datatypes as dt
import ibis.sql.alchemy as alch
from ibis.sql.mssql.compiler import MSSQLDialect


@dt.dtype.register(MSDialect_pyodbc, sa.dialects.mssql.BIT)
def sa_boolean(_, satype, nullable=True):
    return dt.Boolean(nullable=nullable)


class MSSQLTable(alch.AlchemyTable):
    pass


class MSSQLSchema(alch.AlchemyDatabaseSchema):
    pass


class MSSQLDatabase(alch.AlchemyDatabase):
    schema_class = MSSQLSchema


class MSSQLClient(alch.AlchemyClient):

    """The Ibis MSSQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    dialect = MSSQLDialect
    database_class = MSSQLDatabase
    table_class = MSSQLTable

    def __init__(
            self,
            host='localhost',
            user=None,
            password=None,
            port=1433,
            database='mssql',
            url=None,
            driver='pyodbc'
    ):
        if url is None:
            if driver != 'pyodbc':
                raise NotImplementedError(
                    'pyodbc is currently the only supported driver'
                )
            user = user or getpass.getuser()
            url = sa.engine.url.URL(
                'mssql+pyodbc',
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
            )
        else:
            url = sa.engine.url.make_url(url)
        super().__init__(sa.create_engine(url))
        self.database_name = url.database

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            # set timezone utc
            yield bind
            # set timezone previous timezone

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return
            the database named ``self.current_database``.

        Returns
        -------
        db : MSSQLDatabase
            An :class:`ibis.sql.mssql.client.MSSQLDatabase` instance.

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
        schema : MSSQLSchema
            An :class:`ibis.sql.mssql.client.MSSQLSchema` instance.

        """
        return self.database().schema(name)

    @property
    def current_database(self):
        """The name of the current database this client is connected to."""
        return self.database_name

    def list_databases(self):
        return [row.name for row in self.con.execute('SELECT name FROM master.dbo.sysdatabases')]

    def list_schemas(self):
        """List all the schemas in the current database."""
        return self.inspector.get_schema_names()

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with MSSQL client. To use a different'
            ' database, use client.database({!r})'.format(name)
        )

    @property
    def client(self):
        return self

    def table(self, name, database=None, schema=None):
        """Create a table expression that references a particular a table
        called `name` in a MySQL database called `database`.

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
            return self.database(name=database).table(name=name, schema=schema)
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = self.table_class(alch_table, self, self._schemas.get(name))
            return self.table_expr_class(node)

    def list_tables(self, like=None, database=None, schema=None):
        if database is not None and database != self.current_database:
            return self.database(name=database).list_tables(
                like=like, schema=schema
            )
        else:
            parent = super(MSSQLClient, self)
            return parent.list_tables(like=like, schema=schema)
