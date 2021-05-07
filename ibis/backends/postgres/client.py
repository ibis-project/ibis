import contextlib
import getpass
from typing import Optional

import psycopg2  # NOQA fail early if the driver is missing
import sqlalchemy as sa

import ibis.backends.base_sqlalchemy.alchemy as alch
from ibis.backends.postgres import udf
from ibis.backends.postgres.compiler import PostgreSQLDialect


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
    table_class = PostgreSQLTable

    def __init__(
        self,
        host: str = 'localhost',
        user: str = getpass.getuser(),
        password: Optional[str] = None,
        port: int = 5432,
        database: str = 'public',
        url: Optional[str] = None,
        driver: str = 'psycopg2',
    ):
        if url is None:
            if driver != 'psycopg2':
                raise NotImplementedError(
                    'psycopg2 is currently the only supported driver'
                )
            sa_url = sa.engine.url.URL(
                'postgresql+psycopg2',
                host=host,
                port=port,
                username=user,
                password=password,
                database=database,
            )
        else:
            sa_url = sa.engine.url.make_url(url)

        super().__init__(sa.create_engine(sa_url))
        self.database_name = sa_url.database

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
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
            row.datname
            for row in self.con.execute(
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
            parent = super(PostgreSQLClient, self)
            return parent.list_tables(like=like, schema=schema)

    def udf(
        self, pyfunc, in_types, out_type, schema=None, replace=False, name=None
    ):
        """Decorator that defines a PL/Python UDF in-database based on the
        wrapped function and turns it into an ibis function expression.

        Parameters
        ----------
        pyfunc : function
        in_types : List[ibis.expr.datatypes.DataType]
        out_type : ibis.expr.datatypes.DataType
        schema : str
            optionally specify the schema in which to define the UDF
        replace : bool
            replace UDF in database if already exists
        name: str
            name for the UDF to be defined in database

        Returns
        -------
        Callable

        Function that takes in ColumnExpr arguments and returns an instance
        inheriting from PostgresUDFNode
        """

        return udf(
            client=self,
            python_func=pyfunc,
            in_types=in_types,
            out_type=out_type,
            schema=schema,
            replace=replace,
            name=name,
        )
