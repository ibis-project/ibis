import getpass
import contextlib

import sqlalchemy as sa

from ibis.sql.mysql.compiler import MySQLDialect
import ibis.sql.alchemy as alch


class MySQLTable(alch.AlchemyTable):
    pass


class MySQLSchema(alch.AlchemyDatabaseSchema):
    pass


class MySQLDatabase(alch.AlchemyDatabase):
    schema_class = MySQLSchema


class MySQLClient(alch.AlchemyClient):

    """The Ibis MySQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    dialect = MySQLDialect
    database_class = MySQLDatabase
    default_database_name = 'mysql'

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
            if driver is not None and driver != 'pymysql':
                raise NotImplementedError(
                    'pymysql is currently the only supported driver'
                )
            url = sa.engine.url.URL(
                'mysql+pymysql',
                username=user or getpass.getuser(),
                password=password,
                host=host or 'localhost',
                port=port,
                database=database or self.__class__.default_database_name,
            )
        else:
            url = sa.engine.url.make_url(url)

        super(MySQLClient, self).__init__(sa.create_engine(url))
        self.name = url.database
        self.database_name = self.__class__.default_database_name

    @contextlib.contextmanager
    def begin(self):
        with super(MySQLClient, self).begin() as bind:
            previous_timezone = (bind.execute('SELECT @@session.time_zone')
                                     .scalar())
            bind.execute("SET @@session.time_zone = 'UTC'")
            try:
                yield bind
            finally:
                bind.execute("SET @@session.time_zone = "
                             "'{}'".format(previous_timezone))

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return
            the database named ``self.current_database``.

        Returns
        -------
        db : MySQLDatabase
            An :class:`ibis.sql.mysql.client.MySQLDatabase` instance.

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
        schema : MySQLSchema
            An :class:`ibis.sql.mysql.client.MySQLSchema` instance.

        """
        return self.database().schema(name)

    @property
    def current_database(self):
        """The name of the current database this client is connected to."""
        return self.database_name

    def list_databases(self):
        return [row.Database for row in self.con.execute('SHOW DATABASES')]

    def list_schemas(self):
        """List all the schemas in the current database."""
        return self.inspector.get_schema_names()

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with MySQL client. To use a different'
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
            return (
                self.database(name=database)
                    .table(name=name, schema=schema)
            )
        else:
            alch_table = self._get_sqla_table(name, schema=schema)
            node = MySQLTable(alch_table, self, self._schemas.get(name))
            return self._table_expr_klass(node)

    def list_tables(self, like=None, database=None, schema=None):
        if database is not None and database != self.current_database:
            return (
                self.database(name=database)
                    .list_tables(like=like, schema=schema)
            )
        else:
            parent = super(MySQLClient, self)
            return parent.list_tables(like=like, schema=schema)
