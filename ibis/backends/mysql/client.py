import contextlib
import getpass
import warnings

import pymysql  # NOQA fail early if the driver is missing
import sqlalchemy as sa
import sqlalchemy.dialects.mysql as mysql

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import (
    AlchemyClient,
    AlchemyDatabase,
    AlchemyDatabaseSchema,
    AlchemyTable,
)

from .compiler import MySQLCompiler

# TODO(kszucs): unsigned integers


@dt.dtype.register((mysql.DOUBLE, mysql.REAL))
def mysql_double(satype, nullable=True):
    return dt.Double(nullable=nullable)


@dt.dtype.register(mysql.FLOAT)
def mysql_float(satype, nullable=True):
    return dt.Float(nullable=nullable)


@dt.dtype.register(mysql.TINYINT)
def mysql_tinyint(satype, nullable=True):
    return dt.Int8(nullable=nullable)


@dt.dtype.register(mysql.BLOB)
def mysql_blob(satype, nullable=True):
    return dt.Binary(nullable=nullable)


class MySQLTable(AlchemyTable):
    pass


class MySQLSchema(AlchemyDatabaseSchema):
    pass


class MySQLDatabase(AlchemyDatabase):
    schema_class = MySQLSchema


class MySQLClient(AlchemyClient):

    """The Ibis MySQL client class

    Attributes
    ----------
    con : sqlalchemy.engine.Engine
    """

    compiler = MySQLCompiler

    def __init__(
        self,
        backend,
        host='localhost',
        user=None,
        password=None,
        port=3306,
        database='mysql',
        url=None,
        driver='pymysql',
    ):
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        if url is None:
            if driver != 'pymysql':
                raise NotImplementedError(
                    'pymysql is currently the only supported driver'
                )
            user = user or getpass.getuser()
            url = sa.engine.url.URL(
                'mysql+pymysql',
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
            previous_timezone = bind.execute(
                'SELECT @@session.time_zone'
            ).scalar()
            try:
                bind.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn("Couldn't set mysql timezone: {}".format(str(e)))

            try:
                yield bind
            finally:
                query = "SET @@session.time_zone = '{}'"
                bind.execute(query.format(previous_timezone))

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
            parent = super(MySQLClient, self)
            return parent.list_tables(like=like, schema=schema)
