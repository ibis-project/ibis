import errno
import os
from typing import Optional

import sqlalchemy as sa

from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import AlchemyClient, AlchemyTable

from . import udf


class SQLiteTable(AlchemyTable):
    pass


class SQLiteDatabase(Database):
    pass


class SQLiteClient(AlchemyClient):
    """The Ibis SQLite client class."""

    def __init__(self, backend, path=None, create=False):
        super().__init__(sa.create_engine("sqlite://"))
        self.dialect = backend.dialect
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        self.name = path
        self.database_name = "base"

        if path is not None:
            self.attach(self.database_name, path, create=create)

        udf.register_all(self.con)

    @property
    def current_database(self) -> Optional[str]:
        return self.database_name

    def list_databases(self):
        raise NotImplementedError(
            'Listing databases in SQLite is not implemented'
        )

    def set_database(self, name: str) -> None:
        raise NotImplementedError('set_database is not implemented for SQLite')

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

    @property
    def client(self):
        return self

    def _get_sqla_table(self, name, schema=None, autoload=True):
        return sa.Table(
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

    def list_tables(self, like=None, database=None, schema=None):
        if database is None:
            database = self.database_name
        return super().list_tables(like, schema=database)

    def _table_from_schema(
        self, name, schema, database: Optional[str] = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, schema=database, *columns)
