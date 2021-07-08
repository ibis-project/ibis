import errno
import os
from typing import Optional

import sqlalchemy as sa

from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import AlchemyClient, AlchemyTable

from . import udf
from .compiler import SQLiteCompiler


class SQLiteTable(AlchemyTable):
    pass


class SQLiteDatabase(Database):
    pass


class SQLiteClient(AlchemyClient):
    """The Ibis SQLite client class."""

    compiler = SQLiteCompiler

    def __init__(self, backend, path=None, create=False):
        super().__init__(sa.create_engine("sqlite://"))
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

    def _table_from_schema(
        self, name, schema, database: Optional[str] = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, schema=database, *columns)
