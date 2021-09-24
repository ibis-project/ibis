import errno
import os
from typing import Optional

import sqlalchemy as sa

from ibis.backends.base import BaseBackendForClient
from ibis.backends.base.sql.alchemy import AlchemyClient

from . import udf
from .compiler import SQLiteCompiler


class SQLiteClient(AlchemyClient, BaseBackendForClient):
    """The Ibis SQLite client class."""

    compiler = SQLiteCompiler

    def __init__(self, backend, path=None, create=False):
        super().__init__(sa.create_engine("sqlite://"))
        self.backend = backend
        self.database_class = backend.database_class
        self.table_class = backend.table_class
        self.name = path
        self.database_name = "base"

        if path is not None:
            self.attach(self.database_name, path, create=create)

        udf.register_all(self.con)

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

    def _table_from_schema(
        self, name, schema, database: Optional[str] = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, schema=database, *columns)
