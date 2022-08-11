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

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import sqlalchemy as sa

if TYPE_CHECKING:
    import ibis.expr.types as ir

from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.sqlite import udf
from ibis.backends.sqlite.compiler import SQLiteCompiler


class Backend(BaseAlchemyBackend):
    name = 'sqlite'
    # TODO check if there is a reason to not use the parent AlchemyDatabase, or
    # if there is technical debt that makes this required
    database_class = Database
    compiler = SQLiteCompiler

    def __getstate__(self) -> dict:
        r = super().__getstate__()
        r.update(
            dict(
                compiler=self.compiler,
                database_name=self.database_name,
                _con=None,  # clear connection on copy()
                _meta=None,
            )
        )
        return r

    def do_connect(
        self,
        database: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        """Create an Ibis client connected to a SQLite database.

        Multiple database files can be accessed using the `attach()` method.

        Parameters
        ----------
        database
            File path to the SQLite database file. If `None`, creates an
            in-memory transient database and you can use attach() to add more
            files

        Examples
        --------
        >>> import ibis
        >>> ibis.sqlite.connect("path/to/my/sqlite.db")
        """
        if path is not None:
            warnings.warn(
                "The `path` argument is deprecated in 4.0. Use `database=...`"
            )
            database = path

        self.database_name = "main"

        engine = sa.create_engine(
            f"sqlite:///{database if database is not None else ':memory:'}"
        )

        sqlite3.register_adapter(pd.Timestamp, lambda value: value.isoformat())

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Register UDFs on connection."""
            udf.register_all(dbapi_connection)

        super().do_connect(engine)

        self._meta = sa.MetaData(bind=self.con)

    def attach(
        self,
        name: str,
        path: str | Path,
    ) -> None:
        """Connect another SQLite database file to the current connection.

        Parameters
        ----------
        name
            Database name within SQLite
        path
            Path to sqlite3 database file
        """
        quoted_name = self.con.dialect.identifier_preparer.quote(name)
        self.raw_sql(f"ATTACH DATABASE {path!r} AS {quoted_name}")

    def _get_sqla_table(self, name, schema=None, autoload=True):
        return sa.Table(
            name,
            self.meta,
            schema=schema or self.current_database,
            autoload=autoload,
        )

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Create a table expression from a table in the SQLite database.

        Parameters
        ----------
        name
            Table name
        database
            Name of the attached database that the table is located in.

        Returns
        -------
        Table
            Table expression
        """
        alch_table = self._get_sqla_table(name, schema=database)
        node = self.table_class(source=self, sqla_table=alch_table)
        return self.table_expr_class(node)

    def _table_from_schema(
        self, name, schema, database: str | None = None
    ) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, schema=database, *columns)

    @property
    def _current_schema(self) -> str | None:
        return self.current_database
