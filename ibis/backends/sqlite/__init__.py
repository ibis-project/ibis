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

import datetime
import sqlite3
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import DATETIME, TIMESTAMP

if TYPE_CHECKING:
    import ibis.expr.datatypes as dt
    import ibis.expr.types as ir

import ibis.expr.schema as sch
from ibis.backends.base import Database
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend, to_sqla_type
from ibis.backends.sqlite import udf
from ibis.backends.sqlite.compiler import SQLiteCompiler
from ibis.expr.schema import datatype


def to_datetime(value: str | None) -> datetime.datetime | None:
    """Convert a `str` to a `datetime` according to SQLite's rules.

    This function ignores `None` values.
    """
    if value is None:
        return None
    if value.endswith("Z"):
        # Parse and set the timezone as UTC
        o = datetime.datetime.fromisoformat(value[:-1]).replace(
            tzinfo=datetime.timezone.utc
        )
    else:
        o = datetime.datetime.fromisoformat(value)
        if o.tzinfo:
            # Convert any aware datetime to UTC
            return o.astimezone(datetime.timezone.utc)
    return o


class ISODATETIME(DATETIME):
    """A thin `datetime` type to override sqlalchemy's datetime parsing.

    This is to support a wider range of timestamp formats accepted by SQLite.

    See https://sqlite.org/lang_datefunc.html#time_values for the full
    list of datetime formats SQLite accepts.
    """

    def result_processor(self, value, dialect):
        return to_datetime


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
        type_map: dict[str, str | dt.DataType] | None = None,
    ) -> None:
        """Create an Ibis client connected to a SQLite database.

        Multiple database files can be accessed using the `attach()` method.

        Parameters
        ----------
        database
            File path to the SQLite database file. If `None`, creates an
            in-memory transient database and you can use attach() to add more
            files
        path
            Deprecated, use `database`
        type_map
            An optional mapping from a string name of a SQLite "type" to the
            corresponding ibis DataType that it represents. This can be used
            to override schema inference for a given SQLite database.

        Examples
        --------
        >>> import ibis
        >>> ibis.sqlite.connect("path/to/my/sqlite.db")
        """
        import pandas as pd

        if path is not None:
            warnings.warn(
                "The `path` argument is deprecated in 4.0. Use `database=...`"
            )
            database = path

        self.database_name = "main"

        engine = sa.create_engine(
            f"sqlite:///{database if database is not None else ':memory:'}"
        )

        if type_map:
            # Patch out ischema_names for the instantiated dialect. This
            # attribute is required for all SQLAlchemy dialects, but as no
            # public way of modifying it for a given dialect. Patching seems
            # easier than subclassing the builtin SQLite dialect, and achieves
            # the same desired behavior.
            def _to_ischema_val(t):
                sa_type = to_sqla_type(datatype(t))
                if isinstance(sa_type, sa.types.TypeEngine):
                    # SQLAlchemy expects a callable here, rather than an
                    # instance. Use a lambda to work around this.
                    return lambda: sa_type
                return sa_type

            overrides = {k: _to_ischema_val(v) for k, v in type_map.items()}
            engine.dialect.ischema_names = engine.dialect.ischema_names.copy()
            engine.dialect.ischema_names.update(overrides)

        sqlite3.register_adapter(pd.Timestamp, lambda value: value.isoformat())

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Register UDFs on connection."""
            udf.register_all(dbapi_connection)
            dbapi_connection.execute("PRAGMA case_sensitive_like=ON")

        super().do_connect(engine)

        @sa.event.listens_for(self.meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if type(column_info["type"]) is TIMESTAMP:
                column_info["type"] = ISODATETIME()

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

    def _table_from_schema(self, name, schema, database: str | None = None) -> sa.Table:
        columns = self._columns_from_schema(name, schema)
        return sa.Table(name, self.meta, *columns, schema=database)

    @property
    def _current_schema(self) -> str | None:
        return self.current_database

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        raise ValueError(
            "The SQLite backend cannot infer schemas from raw SQL - "
            "please specify the schema directly when calling `.sql` "
            "using the `schema` keyword argument"
        )
