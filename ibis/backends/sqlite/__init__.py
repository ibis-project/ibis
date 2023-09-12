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

import inspect
import sqlite3
from typing import TYPE_CHECKING

import sqlalchemy as sa
import toolz
from sqlalchemy.dialects.sqlite import TIMESTAMP

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base import CanListDatabases
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.sqlite import udf
from ibis.backends.sqlite.compiler import SQLiteCompiler
from ibis.backends.sqlite.datatypes import ISODATETIME, SqliteType

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import ibis.expr.operations as ops
    import ibis.expr.types as ir


class Backend(BaseAlchemyBackend, CanListDatabases):
    name = "sqlite"
    compiler = SQLiteCompiler
    supports_create_or_replace = False
    supports_python_udfs = True

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

    @property
    def current_database(self) -> str:
        # AFAICT there is no notion of a schema in SQLite
        return "main"

    def list_databases(self, like: str | None = None) -> list[str]:
        with self.begin() as con:
            mappings = con.exec_driver_sql("PRAGMA database_list").mappings()
            results = list(toolz.pluck("name", mappings))

        return sorted(self._filter_with_like(results, like))

    def do_connect(
        self,
        database: str | Path | None = None,
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

        self.database_name = "main"

        engine = sa.create_engine(
            f"sqlite:///{database if database is not None else ':memory:'}",
            poolclass=sa.pool.StaticPool,
        )

        if type_map:
            # Patch out ischema_names for the instantiated dialect. This
            # attribute is required for all SQLAlchemy dialects, but as no
            # public way of modifying it for a given dialect. Patching seems
            # easier than subclassing the builtin SQLite dialect, and achieves
            # the same desired behavior.
            def _to_ischema_val(t):
                sa_type = SqliteType.from_ibis(dt.dtype(t))
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

    def attach(self, name: str, path: str | Path) -> None:
        """Connect another SQLite database file to the current connection.

        Parameters
        ----------
        name
            Database name within SQLite
        path
            Path to sqlite3 database files

        Examples
        --------
        >>> con1 = ibis.sqlite.connect("original.db")
        >>> con2 = ibis.sqlite.connect("new.db")
        >>> con1.attach("new", "new.db")
        >>> con1.list_tables(database="new")
        """
        with self.begin() as con:
            con.exec_driver_sql(f"ATTACH DATABASE {str(path)!r} AS {self._quote(name)}")

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if type(column_info["type"]) is TIMESTAMP:
                column_info["type"] = ISODATETIME()

        return meta

    def _table_from_schema(
        self, name, schema, database: str | None = None, temp: bool = True
    ) -> sa.Table:
        prefixes = []
        if temp:
            prefixes.append("TEMPORARY")
        columns = self._columns_from_schema(name, schema)
        return sa.Table(
            name, sa.MetaData(), *columns, schema=database, prefixes=prefixes
        )

    @property
    def _current_schema(self) -> str | None:
        return self.current_database

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        view = f"__ibis_sqlite_metadata_{util.guid()}"

        with self.begin() as con:
            if query in self.list_tables():
                query = f"SELECT * FROM {query}"
            # create a view that should only be visible in this transaction
            con.exec_driver_sql(f"CREATE TEMPORARY VIEW {view} AS {query}")

            # extract table info from the view
            table_info = con.exec_driver_sql(f"PRAGMA table_info({view})")

            # get names and not nullables
            names, notnulls, raw_types = zip(
                *toolz.pluck(["name", "notnull", "type"], table_info.mappings())
            )

            # get the type of the first row if no affinity was returned in
            # `raw_types`; assume that reflects the rest of the rows
            type_queries = ", ".join(map("typeof({})".format, names))
            single_row_types = con.exec_driver_sql(
                f"SELECT {type_queries} FROM {view} LIMIT 1"
            ).fetchone()
            for name, notnull, raw_typ, typ in zip(
                names, notnulls, raw_types, single_row_types
            ):
                ibis_type = SqliteType.from_string(raw_typ or typ)
                yield name, ibis_type(nullable=not notnull)

            # drop the view when we're done with it
            con.exec_driver_sql(f"DROP VIEW IF EXISTS {view}")

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a SQLite SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

    def _register_udfs(self, expr: ir.Expr) -> None:
        import ibis.expr.operations as ops

        with self.begin() as con:
            for udf_node in expr.op().find(ops.ScalarUDF):
                compile_func = getattr(
                    self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
                )

                registration_func = compile_func(udf_node)
                if registration_func is not None:
                    registration_func(con)

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        func = udf_node.__func__
        name = func.__name__

        for argname, arg in zip(udf_node.argnames, udf_node.args):
            dtype = arg.dtype
            if not (
                dtype.is_string()
                or dtype.is_binary()
                or dtype.is_numeric()
                or dtype.is_boolean()
            ):
                raise com.IbisTypeError(
                    "SQLite only supports strings, bytes, booleans and numbers as UDF input and output, "
                    f"got argument `{argname}` with unsupported type {dtype}"
                )

        def register_udf(con):
            return con.connection.create_function(
                name, len(inspect.signature(func).parameters), udf.ignore_nulls(func)
            )

        return register_udf

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"DROP VIEW IF EXISTS {name}"
        yield f"CREATE TEMPORARY VIEW {name} AS {definition}"

    def _get_compiled_statement(self, view: sa.Table, definition: sa.sql.Selectable):
        return super()._get_compiled_statement(
            view, definition, compile_kwargs={"literal_binds": True}
        )
