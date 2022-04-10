"""DuckDB backend."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type

if TYPE_CHECKING:
    import duckdb

import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import parse_type


class Backend(BaseAlchemyBackend):
    name = "duckdb"
    compiler = DuckDBSQLCompiler

    def current_database(self) -> str:
        return "main"

    @property
    def version(self) -> str:
        # TODO: there is a `PRAGMA version` we could use instead
        try:
            import importlib.metadata as importlib_metadata
        except ImportError:
            # TODO: remove this when Python 3.9 support is dropped
            import importlib_metadata
        return importlib_metadata.version("duckdb")

    def do_connect(
        self,
        path: str | Path = ":memory:",
        read_only: bool = False,
    ) -> None:
        """Create an Ibis client connected to a DuckDB database.

        Parameters
        ----------
        path
            Path to a duckdb database
        read_only
            Whether the database is read-only
        """
        if path != ":memory:":
            path = Path(path).absolute()
        super().do_connect(
            sa.create_engine(
                f"duckdb:///{path}",
                connect_args={"read_only": read_only},
            )
        )
        self._meta = sa.MetaData(bind=self.con)

    def fetch_from_cursor(
        self,
        cursor: duckdb.DuckDBPyConnection,
        schema: sch.Schema,
    ):
        df = cursor.cursor.fetch_df()
        return schema.apply_to(df)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a SQL string."""
        with self.con.connect() as con:
            rel = con.connection.c.query(query)
        return sch.Schema.from_dict(
            {
                name: parse_type(type)
                for name, type in zip(rel.columns, rel.types)
            }
        )

    def _get_sqla_table(
        self,
        name: str,
        schema: str | None = None,
        **kwargs: Any,
    ) -> sa.Table:

        with warnings.catch_warnings():
            # don't fail or warn if duckdb-engine fails to discover types
            warnings.filterwarnings(
                "ignore",
                message="Did not recognize type",
                category=sa.exc.SAWarning,
            )

            t = super()._get_sqla_table(name, schema, **kwargs)

        nulltype_cols = frozenset(
            col.name
            for col in t.c.values()
            if isinstance(col.type, sa.types.NullType)
        )

        if nulltype_cols:
            query = (
                f"DESCRIBE {self.con.dialect.identifier_preparer.quote(name)}"
            )
            with self.con.connect() as con:
                metadata = con.connection.c.execute(query).fetchall()

            for colname, type, null, *_ in metadata:
                if colname in nulltype_cols:
                    column = sa.Column(
                        colname,
                        to_sqla_type(parse_type(type)),
                        nullable=null == "YES",
                    )
                    # replace null types discovered by sqlite with non null
                    # types
                    t.append_column(column, replace_existing=True)
        return t

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"
