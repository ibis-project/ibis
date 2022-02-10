"""DuckDB backend."""

from __future__ import annotations

import os

import duckdb
import sqlalchemy as sa

import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

from .compiler import DuckDBSQLCompiler


# TODO: AlchemyTable calls infer on the sqla_table attribute in the super call
# in DuckDBTable -- it wants a sa.Table this is a hack to get it to make it
# through the parent __init__ without barfing
@sch.infer.register(sa.sql.selectable.TableClause)
def schema_passthrough(table, schema=None):
    return schema


class Backend(BaseAlchemyBackend):
    name = "duckdb"
    compiler = DuckDBSQLCompiler

    def current_database(self):
        raise NotImplementedError

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
        path: str = ":memory:",
        read_only: bool = False,
    ) -> None:
        """Create an Ibis client connected to a DuckDB database."""
        if path != ":memory:":
            path = os.path.abspath(path)
        super().do_connect(
            sa.create_engine(
                f"duckdb:///{path}",
                connect_args={"read_only": read_only},
            )
        )
        self._meta = sa.MetaData(bind=self.con)

    def table(self, name: str, database: str | None = None) -> ir.TableExpr:
        """Create a table expression from a table in the SQLite database.

        Parameters
        ----------
        name
            Table name
        database
            Name of the attached database that the table is located in.

        Returns
        -------
        TableExpr
            Table expression
        """
        return self._sqla_table_to_expr(
            self._get_sqla_table(name, schema=database)
        )

    def fetch_from_cursor(
        self, cursor: duckdb.DuckDBPyConnection, schema: sch.Schema
    ):
        df = cursor.cursor.fetch_df()
        df = schema.apply_to(df)
        return df
