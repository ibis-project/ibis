"""DuckDB backend."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple

import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type

if TYPE_CHECKING:
    import duckdb
    import ibis.expr.types as ir

import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import parse
from ibis.common.dispatch import RegexDispatcher


class _ColumnMetadata(NamedTuple):
    name: str
    type: dt.DataType


_generate_view_code = RegexDispatcher("_register")
_dialect = sa.dialects.postgresql.dialect()


def _name_from_path(path: Path) -> str:
    base, *_ = path.name.partition(os.extsep)
    return base.replace("-", "_")


def _quote(name: str):
    return _dialect.identifier_preparer.quote(name)


@_generate_view_code.register(r"parquet://(?P<path>.+)", priority=10)
def _parquet(_, path, table_name=None):
    path = Path(path).absolute()
    table_name = table_name or _name_from_path(path)
    quoted_table_name = _quote(table_name)
    return (
        f"CREATE VIEW {quoted_table_name} as SELECT * from read_parquet('{path}')",  # noqa: E501
        table_name,
    )


@_generate_view_code.register(r"csv(?:\.gz)?://(?P<path>.+)", priority=10)
def _csv(_, path, table_name=None):
    path = Path(path).absolute()
    table_name = table_name or _name_from_path(path)
    quoted_table_name = _quote(table_name)
    return (
        f"CREATE VIEW {quoted_table_name} as SELECT * from read_csv_auto('{path}')",  # noqa: E501
        table_name,
    )


@_generate_view_code.register(r"(?:file://)?(?P<path>.+)", priority=9)
def _file(_, path, table_name=None):
    num_sep_chars = len(os.extsep)
    extension = "".join(Path(path).suffixes)[num_sep_chars:]
    return _generate_view_code(f"{extension}://{path}", table_name=table_name)


@_generate_view_code.register(r".+", priority=1)
def _default(_, **kwargs):
    raise ValueError(
        """
Unrecognized filetype or extension.
Valid prefixes are parquet://, csv://, or file://

Supported filetypes are parquet, csv, and csv.gz
    """
    )


class Backend(BaseAlchemyBackend):
    name = "duckdb"
    compiler = DuckDBSQLCompiler

    def current_database(self) -> str:
        return "main"

    @property
    def version(self) -> str:
        # TODO: there is a `PRAGMA version` we could use instead
        import importlib.metadata

        return importlib.metadata.version("duckdb")

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

    def register(
        self,
        path: str | Path,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register an external file as a table in the current connection
        database

        Parameters
        ----------
        path
            Name of the parquet or CSV file
        table_name
            Name for the created table.  Defaults to filename if not given.
            Any dashes in a user-provided or generated name will be
            replaced with underscores.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        view, table_name = _generate_view_code(path, table_name=table_name)
        self.con.execute(view)
        return self.table(table_name)

    def fetch_from_cursor(
        self,
        cursor: duckdb.DuckDBPyConnection,
        schema: sch.Schema,
    ):
        df = cursor.cursor.fetch_df()
        return schema.apply_to(df)

    def _metadata(self, query: str) -> Iterator[_ColumnMetadata]:
        for name, type, null in toolz.pluck(
            ["column_name", "column_type", "null"],
            self.con.execute(f"DESCRIBE {query}"),
        ):
            yield _ColumnMetadata(
                name=name,
                type=parse(type)(nullable=null.lower() == "yes"),
            )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a DuckDB SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

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

            table = super()._get_sqla_table(name, schema, **kwargs)

        nulltype_cols = frozenset(
            col.name
            for col in table.c
            if isinstance(col.type, sa.types.NullType)
        )

        if not nulltype_cols:
            return table

        quoted_name = self.con.dialect.identifier_preparer.quote(name)

        for colname, type in self._metadata(quoted_name):
            if colname in nulltype_cols:
                # replace null types discovered by sqlalchemy with non null
                # types
                table.append_column(
                    sa.Column(
                        colname,
                        to_sqla_type(type),
                        nullable=type.nullable,
                    ),
                    replace_existing=True,
                )
        return table

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"
