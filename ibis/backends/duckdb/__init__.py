"""DuckDB backend."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple

import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type

if TYPE_CHECKING:
    import duckdb

import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import parse


class _ColumnMetadata(NamedTuple):
    name: str
    type: dt.DataType


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

    def register(
        self,
        file_name: str | Path,
        table_name: str | None = None,
    ) -> None:
        """Register an external file (csv or parquet) as a table in the current
        connection database

        Parameters
        ----------
        file_name
            Name of the parquet or CSV file
        table_name
            Name for the created table.  Defaults to filename if not given
        """
        file_name = Path(file_name)
        suffix = "".join(file_name.suffixes).strip(".")  # handles .csv.gz
        if file_name.parts[0].endswith(":"):
            prefix, *fname = file_name.parts
        else:
            prefix = "file:"
            fname = file_name.parts

        file_name = Path(*fname).absolute()

        # Use prefix for file_type.  If omitted, infer from file extension
        file_type = prefix.strip(":") if prefix != "file:" else suffix
        table_name = table_name or file_name.stem.replace("-", "_")
        if file_type == "parquet":
            view = f"""
            CREATE VIEW {table_name} as SELECT * from
            read_parquet('{file_name}')
            """
        elif file_type.startswith("csv"):
            view = f"""
            CREATE VIEW {table_name} as SELECT * from
            read_csv_auto('{file_name}')
            """
        else:
            raise TypeError(
                "Only csv and parquet files can be registered with DuckDB."
            )

        self.con.execute(view)

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
