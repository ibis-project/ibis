"""Trino backend."""

from __future__ import annotations

from typing import Iterator

import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.trino.compiler import TrinoSQLCompiler
from ibis.backends.trino.datatypes import parse


class Backend(BaseAlchemyBackend):
    name = "trino"
    compiler = TrinoSQLCompiler

    def current_database(self) -> str:
        raise NotImplementedError(type(self))

    @property
    def version(self) -> str:
        # TODO: there is a `PRAGMA version` we could use instead
        import importlib.metadata

        return importlib.metadata.version("trino")

    def do_connect(
        self,
        user: str = "user",
        password: str | None = None,
        host: str = "localhost",
        port: int = 8080,
        database: str | None = None,
        schema: str | None = None,
        **connect_args,
    ) -> None:
        """Create an Ibis client connected to a Trino database."""
        database = "/".join(filter(None, (database, schema)))
        url = sa.engine.URL.create(
            drivername="trino",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        connect_args.setdefault("experimental_python_types", True)
        super().do_connect(sa.create_engine(url, connect_args=connect_args))
        self._meta = sa.MetaData(bind=self.con, schema=schema)

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        tmpname = f"_ibis_trino_output_{util.guid()[:6]}"
        with self.con.begin() as con:
            con.execute(f"PREPARE {tmpname} FROM {query}")
            rows = list(con.execute(f"DESCRIBE OUTPUT {tmpname}"))
            for name, type in toolz.pluck(["Column Name", "Type"], rows):
                ibis_type = parse(type)
                yield name, ibis_type(nullable=True)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a DuckDB SQL string."""
        pairs = self._metadata(query)
        return sch.Schema.from_tuples(pairs)
