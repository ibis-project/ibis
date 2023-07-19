"""Trino backend."""

from __future__ import annotations

import contextlib
import warnings
from functools import cached_property
from typing import Iterator

import sqlalchemy as sa
import toolz
from trino.sqlalchemy.datatype import ROW as _ROW

import ibis.expr.datatypes as dt
from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.base.sql.alchemy.datatypes import ArrayType
from ibis.backends.trino.compiler import TrinoSQLCompiler
from ibis.backends.trino.datatypes import ROW, parse


class Backend(BaseAlchemyBackend):
    name = "trino"
    compiler = TrinoSQLCompiler
    supports_create_or_replace = False
    supports_temporary_tables = False

    def current_database(self) -> str:
        raise NotImplementedError(type(self))

    @cached_property
    def version(self) -> str:
        with self.begin() as con:
            return con.execute(sa.select(sa.func.version())).scalar()

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
        connect_args.setdefault("timezone", "UTC")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The dbapi\(\) classmethod on dialect classes has been renamed",
                category=sa.exc.SADeprecationWarning,
            )
            super().do_connect(
                sa.create_engine(
                    url, connect_args=connect_args, poolclass=sa.pool.StaticPool
                )
            )

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if isinstance(typ := column_info["type"], _ROW):
                column_info["type"] = ROW(typ.attr_types)
            elif isinstance(typ, sa.ARRAY):
                column_info["type"] = toolz.nth(
                    typ.dimensions or 1, toolz.iterate(ArrayType, typ.item_type)
                )

        return meta

    @contextlib.contextmanager
    def _prepare_metadata(self, query: str) -> Iterator[dict[str, str]]:
        name = util.gen_name("ibis_trino_metadata")
        with self.begin() as con:
            con.exec_driver_sql(f"PREPARE {name} FROM {query}")
            try:
                yield con.exec_driver_sql(f"DESCRIBE OUTPUT {name}").mappings()
            finally:
                con.exec_driver_sql(f"DEALLOCATE PREPARE {name}")

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self._prepare_metadata(query) as mappings:
            yield from (
                # trino types appear to be always nullable
                (name, parse(trino_type).copy(nullable=True))
                for name, trino_type in toolz.pluck(["Column Name", "Type"], mappings)
            )

    def _execute_view_creation(self, name, definition):
        from sqlalchemy_views import CreateView

        # NB: trino doesn't support temporary views so we use the less
        # desirable method of cleaning up when the Python process exits using
        # an atexit hook
        #
        # the method that defines the atexit hook is defined in the parent
        # class
        view = CreateView(sa.table(name), definition, or_replace=True)

        with self.begin() as con:
            con.execute(view)
