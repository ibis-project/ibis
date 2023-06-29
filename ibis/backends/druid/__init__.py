"""The Druid backend."""

from __future__ import annotations

import contextlib
import json
import warnings
from typing import Any, Iterable

import sqlalchemy as sa

import ibis.backends.druid.datatypes as ddt
import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.druid.compiler import DruidCompiler


class Backend(BaseAlchemyBackend):
    name = 'druid'
    compiler = DruidCompiler
    supports_create_or_replace = False

    def do_connect(
        self,
        host: str = "localhost",
        port: int = 8082,
        database: str | None = "druid/v2/sql",
        **_: Any,
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        host
            Hostname
        port
            Port
        database
            Database to connect to
        """
        url = sa.engine.url.make_url(f"druid://{host}:{port}/{database}?header=true")

        self.database_name = "default"  # not sure what should go here

        engine = sa.create_engine(url, poolclass=sa.pool.StaticPool)

        super().do_connect(engine)

        # workaround a broken pydruid `has_table` implementation
        engine.dialect.has_table = self._has_table

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if isinstance(typ := column_info["type"], sa.DateTime):
                column_info["type"] = ddt.DruidDateTime()
            elif isinstance(typ, (sa.LargeBinary, sa.BINARY, sa.VARBINARY)):
                column_info["type"] = ddt.DruidBinary()
            elif isinstance(typ, sa.String):
                column_info["type"] = ddt.DruidString()

        return meta

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        query = query.compile(
            dialect=self.con.dialect, compile_kwargs=dict(literal_binds=True)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Dialect druid:rest will not make use of SQL compilation caching",
                category=sa.exc.SAWarning,
            )
            with self.begin() as con:
                yield con.execute(query, *args, **kwargs)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        query = f"EXPLAIN PLAN FOR {query}"
        with self.begin() as con:
            result = con.exec_driver_sql(query).scalar()

        (plan,) = json.loads(result)
        for column in plan["signature"]:
            name, typ = column["name"], column["type"]
            if name == "__time":
                dtype = dt.timestamp
            else:
                dtype = ddt.parse(typ)
            yield name, dtype

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        raise NotImplementedError()

    def _has_table(self, connection, table_name: str, schema) -> bool:
        t = sa.table(
            "TABLES", sa.column("TABLE_NAME", sa.TEXT), schema="INFORMATION_SCHEMA"
        )
        query = sa.select(
            sa.func.sum(sa.cast(t.c.TABLE_NAME == table_name, sa.INTEGER))
        ).compile(dialect=self.con.dialect)

        return bool(connection.execute(query).scalar())

    def _get_sqla_table(
        self, name: str, schema: str | None = None, autoload: bool = True, **kwargs: Any
    ) -> sa.Table:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="|".join(  # noqa: FLY002
                    (
                        "Did not recognize type",
                        "Dialect druid:rest will not make use of SQL compilation caching",
                    )
                ),
                category=sa.exc.SAWarning,
            )
            return super()._get_sqla_table(
                name, schema=schema, autoload=autoload, **kwargs
            )
