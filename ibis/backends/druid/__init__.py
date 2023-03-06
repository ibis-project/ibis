"""The Druid backend."""

from __future__ import annotations

import contextlib
import json
import warnings
from typing import Any, Iterable

import sqlalchemy as sa
from pydruid.db.sqlalchemy import DruidDialect

import ibis.backends.druid.datatypes as ddt
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.druid.compiler import DruidCompiler


class Backend(BaseAlchemyBackend):
    name = 'druid'
    compiler = DruidCompiler

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

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        if not isinstance(query, str):
            query = str(
                query.compile(
                    dialect=DruidDialect(), compile_kwargs=dict(literal_binds=True)
                )
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Dialect druid:rest will not make use of SQL compilation caching",
                category=sa.exc.SAWarning,
            )
            with self.begin() as con:
                yield con.exec_driver_sql(query, *args, **kwargs)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        query = f"EXPLAIN PLAN FOR {query}"
        with self.begin() as con:
            result = con.exec_driver_sql(query).scalar()

        (plan,) = json.loads(result)
        return sch.Schema(
            {
                column["name"]: (
                    dt.timestamp
                    if column["name"] == "__time"
                    else ddt.parse(column["type"])
                )
                for column in plan["signature"]
            }
        )

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        raise NotImplementedError()

    def _has_table(self, connection, table_name: str, schema) -> bool:
        query = sa.text(
            """\
SELECT COUNT(*) > 0 as c
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME = :table_name"""
        ).bindparams(table_name=table_name)

        return bool(connection.execute(query).scalar())

    def _get_sqla_table(
        self, name: str, schema: str | None = None, autoload: bool = True, **kwargs: Any
    ) -> sa.Table:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="|".join(
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
