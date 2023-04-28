"""The Oracle backend."""

from __future__ import annotations

import contextlib
import json
import warnings
from typing import Any, Iterable

import sqlalchemy as sa
from sqlalchemy.dialects import oracle

import ibis.backends.oracle.datatypes as odt
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.oracle.compiler import OracleCompiler


class Backend(BaseAlchemyBackend):
    name = 'oracle'
    compiler = OracleCompiler
    supports_create_or_replace = True

    def do_connect(
        self,
        *,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 1521,
        database: str | None = "FREE",
        **_: Any,
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        user
            Username
        password
            Password
        host
            Hostname
        port
            Port
        database
            Database to connect to
        """
        url = sa.engine.url.make_url(
            f"oracle+oracledb://{user}:{password}@{host}:{port}/{database}"
        )

        # ORACLE IS HORRIBLE
        # SID -- instance identifier -- meant to distinguish oracle instances running on the same machine
        # TABLESPACE -- logical grouping of tables and views, unclear how different from DATABASE
        # DATABASE can be assigned (defaults?) to a tablespace
        #
        # sqlplus ibis/ibis@localhost:1521/IBIS_TESTING
        # for connecting from docker exec

        self.database_name = database  # not sure what should go here

        engine = sa.create_engine(
            url,
            poolclass=sa.pool.StaticPool,
            connect_args={
                "user": user,
                "password": password,
                "service_name": database,
            },
        )

        super().do_connect(engine)

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        query = f"EXPLAIN PLAN FOR {query}"
        with self.begin() as con:
            result = con.exec_driver_sql(query).scalar()

        (plan,) = json.loads(result)
        return sch.Schema(
            {column["name"]: odt.parse(column["type"]) for column in plan["signature"]}
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
        return super()._get_sqla_table(name, schema=schema, autoload=autoload, **kwargs)
