"""The Microsoft Sql Server backend."""

from __future__ import annotations

import atexit
import contextlib
from typing import Literal

import sqlalchemy as sa

import ibis.expr.schema as sch
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mssql.compiler import MsSqlCompiler
from ibis.backends.mssql.datatypes import _FieldDescription, _type_from_result_set_info


class Backend(BaseAlchemyBackend):
    name = "mssql"
    compiler = MsSqlCompiler

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 1433,
        database: str | None = None,
        url: str | None = None,
        driver: Literal["pymssql"] = "pymssql",
    ) -> None:
        if driver != "pymssql":
            raise NotImplementedError("pymssql is currently the only supported driver")
        alchemy_url = self._build_alchemy_url(
            url=url,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            driver=f'mssql+{driver}',
        )
        self.database_name = alchemy_url.database
        super().do_connect(sa.create_engine(alchemy_url))

    @contextlib.contextmanager
    def begin(self):
        with super().begin() as bind:
            previous_datefirst = bind.execute('SELECT @@DATEFIRST').scalar()
            bind.execute('SET DATEFIRST 1')
            try:
                yield bind
            finally:
                bind.execute(f"SET DATEFIRST {previous_datefirst}")

    def _get_schema_using_query(self, query):
        with self.begin() as bind:
            result = bind.execute(
                f"EXEC sp_describe_first_result_set @tsql = N'{query}';"
            )
            result_set_info: list[_FieldDescription] = result.mappings().fetchall()
        fields = [
            (column['name'], _type_from_result_set_info(column))
            for column in result_set_info
        ]
        return sch.Schema.from_tuples(fields)

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f"CREATE OR ALTER VIEW {name} AS {definition}"

    def _register_temp_view_cleanup(self, name: str, raw_name: str) -> None:
        query = f"DROP VIEW IF EXISTS {name}"

        def drop(self, raw_name: str, query: str):
            self.con.execute(query)
            self._temp_views.discard(raw_name)

        atexit.register(drop, self, raw_name, query)
