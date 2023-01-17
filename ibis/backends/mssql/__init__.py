"""The Microsoft Sql Server backend."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mssql.compiler import MsSqlCompiler
from ibis.backends.mssql.datatypes import _type_from_result_set_info

if TYPE_CHECKING:
    pass


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
            previous_datefirst = bind.execute(sa.text('SELECT @@DATEFIRST')).scalar()
            bind.execute(sa.text('SET DATEFIRST 1'))
            try:
                yield bind
            finally:
                bind.execute(sa.text(f"SET DATEFIRST {previous_datefirst}"))

    def _metadata(self, query):
        if query in self.list_tables():
            query = f"SELECT * FROM [{query}]"

        with self.begin() as bind:
            for column in bind.execute(
                sa.text(f"EXEC sp_describe_first_result_set @tsql = N'{query}';")
            ).mappings():
                yield column["name"], _type_from_result_set_info(column)

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        yield f"CREATE OR ALTER VIEW {name} AS {definition}"
