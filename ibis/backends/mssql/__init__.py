"""The Microsoft Sql Server backend."""

from __future__ import annotations

import contextlib
from typing import Literal

import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mssql.compiler import MsSqlCompiler


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
