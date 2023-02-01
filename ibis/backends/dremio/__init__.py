from __future__ import annotations

from functools import cached_property

from ibis.backends.base.sql.adbc import BaseAdbcAlchemyBackend
from ibis.backends.dremio.compiler import DremioCompiler, DremioDialect


class Backend(BaseAdbcAlchemyBackend):
    name = 'dremio'
    compiler = DremioCompiler

    @cached_property
    def dialect(self):
        return DremioDialect()

    def do_connect(
        self, uri: str, *, username: str, password: str, context: str
    ) -> None:
        import adbc_driver_flightsql.dbapi

        super().do_connect(
            adbc_driver_flightsql.dbapi.connect(
                uri, db_kwargs={"username": username, "password": password}
            )
        )
        # Dremio doesn't have catalogs; "contexts" are mapped to
        # ADBC/Flight SQL DB schema.
        self.set_context(catalog=None, db_schema=context)

    def _to_sql_string(self, query) -> str:
        # Dremio does not support bind parameters
        return str(
            query.compile(dialect=self.dialect, compile_kwargs={"literal_binds": True})
        )
