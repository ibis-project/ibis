# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import adbc_driver_manager.dbapi
import sqlalchemy as sa

from ibis.backends.base.sql.alchemy import BaseAlchemyBackend


class Backend(BaseAlchemyBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dialect = None

    @property
    def database_name(self) -> str:
        # TODO(arrow/arrow-adbc#55): retrieve the actual current database
        return "main"

    @property
    def name(self) -> str:
        if self._dialect is None:
            return "adbc"
        return f"adbc_{self._dialect}"

    def do_connect(
        self, *, driver: str, entrypoint: str, dialect=None, db_args=None
    ) -> None:
        """
        Create an Ibis client connected via ADBC.

        Parameters
        ----------
        driver
            The driver name. Example: adbc_driver_sqlite will load
            libadbc_driver_sqlite.so (Unixes) or
            adbc_driver_sqlite.dll (Windows).
        entrypoint
            The driver-specific entrypoint.
        dialect
            The SQLAlchemy dialect name (the scheme of a connection URI).
        db_args
            Driver-specific arguments.

        """
        self._dialect = dialect
        engine = sa.create_engine(
            f"{dialect}://",
            connect_args={
                "driver": driver,
                "entrypoint": entrypoint,
                "db_args": db_args,
            },
        )
        sa.event.listen(engine, "do_connect", self._sqla_connect)

        if dialect == "sqlite":
            from ibis.backends.sqlite.compiler import SQLiteCompiler

            self.compiler = SQLiteCompiler

        super().do_connect(engine)

    def fetch_from_cursor(self, cursor, schema):
        df = cursor.cursor.fetch_df()
        return schema.apply_to(df)

    def _sqla_connect(self, dialect, conn_rec, conn_args, conn_params):
        conn = adbc_driver_manager.dbapi.connect(
            driver=conn_params["driver"],
            entrypoint=conn_params["entrypoint"],
            db_kwargs=conn_params.get("db_args"),
        )
        # XXX: stub out create_function for SQLite backend
        conn.create_function = lambda *args, **kwargs: None
        return conn
