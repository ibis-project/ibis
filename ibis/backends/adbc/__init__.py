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

from typing import Any, Mapping, TYPE_CHECKING

import pyarrow
import sqlalchemy as sa

from ibis import Expr
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend

if TYPE_CHECKING:
    from ibis.expr.typing import TimeContext


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
            self, *, uri: str, dialect: str = "sqlite", db_kwargs=None
    ) -> None:
        """
        Create an Ibis client connected via ADBC.

        Parameters
        ----------
        uri
            The uri of the server to connect to
            Example: grpc+tls://localhost:31337
        dialect
            The SQLAlchemy dialect name (the scheme of a connection URI).
        db_kwargs
            Driver-specific arguments.
        """
        self._dialect = dialect
        engine = sa.create_engine(
            f"{dialect}://",
            connect_args={
                "uri": uri,
                "db_kwargs": db_kwargs,
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

    def execute(self,
                expr: Expr,
                limit: int | str | None = 'default',
                timecontext: TimeContext | None = None,
                params: Mapping[Value, Any] | None = None,
                **kwargs: Any
                ):
        expr_sql = str(expr.compile().compile(compile_kwargs={"literal_binds": True}))
        with self.con.raw_connection().cursor() as cur:
            cur.execute(operation=expr_sql)
            result = cur.fetch_arrow_table().to_pandas()

        return result

    def _sqla_connect(self, dialect, conn_rec, conn_args, conn_params):
        conn = pyarrow.flight_sql.connect(uri=conn_params.get("uri"),
                                          db_kwargs=conn_params.get("db_kwargs")
                                          )
        # XXX: stub out create_function for SQLite backend
        conn.create_function = lambda *args, **kwargs: None

        # Add a notices attribute for the PostgreSQL / DuckDB dialect...
        setattr(conn, "notices", ["n/a"])

        return conn
