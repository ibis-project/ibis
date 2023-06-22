"""The Microsoft Sql Server backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import sqlalchemy as sa
import toolz

from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.mssql.compiler import MsSqlCompiler
from ibis.backends.mssql.datatypes import _type_from_result_set_info

if TYPE_CHECKING:
    import ibis.expr.schema as sch
    import ibis.expr.types as ir


class Backend(BaseAlchemyBackend):
    name = "mssql"
    compiler = MsSqlCompiler
    supports_create_or_replace = False

    _sqlglot_dialect = "tsql"

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

        engine = sa.create_engine(alchemy_url, poolclass=sa.pool.StaticPool)

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            with dbapi_connection.cursor() as cur:
                cur.execute("SET DATEFIRST 1")

        return super().do_connect(engine)

    def _metadata(self, query):
        if query in self.list_tables():
            query = f"SELECT * FROM [{query}]"

        query = sa.text("EXEC sp_describe_first_result_set @tsql = :query").bindparams(
            query=query
        )
        with self.begin() as bind:
            for column in bind.execute(query).mappings():
                yield column["name"], _type_from_result_set_info(column)

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR ALTER VIEW {name} AS {definition}"

    def _table_from_schema(
        self,
        name: str,
        schema: sch.Schema,
        database: str | None = None,
        temp: bool = False,
    ) -> sa.Table:
        return super()._table_from_schema(
            temp * "#" + name, schema=schema, database=database, temp=False
        )

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        with self._safe_raw_sql(sql) as cursor:
            # this is expensive for large result sets
            #
            # see https://github.com/ibis-project/ibis/pull/6513
            batch = cursor.fetchall()

        yield from toolz.partition_all(chunk_size, batch)
