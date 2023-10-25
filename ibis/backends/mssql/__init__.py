"""The Microsoft Sql Server backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import sqlalchemy as sa
import sqlglot as sg
import toolz

from ibis.backends.base import CanCreateDatabase
from ibis.backends.base.sql.alchemy import AlchemyCanCreateSchema, BaseAlchemyBackend
from ibis.backends.mssql.compiler import MsSqlCompiler
from ibis.backends.mssql.datatypes import _type_from_result_set_info

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import ibis.expr.schema as sch
    import ibis.expr.types as ir


class Backend(BaseAlchemyBackend, CanCreateDatabase, AlchemyCanCreateSchema):
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
            driver=f"mssql+{driver}",
        )

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

    @property
    def current_database(self) -> str:
        return self._scalar_query(sa.select(sa.func.db_name()))

    def list_databases(self, like: str | None = None) -> list[str]:
        s = sa.table("databases", sa.column("name", sa.VARCHAR()), schema="sys")
        query = sa.select(s.c.name)

        with self.begin() as con:
            results = list(con.execute(query).scalars())
        return self._filter_with_like(results, like=like)

    @property
    def current_schema(self) -> str:
        return self._scalar_query(sa.select(sa.func.schema_name()))

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> list[str]:
        tablequery = sg.select("name").from_(
            sg.table("tables", db="sys", catalog=database)
        )
        viewquery = sg.select("name").from_(
            sg.table("views", db="sys", catalog=database)
        )

        if schema is not None:
            table_predicate = sg.func(
                "schema_name",
                sg.column("schema_id", table="tables", db="sys", catalog=database),
            ).eq(schema)
            view_predicate = sg.func(
                "schema_name",
                sg.column("schema_id", table="views", db="sys", catalog=database),
            ).eq(schema)
            tablequery = tablequery.where(table_predicate)
            viewquery = viewquery.where(view_predicate)

        tablequery = sa.text(tablequery.sql(dialect="tsql"))
        viewquery = sa.text(viewquery.sql(dialect="tsql"))

        with self.begin() as con:
            tablequery = list(con.execute(tablequery).scalars())
            viewresults = list(con.execute(viewquery).scalars())
        results = tablequery + viewresults

        return self._filter_with_like(results, like)

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

    def create_database(self, name: str, force: bool = False) -> None:
        name = self._quote(name)
        create_stmt = (
            f"""\
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = {name})
BEGIN
  CREATE DATABASE {name};
END;
GO"""
            if force
            else f"CREATE DATABASE {name}"
        )
        with self.con.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
            con.exec_driver_sql(create_stmt)

    def drop_database(self, name: str, force: bool = False) -> None:
        name = self._quote(name)
        if_exists = "IF EXISTS " * force

        with self.con.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
            con.exec_driver_sql(f"DROP DATABASE {if_exists}{name}")

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        current_database = self.current_database
        should_switch_database = database is not None and database != current_database

        name = self._quote(name)

        create_stmt = (
            f"""\
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = {name})
BEGIN
  CREATE SCHEMA {name};
END;
GO"""
            if force
            else f"CREATE SCHEMA {name}"
        )

        with self.begin() as con:
            if should_switch_database:
                con.exec_driver_sql(f"USE {self._quote(database)}")

            con.exec_driver_sql(create_stmt)

            if should_switch_database:
                con.exec_driver_sql(f"USE {self._quote(current_database)}")

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        current_database = self.current_database
        should_switch_database = database is not None and database != current_database

        name = self._quote(name)

        if_exists = "IF EXISTS " * force

        with self.begin() as con:
            if should_switch_database:
                con.exec_driver_sql(f"USE {self._quote(database)}")

            con.exec_driver_sql(f"DROP SCHEMA {if_exists}{name}")

            if should_switch_database:
                con.exec_driver_sql(f"USE {self._quote(current_database)}")
