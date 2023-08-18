from __future__ import annotations

from typing import TYPE_CHECKING

import chdb
import chdb.dataframe as cdf
import chdb.session as chs
import sqlglot as sg
import wurlitzer

from ibis.backends.base import BaseBackend, CanCreateDatabase
from ibis.backends.clickhouse.compiler import translate
from ibis.expr.analysis import p

if TYPE_CHECKING:
    import ibis.expr.types as ir


class Session(chs.Session):
    def query(self, sql, fmt="CSV"):
        with wurlitzer.pipes() as (out, err):
            result = super().query(sql, fmt=fmt)
            if error := err.getvalue():
                raise ValueError(error)
            return result


class Backend(BaseBackend, CanCreateDatabase):
    name = "chdb"

    # ClickHouse itself does, but the client driver does not
    supports_temporary_tables = True

    def raw_sql(self, query: str) -> None:
        """Execute a raw SQL query.

        Parameters
        ----------
        query : str
            Query to execute
        """
        self._session.query(query)

    def create_database(
        self, name: str, *, force: bool = False, engine: str = "Atomic"
    ) -> None:
        if_not_exists = "IF NOT EXISTS " * force
        self.raw_sql(f"CREATE DATABASE {if_not_exists}{name} ENGINE = {engine}")

    @property
    def version(self) -> str:
        return chdb.__version__

    @property
    def current_database(self) -> str:
        return self.raw_sql("SELECT currentDatabase()")

    def create_table(self, *args, **kwargs):
        raise NotImplementedError("Need to implement")

    def create_view(self, *args, **kwargs):
        raise NotImplementedError("Need to implement")

    def drop_table(self, *args, **kwargs):
        raise NotImplementedError("Need to implement")

    def drop_view(self, *args, **kwargs):
        raise NotImplementedError("Need to implement")

    def drop_database(self, *args, **kwargs):
        raise NotImplementedError("Need to implement")

    def list_databases(self, like: str | None = None) -> list[str]:
        return self.raw_sql("SELECT name FROM system.databases")

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SELECT name FROM system.tables WHERE"

        if database is None:
            database = "currentDatabase()"
        else:
            database = f"'{database}'"

        query += f" database = {database} OR is_temporary"
        return self.raw_sql(query)

    def table(self, name: str, database: str | None = None) -> ir.Table:
        raise NotImplementedError("Need to implement")

    def compile(self, expr: ir.Expr, params=None, **kwargs):
        table_expr = expr.as_table()

        sql = translate(table_expr.op(), params=params or {})
        assert not isinstance(sql, sg.exp.Subquery)

        if isinstance(sql, sg.exp.Table):
            sql = sg.select("*").from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql.sql(dialect="clickhouse", pretty=True)

    def execute(self, expr: ir.Expr, **kwargs):
        frames = {}

        def replacer(op, ctx):
            frames[op.name] = op.data._data
            return op.copy(name=f"__{op.name}__")

        node = expr.op().replace(p.InMemoryTable >> replacer)
        sql = self.compile(node.to_expr(), **kwargs)

        with wurlitzer.pipes() as (out, err):
            result = cdf.query(sql, **frames)
            if error := err.getvalue():
                raise ValueError(error)
            else:
                return result.to_pandas()
