"""The StarRocks backend."""

from __future__ import annotations

import contextlib
from functools import cached_property
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.mysql import Backend as MySQLBackend
from ibis.backends.sql.compilers.base import RenameTable
from ibis.backends.sql.compilers.starrocks import compiler

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import MySQLdb
    import pandas as pd
    import polars as pl
    import pyarrow as pa


class Backend(MySQLBackend):
    name = "starrocks"
    compiler = compiler
    supports_create_or_replace = False

    @cached_property
    def version(self) -> str:
        with self._safe_raw_sql("SELECT current_version()") as cur:
            [(version,)] = cur.fetchall()
        return version

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwarg_overrides.setdefault("port", url.port or 9030)
        return super()._from_url(url, **kwarg_overrides)

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = "root",
        password: str | None = None,
        port: int = 9030,
        autocommit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client using the passed StarRocks connection parameters.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            StarRocks FE MySQL protocol port
        autocommit
            Autocommit mode
        kwargs
            Additional keyword arguments passed to `MySQLdb.connect`
        """
        super().do_connect(
            host=host,
            user=user,
            password=password,
            port=port,
            autocommit=autocommit,
            **kwargs,
        )

    @util.experimental
    @classmethod
    def from_connection(cls, con: MySQLdb.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to StarRocks."""
        return super().from_connection(con)

    def _create_table_properties(self, *, temp: bool) -> list[sge.Expression]:
        properties: list[sge.Expression] = []

        if temp:
            properties.append(sge.TemporaryProperty())

        properties.extend(
            [
                sge.EngineProperty(this="OLAP"),
                sge.DistributedByProperty(kind="RANDOM"),
                sge.Property(
                    this=sge.Literal.string("replication_num"),
                    value=sge.Literal.string("1"),
                ),
            ]
        )
        return properties

    @contextlib.contextmanager
    def begin(self):
        with self.con.cursor() as cur:
            yield cur

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.IntoSchema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        table_expr = sg.table(temp_name, catalog=database, quoted=quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot_column_defs(dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(
                expressions=self._create_table_properties(temp=temp)
            ),
        )

        this = sg.table(name, catalog=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite:
                cur.execute(sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect))
                cur.execute(
                    sge.Alter(
                        kind="TABLE",
                        this=table_expr,
                        exists=True,
                        actions=[RenameTable(this=this)],
                    ).sql(dialect)
                )

        if schema is None:
            return self.table(name, database=database)

        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "StarRocks cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted
        dialect = self.dialect

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(dialect),
            ),
            properties=sg.exp.Properties(
                expressions=self._create_table_properties(temp=True)
            ),
        )
        create_stmt_sql = create_stmt.sql(dialect)

        df = op.data.to_frame()
        df = df.replace(float("nan"), None)

        data = df.itertuples(index=False)
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                cur.executemany(sql, data)
