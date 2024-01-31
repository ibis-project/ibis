"""Risingwave backend."""

from __future__ import annotations

from functools import partial
from itertools import repeat
from typing import TYPE_CHECKING

import sqlglot as sg
from psycopg2 import extras

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
from ibis.backends.postgres import Backend as PostgresBackend
from ibis.backends.risingwave.compiler import RisingwaveCompiler
from ibis.common.exceptions import InvalidDecoratorError

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    import ibis
    import ibis.expr.types as ir


def _verify_source_line(func_name: str, line: str):
    if line.startswith("@"):
        raise InvalidDecoratorError(func_name, line)
    return line


class Backend(PostgresBackend):
    name = "risingwave"
    dialect = "postgres"
    compiler = RisingwaveCompiler()
    supports_python_udfs = False

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        # TODO: raise if temp is True
        super().create_table(
            name, obj, schema=schema, database=database, temp=temp, overwrite=overwrite
        )

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [
                            sg.exp.ColumnConstraint(
                                kind=sg.exp.NotNullColumnConstraint()
                            )
                        ]
                    ),
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
            )
            create_stmt_sql = create_stmt.sql(self.dialect)

            columns = schema.keys()
            df = op.data.to_frame()
            data = df.itertuples(index=False)
            cols = ", ".join(
                ident.sql(self.dialect)
                for ident in map(partial(sg.to_identifier, quoted=quoted), columns)
            )
            specs = ", ".join(repeat("%s", len(columns)))
            table = sg.table(name, quoted=quoted)
            sql = f"INSERT INTO {table.sql(self.dialect)} ({cols}) VALUES ({specs})"
            with self.begin() as cur:
                cur.execute(create_stmt_sql)
                extras.execute_batch(cur, sql, data, 128)
