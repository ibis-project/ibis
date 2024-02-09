"""Risingwave backend."""

from __future__ import annotations

from functools import partial
from itertools import repeat
from typing import TYPE_CHECKING

import psycopg2
import sqlglot as sg
import sqlglot.expressions as sge
from psycopg2 import extras

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.postgres import Backend as PostgresBackend
from ibis.backends.risingwave.compiler import RisingwaveCompiler

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


class Backend(PostgresBackend):
    name = "risingwave"
    compiler = RisingwaveCompiler()
    supports_python_udfs = False

    def do_connect(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int = 5432,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create an Ibis client connected to RisingWave database.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port number
        database
            Database to connect to
        schema
            RisingWave schema to use. If `None`, use the default `search_path`.

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_RISINGWAVE_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_RISINGWAVE_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_RISINGWAVE_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_RISINGWAVE_DATABASE", "dev")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        RisingwaveTable[table]
          name: functional_alltypes
          schema:
            id : int32
            bool_col : boolean
            tinyint_col : int16
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32

        """

        self.con = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            options=(f"-csearch_path={schema}" * (schema is not None)) or None,
        )

        with self.begin() as cur:
            cur.execute("SET TIMEZONE = UTC")

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
        """Create a table in Risingwave.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but at least
            one of `obj` or `schema` must be specified
        schema
            The schema of the table to create; optional, but at least one of
            `obj` or `schema` must be specified
        database
            The name of the database in which to create the table; if not
            passed, the current database is used.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists

        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if database is not None and database != self.current_database:
            raise com.UnsupportedOperationError(
                f"Creating tables in other databases is not supported by {self.name}"
            )
        else:
            database = None

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self._to_sqlglot(table)
        else:
            query = None

        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=self.compiler.quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for colname, typ in (schema or table.schema()).items()
        ]

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, catalog=database, quoted=self.compiler.quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table, expression=query).sql(self.dialect)
                cur.execute(insert_stmt)

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.dialect)
                )
                cur.execute(
                    f"ALTER TABLE {table.sql(self.dialect)} RENAME TO {this.sql(self.dialect)}"
                )

        if schema is None:
            return self.table(name, schema=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
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
