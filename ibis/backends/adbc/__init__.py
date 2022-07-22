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

import typing
from typing import Any, Mapping

import adbc_driver_manager
import pandas as pd
import pyarrow as pa
import sqlalchemy as sa

import ibis
import ibis.backends.pyarrow.datatypes as pa_dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import Database
from ibis.backends.base.sql import BaseSQLBackend


class Backend(BaseSQLBackend):
    name = 'adbc'
    database_class = Database
    sqla_dialect = None
    table_expr_class = ir.Table

    def __getstate__(self) -> dict:
        r = super().__getstate__()
        r.update(
            dict(
                compiler=self.compiler,
                sqla_dialect=self.sqla_dialect,
            )
        )
        return r

    def create_table(
        self,
        name: str,
        expr: pa.RecordBatch
        | pa.RecordBatchReader
        | pa.Table
        | pd.DataFrame
        | ir.Table
        | None = None,
        schema: sch.Schema | None = None,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """
        Create a table.

        Parameters
        ----------
        name
            Table name to create
        expr
            DataFrame or table expression to use as the data source
        schema
            An ibis schema
        database
            A database
        force
            Check whether a table exists before creating it
        """
        # TODO: force
        if database is not None:
            raise NotImplementedError(
                'Creating tables from a different database is not yet '
                'implemented'
            )

        if expr is None and schema is None:
            raise ValueError('You must pass either an expression or a schema')

        if expr is not None and schema is not None:
            if not expr.schema().equals(ibis.schema(schema)):
                raise TypeError(
                    'Expression schema is not equal to passed schema. '
                    'Try passing the expression without the schema'
                )
        if schema is None:
            schema = expr.schema()

        if isinstance(expr, ir.Table):
            raise NotImplementedError(
                'Creating tables from an expression is not yet implemented'
            )
        if expr is None:
            pyarrow_schema = pa.schema(
                [
                    (name, pa_dt.to_pyarrow_type(dt))
                    for (name, dt) in schema.items()
                ]
            )
            expr = pa.record_batch(
                [
                    chunked_arr.chunks[0]
                    for chunked_arr in pyarrow_schema.empty_table()
                ],
                schema=pyarrow_schema,
            )
        elif isinstance(expr, pa.Table):
            expr = pa.RecordBatchReader.from_batches(
                expr.to_batches(), expr.schema
            )

        with adbc_driver_manager.AdbcStatement(self._conn) as stmt:
            stmt.set_options(
                **{adbc_driver_manager.INGEST_OPTION_TARGET_TABLE: name}
            )

            if isinstance(expr, pa.RecordBatch):
                array = adbc_driver_manager.ArrowArrayHandle()
                schema = adbc_driver_manager.ArrowSchemaHandle()
                expr._export_to_c(array.address, schema.address)
                stmt.bind(array, schema)
            else:
                stream = adbc_driver_manager.ArrowArrayStreamHandle()
                expr._export_to_c(stream.address)
                stmt.bind_stream(stream)

            stmt.execute()

    @property
    def current_database(self) -> str:
        """The name of the current database this client is connected to."""
        # TODO: switching databases
        # TODO: how to handle this in a generic way? may need some
        # support from ADBC
        return "main"

    def do_connect(
        self, *, driver, entrypoint, db_args=None, dialect=None, **kwargs
    ) -> None:
        """
        Connect to the database.

        Parameters
        ----------
        driver : str
            The ADBC driver to use (e.g. "adbc_driver_flight_sql").

        entrypoint : str
            The entrypoint in the driver (e.g. "AdbcFlightSqlDriverInit").

        db_args : dict, optional
            Arguments to pass to the ADBC driver.

        dialect : str, optional
            Whether to use SQL or Substrait, and if SQL, which dialect
            to use.  If not given, will try to detect Substrait vs SQL
            and default to a generic (but extremely limited) SQL
            dialect.

        kwargs : dict, optional
            Additional arguments.
        """
        # TODO: detect the dialect
        # TODO: need to control compiler
        self._db = adbc_driver_manager.AdbcDatabase(
            driver=driver,
            entrypoint=entrypoint,
            **(db_args or {}),
        )
        self._conn = adbc_driver_manager.AdbcConnection(self._db)
        if dialect == "sqlite":
            from ibis.backends.sqlite.compiler import SQLiteCompiler

            self.compiler = SQLiteCompiler
            self.sqla_dialect = sa.dialects.sqlite.dialect()
        elif not dialect:
            pass
        else:
            raise NotImplementedError(
                f"Unsupported or unknown dialect '{dialect}'"
            )

    def fetch_from_cursor(
        self, cursor: IbisAdbcCursor, schema: sch.Schema
    ) -> pd.DataFrame:
        return cursor.read_pandas()

    def list_databases(self, like=None):
        """List databases in the current server."""
        with self._conn.get_objects(
            depth=adbc_driver_manager.GetObjectsDepth.CATALOGS
        ) as stmt:
            handle = stmt.get_stream()
            reader = pa.RecordBatchReader._import_from_c(handle.address)
            table = reader.read_all()
            catalog_names = table[0]

        names = []
        # TODO: what should SQLite return? See what native backend does
        for chunk in catalog_names.chunks:
            names.extend(
                self._filter_with_like(
                    (x for x in chunk.to_pylist() if x is not None), like
                )
            )
        return names

    def list_tables(self, like=None, database=None):
        with self._conn.get_objects(
            depth=adbc_driver_manager.GetObjectsDepth.TABLES, catalog=database
        ) as stmt:
            handle = stmt.get_stream()
            reader = pa.RecordBatchReader._import_from_c(handle.address)
            table = reader.read_all()
            db_schemas = table[1]

        tables = []
        for chunk in db_schemas.chunks:
            schema_tables = chunk.flatten().flatten()[1].flatten()
            table_names = schema_tables.flatten()[0]
            tables.extend(
                self._filter_with_like(table_names.to_pylist(), like)
            )
        return tables

    def raw_sql(self, query: str) -> Any:
        # Since we also accept sqlalchemy exprs (todo: type annotate)
        query = str(query.compile(dialect=self.sqla_dialect))
        try:
            stmt = adbc_driver_manager.AdbcStatement(self._conn)
            stmt.set_sql_query(query)
            stmt.execute()
            handle = stmt.get_stream()
            reader = pa.RecordBatchReader._import_from_c(handle.address)
        except Exception:
            stmt.close()
            raise
        return IbisAdbcCursor(stmt, reader)

    def table(
        self,
        name: str,
        database: str | None = None,
        schema: str | None = None,
    ) -> ir.Table:
        """Create a table expression from a table in the database.

        Parameters
        ----------
        name
            Table name
        database
            The database the table resides in (mapped to 'catalog')
        schema
            The schema inside `database` where the table resides.

            !!! warning "`schema` refers to database organization"

                The `schema` parameter does **not** refer to the column names
                and types of `table`.

        Returns
        -------
        Table
            Table expression
        """
        handle = self._conn.get_table_schema(database, schema, name)
        pyarrow_schema = pa.Schema._import_from_c(handle.address)
        ibis_schema = pa_dt.infer_pyarrow_schema(pyarrow_schema)
        node = self.table_class(name=name, schema=ibis_schema, source=self)
        return self.table_expr_class(node)

    @property
    def version(self) -> str:
        # TODO:
        return '0.0.1'

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with adbc_driver_manager.AdbcStatement(self._conn) as stmt:
            stmt.set_sql_query(query)
            stmt.execute()
            handle = stmt.get_stream()
            reader = pa.RecordBatchReader._import_from_c(handle.address)
            return pa_dt.infer_pyarrow_schema(reader.schema)


class IbisAdbcCursor(typing.NamedTuple):
    """
    A cursor for a result set.
    """

    statement: adbc_driver_manager.AdbcStatement
    reader: pa.RecordBatchReader

    def read_all(self):
        return self.reader.read_all()

    def read_pandas(self, *args, **kwargs):
        return self.reader.read_pandas(*args, **kwargs)

    def close(self):
        self.reader.close()
        self.statement.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
