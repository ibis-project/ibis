from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge
from chdb.session import Session

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import UrlFromPath
from ibis.backends.clickhouse import Backend as CHBackend
from ibis.backends.sql.compiler import C
from ibis.formats.pyarrow import PyArrowData, PyArrowType

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class ChdbArrowConverter(PyArrowData):
    @classmethod
    def convert_column(cls, column: pa.Array, dtype: dt.DataType) -> pa.Array:
        pa_type = PyArrowType.from_ibis(dtype)
        if dtype.is_date():
            return column.cast(pa.int32()).cast(pa_type)
        elif dtype.is_timestamp():
            if dtype.scale is None:
                pa_type = pa.timestamp("s")
            return column.cast(pa.int64()).cast(pa_type)
        else:
            return column.cast(pa_type)

    @classmethod
    def convert_table(cls, table: pa.Table, schema: sch.Schema) -> pa.Table:
        arrays = []
        for column, dtype in zip(table.columns, schema.values()):
            arrays.append(cls.convert_column(column, dtype))
        return pa.Table.from_arrays(arrays, names=schema.names)


class Backend(CHBackend, UrlFromPath):
    name = "chdb"

    @property
    def version(self) -> str:
        # TODO: there is a `PRAGMA version` we could use instead
        import importlib.metadata

        return importlib.metadata.version("chdb")

    def do_connect(
        self, path: None | str | Path = None, database: None | str = None
    ) -> None:
        self.con = Session(path)
        if database is not None:
            self.raw_sql(f"USE {database}")

    def raw_sql(
        self, query: str | sge.Expression, external_tables=None, **kwargs
    ) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        external_tables
            Mapping of table name to pandas DataFrames providing
            external datasources for the query
        kwargs
            Backend specific query arguments

        Returns
        -------
        Cursor
            Clickhouse cursor

        """
        if external_tables:
            raise NotImplementedError("External tables are not yet supported")
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect, pretty=True)
        self._log(query)
        return self.con.query(query, **kwargs)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        yield self.raw_sql(*args, **kwargs)

    def get_schema(
        self, table_name: str, database: str | None = None, schema: str | None = None
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        database
            Database name
        schema
            Schema name, not supported by ClickHouse

        Returns
        -------
        sch.Schema
            Ibis schema

        """
        if schema is not None:
            raise com.UnsupportedBackendFeatureError(
                "`schema` namespaces are not supported by chdb"
            )
        query = sge.Describe(this=sg.table(table_name, db=database))
        table = self.raw_sql(query, fmt="arrowtable")
        names = table.column("name").to_pylist()
        types = table.column("type").to_pylist()
        dtypes = map(self.compiler.type_mapper.from_string, types)
        return sch.Schema(dict(zip(names, dtypes)))

    def _metadata(self, query: str) -> sch.Schema:
        name = util.gen_name("clickhouse_metadata")
        try:
            self.raw_sql(f"CREATE VIEW {name} AS {query}")
            return self.get_schema(name).items()
        finally:
            self.raw_sql(f"DROP VIEW {name}")

    def execute(self, expr: ir.Expr, **kwargs: Any) -> Any:
        """Execute an expression."""
        table = expr.as_table()
        df = self.to_pyarrow(table, **kwargs).to_pandas()
        return expr.__pandas_result__(table.__pandas_result__(df))

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        external_tables: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        table = expr.as_table()
        sql = self.compile(table, **kwargs)
        self._log(sql)
        external_tables = self._collect_in_memory_tables(expr, external_tables)

        result = self.raw_sql(sql, fmt="arrowtable", external_tables=external_tables)
        result = ChdbArrowConverter.convert_table(result, table.schema())
        return expr.__pyarrow_result__(result)

    def to_pyarrow_batches(self, expr: ir.Expr, **kwargs):
        table = expr.as_table()
        return self.to_pyarrow(table, **kwargs).to_reader()

    def list_databases(self, like: str | None = None) -> list[str]:
        query = sg.select(C.name).from_(sg.table("databases", db="system"))
        result = self.raw_sql(query, fmt="arrowtable")
        databases = result.column("name").to_pylist()
        return self._filter_with_like(databases, like)
