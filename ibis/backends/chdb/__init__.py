"""chDB backend — embedded ClickHouse (in-process).

chDB is an in-process build of ClickHouse. Because the SQL dialect is
identical to ClickHouse, this backend reuses the ClickHouse compiler and the
overwhelming majority of the ClickHouse backend's DDL/SQL construction. Only
the transport layer differs: instead of talking to a ClickHouse server over
HTTP via ``clickhouse_connect``, queries run against the embedded engine
through the ``chdb`` package.

Two chDB specifics are handled here:

* **In-memory tables.** The ClickHouse backend ships ``ibis.memtable`` data to
  the server as HTTP *external tables*. The embedded engine has no such
  transport, so instead each memtable is materialized to an Arrow table,
  injected into this module's namespace under the memtable's (unique) name,
  and referenced from SQL through chDB's ``Python(<name>)`` table function,
  which scans the in-process Python object directly (zero copy for Arrow).

* **Arrow type round-trip.** chDB emits ``DateTime`` (second precision, no
  scale) as Arrow ``uint32`` seconds. :class:`ChdbArrowConverter` restores the
  declared Ibis types on the way out.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, UrlFromPath
from ibis.backends.clickhouse import Backend as CHBackend
from ibis.backends.sql.compilers import ClickHouseCompiler
from ibis.backends.sql.compilers.base import C
from ibis.backends.sql.dialects import ChDB
from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowData, PyArrowType

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# memtable registry
#
# chDB's Python(name) table function resolves ``name`` by walking the calling
# Python stack frames and looking the identifier up in each frame's globals /
# locals. The frames that execute chdb queries live in this module, so a
# memtable injected into this module's globals under its (globally unique)
# name is visible to the engine. A lock guards the shared namespace because
# chDB allows several connections in one process.
# ---------------------------------------------------------------------------
_MEMTABLE_LOCK = threading.Lock()


def _register_memtable(name: str, table: pa.Table) -> None:
    with _MEMTABLE_LOCK:
        globals()[name] = table


def _unregister_memtable(name: str) -> None:
    with _MEMTABLE_LOCK:
        globals().pop(name, None)


class ChdbCompiler(ClickHouseCompiler):
    """ClickHouse compiler that renders in-memory tables as ``Python(name)``."""

    dialect = ChDB

    def visit_InMemoryTable(self, op, *, name, schema, data):
        # Reference the memtable through chDB's Python() table function. The
        # name is emitted as a string literal so it matches chDB's identifier
        # extraction regardless of ClickHouse identifier quoting.
        return sge.Table(this=self.f.Python(sge.convert(name)))


class ChdbArrowConverter(PyArrowData):
    """Restore declared Ibis types on chDB's Arrow output.

    chDB emits ``DateTime`` (no scale) as Arrow ``uint32`` seconds; a plain
    ``cast`` to ``timestamp`` is not allowed by Arrow, so integer-encoded
    temporal columns are cast through ``int64`` first.
    """

    @classmethod
    def convert_column(cls, column: pa.ChunkedArray, dtype: dt.DataType):
        pa_type = PyArrowType.from_ibis(dtype)
        if column.type == pa_type:
            return column
        if dtype.is_timestamp() and pa.types.is_integer(column.type):
            unit = "s" if dtype.scale is None else pa_type.unit
            target = pa.timestamp(unit, tz=dtype.timezone)
            return column.cast(pa.int64()).cast(target).cast(pa_type)
        if (
            dtype.is_array()
            and dtype.value_type.is_timestamp()
            and pa.types.is_list(column.type)
            and pa.types.is_integer(column.type.value_type)
        ):
            value = dtype.value_type
            unit = "s" if value.scale is None else pa_type.value_type.unit
            inner = pa.timestamp(unit, tz=value.timezone)
            return column.cast(pa.list_(pa.int64())).cast(pa.list_(inner)).cast(pa_type)
        with contextlib.suppress(pa.lib.ArrowInvalid, pa.lib.ArrowNotImplementedError):
            return column.cast(pa_type)
        return column

    @classmethod
    def convert_table(cls, table: pa.Table, schema: sch.Schema) -> pa.Table:
        target = schema.to_pyarrow()
        # chDB emits a zero-column table for an empty result set; rebuild the
        # declared (empty) shape instead of indexing missing columns.
        if table.num_columns != len(schema):
            if table.num_rows == 0:
                return target.empty_table()
            raise com.IbisError(
                f"chDB returned {table.num_columns} columns, expected {len(schema)}"
            )
        columns = [
            cls.convert_column(table.column(i), dtype)
            for i, dtype in enumerate(schema.values())
        ]
        # Build with the target schema so field nullability matches the
        # declared Ibis schema (chDB reports columns as non-nullable).
        return pa.Table.from_arrays(columns, schema=target)


# ibis dtype -> chdb.sqltypes constant, for Python scalar UDF registration.
def _chdb_sqltype(dtype: dt.DataType):
    from chdb import sqltypes as st

    mapping = {
        dt.Int8: st.INT8,
        dt.Int16: st.INT16,
        dt.Int32: st.INT32,
        dt.Int64: st.INT64,
        dt.UInt8: st.UINT8,
        dt.UInt16: st.UINT16,
        dt.UInt32: st.UINT32,
        dt.UInt64: st.UINT64,
        dt.Float32: st.FLOAT32,
        dt.Float64: st.FLOAT64,
        dt.String: st.STRING,
        dt.Boolean: st.BOOL,
    }
    for ibis_type, chdb_type in mapping.items():
        if isinstance(dtype, ibis_type):
            return chdb_type
    raise com.UnsupportedBackendType(
        f"chDB Python UDFs do not support the type {dtype}"
    )


class _Con:
    """Adapter over the embedded chDB connection.

    Exposes the small ``clickhouse_connect``-shaped surface that the inherited
    ClickHouse DDL methods (``create_table`` etc.) call directly.
    """

    def __init__(self, session):
        self._session = session

    def raw_query(self, sql, *, external_data=None, **_):
        # Memtables are handled via Python() injection, never external data.
        return self._session.query(sql)

    command = raw_query

    def query(self, sql, *, external_data=None, fmt="CSV", **_):
        return self._session.query(sql, fmt)

    def send_query(self, sql, fmt="Arrow"):
        return self._session.send_query(sql, fmt)

    def cursor(self):
        return self._session.cursor()

    def close(self):
        with contextlib.suppress(Exception):
            self._session.close()


class Backend(CHBackend, CanCreateDatabase, UrlFromPath):
    name = "chdb"
    compiler = ChdbCompiler()

    @property
    def version(self) -> str:
        import chdb

        return chdb.__version__

    def do_connect(self, path: str | Path | None = None, /, **_: Any) -> None:
        """Create an Ibis client connected to an embedded chDB engine.

        Parameters
        ----------
        path
            Directory for a persistent database. Defaults to an ephemeral
            in-memory database. Note that chDB allows multiple connections in
            one process only if they share the same path.

        """
        import chdb

        self.con = _Con(chdb.connect(":memory:" if path is None else str(path)))

    # -- execution ---------------------------------------------------------

    def raw_sql(self, query, external_tables=None, fmt: str = "CSV", **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)
        self._register_in_memory_tables_from_mapping(external_tables)
        self._log(query)
        return self.con.query(query, fmt=fmt)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, external_tables=None, **kwargs):
        yield self.raw_sql(query, external_tables=external_tables, **kwargs)

    def _fetch_arrow(self, expr, /, **kwargs) -> pa.Table:
        table = expr.as_table()
        self._run_pre_execute_hooks(table)
        sql = self.compile(table, **kwargs)
        self._log(sql)
        result = self.con.send_query(sql, "Arrow").record_batch().read_all()
        return ChdbArrowConverter.convert_table(result, table.schema())

    def to_pyarrow(self, expr, /, **kwargs) -> pa.Table:
        table = self._fetch_arrow(expr, **_pop_arrow_kwargs(kwargs))
        return expr.__pyarrow_result__(table, data_mapper=ChdbArrowConverter)

    def to_pyarrow_batches(
        self, expr, /, *, chunk_size: int = 1_000_000, **kwargs
    ) -> pa.ipc.RecordBatchReader:
        table_expr = expr.as_table()
        self._run_pre_execute_hooks(table_expr)
        sql = self.compile(table_expr, **kwargs)
        self._log(sql)
        schema = table_expr.schema()
        arrow_schema = schema.to_pyarrow()

        reader = self.con.send_query(sql, "Arrow").record_batch(chunk_size)

        def batches():
            for batch in reader:
                converted = ChdbArrowConverter.convert_table(
                    pa.Table.from_batches([batch], schema=batch.schema), schema
                )
                yield from converted.to_batches()

        return pa.ipc.RecordBatchReader.from_batches(arrow_schema, batches())

    def execute(self, expr, /, **kwargs):
        table = self._fetch_arrow(expr, **_pop_arrow_kwargs(kwargs))
        df = table.to_pandas(timestamp_as_object=True)
        return expr.__pandas_result__(df, schema=expr.as_table().schema())

    # -- metadata ----------------------------------------------------------

    def get_schema(
        self, table_name, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        if catalog is not None:
            raise com.UnsupportedOperationError(
                "`catalog` namespaces are not supported by chdb"
            )
        query = sge.Describe(this=sg.table(table_name, db=database))
        table = self.raw_sql(query, fmt="ArrowTable")
        names = table.column("name").to_pylist()
        types = table.column("type").to_pylist()
        type_mapper = self.compiler.type_mapper
        return sch.Schema(dict(zip(names, map(type_mapper.from_string, types))))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        name = util.gen_name("chdb_metadata")
        self.raw_sql(f"CREATE OR REPLACE VIEW {name} AS {query}")
        try:
            return self.get_schema(name)
        finally:
            self.raw_sql(f"DROP VIEW IF EXISTS {name}")

    def list_tables(
        self, *, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = sg.select(C.name).from_(sg.table("tables", db="system"))
        if database is None:
            database = self.compiler.f.currentDatabase()
        else:
            database = sge.convert(database)
        query = query.where(C.database.eq(database).or_(C.is_temporary))
        result = self.raw_sql(query, fmt="ArrowTable")
        return self._filter_with_like(result.column("name").to_pylist(), like)

    def list_databases(self, *, like: str | None = None) -> list[str]:
        query = sg.select(C.name).from_(sg.table("databases", db="system"))
        result = self.raw_sql(query, fmt="ArrowTable")
        return self._filter_with_like(result.column("name").to_pylist(), like)

    # -- in-memory tables --------------------------------------------------

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        _register_memtable(op.name, op.data.to_pyarrow(op.schema))

    def _register_in_memory_tables_from_mapping(self, external_tables) -> None:
        for name, obj in (external_tables or {}).items():
            memtable = obj if isinstance(obj, ops.InMemoryTable) else obj.op()
            _register_memtable(name, memtable.data.to_pyarrow(memtable.schema))

    def _normalize_external_tables(self, external_tables=None):
        # chDB has no external-table transport: register each collected
        # memtable for Python() scanning and report that there is no external
        # data to ship.
        self._register_in_memory_tables_from_mapping(external_tables)

    def _make_memtable_finalizer(self, name: str):
        return lambda: _unregister_memtable(name)

    # -- user-defined functions -------------------------------------------

    def _register_udfs(self, expr: ir.Expr) -> None:
        import chdb

        for udf_node in expr.op().find(ops.ScalarUDF):
            if udf_node.__input_type__ != InputType.PYTHON:
                # Only pure-Python scalar UDFs map onto chdb.create_function;
                # builtins need no registration, pandas/pyarrow are unsupported.
                continue
            name = type(udf_node).__name__
            arg_types = [
                _chdb_sqltype(param.annotation.pattern.dtype)
                for param in udf_node.__signature__.parameters.values()
            ]
            return_type = _chdb_sqltype(udf_node.dtype)
            with contextlib.suppress(Exception):
                chdb.drop_function(name)
            chdb.create_function(
                name, udf_node.__func__, arg_types=arg_types, return_type=return_type
            )


def _pop_arrow_kwargs(kwargs: dict) -> dict:
    """Keep only the ``params``/``limit`` kwargs the compiler accepts."""
    return {k: kwargs[k] for k in ("params", "limit") if k in kwargs}
