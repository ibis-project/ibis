"""chDB backend — embedded (in-process) ClickHouse.

Subclasses the ClickHouse backend to reuse its compiler and DDL/SQL, swapping
only the transport: queries run against the embedded ``chdb`` engine instead
of a ClickHouse server. Two chDB specifics are handled here — memtables (via
the ``Python(<name>)`` table function, see below) and the ``DateTime``->uint32
Arrow output fixup (:class:`ChdbArrowConverter`).
"""

from __future__ import annotations

import contextlib
import importlib
import re
import threading
import types
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import UrlFromPath
from ibis.backends.clickhouse import Backend as CHBackend
from ibis.backends.sql.compilers import ClickHouseCompiler
from ibis.backends.sql.compilers.base import C
from ibis.backends.sql.dialects import ChDB
from ibis.expr.operations.udf import InputType
from ibis.formats.pyarrow import PyArrowData, PyArrowType

if TYPE_CHECKING:
    from pathlib import Path


# chDB's Python(name) resolves ``name`` by scanning caller stack frames'
# globals/locals; injecting memtables into this module's globals makes them
# visible to the engine. Locked because one process may hold several connections.
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
        # name as a string literal -> matches chDB's identifier extraction.
        return sge.Table(this=self.f.Python(sge.convert(name)))


class ChdbArrowConverter(PyArrowData):
    """Restore declared Ibis types on chDB's Arrow output.

    chDB emits scale-less ``DateTime`` as ``uint32`` seconds; Arrow forbids a
    direct uint32->timestamp cast, so such columns go through ``int64`` first.
    """

    @classmethod
    def convert_column(cls, column: pa.ChunkedArray, dtype: dt.DataType):
        import ipaddress
        import uuid as uuidlib

        def combined(col):
            return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col

        pa_type = PyArrowType.from_ibis(dtype)
        if column.type == pa_type:
            return column

        n = len(column)
        # dt.null and all-null columns: chDB may return them as binary/other,
        # which Arrow won't cast; rebuild directly as nulls of the target type.
        if dtype.is_null() or pa.types.is_null(pa_type) or (n and column.null_count == n):
            return pa.nulls(n, type=pa_type)

        # scale-less DateTime arrives as uint32 seconds; uint32->timestamp is
        # not a legal Arrow cast, so route through int64.
        if dtype.is_timestamp() and pa.types.is_integer(column.type):
            unit = "s" if dtype.scale is None else pa_type.unit
            target = pa.timestamp(unit, tz=dtype.timezone)
            return column.cast(pa.int64()).cast(target).cast(pa_type)

        if dtype.is_interval() and pa.types.is_integer(column.type):
            return column.cast(pa.int64()).cast(pa_type)

        # UUID arrives as fixed_size_binary(16) / arrow.uuid extension.
        if dtype.is_uuid():
            def fmt_uuid(v):
                if v is None or isinstance(v, str):
                    return v
                if isinstance(v, uuidlib.UUID):
                    return str(v)
                return str(uuidlib.UUID(bytes=bytes(v)))

            return pa.array(
                [fmt_uuid(v) for v in combined(column).to_pylist()], type=pa_type
            )

        # INET arrives as uint32 (IPv4) or fixed_size_binary(16) (IPv6).
        if dtype.is_inet():
            def fmt_ip(v):
                if v is None or isinstance(v, str):
                    return v
                if isinstance(v, (bytes, bytearray)):
                    return str(ipaddress.ip_address(bytes(v)))
                return str(ipaddress.ip_address(v))

            return pa.array(
                [fmt_ip(v) for v in combined(column).to_pylist()], type=pa_type
            )

        # chDB emits anonymous tuples with positional field names ('1','2',...);
        # relabel to the declared field names (Arrow matches struct fields by name).
        if dtype.is_struct() and pa.types.is_struct(column.type):
            arr = combined(column)
            fields = [arr.field(i) for i in range(arr.type.num_fields)]
            renamed = pa.StructArray.from_arrays(fields, names=list(dtype.names))
            with contextlib.suppress(Exception):
                return renamed.cast(pa_type)
            return renamed

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

        # nullable values come back as a dense union; resolve to plain values.
        if pa.types.is_union(column.type):
            return pa.array(combined(column).to_pylist(), type=pa_type)

        # generic: safe cast, then unsafe (e.g. signed<->unsigned), then rebuild.
        with contextlib.suppress(Exception):
            return column.cast(pa_type)
        with contextlib.suppress(Exception):
            return pc.cast(column, pa_type, safe=False)
        with contextlib.suppress(Exception):
            return pa.array(combined(column).to_pylist(), type=pa_type)
        return column

    @classmethod
    def convert_table(cls, table: pa.Table, schema: sch.Schema) -> pa.Table:
        target = schema.to_pyarrow()
        # chDB returns a zero-column table for an empty result.
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
        # schema= aligns field nullability (chDB columns are non-nullable).
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
    """Adapt the chDB connection to a ``clickhouse_connect``-shaped surface.

    The inherited ClickHouse DDL methods call these directly: ``raw_query`` /
    ``command`` run DDL, ``query`` / ``send_query`` / ``cursor`` back execution,
    ``close`` disconnects.
    """

    def __init__(self, session):
        self._session = session

    def raw_query(self, sql, *, external_data=None, **_):
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


class Backend(UrlFromPath, CHBackend):
    # UrlFromPath first: its path-based _from_url (chdb://<path>) must win over
    # ClickHouse's host/port one.
    name = "chdb"
    compiler = ChdbCompiler()

    @property
    def version(self) -> str:
        import chdb

        return chdb.__version__

    def do_connect(self, database: str | Path = ":memory:", **_: Any) -> None:
        """Create an Ibis client connected to an embedded chDB engine.

        Parameters
        ----------
        database
            Directory for a persistent database; defaults to in-memory. chDB is
            one engine per process: the first connection fixes the path, and
            connecting to a different path raises until all are disconnected.

        """
        import chdb

        self.con = _Con(chdb.connect(str(database)))

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
        # Materialize via the non-streaming ArrowTable format: it returns a
        # pyarrow.Table directly, leaving no open stream to poison the shared
        # connection if type conversion raises.
        result = self.con.query(sql, fmt="ArrowTable")
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
        try:
            table = self.raw_sql(query, fmt="ArrowTable")
        except Exception as e:
            if re.search(r"\bUNKNOWN_TABLE\b", str(e)):
                raise com.TableNotFound(table_name) from e
            raise
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

    # -- file readers ------------------------------------------------------
    # The inherited ClickHouse readers stream files over clickhouse_connect;
    # the embedded engine instead reads the local path via file() directly.

    def _read_file(self, path, *, table_name, fmt, engine):
        name = table_name or util.gen_name("read")
        path = sge.convert(str(path)).sql(self.dialect)
        # path is a quoted literal and name is generated, so this is safe.
        sql = f"CREATE OR REPLACE TABLE {name} ENGINE = {engine} AS SELECT * FROM file({path}, '{fmt}')"  # noqa: S608
        self.raw_sql(sql)
        return self.table(name)

    def read_parquet(
        self, path, /, *, table_name=None, engine: str = "MergeTree", **_: Any
    ) -> ir.Table:
        return self._read_file(path, table_name=table_name, fmt="Parquet", engine=engine)

    def read_csv(
        self, path, /, *, table_name=None, engine: str = "MergeTree", **_: Any
    ) -> ir.Table:
        return self._read_file(
            path, table_name=table_name, fmt="CSVWithNames", engine=engine
        )

    def insert(self, name, /, obj, *, database=None, overwrite=False, **_: Any):
        # The embedded engine has no clickhouse_connect insert transport; route
        # pandas/pyarrow/etc. through the Python() memtable path via INSERT SELECT.
        if overwrite:
            self.truncate_table(name, database=database)
        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)
        self._run_pre_execute_hooks(obj)
        query = self._build_insert_from_table(target=name, source=obj, db=database)
        self.raw_sql(query)

    # -- in-memory tables --------------------------------------------------

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        _register_memtable(op.name, op.data.to_pyarrow(op.schema))

    def _register_in_memory_tables_from_mapping(self, external_tables) -> None:
        for name, obj in (external_tables or {}).items():
            memtable = obj if isinstance(obj, ops.InMemoryTable) else obj.op()
            _register_memtable(name, memtable.data.to_pyarrow(memtable.schema))

    def _normalize_external_tables(self, external_tables=None):
        # No external-table transport: register for Python() scanning instead,
        # and ship nothing (return None).
        self._register_in_memory_tables_from_mapping(external_tables)

    def _make_memtable_finalizer(self, name: str):
        return lambda: _unregister_memtable(name)

    # -- user-defined functions -------------------------------------------

    def _register_udfs(self, expr: ir.Expr) -> None:
        import chdb

        for udf_node in expr.op().find(ops.ScalarUDF):
            if udf_node.__input_type__ != InputType.PYTHON:
                # builtins need no registration; pandas/pyarrow are unsupported.
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


# Let ibis.to_sql(expr, dialect="chdb") resolve a compiler the same way it does
# for built-in backends (getattr(compilers, name).compiler).
_compilers = importlib.import_module("ibis.backends.sql.compilers")
if not hasattr(_compilers, "chdb"):
    _compilers.chdb = types.SimpleNamespace(compiler=Backend.compiler)
