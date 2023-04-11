from __future__ import annotations

import contextlib
import itertools
import json
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.base.sql.alchemy.query_builder import _AlchemyTableSetFormatter

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    import ibis.expr.schema as sch


@contextlib.contextmanager
def _handle_pyarrow_warning(*, action: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action,
            message="You have an incompatible version of 'pyarrow' installed",
            category=UserWarning,
        )
        yield


with _handle_pyarrow_warning(action="error"):
    try:
        import pyarrow  # noqa: ICN001
    except ImportError:
        _NATIVE_ARROW = False
    else:
        try:
            import snowflake.connector  # noqa: F401
        except UserWarning:
            _NATIVE_ARROW = False
        else:
            _NATIVE_ARROW = True


with _handle_pyarrow_warning(action="ignore"):
    from snowflake.connector.constants import (
        FIELD_ID_TO_NAME,
        PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT,
    )
    from snowflake.connector.converter import (
        SnowflakeConverter as _BaseSnowflakeConverter,
    )
    from snowflake.connector.pandas_tools import (
        write_pandas,
    )
    from snowflake.sqlalchemy import ARRAY, OBJECT, URL

from ibis.backends.snowflake.datatypes import parse  # noqa: E402
from ibis.backends.snowflake.registry import operation_registry  # noqa: E402


class SnowflakeExprTranslator(AlchemyExprTranslator):
    _registry = operation_registry
    _rewrites = AlchemyExprTranslator._rewrites.copy()
    _has_reduction_filter_syntax = False
    _forbids_frame_clause = (
        *AlchemyExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
    )
    _require_order_by = (*AlchemyExprTranslator._require_order_by, ops.Reduction)
    _dialect_name = "snowflake"
    _quote_column_names = True
    _quote_table_names = True
    supports_unnest_in_select = False


class SnowflakeTableSetFormatter(_AlchemyTableSetFormatter):
    def _format_in_memory_table(self, op, ref_op, translator):
        if _NATIVE_ARROW:
            return super()._format_in_memory_table(op, ref_op, translator)

        import ibis

        schema = ref_op.schema
        selects = (
            sa.select(
                *(
                    translator.translate(ibis.literal(col, typ).op()).label(name)
                    for col, (name, typ) in zip(row, schema.items())
                )
            )
            for row in ref_op.data.to_frame().itertuples(index=False)
        )
        return sa.union_all(*selects)


class SnowflakeCompiler(AlchemyCompiler):
    cheap_in_memory_tables = _NATIVE_ARROW
    translator_class = SnowflakeExprTranslator
    table_set_formatter_class = SnowflakeTableSetFormatter


class _SnowFlakeConverter(_BaseSnowflakeConverter):
    def _VARIANT_to_python(self, _):
        return json.loads

    _ARRAY_to_python = _OBJECT_to_python = _VARIANT_to_python


_SNOWFLAKE_MAP_UDFS = {
    "ibis_udfs.public.object_merge": {
        "inputs": {"obj1": OBJECT, "obj2": OBJECT},
        "returns": OBJECT,
        "source": "return Object.assign(obj1, obj2)",
    },
    "ibis_udfs.public.object_values": {
        "inputs": {"obj": OBJECT},
        "returns": ARRAY,
        "source": "return Object.values(obj)",
    },
    "ibis_udfs.public.object_from_arrays": {
        "inputs": {"ks": ARRAY, "vs": ARRAY},
        "returns": OBJECT,
        "source": "return Object.assign(...ks.map((k, i) => ({[k]: vs[i]})))",
    },
}


def _make_udf(name, defn, *, quote) -> str:
    signature = ", ".join(
        f'{quote(argname)} {sa.types.to_instance(typ)}'
        for argname, typ in defn["inputs"].items()
    )
    return f"""\
CREATE FUNCTION IF NOT EXISTS {name}({signature})
RETURNS {sa.types.to_instance(defn["returns"])}
LANGUAGE JAVASCRIPT
AS
$$ {defn["source"]} $$"""


class Backend(BaseAlchemyBackend):
    name = "snowflake"
    compiler = SnowflakeCompiler

    @property
    def _current_schema(self) -> str:
        with self.begin() as con:
            return con.execute(sa.select(sa.func.current_schema())).scalar()

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    @property
    def version(self) -> str:
        with self.begin() as con:
            return con.exec_driver_sql("SELECT CURRENT_VERSION()").scalar()

    def do_connect(
        self,
        user: str,
        password: str,
        account: str,
        database: str,
        connect_args: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )
        url = URL(account=account, user=user, password=password, **dbparams, **kwargs)
        self.database_name = dbparams["database"]
        if connect_args is None:
            connect_args = {}
        connect_args.setdefault("converter_class", _SnowFlakeConverter)
        connect_args.setdefault(
            "session_parameters",
            {
                PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT: "JSON",
                "STRICT_JSON_OUTPUT": "TRUE",
            },
        )
        self._default_connector_format = connect_args["session_parameters"].setdefault(
            PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT, "JSON"
        )
        engine = sa.create_engine(
            url,
            connect_args=connect_args,
            poolclass=sa.pool.StaticPool,
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Register UDFs on a `"connect"` event."""
            dialect = engine.dialect
            quote = dialect.preparer(dialect).quote_identifier
            with dbapi_connection.cursor() as cur:
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                (database, schema) = cur.execute(
                    "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()"
                ).fetchone()
                try:
                    cur.execute("CREATE DATABASE IF NOT EXISTS ibis_udfs")
                    for name, defn in _SNOWFLAKE_MAP_UDFS.items():
                        cur.execute(_make_udf(name, defn, quote=quote))
                    cur.execute(f"USE SCHEMA {quote(database)}.{quote(schema)}")
                except Exception as e:  # noqa: BLE001
                    warnings.warn(
                        f"Unable to create map UDFs, some functionality will not work: {e}"
                    )

        res = super().do_connect(engine)

        def normalize_name(name):
            if name is None:
                return None
            elif not name:
                return ""
            elif name.lower() == name:
                return sa.sql.quoted_name(name, quote=True)
            else:
                return name

        self.con.dialect.normalize_name = normalize_name
        return res

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        if not _NATIVE_ARROW:
            return super().to_pyarrow(expr, params=params, limit=limit, **kwargs)

        import pyarrow as pa

        self._register_in_memory_tables(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        with self.begin() as con:
            con.exec_driver_sql(
                f"ALTER SESSION SET {PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT} = 'ARROW'"
            )
            res = con.execute(sql).cursor.fetch_arrow_all()
            con.exec_driver_sql(
                f"ALTER SESSION SET {PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT} = {self._default_connector_format!r}"
            )

        target_schema = expr.as_table().schema().to_pyarrow()
        if res is None:
            res = pa.Table.from_pylist([], schema=target_schema)

        if not res.schema.equals(target_schema, check_metadata=False):
            res = res.rename_columns(target_schema.names).cast(target_schema)

        if isinstance(expr, ir.Column):
            return res[expr.get_name()]
        elif isinstance(expr, ir.Scalar):
            return res[expr.get_name()][0]
        return res

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if _NATIVE_ARROW and self._default_connector_format == "ARROW":
            return cursor.cursor.fetch_pandas_all()
        return super().fetch_from_cursor(cursor, schema)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1000000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        if not _NATIVE_ARROW:
            return super().to_pyarrow_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size, **kwargs
            )

        import pyarrow as pa

        self._register_in_memory_tables(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        con = self.con.connect()
        con.exec_driver_sql(
            f"ALTER SESSION SET {PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT} = 'ARROW'"
        )
        cursor = con.execute(sql)
        con.exec_driver_sql(
            f"ALTER SESSION SET {PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT} = {self._default_connector_format!r}"
        )
        raw_cursor = cursor.cursor
        target_schema = expr.as_table().schema().to_pyarrow()
        target_columns = target_schema.names
        reader = pa.RecordBatchReader.from_batches(
            target_schema,
            itertools.chain.from_iterable(
                (
                    t.rename_columns(target_columns)
                    .cast(target_schema)
                    .to_batches(max_chunksize=chunk_size)
                )
                for t in raw_cursor.fetch_arrow_batches()
            ),
        )

        def close(cursor=cursor, con=con):
            cursor.close()
            con.close()

        weakref.finalize(reader, close)
        return reader

    def _get_sqla_table(
        self,
        name: str,
        schema: str | None = None,
        database: str | None = None,
        autoload: bool = True,
        **kwargs: Any,
    ) -> sa.Table:
        default_db, default_schema = self.con.url.database.split("/", 1)
        if schema is None:
            schema = default_schema
        *db, schema = schema.split(".")
        db = "".join(db) or database or default_db
        ident = ".".join(filter(None, (db, schema)))
        if ident:
            with self.begin() as con:
                con.exec_driver_sql(f"USE {ident}")
        result = super()._get_sqla_table(
            name, schema=schema, autoload=autoload, database=db, **kwargs
        )

        with self.begin() as con:
            con.exec_driver_sql(f"USE {default_db}.{default_schema}")
        result.schema = ident
        return result

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        with self.begin() as con, con.connection.cursor() as cur:
            result = cur.describe(query)

        for name, type_code, _, _, precision, scale, is_nullable in result:
            if precision is not None and scale is not None:
                typ = dt.Decimal(precision=precision, scale=scale, nullable=is_nullable)
            else:
                typ = parse(FIELD_ID_TO_NAME[type_code]).copy(nullable=is_nullable)
            yield name, typ

    def list_databases(self, like=None) -> list[str]:
        with self.begin() as con:
            databases = con.exec_driver_sql(
                "SELECT database_name FROM information_schema.databases"
            ).scalars()
        return self._filter_with_like(databases, like)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        df = op.data.to_frame()
        with self.begin() as con:
            write_pandas(
                conn=con.connection.connection,
                df=df,
                table_name=op.name,
                table_type="temp",
                auto_create_table=True,
                quote_identifiers=True,
            )

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"
