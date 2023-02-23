from __future__ import annotations

import contextlib
import json
import warnings
from typing import Any, Iterable

import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)


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
        import pyarrow  # noqa: F401, ICN001
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
    supports_unnest_in_select = False


class SnowflakeCompiler(AlchemyCompiler):
    cheap_in_memory_tables = _NATIVE_ARROW
    translator_class = SnowflakeExprTranslator


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

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    @property
    def version(self) -> str:
        with self.begin() as con:
            return con.exec_driver_sql("SELECT CURRENT_VERSION()").scalar()

    def do_connect(
        self, user: str, password: str, account: str, database: str, **kwargs: Any
    ):
        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )
        url = URL(account=account, user=user, password=password, **dbparams, **kwargs)
        self.database_name = dbparams["database"]
        engine = sa.create_engine(
            url,
            connect_args={
                "converter_class": _SnowFlakeConverter,
                "session_parameters": {
                    PARAMETER_PYTHON_CONNECTOR_QUERY_RESULT_FORMAT: "JSON",
                    "STRICT_JSON_OUTPUT": "TRUE",
                },
            },
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

        return super().do_connect(engine)

    def _get_sqla_table(
        self, name: str, schema: str | None = None, **_: Any
    ) -> sa.Table:
        different_schema = schema is not None
        with self.begin() as con:
            if different_schema:
                con.exec_driver_sql(f"USE {schema}")
        try:
            inspected = self.inspector.get_columns(
                name,
                schema=schema.split(".", 1)[1] if schema is not None else schema,
            )
        finally:
            if different_schema:
                path = ".".join(self.con.url.database.split("/", 1))
                with self.begin() as con:
                    con.exec_driver_sql(f"USE {path}")
        cols = []
        identifier = name if schema is None else schema + "." + name
        with self.begin() as con:
            cur = con.exec_driver_sql(f"DESCRIBE TABLE {identifier}").mappings()
            for colname, colinfo in zip(toolz.pluck("name", cur), inspected):
                colinfo["name"] = colname
                colinfo["type_"] = colinfo.pop("type")
                cols.append(sa.Column(**colinfo, quote=True))
        return sa.Table(
            name,
            self.meta,
            *cols,
            schema=schema,
            extend_existing=True,
            keep_existing=False,
        )

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
        with self.begin() as con:
            write_pandas(
                conn=con.connection.connection,
                df=op.data.to_frame(),
                table_name=op.name,
                table_type="temp",
                auto_create_table=True,
                quote_identifiers=False,
            )
