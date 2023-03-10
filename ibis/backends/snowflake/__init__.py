from __future__ import annotations

import contextlib
import json
import warnings
from typing import Any, Iterable

import sqlalchemy as sa

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.base.sql.alchemy.query_builder import _AlchemyTableSetFormatter


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
    _always_quote_columns = True
    supports_unnest_in_select = False


class SnowflakeTableSetFormatter(_AlchemyTableSetFormatter):
    def _format_in_memory_table(self, _, ref_op, translator):
        columns = translator._schema_to_sqlalchemy_columns(ref_op.schema)
        rows = list(ref_op.data.to_frame().itertuples(index=False))
        pos_columns = [
            sa.column(f"${idx}") for idx in range(1, len(ref_op.schema.names) + 1)
        ]
        return sa.select(*pos_columns).select_from(sa.values(*columns).data(rows))


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
    quote_table_names = True

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
            elif name == "":
                return ""
            elif name.lower() == name:
                return sa.sql.quoted_name(name, quote=True)
            else:
                return name

        self.con.dialect.normalize_name = normalize_name
        return res

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
        with self.begin() as con:
            write_pandas(
                conn=con.connection.connection,
                df=op.data.to_frame(),
                table_name=op.name,
                table_type="temp",
                auto_create_table=True,
                quote_identifiers=False,
            )

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"
