from __future__ import annotations

import contextlib
import functools
import itertools
import os
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pyarrow as pa
import sqlalchemy as sa
from snowflake.connector.constants import FIELD_ID_TO_NAME
from snowflake.connector.converter import SnowflakeConverter
from snowflake.sqlalchemy import ARRAY, OBJECT, URL

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sql.alchemy import (
    AlchemyCompiler,
    AlchemyExprTranslator,
    BaseAlchemyBackend,
)
from ibis.backends.snowflake.datatypes import (
    dtype_from_snowflake,
    dtype_to_snowflake,
    parse,
)
from ibis.backends.snowflake.registry import operation_registry

if TYPE_CHECKING:
    import pandas as pd
    import snowflake.snowpark as sp

    import ibis.expr.schema as sch


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

    get_sqla_type = staticmethod(dtype_to_snowflake)
    get_ibis_type = staticmethod(dtype_from_snowflake)


class SnowflakeCompiler(AlchemyCompiler):
    cheap_in_memory_tables = True
    translator_class = SnowflakeExprTranslator


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
    "ibis_udfs.public.array_zip": {
        "inputs": {"arrays": ARRAY},
        "returns": ARRAY,
        "source": """\
const longest = arrays.reduce((a, b) => a.length > b.length ? a : b, []);
const keys = Array.from(Array(arrays.length).keys()).map(key => `f${key + 1}`);
return longest.map((_, i) => {
    return Object.assign(...keys.map((key, j) => ({[key]: arrays[j][i]})));
})""",
    },
}


class IbisSnowflakeConverter(SnowflakeConverter):
    # when using snowpark, con._paramstyle is set to qmark and is inconsistent
    # with snowflake-sqlalchemy which uses pyformat this allows strings that
    # are used in metadata queries to be used in qmark style queries which are
    # supported by snowflake-sqlalchemy (just not the default)
    def _quoted_name_to_snowflake_bindings(self, snowflake_type, value):
        return self._str_to_snowflake_bindings(snowflake_type, value)


class Backend(BaseAlchemyBackend):
    name = "snowflake"
    compiler = SnowflakeCompiler
    supports_create_or_replace = True

    _top_level_methods = ("from_snowpark",)

    @functools.cached_property
    def _pandas_converter(self):
        from ibis.backends.snowflake.converter import SnowflakePandasConverter

        return SnowflakePandasConverter

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
            return con.execute(sa.select(sa.func.current_version())).scalar()

    def _compile_sqla_type(self, typ) -> str:
        return sa.types.to_instance(typ).compile(dialect=self.con.dialect)

    def _make_udf(self, name: str, defn) -> str:
        dialect = self.con.dialect
        quote = dialect.preparer(dialect).quote_identifier
        signature = ", ".join(
            f"{quote(argname)} {self._compile_sqla_type(typ)}"
            for argname, typ in defn["inputs"].items()
        )
        return_type = self._compile_sqla_type(defn["returns"])
        return f"""\
CREATE OR REPLACE FUNCTION {name}({signature})
RETURNS {return_type}
LANGUAGE JAVASCRIPT
RETURNS NULL ON NULL INPUT
IMMUTABLE
AS
$$ {defn["source"]} $$"""

    def do_connect(
        self,
        *,
        database: str,
        user: str | None = None,
        account: str | None = None,
        password: str | None = None,
        authenticator: str | None = None,
        connect_args: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Connect to Snowflake.

        Parameters
        ----------
        user
            Username
        account
            A Snowflake organization ID and a Snowflake user ID, separated by a hyphen.
            Note that a Snowflake user ID is a separate identifier from a username.
            See https://ibis-project.org/backends/Snowflake/ for details
        database
            A Snowflake database and a Snowflake schema, separated by a `/`.
            See https://ibis-project.org/backends/Snowflake/ for details
        password
            Password. If empty or `None` then `authenticator` must be passed.
        authenticator
            String indicating authentication method. See
            https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example#connecting-with-oauth
            for details.

            Note that the authentication flow **will not take place** until a
            database connection is made. This means that
            `ibis.snowflake.connect(...)` can succeed, while subsequent API
            calls fail if the authentication fails for any reason.
        connect_args
            Additional arguments passed to the SQLAlchemy engine creation call.
        kwargs:
            Additiional arguments passed to the SQLAlchemy URL constructor.
            See https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#additional-connection-parameters
            for more details
        """
        if connect_args is None:
            connect_args = {}

        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )

        if (creator := kwargs.pop("__creator__", None)) is None:
            if authenticator is not None:
                connect_args.setdefault("authenticator", authenticator)

            # snowflake-connector-python does not handle `None` for passyword, but
            # accepts the empty string
            url = URL(
                account=account,
                user=user,
                password=password or "",
                **dbparams,
                **kwargs,
            )
            engine = sa.create_engine(
                url, connect_args=connect_args, poolclass=sa.pool.StaticPool
            )
        else:
            # we're connecting from a snowpark session most likely
            engine = sa.create_engine(
                "snowflake://",
                creator=creator,
                connect_args=connect_args,
                poolclass=sa.pool.StaticPool,
                paramstyle="qmark",
            )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Register UDFs and ibis-specific configuration on a `"connect"` event."""
            dialect = engine.dialect
            quote = dialect.preparer(dialect).quote_identifier
            with dbapi_connection.cursor() as cur:
                cur.execute(
                    """
                    ALTER SESSION SET
                        JSON_INDENT = 0
                        PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'ARROW'
                        STRICT_JSON_OUTPUT = TRUE
                        TIMEZONE = 'UTC'
                    """
                )
                (database, schema) = cur.execute(
                    "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()"
                ).fetchone()
                try:
                    cur.execute("CREATE DATABASE IF NOT EXISTS ibis_udfs")
                    for name, defn in _SNOWFLAKE_MAP_UDFS.items():
                        cur.execute(self._make_udf(name, defn))
                    cur.execute(f"USE SCHEMA {quote(database)}.{quote(schema)}")
                except Exception as e:  # noqa: BLE001
                    warnings.warn(
                        f"Unable to create map UDFs, some functionality will not work: {e}"
                    )

        super().do_connect(engine)

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
        self.database_name = self._default_database = dbparams["database"]
        self._default_schema = dbparams["schema"]

    @classmethod
    def from_snowpark(cls, session: sp.Session) -> Backend:
        """Create an Ibis Snowflake backend from a Snowpark session.

        Parameters
        ----------
        session
            A snowpark session.

        Returns
        -------
        Backend
            An Ibis Snowflake backend instance.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import snowflake.snowpark as sp  # doctest: +SKIP
        >>> session = sp.Session.builder.configs(...).create()  # doctest: +SKIP
        >>> con = ibis.snowflake.from_snowpark(session)  # doctest: +SKIP
        >>> batting = con.tables.BATTING  # doctest: +SKIP
        >>> batting[["playerID", "RBI"]].head()  # doctest: +SKIP
        ┏━━━━━━━━━━━┳━━━━━━━┓
        ┃ playerID  ┃ RBI   ┃
        ┡━━━━━━━━━━━╇━━━━━━━┩
        │ string    │ int64 │
        ├───────────┼───────┤
        │ abercda01 │     0 │
        │ addybo01  │    13 │
        │ allisar01 │    19 │
        │ allisdo01 │    27 │
        │ ansonca01 │    16 │
        └───────────┴───────┘
        """
        con = session._conn._conn
        con.converter = IbisSnowflakeConverter()
        backend = cls(
            database=f"{con.database}/{con.schema}", __creator__=lambda con=con: con
        )
        backend.reconnect()
        return backend

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **_: Any,
    ) -> pa.Table:
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        with self.begin() as con:
            res = con.execute(sql).cursor.fetch_arrow_all()

        target_schema = expr.as_table().schema().to_pyarrow()
        if res is None:
            res = pa.Table.from_pylist([], schema=target_schema)

        res = res.rename_columns(target_schema.names).cast(target_schema)

        if isinstance(expr, ir.Column):
            return res[0]
        elif isinstance(expr, ir.Scalar):
            return res[0][0]
        else:
            return res

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if (table := cursor.cursor.fetch_arrow_all()) is None:
            table = pa.Table.from_pylist([], schema=schema.to_pyarrow())
        df = table.to_pandas(timestamp_as_object=True)
        return self._pandas_converter.convert_frame(df, schema)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        target_schema = expr.as_table().schema().to_pyarrow()
        target_columns = target_schema.names

        def batch_producer(con):
            with con.begin() as c, contextlib.closing(c.execute(sql)) as cur:
                yield from itertools.chain.from_iterable(
                    t.rename_columns(target_columns)
                    .cast(target_schema)
                    .to_batches(max_chunksize=chunk_size)
                    # yields pyarrow.Table objects, which are then converted to record batches
                    for t in cur.cursor.fetch_arrow_batches()
                )

        return pa.RecordBatchReader.from_batches(
            target_schema, batch_producer(self.con)
        )

    def _get_sqla_table(
        self,
        name: str,
        schema: str | None = None,
        database: str | None = None,
        autoload: bool = True,
        **kwargs: Any,
    ) -> sa.Table:
        default_db = self._default_database
        default_schema = self._default_schema
        if schema is None:
            schema = default_schema
        *db, schema = schema.split(".")
        db = "".join(db) or database or default_db
        ident = ".".join(filter(None, (db, schema)))
        if ident:
            with self.begin() as con:
                con.exec_driver_sql(f"USE {ident}")
        try:
            result = super()._get_sqla_table(
                name, schema=schema, autoload=autoload, database=db, **kwargs
            )
        except sa.exc.NoSuchTableError:
            raise sa.exc.NoSuchTableError(name)

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
        import pyarrow.parquet as pq

        from ibis.backends.snowflake.datatypes import dtype_to_snowflake

        dialect = self.con.dialect
        quote = dialect.preparer(dialect).quote_identifier
        raw_name = op.name
        table = quote(raw_name)

        with self.begin() as con:
            if con.exec_driver_sql(f"SHOW TABLES LIKE '{raw_name}'").scalar() is None:
                # 1. create a temporary stage for holding parquet files
                stage = util.gen_name("stage")
                con.exec_driver_sql(f"CREATE TEMP STAGE {stage}")

                tmpdir = tempfile.TemporaryDirectory()
                try:
                    path = os.path.join(tmpdir.name, f"{raw_name}.parquet")
                    # optimize for bandwidth so use zstd which typically compresses
                    # better than the other options without much loss in speed
                    pq.write_table(
                        op.data.to_pyarrow(schema=op.schema), path, compression="zstd"
                    )

                    # 2. copy the parquet file into the stage
                    #
                    # disable the automatic compression to gzip because we've
                    # already compressed the data with zstd
                    #
                    # 99 is the limit on the number of threads use to upload data,
                    # who knows why?
                    con.exec_driver_sql(
                        f"""
                        PUT 'file://{path}' @{stage}
                        PARALLEL = {min((os.cpu_count() or 2) // 2, 99)}
                        AUTO_COMPRESS = FALSE
                        """
                    )
                finally:
                    with contextlib.suppress(Exception):
                        shutil.rmtree(tmpdir.name)

                # 3. create a temporary table
                schema = ", ".join(
                    "{name} {typ}".format(
                        name=quote(col),
                        typ=sa.types.to_instance(dtype_to_snowflake(typ)).compile(
                            dialect=dialect
                        ),
                    )
                    for col, typ in op.schema.items()
                )
                con.exec_driver_sql(f"CREATE TEMP TABLE {table} ({schema})")
                # 4. copy the data into the table
                columns = op.schema.names
                column_names = ", ".join(map(quote, columns))
                parquet_column_names = ", ".join(f"$1:{col}" for col in columns)
                con.exec_driver_sql(
                    f"""
                    COPY INTO {table} ({column_names})
                    FROM (SELECT {parquet_column_names} FROM @{stage})
                    FILE_FORMAT = (TYPE = PARQUET COMPRESSION = AUTO)
                    PURGE = TRUE
                    """
                )

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"
