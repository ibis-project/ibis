from __future__ import annotations

import contextlib
import functools
import glob
import importlib
import inspect
import itertools
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import sqlalchemy as sa
import sqlglot as sg
from packaging.version import parse as vparse
from sqlalchemy.ext.compiler import compiles

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateDatabase
from ibis.backends.base.sql.alchemy import (
    AlchemyCanCreateSchema,
    AlchemyCompiler,
    AlchemyCrossSchemaBackend,
    AlchemyExprTranslator,
)

with warnings.catch_warnings():
    if vparse(importlib.metadata.version("snowflake-connector-python")) >= vparse(
        "3.3.0"
    ):
        warnings.filterwarnings(
            "ignore",
            message="You have an incompatible version of 'pyarrow' installed",
            category=UserWarning,
        )
    from snowflake.sqlalchemy import ARRAY, DOUBLE, OBJECT, URL

    from ibis.backends.snowflake.converter import SnowflakePandasData
    from ibis.backends.snowflake.datatypes import SnowflakeType
    from ibis.backends.snowflake.registry import operation_registry

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    import pandas as pd

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
    type_mapper = SnowflakeType


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
    "ibis_udfs.public.array_sort": {
        "inputs": {"array": ARRAY},
        "returns": ARRAY,
        "source": """return array.sort();""",
    },
    "ibis_udfs.public.array_repeat": {
        # Integer inputs are not allowed because JavaScript only supports
        # doubles
        "inputs": {"value": ARRAY, "count": DOUBLE},
        "returns": ARRAY,
        "source": """return Array(count).fill(value).flat();""",
    },
}


class Backend(AlchemyCrossSchemaBackend, CanCreateDatabase, AlchemyCanCreateSchema):
    name = "snowflake"
    compiler = SnowflakeCompiler
    supports_create_or_replace = True
    supports_python_udfs = True

    _latest_udf_python_version = (3, 10)

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    @property
    def version(self) -> str:
        return self._scalar_query(sa.select(sa.func.current_version()))

    @property
    def current_schema(self) -> str:
        with self.con.connect() as con:
            return con.connection.schema

    @property
    def current_database(self) -> str:
        with self.con.connect() as con:
            return con.connection.database

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
        user: str,
        account: str,
        database: str,
        password: str | None = None,
        authenticator: str | None = None,
        connect_args: Mapping[str, Any] | None = None,
        create_object_udfs: bool = True,
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
        create_object_udfs
            Enable object UDF extensions defined by ibis on the first
            connection to the database.
        connect_args
            Additional arguments passed to the SQLAlchemy engine creation call.
        kwargs
            Additional arguments passed to the SQLAlchemy URL constructor.
            See https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy#additional-connection-parameters
            for more details
        """
        dbparams = dict(zip(("database", "schema"), database.split("/", 1)))
        if dbparams.get("schema") is None:
            raise ValueError(
                "Schema must be non-None. Pass the schema as part of the "
                f"database e.g., {dbparams['database']}/my_schema"
            )

        # snowflake-connector-python does not handle `None` for password, but
        # accepts the empty string
        url = URL(
            account=account, user=user, password=password or "", **dbparams, **kwargs
        )
        if connect_args is None:
            connect_args = {}

        session_parameters = connect_args.setdefault("session_parameters", {})

        # enable multiple SQL statements by default
        session_parameters.setdefault("MULTI_STATEMENT_COUNT", "0")
        # don't format JSON output by default
        session_parameters.setdefault("JSON_INDENT", "0")

        # overwrite session parameters that are required for ibis + snowflake
        # to work
        session_parameters.update(
            dict(
                # Use Arrow for query results
                PYTHON_CONNECTOR_QUERY_RESULT_FORMAT="ARROW",
                # JSON output must be strict for null versus undefined
                STRICT_JSON_OUTPUT="TRUE",
                # Timezone must be UTC
                TIMEZONE="UTC",
            ),
        )

        if authenticator is not None:
            connect_args.setdefault("authenticator", authenticator)

        engine = sa.create_engine(
            url, connect_args=connect_args, poolclass=sa.pool.StaticPool
        )

        @sa.event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Register UDFs on a `"connect"` event."""
            if create_object_udfs:
                with dbapi_connection.cursor() as cur:
                    database, schema = cur.execute(
                        "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()"
                    ).fetchone()
                    try:
                        cur.execute("CREATE DATABASE IF NOT EXISTS ibis_udfs")
                        # snowflake activates a database on creation, so reset
                        # it back to the original database and schema
                        cur.execute(f"USE SCHEMA {database}.{schema}")
                        for name, defn in _SNOWFLAKE_MAP_UDFS.items():
                            cur.execute(self._make_udf(name, defn))
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

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        name = type(udf_node).__name__
        signature = ", ".join(
            f"{name} {self._compile_type(arg.dtype)}"
            for name, arg in zip(udf_node.argnames, udf_node.args)
        )
        return_type = self._compile_type(udf_node.dtype)
        source = textwrap.dedent(inspect.getsource(udf_node.__func__)).strip()
        source = "\n".join(
            line for line in source.splitlines() if not line.startswith("@udf")
        )
        return dict(
            source=source,
            name=name,
            signature=signature,
            return_type=return_type,
            comment=f"Generated by ibis {ibis.__version__} using Python {platform.python_version()}",
            version=".".join(
                map(str, min(sys.version_info[:2], self._latest_udf_python_version))
            ),
        )

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
CREATE OR REPLACE TEMPORARY FUNCTION {name}({signature})
RETURNS {return_type}
LANGUAGE PYTHON
IMMUTABLE
RUNTIME_VERSION = '{version}'
COMMENT = '{comment}'
HANDLER = '{name}'
AS $$
from __future__ import annotations

from typing import *

{source}
$$""".format(
            **self._get_udf_source(udf_node)
        )

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
CREATE OR REPLACE TEMPORARY FUNCTION {name}({signature})
RETURNS {return_type}
LANGUAGE PYTHON
IMMUTABLE
RUNTIME_VERSION = '{version}'
COMMENT = '{comment}'
PACKAGES = ('pandas')
HANDLER = 'wrapper'
AS $$
from __future__ import annotations

from typing import *

import _snowflake
import pandas as pd

{source}

@_snowflake.vectorized(input=pd.DataFrame)
def wrapper(df):
    return {name}(*(col for _, col in df.items()))
$$""".format(
            **self._get_udf_source(udf_node)
        )

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
            res = target_schema.empty_table()

        res = res.rename_columns(target_schema.names).cast(target_schema)

        return expr.__pyarrow_result__(res)

    def fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if (table := cursor.cursor.fetch_arrow_all()) is None:
            table = schema.to_pyarrow().empty_table()
        df = table.to_pandas(timestamp_as_object=True)
        return SnowflakePandasData.convert_table(df, schema)

    def to_pandas_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **_: Any,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        target_schema = expr.as_table().schema()
        converter = functools.partial(
            SnowflakePandasData.convert_table, schema=target_schema
        )

        with self.begin() as con, contextlib.closing(con.execute(sql)) as cur:
            yield from map(
                expr.__pandas_result__,
                map(converter, cur.cursor.fetch_pandas_batches()),
            )

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

        return pa.RecordBatchReader.from_batches(
            target_schema,
            self._make_batch_iter(
                sql, target_schema=target_schema, chunk_size=chunk_size
            ),
        )

    def _make_batch_iter(
        self, sql: str, *, target_schema: sch.Schema, chunk_size: int
    ) -> Iterator[pa.RecordBatch]:
        with self.begin() as con, contextlib.closing(con.execute(sql)) as cur:
            yield from itertools.chain.from_iterable(
                t.rename_columns(target_schema.names)
                .cast(target_schema)
                .to_batches(max_chunksize=chunk_size)
                for t in cur.cursor.fetch_arrow_batches()
            )

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        with self.begin() as con:
            con.exec_driver_sql(query)
            result = con.exec_driver_sql("DESC RESULT last_query_id()").mappings().all()

        for field in result:
            name = field["name"]
            type_string = field["type"]
            is_nullable = field["null?"] == "Y"
            yield name, SnowflakeType.from_string(type_string, nullable=is_nullable)

    def list_databases(self, like: str | None = None) -> list[str]:
        with self.begin() as con:
            databases = [
                row["name"] for row in con.exec_driver_sql("SHOW DATABASES").mappings()
            ]
        return self._filter_with_like(databases, like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if database is not None:
            query += f" IN {self._quote(database)}"

        with self.begin() as con:
            schemata = [row["name"] for row in con.exec_driver_sql(query).mappings()]

        return self._filter_with_like(schemata, like)

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> list[str]:
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            The database (catalog) to perform the list against.
        schema
            The schema inside `database` to perform the list against.

            ::: {.callout-warning}
            ## `schema` refers to database hierarchy

            The `schema` parameter does **not** refer to the column names and
            types of `table`.
            :::
        """

        if database is not None and schema is None:
            util.warn_deprecated(
                "database",
                instead=(
                    f"{self.name} cannot list tables only using `database` specifier. "
                    "Include a `schema` argument."
                ),
                as_of="7.1",
                removed_in="8.0",
            )
            database = sg.parse_one(database, into=sg.exp.Table).sql(dialect=self.name)
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sg.exp.Table).sql(dialect=self.name)
        else:
            database = (
                sg.table(schema, db=database, quoted=True).sql(dialect=self.name)
                or None
            )

        tables_query = "SHOW TABLES"
        views_query = "SHOW VIEWS"

        if database is not None:
            tables_query += f" IN {database}"
            views_query += f" IN {database}"

        with self.begin() as con:
            # TODO: considering doing this with a single query using information_schema
            tables = [
                row["name"] for row in con.exec_driver_sql(tables_query).mappings()
            ]
            views = [row["name"] for row in con.exec_driver_sql(views_query).mappings()]

        return self._filter_with_like(tables + views, like=like)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        import pyarrow.parquet as pq

        raw_name = op.name

        with self.begin() as con:
            if con.exec_driver_sql(f"SHOW TABLES LIKE '{raw_name}'").scalar() is None:
                tmpdir = tempfile.TemporaryDirectory()
                try:
                    path = os.path.join(tmpdir.name, f"{raw_name}.parquet")
                    # optimize for bandwidth so use zstd which typically compresses
                    # better than the other options without much loss in speed
                    pq.write_table(
                        op.data.to_pyarrow(schema=op.schema), path, compression="zstd"
                    )
                    self.read_parquet(path, table_name=raw_name)
                finally:
                    with contextlib.suppress(Exception):
                        shutil.rmtree(tmpdir.name)

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"

    def create_database(self, name: str, force: bool = False) -> None:
        current_database = self.current_database
        current_schema = self.current_schema
        name = self._quote(name)
        if_not_exists = "IF NOT EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE DATABASE {if_not_exists}{name}")
            # Snowflake automatically switches to the new database after creating
            # it per
            # https://docs.snowflake.com/en/sql-reference/sql/create-database#general-usage-notes
            # so we switch back to the original database and schema
            con.exec_driver_sql(
                f"USE SCHEMA {self._quote(current_database)}.{self._quote(current_schema)}"
            )

    def drop_database(self, name: str, force: bool = False) -> None:
        current_database = self.current_database
        if name == current_database:
            raise com.UnsupportedOperationError(
                "Dropping the current database is not supported because its behavior is undefined"
            )
        name = self._quote(name)
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP DATABASE {if_exists}{name}")

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        name = ".".join(map(self._quote, filter(None, [database, name])))
        if_not_exists = "IF NOT EXISTS " * force
        current_database = self.current_database
        current_schema = self.current_schema
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA {if_not_exists}{name}")
            # Snowflake automatically switches to the new schema after creating
            # it per
            # https://docs.snowflake.com/en/sql-reference/sql/create-schema#usage-notes
            # so we switch back to the original schema
            con.exec_driver_sql(
                f"USE SCHEMA {self._quote(current_database)}.{self._quote(current_schema)}"
            )

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if self.current_schema == name and (
            database is None or self.current_database == database
        ):
            raise com.UnsupportedOperationError(
                "Dropping the current schema is not supported because its behavior is undefined"
            )

        name = ".".join(map(self._quote, filter(None, [database, name])))
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP SCHEMA {if_exists}{name}")

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        comment: str | None = None,
    ) -> ir.Table:
        """Create a table in Snowflake.

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
        comment
            Add a comment to the table
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        create_stmt = "CREATE"

        if overwrite:
            create_stmt += " OR REPLACE"

        if temp:
            create_stmt += " TEMPORARY"

        ident = self._quote(name)
        create_stmt += f" TABLE {ident}"

        if schema is not None:
            schema_sql = ", ".join(
                f"{name} {SnowflakeType.to_string(typ) + ' NOT NULL' * (not typ.nullable)}"
                for name, typ in zip(map(self._quote, schema.keys()), schema.values())
            )
            create_stmt += f" ({schema_sql})"

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compile(table).compile(
                dialect=self.con.dialect, compile_kwargs=dict(literal_binds=True)
            )
            create_stmt += f" AS {query}"

        if comment is not None:
            create_stmt += f" COMMENT '{comment}'"

        with self.begin() as con:
            con.exec_driver_sql(create_stmt)

        return self.table(name, schema=database)

    def drop_table(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        name = self._quote(name)
        # TODO: handle database quoting
        if database is not None:
            name = f"{database}.{name}"
        drop_stmt = "DROP TABLE" + (" IF EXISTS" * force) + f" {name}"
        with self.begin() as con:
            con.exec_driver_sql(drop_stmt)

    def read_csv(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Register a CSV file as a table in the Snowflake backend.

        Parameters
        ----------
        path
            A string or Path to a CSV file; globs are supported
        table_name
            Optional name for the table; if not passed, a random name will be generated
        kwargs
            Snowflake-specific file format configuration arguments. See the documentation for
            the full list of options: https://docs.snowflake.com/en/sql-reference/sql/create-file-format#type-csv

        Returns
        -------
        Table
            The table that was read from the CSV file
        """
        stage = ibis.util.gen_name("stage")
        file_format = ibis.util.gen_name("format")
        # 99 is the maximum allowed number of threads by Snowflake:
        # https://docs.snowflake.com/en/sql-reference/sql/put#optional-parameters
        threads = min((os.cpu_count() or 2) // 2, 99)
        table = table_name or ibis.util.gen_name("read_csv_snowflake")
        qtable = self._quote(table)

        parse_header = header = kwargs.pop("parse_header", True)
        skip_header = kwargs.pop("skip_header", True)

        if int(parse_header) != int(skip_header):
            raise com.IbisInputError(
                "`parse_header` and `skip_header` must match: "
                f"parse_header = {parse_header}, skip_header = {skip_header}"
            )

        options = " " * bool(kwargs) + " ".join(
            f"{name.upper()} = {value!r}" for name, value in kwargs.items()
        )

        with self.begin() as con:
            # create a temporary stage for the file
            con.exec_driver_sql(f"CREATE TEMP STAGE {stage}")

            # create a temporary file format for CSV schema inference
            create_infer_fmt = (
                f"CREATE TEMP FILE FORMAT {file_format} TYPE = CSV PARSE_HEADER = {str(header).upper()}"
                + options
            )
            con.exec_driver_sql(create_infer_fmt)

            # copy the local file to the stage
            con.exec_driver_sql(
                f"PUT 'file://{Path(path).absolute()}' @{stage} PARALLEL = {threads:d}"
            )

            # handle setting up the schema in python because snowflake is
            # broken for csv globs: it cannot parse the result of the following
            # query in  USING TEMPLATE
            fields = json.loads(
                con.exec_driver_sql(
                    f"""
                    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                    WITHIN GROUP (ORDER BY ORDER_ID ASC)
                    FROM TABLE(
                        INFER_SCHEMA(
                            LOCATION => '@{stage}',
                            FILE_FORMAT => '{file_format}'
                        )
                    )
                    """
                ).scalar()
            )
            fields = [
                (self._quote(field["COLUMN_NAME"]), field["TYPE"], field["NULLABLE"])
                for field in fields
            ]
            columns = ", ".join(
                f"{quoted_name} {typ}{' NOT NULL' * (not nullable)}"
                for quoted_name, typ, nullable in fields
            )
            # create a temporary table using the stage and format inferred
            # from the CSV
            con.exec_driver_sql(f"CREATE TEMP TABLE {qtable} ({columns})")

            # load the CSV into the table
            con.exec_driver_sql(
                f"""
                COPY INTO {qtable}
                FROM @{stage}
                FILE_FORMAT = (TYPE = CSV SKIP_HEADER = {int(header)}{options})
                """
            )

        return self.table(table)

    def read_json(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Read newline-delimited JSON into an ibis table, using Snowflake.

        Parameters
        ----------
        path
            A string or Path to a JSON file; globs are supported
        table_name
            Optional table name
        kwargs
            Additional keyword arguments. See
            https://docs.snowflake.com/en/sql-reference/sql/create-file-format#type-json
            for the full list of options.

        Returns
        -------
        Table
            An ibis table expression
        """
        stage = util.gen_name("read_json_stage")
        file_format = util.gen_name("read_json_format")
        table = table_name or util.gen_name("read_json_snowflake")
        qtable = self._quote(table)
        threads = min((os.cpu_count() or 2) // 2, 99)

        kwargs.setdefault("strip_outer_array", True)
        match_by_column_name = kwargs.pop("match_by_column_name", "case_sensitive")

        options = " " * bool(kwargs) + " ".join(
            f"{name.upper()} = {value!r}" for name, value in kwargs.items()
        )

        with self.begin() as con:
            con.exec_driver_sql(
                f"CREATE TEMP FILE FORMAT {file_format} TYPE = JSON" + options
            )

            con.exec_driver_sql(
                f"CREATE TEMP STAGE {stage} FILE_FORMAT = {file_format}"
            )
            con.exec_driver_sql(
                f"PUT 'file://{Path(path).absolute()}' @{stage} PARALLEL = {threads:d}"
            )

            con.exec_driver_sql(
                f"""
                CREATE TEMP TABLE {qtable}
                USING TEMPLATE (
                    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                    WITHIN GROUP (ORDER BY ORDER_ID ASC)
                    FROM TABLE(
                        INFER_SCHEMA(
                            LOCATION => '@{stage}',
                            FILE_FORMAT => '{file_format}'
                        )
                    )
                )
                """
            )

            # load the JSON file into the table
            con.exec_driver_sql(
                f"""
                COPY INTO {qtable}
                FROM @{stage}
                MATCH_BY_COLUMN_NAME = {str(match_by_column_name).upper()}
                """
            )

        return self.table(table)

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Read a Parquet file into an ibis table, using Snowflake.

        Parameters
        ----------
        path
            A string or Path to a Parquet file; globs are supported
        table_name
            Optional table name
        kwargs
            Additional keyword arguments. See
            https://docs.snowflake.com/en/sql-reference/sql/create-file-format#type-parquet
            for the full list of options.

        Returns
        -------
        Table
            An ibis table expression
        """
        import pyarrow.dataset as ds

        from ibis.formats.pyarrow import PyArrowSchema

        abspath = Path(path).absolute()
        schema = PyArrowSchema.to_ibis(
            ds.dataset(glob.glob(str(abspath)), format="parquet").schema
        )

        stage = util.gen_name("read_parquet_stage")
        table = table_name or util.gen_name("read_parquet_snowflake")
        qtable = self._quote(table)
        threads = min((os.cpu_count() or 2) // 2, 99)

        options = " " * bool(kwargs) + " ".join(
            f"{name.upper()} = {value!r}" for name, value in kwargs.items()
        )

        # we can't infer the schema from the format alone because snowflake
        # doesn't support logical timestamp types in parquet files
        #
        # see
        # https://community.snowflake.com/s/article/How-to-load-logical-type-TIMESTAMP-data-from-Parquet-files-into-Snowflake
        names_types = [
            (name, SnowflakeType.to_string(typ), typ.nullable, typ.is_timestamp())
            for name, typ in schema.items()
        ]
        snowflake_schema = ", ".join(
            f"{self._quote(col)} {typ}{' NOT NULL' * (not nullable)}"
            for col, typ, nullable, _ in names_types
        )
        cols = ", ".join(
            f"$1:{col}{'::VARCHAR' * is_timestamp}::{typ}"
            for col, typ, _, is_timestamp in names_types
        )

        with self.begin() as con:
            con.exec_driver_sql(
                f"CREATE TEMP STAGE {stage} FILE_FORMAT = (TYPE = PARQUET{options})"
            )
            con.exec_driver_sql(
                f"PUT 'file://{abspath}' @{stage} PARALLEL = {threads:d}"
            )
            con.exec_driver_sql(f"CREATE TEMP TABLE {qtable} ({snowflake_schema})")
            con.exec_driver_sql(
                f"COPY INTO {qtable} FROM (SELECT {cols} FROM @{stage})"
            )

        return self.table(table)


@compiles(sa.sql.Join, "snowflake")
def compile_join(element, compiler, **kw):
    """Override compilation of LATERAL joins.

    Snowflake doesn't support lateral joins with ON clauses as of
    https://docs.snowflake.com/en/release-notes/bcr-bundles/2023_04/bcr-1057
    even if they are trivial boolean literals.
    """
    result = compiler.visit_join(element, **kw)

    if element.right._is_lateral:
        return re.sub(r"^(.+) ON true$", r"\1", result, flags=re.IGNORECASE | re.DOTALL)
    return result
