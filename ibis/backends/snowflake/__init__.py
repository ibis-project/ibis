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
import shutil
import sys
import tempfile
import textwrap
import warnings
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge
from packaging.version import parse as vparse

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, CanCreateSchema
from ibis.backends.snowflake.compiler import SnowflakeCompiler
from ibis.backends.snowflake.converter import SnowflakePandasData
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.datatypes import SnowflakeType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    import pandas as pd

    import ibis.expr.schema as sch


_SNOWFLAKE_MAP_UDFS = {
    "ibis_udfs.public.object_merge": {
        "inputs": {"obj1": "OBJECT", "obj2": "OBJECT"},
        "returns": "OBJECT",
        "source": "return Object.assign(obj1, obj2)",
    },
    "ibis_udfs.public.object_values": {
        "inputs": {"obj": "OBJECT"},
        "returns": "ARRAY",
        "source": "return Object.values(obj)",
    },
    "ibis_udfs.public.array_zip": {
        "inputs": {"arrays": "ARRAY"},
        "returns": "ARRAY",
        "source": """\
const longest = arrays.reduce((a, b) => a.length > b.length ? a : b, []);
const keys = Array.from(Array(arrays.length).keys()).map(key => `f${key + 1}`);
return longest.map((_, i) => {
    return Object.assign(...keys.map((key, j) => ({[key]: arrays[j][i]})));
})""",
    },
    "ibis_udfs.public.array_repeat": {
        # Integer inputs are not allowed because JavaScript only supports
        # doubles
        "inputs": {"value": "ARRAY", "count": "DOUBLE"},
        "returns": "ARRAY",
        "source": """return Array(count).fill(value).flat();""",
    },
}


class Backend(SQLBackend, CanCreateDatabase, CanCreateSchema):
    name = "snowflake"
    compiler = SnowflakeCompiler()
    supports_python_udfs = True

    _latest_udf_python_version = (3, 10)

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    def _from_url(self, url: str, **kwargs):
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments

        Returns
        -------
        BaseBackend
            A backend instance

        """

        url = urlparse(url)
        if url.path:
            database, schema = url.path[1:].split("/", 1)
            query_params = parse_qs(url.query)
            (warehouse,) = query_params.pop("warehouse", (None,))
            connect_args = {
                "user": url.username,
                "password": url.password or "",
                "account": url.hostname,
                "warehouse": warehouse,
                "database": database or "",
                "schema": schema or "",
            }
        else:
            connect_args = {}
            query_params = {}

        for name, value in query_params.items():
            if len(value) > 1:
                connect_args[name] = value
            elif len(value) == 1:
                connect_args[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        session_parameters = kwargs.setdefault("session_parameters", {})

        session_parameters["MULTI_STATEMENT_COUNT"] = 0
        session_parameters["JSON_INDENT"] = 0
        session_parameters["PYTHON_CONNECTOR_QUERY_RESULT_FORMAT"] = "arrow_force"

        kwargs.update(connect_args)
        self._convert_kwargs(kwargs)

        if "database" in kwargs and not kwargs["database"]:
            del kwargs["database"]

        if "schema" in kwargs and not kwargs["schema"]:
            del kwargs["schema"]

        if "password" in kwargs and kwargs["password"] is None:
            kwargs["password"] = ""
        return self.connect(**kwargs)

    @property
    def version(self) -> str:
        with self._safe_raw_sql(sg.select(sg.func("current_version"))) as cur:
            (version,) = cur.fetchone()
        return version

    @property
    def current_schema(self) -> str:
        return self.con.schema

    @property
    def current_database(self) -> str:
        return self.con.database

    def _make_udf(self, name: str, defn) -> str:
        signature = ", ".join(
            "{} {}".format(
                sg.to_identifier(argname, quoted=self.compiler.quoted).sql(self.name),
                typ,
            )
            for argname, typ in defn["inputs"].items()
        )
        return_type = defn["returns"]
        return f"""\
CREATE OR REPLACE FUNCTION {name}({signature})
RETURNS {return_type}
LANGUAGE JAVASCRIPT
RETURNS NULL ON NULL INPUT
IMMUTABLE
AS
$$ {defn["source"]} $$"""

    def do_connect(self, create_object_udfs: bool = True, **kwargs: Any):
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
            Additional arguments passed to the DBAPI connection call.
        kwargs
            Additional arguments passed to the URL constructor.

        """
        with warnings.catch_warnings():
            if vparse(
                importlib.metadata.version("snowflake-connector-python")
            ) >= vparse("3.3.0"):
                warnings.filterwarnings(
                    "ignore",
                    message="You have an incompatible version of 'pyarrow' installed",
                    category=UserWarning,
                )
                import snowflake.connector as sc

        connect_args = kwargs.copy()
        session_parameters = connect_args.pop("session_parameters", {})

        # enable multiple SQL statements by default
        session_parameters.setdefault("MULTI_STATEMENT_COUNT", 0)
        # don't format JSON output by default
        session_parameters.setdefault("JSON_INDENT", 0)

        # overwrite session parameters that are required for ibis + snowflake
        # to work
        session_parameters.update(
            dict(
                # Use Arrow for query results
                PYTHON_CONNECTOR_QUERY_RESULT_FORMAT="arrow_force",
                # JSON output must be strict for null versus undefined
                STRICT_JSON_OUTPUT=True,
                # Timezone must be UTC
                TIMEZONE="UTC",
            ),
        )

        con = sc.connect(**connect_args)

        with contextlib.closing(con.cursor()) as cur:
            cur.execute(
                "ALTER SESSION SET {}".format(
                    " ".join(f"{k} = {v!r}" for k, v in session_parameters.items())
                )
            )

        if create_object_udfs:
            database = con.database
            schema = con.schema
            dialect = self.name
            create_stmt = sge.Create(
                kind="DATABASE", this="ibis_udfs", exists=True
            ).sql(dialect)
            use_stmt = sge.Use(
                kind="SCHEMA",
                this=sg.table(schema, db=database, quoted=self.compiler.quoted),
            ).sql(dialect)

            stmts = [
                create_stmt,
                # snowflake activates a database on creation, so reset it back
                # to the original database and schema
                use_stmt,
                *itertools.starmap(self._make_udf, _SNOWFLAKE_MAP_UDFS.items()),
            ]

            stmt = ";\n".join(stmts)
            with contextlib.closing(con.cursor()) as cur:
                try:
                    cur.execute(stmt)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(
                        f"Unable to create Ibis UDFs, some functionality will not work: {e}"
                    )
        self.con = con

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        name = type(udf_node).__name__
        signature = ", ".join(
            f"{name} {self.compiler.type_mapper.to_string(arg.dtype)}"
            for name, arg in zip(udf_node.argnames, udf_node.args)
        )
        return_type = SnowflakeType.to_string(udf_node.dtype)
        lines, _ = inspect.getsourcelines(udf_node.__func__)
        source = textwrap.dedent(
            "".join(
                itertools.dropwhile(
                    lambda line: not line.lstrip().startswith("def "), lines
                )
            )
        ).strip()

        config = udf_node.__config__

        preamble_lines = [*self._UDF_PREAMBLE_LINES]

        if imports := config.get("imports"):
            preamble_lines.append(f"IMPORTS = ({', '.join(map(repr, imports))})")

        packages = "({})".format(
            ", ".join(map(repr, ("pandas", *config.get("packages", ()))))
        )
        preamble_lines.append(f"PACKAGES = {packages}")

        return dict(
            source=source,
            name=name,
            func_name=udf_node.__func_name__,
            preamble="\n".join(preamble_lines).format(
                name=name,
                signature=signature,
                return_type=return_type,
                comment=f"Generated by ibis {ibis.__version__} using Python {platform.python_version()}",
                version=".".join(
                    map(str, min(sys.version_info[:2], self._latest_udf_python_version))
                ),
            ),
        )

    _UDF_PREAMBLE_LINES = (
        "CREATE OR REPLACE TEMPORARY FUNCTION {name}({signature})",
        "RETURNS {return_type}",
        "LANGUAGE PYTHON",
        "IMMUTABLE",
        "RUNTIME_VERSION = '{version}'",
        "COMMENT = '{comment}'",
    )

    def _define_udf_translation_rules(self, expr):
        """No-op, these are defined in the compiler."""

    def _register_udfs(self, expr: ir.Expr) -> None:
        udf_sources = []
        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            if sql := compile_func(udf_node):
                udf_sources.append(sql)
        if udf_sources:
            # define every udf in one execution to avoid the overhead of db
            # round trips per udf
            with self._safe_raw_sql(";\n".join(udf_sources)):
                pass

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        return """\
{preamble}
HANDLER = '{func_name}'
AS $$
from __future__ import annotations

from typing import *

{source}
$$""".format(**self._get_udf_source(udf_node))

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        template = """\
{preamble}
HANDLER = 'wrapper'
AS $$
from __future__ import annotations

from typing import *

import _snowflake
import pandas as pd

{source}

@_snowflake.vectorized(input=pd.DataFrame)
def wrapper(df):
    return {func_name}(*(col for _, col in df.items()))
$$"""
        return template.format(**self._get_udf_source(udf_node))

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        from ibis.backends.snowflake.converter import SnowflakePyArrowData

        self._run_pre_execute_hooks(expr)

        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        with self._safe_raw_sql(sql) as cur:
            res = cur.fetch_arrow_all()

        target_schema = expr.as_table().schema().to_pyarrow()
        if res is None:
            res = target_schema.empty_table()

        return expr.__pyarrow_result__(res, data_mapper=SnowflakePyArrowData)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if (table := cursor.fetch_arrow_all()) is None:
            table = schema.to_pyarrow().empty_table()
        df = table.to_pandas(timestamp_as_object=True)
        df.columns = list(schema.names)
        return SnowflakePandasData.convert_table(df, schema)

    def to_pandas_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        self._run_pre_execute_hooks(expr)
        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        target_schema = expr.as_table().schema()
        converter = functools.partial(
            SnowflakePandasData.convert_table, schema=target_schema
        )

        with self._safe_raw_sql(sql) as cur:
            yield from map(
                expr.__pandas_result__, map(converter, cur.fetch_pandas_batches())
            )

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        self._run_pre_execute_hooks(expr)
        sql = self.compile(expr, limit=limit, params=params, **kwargs)
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
        with self._safe_raw_sql(sql) as cur:
            yield from itertools.chain.from_iterable(
                t.rename_columns(target_schema.names)
                .cast(target_schema)
                .to_batches(max_chunksize=chunk_size)
                for t in cur.fetch_arrow_batches()
            )

    def get_schema(
        self, table_name: str, schema: str | None = None, database: str | None = None
    ) -> Iterable[tuple[str, dt.DataType]]:
        table = sg.table(
            table_name, db=schema, catalog=database, quoted=self.compiler.quoted
        )
        with self._safe_raw_sql(sge.Describe(kind="TABLE", this=table)) as cur:
            result = cur.fetchall()

        type_mapper = self.compiler.type_mapper
        return ibis.schema(
            {
                name: type_mapper.from_string(typ, nullable=nullable == "Y")
                for name, typ, _, nullable, *_ in result
            }
        )

    def _metadata(self, query: str) -> Iterable[tuple[str, dt.DataType]]:
        dialect = self.name
        sql = sge.Describe(kind="RESULT", this=self.compiler.f.last_query_id()).sql(
            dialect
        )
        with self._safe_raw_sql(sg.parse_one(query, read=dialect).limit(0)) as cur:
            rows = cur.execute(sql).fetchall()

        type_mapper = self.compiler.type_mapper
        return (
            (name, type_mapper.from_string(type_name, nullable=nullable == "Y"))
            for name, type_name, _, nullable, *_ in rows
        )

    def list_databases(self, like: str | None = None) -> list[str]:
        with self._safe_raw_sql("SHOW DATABASES") as con:
            databases = list(map(itemgetter(1), con))
        return self._filter_with_like(databases, like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if database is not None:
            db = sg.to_identifier(database, quoted=self.compiler.quoted).sql(self.name)
            query += f" IN {db}"

        with self._safe_raw_sql(query) as con:
            schemata = list(map(itemgetter(1), con))

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
            raise com.IbisInputError(
                f"{self.name} cannot list tables only using `database` specifier. "
                "Include a `schema` argument."
            )
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sge.Table).sql(dialect=self.name)
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

        with self.con.cursor() as cur:
            # TODO: considering doing this with a single query using information_schema
            tables = list(map(itemgetter(1), cur.execute(tables_query)))
            views = list(map(itemgetter(1), cur.execute(views_query)))

        return self._filter_with_like(tables + views, like=like)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        import pyarrow.parquet as pq

        raw_name = op.name

        with self.con.cursor() as con:
            if not con.execute(f"SHOW TABLES LIKE '{raw_name}'").fetchone():
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

    def create_database(self, name: str, force: bool = False) -> None:
        current_database = self.current_database
        current_schema = self.current_schema
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.to_identifier(name, quoted=quoted), kind="DATABASE", exists=force
        )
        use_stmt = sge.Use(
            kind="SCHEMA",
            this=sg.table(current_schema, db=current_database, quoted=quoted),
        ).sql(self.name)
        with self._safe_raw_sql(create_stmt) as cur:
            # Snowflake automatically switches to the new database after creating
            # it per
            # https://docs.snowflake.com/en/sql-reference/sql/create-database#general-usage-notes
            # so we switch back to the original database and schema
            cur.execute(use_stmt)

    def drop_database(self, name: str, force: bool = False) -> None:
        current_database = self.current_database
        if name == current_database:
            raise com.UnsupportedOperationError(
                "Dropping the current database is not supported because its behavior is undefined"
            )
        drop_stmt = sge.Drop(
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            kind="DATABASE",
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        current_database = self.current_database
        current_schema = self.current_schema
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.table(name, db=database, quoted=quoted), kind="SCHEMA", exists=force
        )
        use_stmt = sge.Use(
            kind="SCHEMA",
            this=sg.table(current_schema, db=current_database, quoted=quoted),
        ).sql(self.name)
        with self._safe_raw_sql(create_stmt) as cur:
            # Snowflake automatically switches to the new schema after creating
            # it per
            # https://docs.snowflake.com/en/sql-reference/sql/create-schema#usage-notes
            # so we switch back to the original schema
            cur.execute(use_stmt)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

        with contextlib.closing(self.raw_sql(query, **kwargs)) as cur:
            yield cur

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)
        cur = self.con.cursor()
        try:
            cur.execute(query, **kwargs)
        except Exception:
            cur.close()
            raise
        else:
            return cur

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if self.current_schema == name and (
            database is None or self.current_database == database
        ):
            raise com.UnsupportedOperationError(
                "Dropping the current schema is not supported because its behavior is undefined"
            )

        drop_stmt = sge.Drop(
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            kind="SCHEMA",
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

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

        quoted = self.compiler.quoted

        if database is None:
            target = sg.table(name, quoted=quoted)
            catalog = db = database
        else:
            db = sg.parse_one(database, into=sge.Table, read=self.name)
            catalog = db.db
            db = db.name
            target = sg.table(name, db=db, catalog=catalog, quoted=quoted)

        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(name, quoted=quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for name, typ in (schema or {}).items()
        ]

        if column_defs:
            target = sge.Schema(this=target, expressions=column_defs)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if comment is not None:
            properties.append(sge.SchemaCommentProperty(this=sge.convert(comment)))

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self._to_sqlglot(table)
        else:
            query = None

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            replace=overwrite,
            properties=sge.Properties(expressions=properties),
            expression=query,
        )

        with self._safe_raw_sql(create_stmt):
            pass

        return self.table(name, schema=db, database=catalog)

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
        quoted = self.compiler.quoted
        qtable = sg.to_identifier(table, quoted=quoted)

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

        stmts = [
            # create a temporary stage for the file
            f"CREATE TEMP STAGE {stage}",
            # create a temporary file format for CSV schema inference
            (
                f"CREATE TEMP FILE FORMAT {file_format} TYPE = CSV PARSE_HEADER = {str(header).upper()}"
                + options
            ),
        ]

        with self._safe_raw_sql(";\n".join(stmts)) as cur:
            # copy the local file to the stage
            if str(path).startswith("https://"):
                with tempfile.NamedTemporaryFile() as tmp:
                    tmpname = tmp.name
                    urlretrieve(path, filename=tmpname)
                    tmp.flush()
                    cur.execute(
                        f"PUT 'file://{tmpname}' @{stage} PARALLEL = {threads:d}"
                    )
            else:
                cur.execute(
                    f"PUT 'file://{Path(path).absolute()}' @{stage} PARALLEL = {threads:d}"
                )

            # handle setting up the schema in python because snowflake is
            # broken for csv globs: it cannot parse the result of the following
            # query in  USING TEMPLATE
            (info,) = cur.execute(
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
            ).fetchone()
            columns = ", ".join(
                "{} {}{}".format(
                    sg.to_identifier(field["COLUMN_NAME"], quoted=quoted).sql(
                        self.name
                    ),
                    field["TYPE"],
                    " NOT NULL" if not field["NULLABLE"] else "",
                )
                for field in json.loads(info)
            )
            stmts = [
                # create a temporary table using the stage and format inferred
                # from the CSV
                f"CREATE TEMP TABLE {qtable} ({columns})",
                # load the CSV into the table
                f"""
                COPY INTO {qtable}
                FROM @{stage}
                FILE_FORMAT = (TYPE = CSV SKIP_HEADER = {int(header)}{options})
                """,
            ]
            cur.execute(";\n".join(stmts))

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
        qtable = sg.to_identifier(table, quoted=self.compiler.quoted)
        threads = min((os.cpu_count() or 2) // 2, 99)

        kwargs.setdefault("strip_outer_array", True)
        match_by_column_name = kwargs.pop("match_by_column_name", "case_sensitive")

        options = " " * bool(kwargs) + " ".join(
            f"{name.upper()} = {value!r}" for name, value in kwargs.items()
        )

        stmts = [
            f"CREATE TEMP FILE FORMAT {file_format} TYPE = JSON" + options,
            f"CREATE TEMP STAGE {stage} FILE_FORMAT = {file_format}",
        ]

        with self._safe_raw_sql(";\n".join(stmts)) as cur:
            cur.execute(
                f"PUT 'file://{Path(path).absolute()}' @{stage} PARALLEL = {threads:d}"
            )
            cur.execute(
                ";\n".join(
                    [
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
                        """,
                        # load the JSON file into the table
                        f"""
                        COPY INTO {qtable}
                        FROM @{stage}
                        MATCH_BY_COLUMN_NAME = {str(match_by_column_name).upper()}
                        """,
                    ]
                )
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
        quoted = self.compiler.quoted
        qtable = sg.to_identifier(table, quoted=quoted)
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
            (
                name,
                self.compiler.type_mapper.to_string(typ),
                typ.nullable,
                typ.is_timestamp(),
            )
            for name, typ in schema.items()
        ]
        snowflake_schema = ", ".join(
            f"{sg.to_identifier(col, quoted=quoted)} {typ}{' NOT NULL' * (not nullable)}"
            for col, typ, nullable, _ in names_types
        )
        cols = ", ".join(
            f"$1:{col}{'::VARCHAR' * is_timestamp}::{typ}"
            for col, typ, _, is_timestamp in names_types
        )

        stmts = [
            f"CREATE TEMP STAGE {stage} FILE_FORMAT = (TYPE = PARQUET{options})",
            f"CREATE TEMP TABLE {qtable} ({snowflake_schema})",
        ]

        query = ";\n".join(stmts)
        with self._safe_raw_sql(query) as cur:
            cur.execute(f"PUT 'file://{abspath}' @{stage} PARALLEL = {threads:d}")
            cur.execute(f"COPY INTO {qtable} FROM (SELECT {cols} FROM @{stage})")

        return self.table(table)

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        schema: str | None = None,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        schema
            The name of the schema that the table is located in
        database
            Name of the attached database that the table is located in.
        overwrite
            If `True` then replace existing contents of table

        """
        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        table = sg.table(table_name, db=schema, catalog=database, quoted=True)
        self._run_pre_execute_hooks(obj)
        query = sg.exp.insert(
            expression=self.compile(obj),
            into=table,
            columns=[sg.to_identifier(col, quoted=True) for col in obj.columns],
            dialect=self.name,
        )

        statements = []
        if overwrite:
            statements.append(f"TRUNCATE TABLE {table.sql(self.name)}")
        statements.append(query.sql(self.name))

        statement = ";".join(statements)
        with self._safe_raw_sql(statement):
            pass
