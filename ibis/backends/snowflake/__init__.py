from __future__ import annotations

import contextlib
import functools
import glob
import itertools
import json
import os
import tempfile
import warnings
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus
from urllib.request import urlretrieve

import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateCatalog, CanCreateDatabase
from ibis.backends.snowflake.converter import SnowflakePandasData
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import snowflake.connector
    import snowflake.snowpark


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
    "ibis_udfs.public.array_sum": {
        "inputs": {"array": "ARRAY"},
        "returns": "DOUBLE",
        "source": """\
let total = 0.0;
let allNull = true;

for (val of array) {
  if (val !== null) {
    total += val;
    allNull = false;
  }
}

return !allNull ? total : null;""",
    },
    "ibis_udfs.public.array_avg": {
        "inputs": {"array": "ARRAY"},
        "returns": "DOUBLE",
        "source": """\
let count = 0;
let total = 0.0;

for (val of array) {
  if (val !== null) {
    total += val;
    ++count;
  }
}

return count !== 0 ? total / count : null;""",
    },
    "ibis_udfs.public.array_any": {
        "inputs": {"array": "ARRAY"},
        "returns": "BOOLEAN",
        "source": """\
let count = 0;

for (val of array) {
  if (val === true) {
    return true;
  } else if (val === false) {
    ++count;
  }
}

return count !== 0 ? false : null;""",
    },
    "ibis_udfs.public.array_all": {
        "inputs": {"array": "ARRAY"},
        "returns": "BOOLEAN",
        "source": """\
let count = 0;

for (val of array) {
  if (val === false) {
    return false;
  } else if (val === true) {
    ++count;
  }
}

return count !== 0 ? true : null;""",
    },
}


class Backend(SQLBackend, CanCreateCatalog, CanCreateDatabase):
    name = "snowflake"
    compiler = sc.snowflake.compiler
    supports_python_udfs = True

    _top_level_methods = ("from_connection", "from_snowpark")

    def __init__(self, *args, _from_snowpark: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._from_snowpark = _from_snowpark

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["account"] = kwargs.pop("host")

    def _from_url(self, url: ParseResult, **kwargs):
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
        if url.path:
            database, schema = url.path[1:].split("/", 1)
            warehouse = kwargs.pop("warehouse", None)
            connect_args = {
                "user": url.username,
                "password": unquote_plus(url.password or ""),
                "account": url.hostname,
                "warehouse": warehouse,
                "database": database or "",
                "schema": schema or "",
            }
        else:
            connect_args = {}

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
    def current_database(self) -> str:
        return self.con.schema

    @property
    def current_catalog(self) -> str:
        return self.con.database

    def _make_udf(self, name: str, defn) -> str:
        signature = ", ".join(
            f"{sg.to_identifier(argname, quoted=self.compiler.quoted).sql(self.name)} {typ}"
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
            Enable object UDF extensions defined by Ibis on the first
            connection to the database.
        kwargs
            Additional arguments passed to the DBAPI connection call.
        """
        import snowflake.connector as sc

        connect_args = kwargs.copy()
        session_parameters = connect_args.pop("session_parameters", {})

        self.con = sc.connect(**connect_args)
        self._setup_session(
            session_parameters=session_parameters,
            create_object_udfs=create_object_udfs,
        )

    def _setup_session(self, *, session_parameters, create_object_udfs: bool):
        con = self.con

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

        with contextlib.closing(con.cursor()) as cur:
            cur.execute(
                "ALTER SESSION SET {}".format(
                    " ".join(f"{k} = {v!r}" for k, v in session_parameters.items())
                )
            )

        if create_object_udfs:
            dialect = self.name
            create_stmt = sge.Create(
                kind="DATABASE", this="ibis_udfs", exists=True
            ).sql(dialect)
            if "/" in con.database:
                (catalog, db) = con.database.split("/")
                use_stmt = sge.Use(
                    kind="SCHEMA",
                    this=sg.table(db, catalog=catalog, quoted=self.compiler.quoted),
                ).sql(dialect)
            else:
                use_stmt = ""

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

    @util.deprecated(as_of="10.0", instead="use from_connection instead")
    @classmethod
    def from_snowpark(
        cls, session: snowflake.snowpark.Session, *, create_object_udfs: bool = True
    ) -> Backend:
        """Create an Ibis Snowflake backend from a Snowpark session.

        Parameters
        ----------
        session
            A Snowpark session instance.
        create_object_udfs
            Enable object UDF extensions defined by Ibis on the first
            connection to the database.

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
        import snowflake.connector

        backend = cls(_from_snowpark=True)
        backend.con = session._conn._conn
        with contextlib.suppress(snowflake.connector.errors.ProgrammingError):
            # stored procs on snowflake don't allow session mutation it seems
            backend._setup_session(
                session_parameters={}, create_object_udfs=create_object_udfs
            )
        return backend

    @util.experimental
    @classmethod
    def from_connection(
        cls,
        con: snowflake.connector.SnowflakeConnection | snowflake.snowpark.Session,
        *,
        create_object_udfs: bool = True,
    ) -> Backend:
        """Create an Ibis Snowflake backend from an existing connection.

        Parameters
        ----------
        con
            A Snowflake Connector for Python connection or a Snowpark
            session instance.
        create_object_udfs
            Enable object UDF extensions defined by Ibis on the first
            connection to the database.

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
        >>> con = ibis.snowflake.from_connection(session)  # doctest: +SKIP
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
        import snowflake.connector

        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = (
            con
            if isinstance(con, snowflake.connector.SnowflakeConnection)
            else con._conn._conn
        )
        with contextlib.suppress(snowflake.connector.errors.ProgrammingError):
            # stored procs on snowflake don't allow session mutation it seems
            new_backend._setup_session(
                session_parameters={}, create_object_udfs=create_object_udfs
            )
        return new_backend

    def reconnect(self) -> None:
        if self._from_snowpark:
            raise com.IbisError(
                "Reconnecting is not supported when using a Snowpark session"
            )
        super().reconnect()

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
        chunk_size: int = 1_000_000,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        self._run_pre_execute_hooks(expr)
        sql = self.compile(expr, limit=limit, params=params)
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

        return pa.ipc.RecordBatchReader.from_batches(
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
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> Iterable[tuple[str, dt.DataType]]:
        import snowflake.connector

        # this will always show temp tables with the same name as a non-temp
        # table first
        #
        # snowflake puts temp tables in the same catalog and database as
        # non-temp tables and differentiates between them using a different
        # mechanism than other database that often put temp tables in a hidden
        # or intentionally-difficult-to-access catalog/database
        table = sg.table(
            table_name, db=database, catalog=catalog, quoted=self.compiler.quoted
        )
        query = sge.Describe(kind="TABLE", this=table)

        try:
            with self._safe_raw_sql(query) as cur:
                result = cur.fetchall()
        except snowflake.connector.errors.ProgrammingError as e:
            # apparently sqlstate codes are "standard", in the same way that
            # SQL is standard, because sqlstate codes are part of the SQL
            # standard
            #
            # Nowhere does this exist in Snowflake's documentation but this
            # exists in MariaDB's docs and matches the SQLSTATE error code
            #
            # https://mariadb.com/kb/en/sqlstate/
            # https://mariadb.com/kb/en/mariadb-error-code-reference/
            # and the least helpful version: https://docs.snowflake.com/en/developer-guide/snowflake-scripting/exceptions#handling-an-exception
            if e.sqlstate == "42S02":
                raise com.TableNotFound(table.sql(self.dialect)) from e
            raise

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(typ, nullable=nullable == "Y")
                for name, typ, _, nullable, *_ in result
            }
        )

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        dialect = self.dialect
        sql = sge.Describe(kind="RESULT", this=self.compiler.f.last_query_id()).sql(
            dialect
        )
        with self._safe_raw_sql(sg.parse_one(query, read=dialect).limit(0)) as cur:
            rows = cur.execute(sql).fetchall()

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(type_name, nullable=nullable == "Y")
                for name, type_name, _, nullable, *_ in rows
            }
        )

    def list_catalogs(self, like: str | None = None) -> list[str]:
        with self._safe_raw_sql("SHOW DATABASES") as con:
            catalogs = list(map(itemgetter(1), con))
        return self._filter_with_like(catalogs, like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if catalog is not None:
            sg_cat = sg.to_identifier(catalog, quoted=self.compiler.quoted).sql(
                self.name
            )
            query += f" IN {sg_cat}"

        with self._safe_raw_sql(query) as con:
            schemata = list(map(itemgetter(1), con))

        return self._filter_with_like(schemata, like)

    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        """List the tables in the database.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of tables is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Table location. If not passed, uses the current catalog and database.

            To specify a table in a separate Snowflake catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)

        tables_query = "SHOW TABLES"
        views_query = "SHOW VIEWS"

        if table_loc.catalog or table_loc.db:
            tables_query += f" IN {table_loc}"
            views_query += f" IN {table_loc}"

        with self.con.cursor() as cur:
            # TODO: considering doing this with a single query using information_schema
            tables = list(map(itemgetter(1), cur.execute(tables_query)))
            views = list(map(itemgetter(1), cur.execute(views_query)))

        return self._filter_with_like(tables + views, like=like)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        import pyarrow.parquet as pq

        name = op.name
        data = op.data.to_pyarrow(schema=op.schema)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = Path(tmpdir, f"{name}.parquet")
            # optimize for bandwidth so use zstd which typically compresses
            # better than the other options without much loss in speed
            pq.write_table(data, path, compression="zstd")
            self.read_parquet(path, table_name=name)

    def create_catalog(self, name: str, force: bool = False) -> None:
        current_catalog = self.current_catalog
        current_database = self.current_database
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.to_identifier(name, quoted=quoted), kind="DATABASE", exists=force
        )
        use_stmt = sge.Use(
            kind="SCHEMA",
            this=sg.table(current_database, db=current_catalog, quoted=quoted),
        ).sql(self.name)
        with self._safe_raw_sql(create_stmt) as cur:
            # Snowflake automatically switches to the new database after creating
            # it per
            # https://docs.snowflake.com/en/sql-reference/sql/create-database#general-usage-notes
            # so we switch back to the original database and schema
            cur.execute(use_stmt)

    def drop_catalog(self, name: str, force: bool = False) -> None:
        current_catalog = self.current_catalog
        if name == current_catalog:
            raise com.UnsupportedOperationError(
                "Dropping the current catalog is not supported because its behavior is undefined"
            )
        drop_stmt = sge.Drop(
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            kind="DATABASE",
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        current_catalog = self.current_catalog
        current_database = self.current_database
        quoted = self.compiler.quoted
        create_stmt = sge.Create(
            this=sg.table(name, db=catalog, quoted=quoted), kind="SCHEMA", exists=force
        )
        use_stmt = sge.Use(
            kind="SCHEMA",
            this=sg.table(current_database, db=current_catalog, quoted=quoted),
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

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if self.current_database == name and (
            catalog is None or self.current_catalog == catalog
        ):
            raise com.UnsupportedOperationError(
                "Dropping the current database is not supported because its behavior is undefined"
            )

        drop_stmt = sge.Drop(
            this=sg.table(name, db=catalog, quoted=self.compiler.quoted),
            kind="SCHEMA",
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def create_table(
        self,
        name: str,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.SchemaLike | None = None,
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
        if schema is not None:
            schema = ibis.schema(schema)

        quoted = self.compiler.quoted

        if database is None:
            target = sg.table(name, quoted=quoted)
            catalog = db = database
        else:
            db = sg.parse_one(database, into=sge.Table, read=self.name)
            catalog = db.db
            db = db.name
            target = sg.table(name, db=db, catalog=catalog, quoted=quoted)

        if schema:
            target = sge.Schema(
                this=target, expressions=schema.to_sqlglot(self.dialect)
            )

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

            query = self.compiler.to_sqlglot(table)
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

        return self.table(name, database=(catalog, db))

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
        compiler = self.compiler
        quoted = compiler.quoted
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
                    urlretrieve(path, filename=tmpname)  # noqa: S310
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
            # query in USING TEMPLATE
            query = sg.select(
                sge.WithinGroup(
                    this=sge.ArrayAgg(this=sge.StarMap(this=STAR)),
                    expression=sge.Order(
                        expressions=[sge.Ordered(this=sg.column("ORDER_ID"))]
                    ),
                )
            ).from_(
                compiler.f.anon.TABLE(
                    compiler.f.anon.INFER_SCHEMA(
                        sge.Kwarg(
                            this=compiler.v.LOCATION,
                            expression=sge.convert(f"@{stage}"),
                        ),
                        sge.Kwarg(
                            this=compiler.v.FILE_FORMAT,
                            expression=sge.convert(file_format),
                        ),
                    )
                )
            )
            (info,) = cur.execute(query.sql(self.dialect)).fetchone()
            stmts = [
                # create a temporary table using the stage and format inferred
                # from the CSV
                sge.Create(
                    kind="TABLE",
                    this=sge.Schema(
                        this=qtable,
                        expressions=[
                            sge.ColumnDef(
                                this=sg.to_identifier(
                                    field["COLUMN_NAME"], quoted=quoted
                                ),
                                kind=field["TYPE"],
                                constraints=(
                                    [sge.NotNullColumnConstraint()]
                                    if not field["NULLABLE"]
                                    else None
                                ),
                            )
                            for field in json.loads(info)
                        ],
                    ),
                    properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
                ).sql(self.dialect),
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
        quoted = self.compiler.quoted
        qtable = sg.table(table, quoted=quoted)
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

        compiler = self.compiler
        query = sg.select(
            sge.WithinGroup(
                this=sge.ArrayAgg(this=sge.StarMap(this=STAR)),
                expression=sge.Order(
                    expressions=[sge.Ordered(this=sg.column("ORDER_ID"))]
                ),
            )
        ).from_(
            compiler.f.anon.TABLE(
                compiler.f.anon.INFER_SCHEMA(
                    sge.Kwarg(
                        this=compiler.v.LOCATION,
                        expression=sge.convert(f"@{stage}"),
                    ),
                    sge.Kwarg(
                        this=compiler.v.FILE_FORMAT,
                        expression=sge.convert(file_format),
                    ),
                )
            )
        )
        with self._safe_raw_sql(";\n".join(stmts)) as cur:
            cur.execute(
                f"PUT 'file://{Path(path).absolute()}' @{stage} PARALLEL = {threads:d}"
            )
            cur.execute(
                ";\n".join(
                    [
                        f"CREATE TEMP TABLE {qtable} USING TEMPLATE ({query.sql(self.dialect)})",
                        # load the JSON file into the table
                        sge.Copy(
                            this=qtable,
                            kind=True,
                            files=[sge.Table(this=sge.Var(this=f"@{stage}"))],
                            params=[
                                sge.CopyParameter(
                                    this=self.compiler.v.MATCH_BY_COLUMN_NAME,
                                    expression=sge.convert(match_by_column_name),
                                )
                            ],
                        ).sql(self.dialect),
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
        qtable = sg.table(table, quoted=quoted)
        threads = min((os.cpu_count() or 2) // 2, 99)

        kwargs.setdefault("USE_LOGICAL_TYPE", True)
        options = " ".join(
            f"{name.upper()} = {value!r}" for name, value in kwargs.items()
        )

        type_mapper = self.compiler.type_mapper

        stmts = [
            f"CREATE TEMP STAGE {stage} FILE_FORMAT = (TYPE = PARQUET {options})",
            sge.Create(
                kind="TABLE",
                this=sge.Schema(
                    this=qtable, expressions=schema.to_sqlglot(self.dialect)
                ),
                properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
            ).sql(self.dialect),
        ]

        query = ";\n".join(stmts)

        param = sge.Parameter(this=sge.convert(1))
        copy_select = (
            sg.select(
                *(
                    sg.cast(
                        self.compiler.f.get_path(param, sge.convert(col)),
                        type_mapper.from_ibis(typ),
                    )
                    for col, typ in schema.items()
                )
            )
            .from_(sge.Table(this=sge.Var(this=f"@{stage}")))
            .subquery()
        )
        copy_query = sge.Copy(this=qtable, kind=True, files=[copy_select]).sql(
            self.dialect
        )
        with self._safe_raw_sql(query) as cur:
            cur.execute(f"PUT 'file://{abspath}' @{stage} PARALLEL = {threads:d}")
            cur.execute(copy_query)

        return self.table(table)

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.
        A collection of tables is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.
        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.

            For multi-level table hierarchies, you can pass in a dotted string
            path like `"catalog.database"` or a tuple of strings like
            `("catalog", "database")`.
        overwrite
            If `True` then replace existing contents of table

        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        self._run_pre_execute_hooks(obj)

        query = self._build_insert_from_table(
            target=table_name, source=obj, db=db, catalog=catalog
        )
        table = sg.table(
            table_name, db=db, catalog=catalog, quoted=self.compiler.quoted
        )

        statements = []
        if overwrite:
            statements.append(f"TRUNCATE TABLE {table.sql(self.name)}")
        statements.append(query.sql(self.name))

        statement = ";".join(statements)
        with self._safe_raw_sql(statement):
            pass
