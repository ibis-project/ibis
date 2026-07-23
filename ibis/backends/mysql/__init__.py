"""The MySQL backend."""

from __future__ import annotations

import contextlib
import getpass
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge
from adbc_driver_manager import dbapi as adbc_dbapi

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import RenameTable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "mysql"
    compiler = sc.mysql.compiler
    supports_create_or_replace = False

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database := url.path[1:].split("/", 1)[0]:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    @cached_property
    def version(self):
        with self.con.cursor() as cur:
            cur.execute("SELECT VERSION()")
            result = cur.fetch_arrow_table()
        return result.column(0)[0].as_py()

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        database: str | None = None,
        autocommit: Literal[True] = True,
        **kwargs,
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port
        database
            Database to connect to
        autocommit
            Whether to use autocommit mode. Only ``True`` is supported at this
            time due to a limitation of the ADBC MySQL driver.
        kwargs
            Additional keyword arguments

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_MYSQL_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MYSQL_USER", "ibis")
        >>> password = os.environ.get("IBIS_TEST_MYSQL_PASSWORD", "ibis")
        >>> database = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis-testing")
        >>> con = ibis.mysql.connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int32
          bool_col        int8
          tinyint_col     int8
          smallint_col    int16
          int_col         int32
          bigint_col      int64
          float_col       float32
          double_col      float64
          date_string_col string
          string_col      string
          timestamp_col   timestamp
          year            int32
          month           int32
        """
        user = user or getpass.getuser()
        host = "127.0.0.1" if host == "localhost" else host
        password = password or ""

        # Also accept db from kwargs for backwards compat
        if database is None:
            db = kwargs.pop("db", None)
            if db is not None:
                warnings.warn(
                    "Passing `db` is deprecated, use `database` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                database = db

        autocommit = bool(autocommit)
        if not autocommit:
            raise ValueError(
                "The MySQL backend only supports `autocommit=True` at this time. "
                "See https://github.com/ibis-project/ibis/pull/11958 for details."
            )

        uri = f"{user}:{password}@tcp({host}:{port})/{database or ''}"
        self.con = adbc_dbapi.connect(
            driver="mysql", db_kwargs={"uri": uri}, autocommit=autocommit
        )

        self._post_connect()

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @classmethod
    def from_connection(cls, con: adbc_dbapi.Connection, /, **kwargs) -> Backend:
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def disconnect(self) -> None:
        self.con.close()

    @property
    def current_database(self) -> str:
        sql = sg.select(self.compiler.f.database()).sql(self.dialect)
        with self.con.cursor() as cur:
            cur.execute(sql)
            table = cur.fetch_arrow_table()
        return table.column(0)[0].as_py()

    def list_databases(self, *, like: str | None = None) -> list[str]:
        # In MySQL, "database" and "schema" are synonymous
        result = self.con.adbc_get_objects(depth="catalogs").read_all()
        databases = result.column("catalog_name").to_pylist()
        return self._filter_with_like(databases, like)

    @staticmethod
    def _schema_from_adbc_execute_schema(pyarrow_schema) -> sch.Schema:
        from ibis.formats.pyarrow import PyArrowType

        fields = {}
        for field in pyarrow_schema:
            meta = {k.decode(): v.decode() for k, v in (field.metadata or {}).items()}
            db_type = meta.get("sql.database_type_name", "")

            if db_type.startswith("UNSIGNED"):
                base = db_type.removeprefix("UNSIGNED ").lower()
                fields[field.name] = sc.mysql.MySQLType.from_string(
                    f"{base} unsigned", nullable=field.nullable
                )
            elif db_type == "DECIMAL":
                p = int(meta["sql.precision"])
                s = int(meta["sql.scale"])
                fields[field.name] = dt.Decimal(p, s, nullable=field.nullable)
            elif db_type in ("DATETIME", "TIMESTAMP"):
                scale = int(meta.get("sql.fractional_seconds_precision", 0))
                tz = "UTC" if db_type == "TIMESTAMP" else None
                fields[field.name] = dt.Timestamp(
                    timezone=tz, scale=scale or None, nullable=field.nullable
                )
            elif db_type == "YEAR":
                fields[field.name] = dt.UInt8(nullable=field.nullable)
            elif db_type == "SET":
                fields[field.name] = dt.Array(dt.string, nullable=field.nullable)
            else:
                fields[field.name] = PyArrowType.to_ibis(field.type, field.nullable)

        return sch.Schema(fields)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with self.con.cursor() as cur:
            pyarrow_schema = cur.adbc_execute_schema(str(query))
        return self._schema_from_adbc_execute_schema(pyarrow_schema)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        )
        query = sg.select("*").from_(table).sql(self.dialect)
        try:
            return self._get_schema_using_query(query)
        except Exception as e:
            if getattr(e, "sqlstate", None) == "42S02":
                raise com.TableNotFound(name) from e
            raise

    def create_database(self, name: str, force: bool = False) -> None:
        sql = sge.Create(
            kind="DATABASE", exists=force, this=sg.to_identifier(name)
        ).sql(self.name)
        with self.con.cursor() as cur:
            cur.execute(sql)

    def drop_database(
        self, name: str, *, catalog: str | None = None, force: bool = False
    ) -> None:
        sql = sge.Drop(
            kind="DATABASE", exists=force, this=sg.table(name, catalog=catalog)
        ).sql(self.name)
        with self.con.cursor() as cur:
            cur.execute(sql)

    @contextlib.contextmanager
    def begin(self):
        cur = self.con.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.raw_sql(*args, **kwargs) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

        cursor = self.con.cursor()
        try:
            cursor.execute(query, **kwargs)
        except Exception:
            cursor.close()
            raise
        else:
            return cursor

    # TODO: disable positional arguments
    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        if database is not None:
            table_loc = self._to_sqlglot_table(database)
            # In MySQL, catalog and db are both "database"
            catalog = table_loc.catalog or table_loc.db
        else:
            catalog = self.current_database

        result = self.con.adbc_get_objects(
            depth="tables", catalog_filter=catalog
        ).read_all()
        catalogs = result.to_pydict()
        tables = [
            table["table_name"]
            for schemas in catalogs.get("catalog_db_schemas", [])
            if schemas is not None
            for schema in schemas
            for table in schema.get("db_schema_tables") or []
        ]

        return self._filter_with_like(tables, like)

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments
        """

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params, **kwargs)

        schema = table.schema()

        with self.con.cursor() as cur:
            cur.execute(sql)
            arrow_table = cur.fetch_arrow_table()

        import pandas as pd

        from ibis.formats.pandas import PandasData

        df = arrow_table.to_pandas(timestamp_as_object=False)
        if df.empty:
            df = pd.DataFrame(columns=schema.names)
        result = PandasData.convert_table(df, schema)
        return expr.__pandas_result__(result)

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.IntoSchema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        table_expr = sg.table(temp_name, catalog=database, quoted=quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot_column_defs(dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE", this=target, properties=sge.Properties(expressions=properties)
        )

        this = sg.table(name, catalog=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite:
                cur.execute(sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect))
                cur.execute(
                    sge.Alter(
                        kind="TABLE",
                        this=table_expr,
                        exists=True,
                        actions=[RenameTable(this=this)],
                    ).sql(dialect)
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "MySQL cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        arrow_table = op.data.to_pyarrow(schema)

        with self.con.cursor() as cur:
            cur.adbc_ingest(op.name, arrow_table, mode="create", temporary=True)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        import pyarrow as pa

        self._run_pre_execute_hooks(expr)

        table_expr = expr.as_table()
        sql = self.compile(table_expr, limit=limit, params=params)
        target_schema = table_expr.schema().to_pyarrow()

        cur = self.raw_sql(sql)
        reader = cur.fetch_record_batch()

        def batch_producer():
            try:
                for batch in reader:
                    yield batch.rename_columns(target_schema.names)
            finally:
                cur.close()

        return pa.ipc.RecordBatchReader.from_batches(target_schema, batch_producer())
