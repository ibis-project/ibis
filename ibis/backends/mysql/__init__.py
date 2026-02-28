"""The MySQL backend."""

from __future__ import annotations

import contextlib
import getpass
import math
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge
from adbc_driver_manager import dbapi as adbc_dbapi

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
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
from ibis.backends.sql.compilers.base import TRUE, C, RenameTable

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

        # Also accept database/db from kwargs for backwards compat
        if database is None:
            database = kwargs.pop("database", kwargs.pop("db", None))

        uri = f"{user}:{password}@tcp({host}:{port})/{database or ''}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.con = adbc_dbapi.connect(
                driver="mysql", db_kwargs={"uri": uri}, autocommit=True
            )

        self._post_connect()

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

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
        with self.con.cursor() as cur:
            cur.execute("SHOW DATABASES")
            table = cur.fetch_arrow_table()
        databases = table.column(0).to_pylist()
        return self._filter_with_like(databases, like)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        tmp_name = util.gen_name("mysql_schema")
        quoted_tmp = sg.to_identifier(tmp_name, quoted=self.compiler.quoted).sql(
            self.dialect
        )
        create_sql = f"CREATE TEMPORARY TABLE {quoted_tmp} AS {query} LIMIT 0"
        describe_sql = f"DESCRIBE {quoted_tmp}"
        drop_sql = f"DROP TEMPORARY TABLE IF EXISTS {quoted_tmp}"

        type_mapper = self.compiler.type_mapper
        with self.con.cursor() as cur:
            try:
                cur.execute(create_sql)
                cur.execute(describe_sql)
                result = cur.fetch_arrow_table()
            finally:
                cur.execute(drop_sql)

        fields = {}
        for i in range(result.num_rows):
            col_name = result.column(0)[i].as_py()
            type_string = result.column(1)[i].as_py()
            is_nullable = result.column(2)[i].as_py()
            fields[col_name] = type_mapper.from_string(
                type_string, nullable=is_nullable == "YES"
            )

        return sch.Schema(fields)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)

        describe_sql = sge.Describe(this=table).sql(self.dialect)
        with self.con.cursor() as cur:
            try:
                cur.execute(describe_sql)
                result = cur.fetch_arrow_table()
            except Exception as e:
                if "doesn't exist" in str(e):
                    raise com.TableNotFound(name) from e
                raise

        type_mapper = self.compiler.type_mapper
        fields = {}
        for i in range(result.num_rows):
            col_name = result.column(0)[i].as_py()
            type_string = result.column(1)[i].as_py()
            is_nullable = result.column(2)[i].as_py()
            fields[col_name] = type_mapper.from_string(
                type_string, nullable=is_nullable == "YES"
            )

        return sch.Schema(fields)

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
        else:
            table_loc = sge.Table(
                db=sg.to_identifier(self.current_database, quoted=self.compiler.quoted),
                catalog=None,
            )

        conditions = [TRUE]

        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
        if table_loc.catalog or table_loc.db:
            conditions = [C.table_schema.eq(sge.convert(table_loc.sql(self.name)))]

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql(self.name)
        )

        with self.con.cursor() as cur:
            cur.execute(sql)
            table = cur.fetch_arrow_table()

        return self._filter_with_like(table.column(0).to_pylist(), like)

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
        else:
            df.columns = list(schema.names)
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

        name = op.name
        quoted = self.compiler.quoted
        dialect = self.dialect

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(dialect),
            ),
            properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
        )
        create_stmt_sql = create_stmt.sql(dialect)

        df = op.data.to_frame()
        # nan can not be used with MySQL
        df = df.replace(float("nan"), None)

        insert_sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="?"
        )
        with self.con.cursor() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                for row in df.itertuples(index=False):
                    # Convert values: replace NaN/None with None, handle types
                    values = []
                    for v in row:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            values.append(None)
                        else:
                            values.append(v)
                    cur.execute(insert_sql, values)

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
        self._run_pre_execute_hooks(expr)

        sql = self.compile(expr, limit=limit, params=params)
        cur = self.con.cursor()
        cur.execute(sql)
        return cur.fetch_record_batch()
