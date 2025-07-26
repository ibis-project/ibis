"""The MySQL backend."""

from __future__ import annotations

import contextlib
import warnings
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import MySQLdb
import sqlglot as sg
import sqlglot.expressions as sge
from MySQLdb import ProgrammingError
from MySQLdb.constants import ER

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
from ibis.backends.sql.compilers.base import STAR, TRUE, C, RenameTable

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
        return ".".join(map(str, self.con._server_version))

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        autocommit: bool = True,
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
        autocommit
            Autocommit mode
        kwargs
            Additional keyword arguments passed to `MySQLdb.connect`

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
        self.con = MySQLdb.connect(
            user=user,
            host="127.0.0.1" if host == "localhost" else host,
            port=port,
            password=password,
            autocommit=autocommit,
            **kwargs,
        )

        self._post_connect()

    @util.experimental
    @classmethod
    def from_connection(cls, con: MySQLdb.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to a MySQL database.

        Parameters
        ----------
        con
            An existing connection to a MySQL database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.database())) as cur:
            [(database,)] = cur.fetchall()
        return database

    def list_databases(self, *, like: str | None = None) -> list[str]:
        # In MySQL, "database" and "schema" are synonymous
        with self._safe_raw_sql("SHOW DATABASES") as cur:
            databases = list(map(itemgetter(0), cur.fetchall()))
        return self._filter_with_like(databases, like)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        from ibis.backends.mysql.datatypes import _type_from_cursor_info

        char_set_info = self.con.get_character_set_info()
        multi_byte_maximum_length = char_set_info["mbmaxlen"]

        sql = (
            sg.select(STAR)
            .from_(
                sg.parse_one(query, dialect=self.dialect).subquery(
                    sg.to_identifier(
                        util.gen_name("query_schema"), quoted=self.compiler.quoted
                    )
                )
            )
            .limit(0)
            .sql(self.dialect)
        )
        with self.begin() as cur:
            cur.execute(sql)
            descr, flags = cur.description, cur.description_flags

        items = {}
        for (name, type_code, _, _, field_length, scale, _), raw_flags in zip(
            descr, flags
        ):
            items[name] = _type_from_cursor_info(
                flags=raw_flags,
                type_code=type_code,
                field_length=field_length,
                scale=scale,
                multi_byte_maximum_length=multi_byte_maximum_length,
            )
        return sch.Schema(items)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)

        with self.begin() as cur:
            try:
                cur.execute(sge.Describe(this=table).sql(self.dialect))
            except ProgrammingError as e:
                if e.args[0] == ER.NO_SUCH_TABLE:
                    raise com.TableNotFound(name) from e
            else:
                result = cur.fetchall()

        type_mapper = self.compiler.type_mapper
        fields = {
            name: type_mapper.from_string(type_string, nullable=is_nullable == "YES")
            for name, type_string, is_nullable, *_ in result
        }

        return sch.Schema(fields)

    def create_database(self, name: str, force: bool = False) -> None:
        sql = sge.Create(
            kind="DATABASE", exists=force, this=sg.to_identifier(name)
        ).sql(self.name)
        with self.begin() as cur:
            cur.execute(sql)

    def drop_database(
        self, name: str, *, catalog: str | None = None, force: bool = False
    ) -> None:
        sql = sge.Drop(
            kind="DATABASE", exists=force, this=sg.table(name, catalog=catalog)
        ).sql(self.name)
        with self.begin() as cur:
            cur.execute(sql)

    @contextlib.contextmanager
    def begin(self):
        con = self.con
        cur = con.cursor()
        autocommit = con.get_autocommit()

        if not autocommit:
            con.begin()

        try:
            yield cur
        except Exception:
            if not autocommit:
                con.rollback()
            raise
        else:
            if not autocommit:
                con.commit()
        finally:
            cur.close()

    # TODO(kszucs): should make it an abstract method or remove the use of it
    # from .execute()
    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.raw_sql(*args, **kwargs) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

        con = self.con
        autocommit = con.get_autocommit()

        cursor = con.cursor()

        if not autocommit:
            con.begin()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            if not autocommit:
                con.rollback()
            cursor.close()
            raise
        else:
            if not autocommit:
                con.commit()
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

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

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

        with self._safe_raw_sql(sql) as cur:
            result = self._fetch_from_cursor(cur, schema)
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
        schema: sch.SchemaLike | None = None,
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

        data = df.itertuples(index=False)
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                cur.executemany(sql, data)

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

        schema = expr.as_table().schema()
        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            df = self._fetch_from_cursor(cursor, schema)
        table = pa.Table.from_pandas(
            df, schema=schema.to_pyarrow(), preserve_index=False
        )
        return table.to_reader(max_chunksize=chunk_size)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.backends.mysql.converter import MySQLPandasData

        df = pd.DataFrame.from_records(
            cursor.fetchall(), columns=schema.names, coerce_float=True
        )
        return MySQLPandasData.convert_table(df, schema)
