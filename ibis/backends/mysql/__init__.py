"""The MySQL backend."""

from __future__ import annotations

import contextlib
import re
import warnings
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import numpy as np
import pymysql
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase
from ibis.backends.mysql.datatypes import _type_from_cursor_info
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers import MySQLCompiler
from ibis.backends.sql.compilers.base import STAR, TRUE, C

if TYPE_CHECKING:
    from collections.abc import Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


class Backend(SQLBackend, CanCreateDatabase):
    name = "mysql"
    compiler = MySQLCompiler()
    supports_create_or_replace = False

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
        database, *_ = url.path[1:].split("/", 1)
        connect_args = {
            "user": url.username,
            "password": unquote_plus(url.password or ""),
            "host": url.hostname,
            "database": database or "",
            "port": url.port or None,
        }

        kwargs.update(connect_args)
        self._convert_kwargs(kwargs)

        if "user" in kwargs and not kwargs["user"]:
            del kwargs["user"]

        if "host" in kwargs and not kwargs["host"]:
            del kwargs["host"]

        if "database" in kwargs and not kwargs["database"]:
            del kwargs["database"]

        if "password" in kwargs and kwargs["password"] is None:
            del kwargs["password"]

        if "port" in kwargs and kwargs["port"] is None:
            del kwargs["port"]

        return self.connect(**kwargs)

    @cached_property
    def version(self):
        matched = re.search(r"(\d+)\.(\d+)\.(\d+)", self.con.server_version)
        return ".".join(matched.groups())

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        database: str | None = None,
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
        database
            Database to connect to
        autocommit
            Autocommit mode
        kwargs
            Additional keyword arguments passed to `pymysql.connect`

        Examples
        --------
        >>> import os
        >>> import getpass
        >>> host = os.environ.get("IBIS_TEST_MYSQL_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MYSQL_USER", getpass.getuser())
        >>> password = os.environ.get("IBIS_TEST_MYSQL_PASSWORD")
        >>> database = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis_testing")
        >>> con = connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        MySQLTable[table]
          name: functional_alltypes
          schema:
            id : int32
            bool_col : int8
            tinyint_col : int8
            smallint_col : int16
            int_col : int32
            bigint_col : int64
            float_col : float32
            double_col : float64
            date_string_col : string
            string_col : string
            timestamp_col : timestamp
            year : int32
            month : int32

        """
        self.con = pymysql.connect(
            user=user,
            host=host,
            port=port,
            password=password,
            database=database,
            autocommit=autocommit,
            conv=pymysql.converters.conversions,
            **kwargs,
        )

        self._post_connect()

    @util.experimental
    @classmethod
    def from_connection(cls, con: pymysql.Connection) -> Backend:
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
        with contextlib.closing(self.con.cursor()) as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.database())) as cur:
            [(database,)] = cur.fetchall()
        return database

    def list_databases(self, like: str | None = None) -> list[str]:
        # In MySQL, "database" and "schema" are synonymous
        with self._safe_raw_sql("SHOW DATABASES") as cur:
            databases = list(map(itemgetter(0), cur.fetchall()))
        return self._filter_with_like(databases, like)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with self.begin() as cur:
            cur.execute(
                sg.select(STAR)
                .from_(
                    sg.parse_one(query, dialect=self.dialect).subquery(
                        sg.to_identifier("tmp", quoted=self.compiler.quoted)
                    )
                )
                .limit(0)
                .sql(self.dialect)
            )

            return sch.Schema(
                {
                    field.name: _type_from_cursor_info(descr, field)
                    for descr, field in zip(cur.description, cur._result.fields)
                }
            )

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)

        with self.begin() as cur:
            cur.execute(sge.Describe(this=table).sql(self.dialect))
            result = cur.fetchall()

        type_mapper = self.compiler.type_mapper
        fields = {
            name: type_mapper.from_string(type_string, nullable=is_nullable == "YES")
            for name, type_string, is_nullable, *_ in result
        }

        return sch.Schema(fields)

    def create_database(self, name: str, force: bool = False) -> None:
        sql = sge.Create(kind="DATABASE", exist=force, this=sg.to_identifier(name)).sql(
            self.name
        )
        with self.begin() as cur:
            cur.execute(sql)

    def drop_database(self, name: str, force: bool = False) -> None:
        sql = sge.Drop(kind="DATABASE", exist=force, this=sg.to_identifier(name)).sql(
            self.name
        )
        with self.begin() as cur:
            cur.execute(sql)

    @contextlib.contextmanager
    def begin(self):
        con = self.con
        cur = con.cursor()
        try:
            yield cur
        except Exception:
            con.rollback()
            raise
        else:
            con.commit()
        finally:
            cur.close()

    # TODO(kszucs): should make it an abstract method or remove the use of it
    # from .execute()
    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

        con = self.con
        cursor = con.cursor()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            con.rollback()
            cursor.close()
            raise
        else:
            con.commit()
            return cursor

    # TODO: disable positional arguments
    def list_tables(
        self,
        like: str | None = None,
        schema: str | None = None,
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
        schema
            [deprecated] The schema to perform the list against.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database (`self.current_database`).
        """
        if schema is not None:
            self._warn_schema()

        if schema is not None and database is not None:
            raise ValueError(
                "Using both the `schema` and `database` kwargs is not supported. "
                "`schema` is deprecated and will be removed in Ibis 10.0"
                "\nUse the `database` kwarg with one of the following patterns:"
                '\ndatabase="database"'
                '\ndatabase=("catalog", "database")'
                '\ndatabase="catalog.database"',
            )
        elif schema is not None:
            table_loc = schema
        elif database is not None:
            table_loc = database
        else:
            table_loc = self.current_database

        table_loc = self._to_sqlglot_table(table_loc)

        conditions = [TRUE]

        if table_loc is not None:
            if (sg_cat := table_loc.args["catalog"]) is not None:
                sg_cat.args["quoted"] = False
            if (sg_db := table_loc.args["db"]) is not None:
                sg_db.args["quoted"] = False
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
        self, expr: ir.Expr, limit: str | None = "default", **kwargs: Any
    ) -> Any:
        """Execute an expression."""

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, **kwargs)

        schema = table.schema()

        with self._safe_raw_sql(sql) as cur:
            result = self._fetch_from_cursor(cur, schema)
        return expr.__pandas_result__(result)

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
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if database is not None and database != self.current_database:
            raise com.UnsupportedOperationError(
                "Creating tables in other databases is not supported by Postgres"
            )
        else:
            database = None

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        temp_memtable_view = None
        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
                temp_memtable_view = table.op().name
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self._to_sqlglot(table)
        else:
            query = None

        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=self.compiler.quoted),
                kind=self.compiler.type_mapper.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for colname, typ in (schema or table.schema()).items()
        ]

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        table = sg.table(temp_name, catalog=database, quoted=self.compiler.quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.Insert(this=table, expression=query).sql(self.name)
                cur.execute(insert_stmt)

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.name)
                )
                cur.execute(
                    f"ALTER TABLE IF EXISTS {table.sql(self.name)} RENAME TO {this.sql(self.name)}"
                )

        if schema is None:
            # Clean up temporary memtable if we've created one
            # for in-memory reads
            if temp_memtable_view is not None:
                self.drop_table(temp_memtable_view)

            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                "MySQL cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    constraints=(
                        None
                        if typ.nullable
                        else [
                            sg.exp.ColumnConstraint(
                                kind=sg.exp.NotNullColumnConstraint()
                            )
                        ]
                    ),
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
                properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
            )
            create_stmt_sql = create_stmt.sql(self.name)

            df = op.data.to_frame()
            # nan can not be used with MySQL
            df = df.replace(np.nan, None)

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

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = MySQLPandasData.convert_table(df, schema)
        return df
