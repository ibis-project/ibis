"""The Microsoft Sql Server backend."""

from __future__ import annotations

import contextlib
import datetime
import struct
from contextlib import closing
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import pyodbc
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
from ibis.backends import (
    CanCreateCatalog,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    PyArrowExampleLoader,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR, C

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


def datetimeoffset_to_datetime(value):
    """Convert a datetimeoffset value to a datetime.

    Adapted from https://github.com/mkleehammer/pyodbc/issues/1141
    """
    # ref: https://github.com/mkleehammer/pyodbc/issues/134#issuecomment-281739794
    year, month, day, hour, minute, second, frac, tz_hour, tz_minutes = struct.unpack(
        "<6hI2h", value
    )  # e.g., (2017, 3, 16, 10, 35, 18, 500000000, -6, 0)
    return datetime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        frac // 1000,
        datetime.timezone(datetime.timedelta(hours=tz_hour, minutes=tz_minutes)),
    )


# For testing we use the collation "Latin1_General_100_BIN2_UTF8"
# which is case-sensitive and supports UTF8.
# This allows us to (hopefully) support both case-sensitive and case-insensitive
# collations.
# It DOES mean, though, that we need to be correct in our usage of case when
# referring to system tables and views.
# So, the correct casing for the tables and views we use often (and the
# corresponding columns):
#
#
# Info schema tables:
# - INFORMATION_SCHEMA.COLUMNS
# - INFORMATION_SCHEMA.SCHEMATA
# - INFORMATION_SCHEMA.TABLES
# Temp table location: tempdb.dbo
# Catalogs: sys.databases
# Databases: sys.schemas


class Backend(
    SQLBackend,
    CanCreateCatalog,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "mssql"
    compiler = sc.mssql.compiler
    supports_create_or_replace = False

    @property
    def version(self) -> str:
        with self._safe_raw_sql("SELECT @@VERSION") as cur:
            [(version,)] = cur.fetchall()
        return version

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 1433,
        database: str | None = None,
        driver: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Connect to MSSQL database.

        Parameters
        ----------
        host
            Address of MSSQL server to connect to.
        user
            Username.  Leave blank to use Integrated Authentication.
        password
            Password.  Leave blank to use Integrated Authentication.
        port
            Port of MSSQL server to connect to.
        database
            The MSSQL database to connect to.
        driver
            ODBC Driver to use.

            On Mac and Linux this is usually 'FreeTDS'.

            On Windows, it is usually one of:

            - ODBC Driver 11 for SQL Server
            - ODBC Driver 13 for SQL Server (for both 13 and 13.1)
            - ODBC Driver 17 for SQL Server
            - ODBC Driver 18 for SQL Server

            See https://learn.microsoft.com/en-us/sql/connect/odbc/windows/system-requirements-installation-and-driver-files
        kwargs
            Additional keyword arguments to pass to PyODBC.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_MSSQL_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MSSQL_USER", "sa")
        >>> password = os.environ.get("IBIS_TEST_MSSQL_PASSWORD", "1bis_Testing!")
        >>> database = os.environ.get("IBIS_TEST_MSSQL_DATABASE", "ibis-testing")
        >>> driver = os.environ.get("IBIS_TEST_MSSQL_PYODBC_DRIVER", "FreeTDS")
        >>> con = ibis.mssql.connect(
        ...     database=database,
        ...     host=host,
        ...     user=user,
        ...     password=password,
        ...     driver=driver,
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int32
          bool_col        boolean
          tinyint_col     int16
          smallint_col    int16
          int_col         int32
          bigint_col      int64
          float_col       float32
          double_col      float64
          date_string_col string
          string_col      string
          timestamp_col   timestamp(7)
          year            int32
          month           int32
        """

        # If no user/password given, assume Windows Integrated Authentication
        # and set "Trusted_Connection" accordingly
        # see: https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/using-integrated-authentication
        if user is None and password is None:
            kwargs.setdefault("Trusted_Connection", "yes")

        if database is not None:
            # passing database=None tries to interpolate "None" into the
            # connection string and use it as a database
            kwargs["database"] = database

        if password is not None:
            password = self._escape_special_characters(password)

        self.con = pyodbc.connect(
            user=user,
            server=f"{host},{port}",
            password=password,
            driver=driver,
            **kwargs,
        )

        self._post_connect()

    @staticmethod
    def _escape_special_characters(value: str) -> str:
        return "{" + value.replace("}", "}}") + "}"

    @util.experimental
    @classmethod
    def from_connection(cls, con: pyodbc.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to a MSSQL database.

        Parameters
        ----------
        con
            An existing connection to a MSSQL database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def _post_connect(self):
        # -155 is the code for datetimeoffset
        self.con.add_output_converter(-155, datetimeoffset_to_datetime)

        with closing(self.con.cursor()) as cur:
            cur.execute("SET DATEFIRST 1")

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if url.port:
            kwargs["port"] = url.port
        if database := url.path[1:].split("/")[0]:
            kwargs["database"] = database
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        # TODO: this is brittle and should be improved. We want to be able to
        # identify if a given table is a temp table and update the search
        # location accordingly.
        if name.startswith("ibis_cache_"):
            catalog, database = ("tempdb", "dbo")
            name = "##" + name

        query = (
            sg.select(
                C.column_name,
                C.data_type,
                C.is_nullable,
                C.numeric_precision,
                C.numeric_scale,
                C.datetime_precision,
                C.character_maximum_length,
            )
            .from_(
                sg.table(
                    "COLUMNS",
                    db="INFORMATION_SCHEMA",
                    catalog=catalog or self.current_catalog,
                )
            )
            .where(
                C.table_name.eq(sge.convert(name)),
                C.table_schema.eq(sge.convert(database or self.current_database)),
            )
            .order_by(C.ordinal_position)
        )

        with self._safe_raw_sql(query) as cur:
            meta = cur.fetchall()

        if not meta:
            fqn = sg.table(name, db=database, catalog=catalog).sql(self.dialect)
            raise com.TableNotFound(fqn)

        mapping = {}
        for (
            col,
            typ,
            is_nullable,
            numeric_precision,
            numeric_scale,
            datetime_precision,
            character_maximum_length,
        ) in meta:
            newtyp = self.compiler.type_mapper.from_string(
                typ, nullable=is_nullable == "YES"
            )

            if (
                typ.lower() != "hierarchyid"
                and character_maximum_length is not None
                and character_maximum_length != -1
                and newtyp.is_string()
            ):
                newtyp = newtyp.copy(length=character_maximum_length)
            elif typ == "float":
                newcls = dt.Float64 if numeric_precision == 53 else dt.Float32
                newtyp = newcls(nullable=newtyp.nullable)
            elif newtyp.is_decimal():
                newtyp = newtyp.copy(precision=numeric_precision, scale=numeric_scale)
            elif newtyp.is_timestamp():
                newtyp = newtyp.copy(scale=datetime_precision)
            mapping[col] = newtyp

        return sch.Schema(mapping)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        # Docs describing usage of dm_exec_describe_first_result_set
        # https://learn.microsoft.com/en-us/sql/relational-databases/system-dynamic-management-views/sys-dm-exec-describe-first-result-set-transact-sql?view=sql-server-ver16
        tsql = sge.convert(str(query)).sql(self.dialect)

        # For some reason when using "Latin1_General_100_BIN2_UTF8"
        # the stored procedure `sp_describe_first_result_set` starts throwing errors about DLL loading.
        # This "dynamic management function" uses the same algorithm and allows
        # us to pre-filter the columns we want back.
        # The syntax is:
        # `sys.dm_exec_describe_first_result_set(@tsql, @params, @include_browse_information)`
        #
        # Yes, this *is* a SQL injection risk, but it's not clear how to avoid
        # that since we allow users to pass arbitrary SQL.
        #
        # SQLGlot has a bug that forces capitalization of
        # `dm_exec_describe_first_result_set`, so we can't even use its builder
        # APIs. That doesn't really solve the injection problem though.
        query = f"""
        SELECT
          name,
          is_nullable,
          system_type_name,
          precision,
          scale,
          error_number,
          error_message
        FROM sys.dm_exec_describe_first_result_set(N{tsql}, NULL, 0)
        ORDER BY column_ordinal
        """  # noqa: S608
        with self._safe_raw_sql(query) as cur:
            rows = cur.fetchall()

        schema = {}
        for (
            name,
            nullable,
            system_type_name,
            precision,
            scale,
            err_num,
            err_msg,
        ) in rows:
            if err_num is not None:
                raise com.IbisInputError(f".sql failed with message: {err_msg}")

            newtyp = self.compiler.type_mapper.from_string(
                system_type_name, nullable=nullable
            )

            if system_type_name == "float":
                newcls = dt.Float64 if precision == 53 else dt.Float32
                newtyp = newcls(nullable=newtyp.nullable)
            elif newtyp.is_decimal():
                newtyp = newtyp.copy(precision=precision, scale=scale)
            elif newtyp.is_timestamp():
                newtyp = newtyp.copy(scale=scale)

            if name is None:
                name = util.gen_name("col")

            schema[name] = newtyp

        return sch.Schema(schema)

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.db_name())) as cur:
            [(database,)] = cur.fetchall()
        return database

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        s = sg.table("databases", db="sys")

        with self._safe_raw_sql(sg.select(C.name).from_(s)) as cur:
            results = list(map(itemgetter(0), cur.fetchall()))

        return self._filter_with_like(results, like=like)

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.schema_name())) as cur:
            [(schema,)] = cur.fetchall()
        return schema

    @contextlib.contextmanager
    def begin(self):
        with contextlib.closing(self.con.cursor()) as cur:
            yield cur

    @contextlib.contextmanager
    def _ddl_begin(self):
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

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)

        with self.begin() as cur:
            cur.execute(query, *args, **kwargs)
            yield cur

    @contextlib.contextmanager
    def _safe_ddl(self, query, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)

        with self._ddl_begin() as cur:
            cur.execute(query, *args, **kwargs)
            yield cur

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)

        con = self.con
        cursor = con.cursor()

        cursor.execute(query, **kwargs)
        return cursor

    def create_catalog(self, name: str, /, *, force: bool = False) -> None:
        expr = (
            sg.select(STAR)
            .from_(sg.table("databases", db="sys"))
            .where(C.name.eq(sge.convert(name)))
        )
        stmt = sge.Create(
            kind="DATABASE", this=sg.to_identifier(name, quoted=self.compiler.quoted)
        ).sql(self.dialect)
        create_stmt = (
            f"""\
IF NOT EXISTS ({expr.sql(self.dialect)})
BEGIN
  {stmt};
END;
GO"""
            if force
            else stmt
        )
        with self._safe_ddl(create_stmt):
            pass

    def drop_catalog(self, name: str, /, *, force: bool = False) -> None:
        with self._safe_ddl(
            sge.Drop(
                kind="DATABASE",
                this=sg.to_identifier(name, quoted=self.compiler.quoted),
                exists=force,
            )
        ):
            pass

    def create_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        current_catalog = self.current_catalog
        should_switch_catalog = catalog is not None and catalog != current_catalog
        quoted = self.compiler.quoted

        expr = (
            sg.select(STAR)
            .from_(sg.table("schemas", db="sys"))
            .where(C.name.eq(sge.convert(name)))
        )
        stmt = sge.Create(
            kind="SCHEMA", this=sg.to_identifier(name, quoted=quoted)
        ).sql(self.dialect)

        create_stmt = (
            f"""\
IF NOT EXISTS ({expr.sql(self.dialect)})
BEGIN
  {stmt};
END;
GO"""
            if force
            else stmt
        )

        with self.begin() as cur:
            if should_switch_catalog:
                cur.execute(
                    sge.Use(this=sg.to_identifier(catalog, quoted=quoted)).sql(
                        self.dialect
                    )
                )

            cur.execute(create_stmt)

            if should_switch_catalog:
                cur.execute(
                    sge.Use(this=sg.to_identifier(current_catalog, quoted=quoted)).sql(
                        self.dialect
                    )
                )

    def drop_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        current_catalog = self.current_catalog
        should_switch_catalog = catalog is not None and catalog != current_catalog

        quoted = self.compiler.quoted

        with self.begin() as cur:
            if should_switch_catalog:
                cur.execute(
                    sge.Use(this=sg.to_identifier(catalog, quoted=quoted)).sql(
                        self.dialect
                    )
                )

            cur.execute(
                sge.Drop(
                    kind="SCHEMA",
                    exists=force,
                    this=sg.to_identifier(name, quoted=quoted),
                ).sql(self.dialect)
            )

            if should_switch_catalog:
                cur.execute(
                    sge.Use(this=sg.to_identifier(current_catalog, quoted=quoted)).sql(
                        self.dialect
                    )
                )

    def list_tables(
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        sql = (
            sg.select(C.table_name)
            .from_(
                sg.table(
                    "TABLES",
                    db="INFORMATION_SCHEMA",
                    catalog=catalog if catalog is not None else self.current_catalog,
                )
            )
            .where(C.table_schema.eq(sge.convert(db or self.current_database)))
            .distinct()
        )

        sql = sql.sql(self.dialect)

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        query = sg.select(C.schema_name).from_(
            sg.table(
                "SCHEMATA",
                db="INFORMATION_SCHEMA",
                catalog=catalog or self.current_catalog,
            )
        )
        with self._safe_raw_sql(query) as cur:
            results = list(map(itemgetter(0), cur.fetchall()))
        return self._filter_with_like(results, like=like)

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
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression or pandas table that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `obj` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.

            To specify a location in a separate catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.
        temp
            Whether a table is temporary or not.
            All created temp tables are "Global Temporary Tables". They will be
            created in "tempdb.dbo" and will be prefixed with "##".
        overwrite
            Whether to clobber existing data.
            `overwrite` and `temp` cannot be used together with MSSQL.

        Returns
        -------
        Table
            The table that was created.
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        if temp and overwrite:
            raise ValueError(
                "MSSQL doesn't support overwriting temp tables, create a new temp table instead."
            )

        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())
            catalog, db = None, None

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
        raw_table = sg.table(temp_name, catalog=catalog, db=db, quoted=False)
        target = sge.Schema(
            this=sg.table(
                "#" * bool(temp) + temp_name, catalog=catalog, db=db, quoted=quoted
            ),
            expressions=schema.to_sqlglot_column_defs(self.dialect),
        )

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=catalog, db=db, quoted=quoted)
        raw_this = sg.table(name, catalog=catalog, db=db, quoted=False)
        with self._safe_ddl(create_stmt) as cur:
            if query is not None:
                # You can specify that a table is temporary for the sqlglot `Create` but not
                # for the subsequent `Insert`, so we need to shove a `#` in
                # front of the table identifier.
                _table = sg.table(
                    "##" * bool(temp) + temp_name,
                    catalog=catalog,
                    db=db,
                    quoted=self.compiler.quoted,
                )
                insert_stmt = sge.Insert(this=_table, expression=query).sql(
                    self.dialect
                )
                cur.execute(insert_stmt)

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.dialect)
                )
                old = raw_table.sql(self.dialect)
                new = raw_this.sql(self.dialect)
                cur.execute(f"EXEC sp_rename '{old}', '{new}'")

        if temp:
            # If a temporary table, amend the output name/catalog/db accordingly
            name = "##" + name
            catalog = "tempdb"
            db = "dbo"

        if schema is None:
            return self.table(name, database=(catalog, db))

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=db),
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "MS SQL cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        quoted = self.compiler.quoted

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(self.dialect),
            ),
        )

        df = op.data.to_frame()
        data = df.itertuples(index=False)

        insert_stmt = self._build_insert_template(name, schema=schema, columns=True)
        with self._safe_ddl(create_stmt) as cur:
            if not df.empty:
                cur.executemany(insert_stmt, data)

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list[tuple]]:
        def process_value(value, dtype):
            return bool(value) if dtype.is_boolean() else value

        types = expr.as_table().schema().types

        for batch in super()._cursor_batches(
            expr, params=params, limit=limit, chunk_size=chunk_size
        ):
            yield [tuple(map(process_value, row, types)) for row in batch]
