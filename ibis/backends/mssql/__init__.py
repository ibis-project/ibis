"""The Microsoft Sql Server backend."""

from __future__ import annotations

import contextlib
import datetime
import struct
from contextlib import closing
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import pyodbc
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateCatalog, CanCreateDatabase, CanCreateSchema, NoUrl
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers import MSSQLCompiler
from ibis.backends.sql.compilers.base import STAR, C

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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


class Backend(SQLBackend, CanCreateCatalog, CanCreateDatabase, CanCreateSchema, NoUrl):
    name = "mssql"
    compiler = MSSQLCompiler()
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
        """

        # If no user/password given, assume Windows Integrated Authentication
        # and set "Trusted_Connection" accordingly
        # see: https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/using-integrated-authentication
        if user is None and password is None:
            kwargs.setdefault("Trusted_Connection", "yes")

        self.con = pyodbc.connect(
            user=user,
            server=f"{host},{port}",
            password=password,
            database=database,
            driver=driver,
            **kwargs,
        )

        self._post_connect()

    @util.experimental
    @classmethod
    def from_connection(cls, con: pyodbc.Connection) -> Backend:
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

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        # TODO: this is brittle and should be improved. We want to be able to
        # identify if a given table is a temp table and update the search
        # location accordingly.
        if name.startswith("ibis_cache_"):
            catalog, database = ("tempdb", "dbo")
            name = "##" + name
        conditions = [sg.column("table_name").eq(sge.convert(name))]

        if database is not None:
            conditions.append(sg.column("table_schema").eq(sge.convert(database)))

        query = (
            sg.select(
                "column_name",
                "data_type",
                "is_nullable",
                "numeric_precision",
                "numeric_scale",
                "datetime_precision",
            )
            .from_(
                sg.table(
                    "columns",
                    db="information_schema",
                    catalog=catalog or self.current_catalog,
                )
            )
            .where(*conditions)
            .order_by("ordinal_position")
        )

        with self._safe_raw_sql(query) as cur:
            meta = cur.fetchall()

        if not meta:
            fqn = sg.table(name, db=database, catalog=catalog).sql(self.dialect)
            raise com.IbisError(f"Table not found: {fqn}")

        mapping = {}
        for (
            col,
            typ,
            is_nullable,
            numeric_precision,
            numeric_scale,
            datetime_precision,
        ) in meta:
            newtyp = self.compiler.type_mapper.from_string(
                typ, nullable=is_nullable == "YES"
            )

            if typ == "float":
                newcls = dt.Float64 if numeric_precision == 53 else dt.Float32
                newtyp = newcls(nullable=newtyp.nullable)
            elif newtyp.is_decimal():
                newtyp = newtyp.copy(precision=numeric_precision, scale=numeric_scale)
            elif newtyp.is_timestamp():
                newtyp = newtyp.copy(scale=datetime_precision)
            mapping[col] = newtyp

        return sch.Schema(mapping)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        tsql = sge.convert(str(query)).sql(self.dialect)
        query = f"EXEC sp_describe_first_result_set @tsql = N{tsql}"
        with self._safe_raw_sql(query) as cur:
            rows = cur.fetchall()

        schema = {}
        for (
            _,
            _,
            name,
            nullable,
            _,
            system_type_name,
            _,
            precision,
            scale,
            *_,
        ) in sorted(rows, key=itemgetter(1)):
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

            schema[name] = newtyp

        return sch.Schema(schema)

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.db_name())) as cur:
            [(database,)] = cur.fetchall()
        return database

    def list_catalogs(self, like: str | None = None) -> list[str]:
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

    def create_catalog(self, name: str, force: bool = False) -> None:
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

    def drop_catalog(self, name: str, force: bool = False) -> None:
        with self._safe_ddl(
            sge.Drop(
                kind="DATABASE",
                this=sg.to_identifier(name, quoted=self.compiler.quoted),
                exists=force,
            )
        ):
            pass

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
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
        self, name: str, catalog: str | None = None, force: bool = False
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
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
        schema: str | None = None,
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

            To specify a table in a separate catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.
        schema
            [deprecated] The schema inside `database` to perform the list against.
        """
        table_loc = self._warn_and_create_table_loc(database, schema)
        catalog, db = self._to_catalog_db_tuple(table_loc)
        conditions = []

        if table_loc is not None:
            conditions.append(C.table_schema.eq(sge.convert(db)))

        sql = (
            sg.select("table_name")
            .from_(
                sg.table(
                    "tables",
                    db="information_schema",
                    catalog=catalog if catalog is not None else self.current_catalog,
                )
            )
            .distinct()
        )

        if conditions:
            sql = sql.where(*conditions)

        sql = sql.sql(self.dialect)

        with self._safe_raw_sql(sql) as cur:
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        query = sg.select(C.schema_name).from_(
            sg.table(
                "schemata",
                db="information_schema",
                catalog=catalog or self.current_catalog,
            )
        )
        with self._safe_raw_sql(query) as cur:
            results = list(map(itemgetter(0), cur.fetchall()))
        return self._filter_with_like(results, like=like)

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
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
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

        table = sg.table(
            "#" * temp + temp_name, catalog=catalog, db=db, quoted=self.compiler.quoted
        )
        raw_table = sg.table(temp_name, catalog=catalog, db=db, quoted=False)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        this = sg.table(name, catalog=catalog, db=db, quoted=self.compiler.quoted)
        raw_this = sg.table(name, catalog=catalog, db=db, quoted=False)
        with self._safe_ddl(create_stmt) as cur:
            if query is not None:
                # You can specify that a table is temporary for the sqlglot `Create` but not
                # for the subsequent `Insert`, so we need to shove a `#` in
                # front of the table identifier.
                _table = sg.table(
                    "##" * temp + temp_name,
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

        if schema is None:
            # Clean up temporary memtable if we've created one
            # for in-memory reads
            if temp_memtable_view is not None:
                self.drop_table(temp_memtable_view)
            return self.table(
                "##" * temp + name,
                database=("tempdb" * temp or catalog, "dbo" * temp or db),
            )

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=db),
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                "MS SQL cannot yet reliably handle `null` typed columns; "
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
                # properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
            )

            df = op.data.to_frame()
            data = df.itertuples(index=False)

            insert_stmt = self._build_insert_template(name, schema=schema, columns=True)
            with self._safe_ddl(create_stmt) as cur:
                if not df.empty:
                    cur.executemany(insert_stmt, data)

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
        table_expr = expr.as_table()
        conversions = {
            name: ibis.ifelse(table_expr[name], 1, 0).cast("boolean")
            for name, typ in table_expr.schema().items()
            if typ.is_boolean()
        }

        if conversions:
            table_expr = table_expr.mutate(**conversions)
        return super()._to_sqlglot(table_expr, limit=limit, params=params)

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
