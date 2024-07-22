from __future__ import annotations

import atexit
import contextlib
import datetime
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import pyexasol
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, CanCreateSchema
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers import ExasolCompiler
from ibis.backends.sql.compilers.base import STAR, C

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from ibis.backends import BaseBackend

# strip trailing encodings e.g., UTF8
_VARCHAR_REGEX = re.compile(r"^((VAR)?CHAR(?:\(\d+\)))?(?:\s+.+)?$")


class Backend(SQLBackend, CanCreateDatabase, CanCreateSchema):
    name = "exasol"
    compiler = ExasolCompiler()
    supports_temporary_tables = False
    supports_create_or_replace = False
    supports_in_memory_tables = False
    supports_python_udfs = False

    @property
    def version(self) -> str:
        # https://stackoverflow.com/a/67500385
        query = (
            sg.select("param_value")
            .from_(sg.table("EXA_METADATA", catalog="SYS"))
            .where(C.param_name.eq(sge.convert("databaseProductVersion")))
        )
        with self._safe_raw_sql(query) as result:
            [(version,)] = result.fetchall()
        return version

    def do_connect(
        self,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 8563,
        timezone: str = "UTC",
        **kwargs: Any,
    ) -> None:
        """Create an Ibis client connected to an Exasol database.

        Parameters
        ----------
        user
            Username used for authentication.
        password
            Password used for authentication.
        host
            Hostname to connect to.
        port
            Port number to connect to.
        timezone
            The session timezone.
        kwargs
            Additional keyword arguments passed to `pyexasol.connect`.

        """
        if kwargs.pop("quote_ident", None) is not None:
            raise com.UnsupportedArgumentError(
                "Setting `quote_ident` to anything other than `True` is not supported. "
                "Ibis requires all identifiers to be quoted to work correctly."
            )

        self.con = pyexasol.connect(
            dsn=f"{host}:{port}",
            user=user,
            password=password,
            quote_ident=True,
            **kwargs,
        )
        self._post_connect(timezone)

    @util.experimental
    @classmethod
    def from_connection(
        cls, con: pyexasol.ExaConnection, timezone: str | None = None
    ) -> Backend:
        """Create an Ibis client from an existing connection to an Exasol database.

        Parameters
        ----------
        con
            An existing connection to an Exasol database.
        timezone
            The session timezone.
        """
        if timezone is None:
            timezone = (con.execute("SELECT SESSIONTIMEZONE").fetchone() or ("UTC",))[0]

        new_backend = cls(timezone=timezone)
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect(timezone)
        return new_backend

    def _post_connect(self, timezone: str = "UTC") -> None:
        with self.begin() as con:
            con.execute(f"ALTER SESSION SET TIME_ZONE = {timezone!r}")

    def _from_url(self, url: ParseResult, **kwargs) -> BaseBackend:
        """Construct an ibis backend from a URL."""
        kwargs = {
            "user": url.username,
            "password": unquote_plus(url.password)
            if url.password is not None
            else None,
            "schema": url.path[1:] or None,
            "host": url.hostname,
            "port": url.port,
            **kwargs,
        }

        self._convert_kwargs(kwargs)

        return self.connect(**kwargs)

    @contextlib.contextmanager
    def begin(self):
        # pyexasol doesn't have a cursor method
        con = self.con
        try:
            yield con
        except Exception:
            con.rollback()
            raise
        else:
            con.commit()

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        with self.begin() as cur:
            yield cur.execute(query, *args, **kwargs)

    def list_tables(self, like=None, database=None):
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            Database to list tables from. Default behavior is to show tables in
            the current database.
        """
        tables = sg.select("table_name").from_(
            sg.table("EXA_ALL_TABLES", catalog="SYS")
        )
        views = sg.select(sg.column("view_name").as_("table_name")).from_(
            sg.table("EXA_ALL_VIEWS", catalog="SYS")
        )

        if database is not None:
            tables = tables.where(sg.column("table_schema").eq(sge.convert(database)))
            views = views.where(sg.column("view_schema").eq(sge.convert(database)))

        query = sg.union(tables, views)

        with self._safe_raw_sql(query) as con:
            tables = con.fetchall()

        return self._filter_with_like([table for (table,) in tables], like=like)

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        return self._get_schema_using_query(
            sg.select(STAR)
            .from_(
                sg.table(
                    table_name,
                    db=database,
                    catalog=catalog,
                    quoted=self.compiler.quoted,
                )
            )
            .sql(self.dialect)
        )

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.backends.exasol.converter import ExasolPandasData

        df = pd.DataFrame.from_records(cursor, columns=schema.names, coerce_float=True)
        df = ExasolPandasData.convert_table(df, schema)
        return df

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        table = sg.table(
            util.gen_name(f"{self.name}_metadata"), quoted=self.compiler.quoted
        )
        dialect = self.dialect
        create_view = sg.exp.Create(
            kind="VIEW",
            this=table,
            expression=sg.parse_one(query, dialect=dialect),
        )
        drop_view = sg.exp.Drop(kind="VIEW", this=table).sql(dialect)
        describe = sg.exp.Describe(this=table).sql(dialect)
        type_mapper = self.compiler.type_mapper
        with self._safe_raw_sql(create_view):
            try:
                return sch.Schema(
                    {
                        name: type_mapper.from_string(_VARCHAR_REGEX.sub(r"\1", typ))
                        for name, typ, *_ in self.con.execute(describe).fetchall()
                    }
                )
            finally:
                self.con.execute(drop_view)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                "Exasol cannot yet reliably handle `null` typed columns; "
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

            ident = sg.to_identifier(name, quoted=quoted)
            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(this=ident, expressions=column_defs),
            )
            create_stmt_sql = create_stmt.sql(self.name)

            df = op.data.to_frame()
            data = df.itertuples(index=False, name=None)

            def process_item(item: Any):
                """Handle inserting timestamps with timezones."""
                if isinstance(item, datetime.datetime):
                    if item.tzinfo is not None:
                        item = item.tz_convert("UTC").tz_localize(None)
                    return item.isoformat(sep=" ", timespec="milliseconds")
                return item

            rows = (tuple(map(process_item, row)) for row in data)
            with self._safe_raw_sql(create_stmt_sql):
                if not df.empty:
                    self.con.ext.insert_multi(name, rows)

            atexit.register(self._clean_up_tmp_table, ident)

    def _clean_up_tmp_table(self, ident: sge.Identifier) -> None:
        with self._safe_raw_sql(
            sge.Drop(kind="TABLE", this=ident, exists=True, cascade=True)
        ):
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
        schema: sch.Schema | None = None,
        database: str | None = None,
        overwrite: bool = False,
        temp: bool = False,
    ) -> ir.Table:
        """Create a table in Exasol.

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
            The database in which to create the table; optional
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        temp
            Create a temporary table (not supported)

        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if temp:
            raise com.UnsupportedOperationError(
                "Creating temp tables is not supported by Exasol."
            )

        if database is not None and database != self.current_database:
            raise com.UnsupportedOperationError(
                "Creating tables in other databases is not supported by Exasol"
            )
        else:
            database = None

        quoted = self.compiler.quoted

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

        type_mapper = self.compiler.type_mapper
        column_defs = [
            sge.ColumnDef(
                this=sg.to_identifier(colname, quoted=quoted),
                kind=type_mapper.from_ibis(typ),
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

        table = sg.table(temp_name, catalog=database, quoted=quoted)
        target = sge.Schema(this=table, expressions=column_defs)

        create_stmt = sge.Create(kind="TABLE", this=target)

        this = sg.table(name, catalog=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt):
            if query is not None:
                self.con.execute(
                    sge.Insert(this=table, expression=query).sql(self.name)
                )

            if overwrite:
                self.con.execute(
                    sge.Drop(kind="TABLE", this=this, exists=True).sql(self.name)
                )
                self.con.execute(
                    f"RENAME TABLE {table.sql(self.name)} TO {this.sql(self.name)}"
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

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql("SELECT CURRENT_SCHEMA") as cur:
            [(schema,)] = cur.fetchall()
        return schema

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None:
            raise NotImplementedError(
                "`catalog` argument is not supported for the Exasol backend"
            )
        drop_schema = sg.exp.Drop(kind="SCHEMA", this=name, exists=force)
        with self.begin() as con:
            con.execute(drop_schema.sql(dialect=self.dialect))

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None:
            raise NotImplementedError(
                "`catalog` argument is not supported for the Exasol backend"
            )
        create_database = sg.exp.Create(kind="SCHEMA", this=name, exists=force)
        open_database = self.current_database
        with self.begin() as con:
            con.execute(create_database.sql(dialect=self.dialect))
            # Exasol implicitly opens the created schema, therefore we need to restore
            # the previous context.
            con.execute(
                f"OPEN SCHEMA {open_database}"
                if open_database
                else f"CLOSE SCHEMA {name}"
            )

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        if catalog is not None:
            raise NotImplementedError(
                "`catalog` argument is not supported for the Exasol backend"
            )

        query = sg.select("schema_name").from_(sg.table("EXA_SCHEMAS", catalog="SYS"))

        with self._safe_raw_sql(query) as con:
            databases = con.fetchall()
        return self._filter_with_like([db for (db,) in databases], like=like)

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)

        dtypes = expr.as_table().schema().values()

        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            while batch := cursor.fetchmany(chunk_size):
                yield (tuple(map(dt.normalize, dtypes, row)) for row in batch)
