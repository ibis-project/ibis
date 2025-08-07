"""Trino backend."""

from __future__ import annotations

import contextlib
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus, urlparse

import sqlglot as sg
import sqlglot.expressions as sge
import trino
from trino.auth import BasicAuthentication

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    CanListCatalog,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoExampleLoader,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import AlterTable, C, RenameTable

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa

    import ibis.expr.operations as ops


class Backend(
    SQLBackend,
    CanListCatalog,
    CanCreateDatabase,
    HasCurrentCatalog,
    HasCurrentDatabase,
    NoExampleLoader,
):
    name = "trino"
    compiler = sc.trino.compiler
    supports_create_or_replace = False

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        database, *schema = url.path.strip("/").split("/", 1)
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["auth"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        if schema:
            kwargs["schema"] = schema[0]
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    def raw_sql(self, query: str | sg.Expression) -> Any:
        """Execute a raw SQL query."""
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name, pretty=True)

        con = self.con
        cur = con.cursor()
        try:
            cur.execute(query)
        except Exception:
            if con.transaction is not None:
                con.rollback()
            if cur._query:
                cur.close()
            raise
        else:
            if con.transaction is not None:
                con.commit()
            return cur

    @contextlib.contextmanager
    def begin(self):
        con = self.con
        cur = con.cursor()
        try:
            yield cur
        except Exception:
            if con.transaction is not None:
                con.rollback()
            raise
        else:
            if con.transaction is not None:
                con.commit()
        finally:
            if cur._query:
                cur.close()

    @contextlib.contextmanager
    def _safe_raw_sql(
        self, query: str | sge.Expression
    ) -> Iterator[trino.dbapi.Cursor]:
        """Execute a raw SQL query, yielding the cursor.

        Parameters
        ----------
        query
            The query to execute.

        Yields
        ------
        trino.dbapi.Cursor
            The cursor of the executed query.

        """
        cur = self.raw_sql(query)
        try:
            yield cur
        finally:
            if cur._query:
                cur.close()

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Compute the schema of a `table`.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        catalog
            Catalog name
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema

        """
        query = (
            sg.select(
                C.column_name,
                C.data_type,
                C.is_nullable.eq(sge.convert("YES")).as_("nullable"),
            )
            .from_(sg.table("columns", db="information_schema", catalog=catalog))
            .where(
                C.table_name.eq(sge.convert(table_name)),
                C.table_schema.eq(sge.convert(database or self.current_database)),
            )
            .order_by(C.ordinal_position)
        )

        with self._safe_raw_sql(query) as cur:
            meta = cur.fetchall()

        if not meta:
            fqn = sg.table(table_name, db=database, catalog=catalog).sql(self.name)
            raise com.TableNotFound(fqn)

        type_mapper = self.compiler.type_mapper

        return sch.Schema(
            {
                name: type_mapper.from_string(typ, nullable=nullable)
                for name, typ, nullable in meta
            }
        )

    @cached_property
    def version(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.version())) as cur:
            [(version,)] = cur.fetchall()
        return version

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(C.current_catalog)) as cur:
            [(database,)] = cur.fetchall()
        return database

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(C.current_schema)) as cur:
            [(schema,)] = cur.fetchall()
        return schema

    def list_catalogs(self, *, like: str | None = None) -> list[str]:
        query = "SHOW CATALOGS"
        with self._safe_raw_sql(query) as cur:
            catalogs = cur.fetchall()
        return self._filter_with_like(list(map(itemgetter(0), catalogs)), like=like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if catalog is not None:
            catalog = sg.to_identifier(catalog, quoted=self.compiler.quoted).sql(
                self.name
            )
            query += f" IN {catalog}"

        with self._safe_raw_sql(query) as cur:
            databases = cur.fetchall()
        return self._filter_with_like(list(map(itemgetter(0), databases)), like)

    def list_tables(
        self, *, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        table_loc = self._to_sqlglot_table(database)

        query = "SHOW TABLES"

        if table_loc.catalog or table_loc.db:
            table_loc = table_loc.sql(dialect=self.dialect)
            query += f" IN {table_loc}"

        with self._safe_raw_sql(query) as cur:
            tables = cur.fetchall()

        return self._filter_with_like(list(map(itemgetter(0), tables)), like=like)

    def do_connect(
        self,
        user: str = "user",
        auth: str | None = None,
        host: str = "localhost",
        port: int = 8080,
        database: str | None = None,
        schema: str | None = None,
        source: str | None = None,
        timezone: str = "UTC",
        **kwargs,
    ) -> None:
        """Connect to Trino.

        Parameters
        ----------
        user
            Username to connect with
        auth
            Authentication method or password to use for the connection.
        host
            Hostname of the Trino server
        port
            Port of the Trino server
        database
            Catalog to use on the Trino server
        schema
            Schema to use on the Trino server
        source
            Application name passed to Trino
        timezone
            Timezone to use for the connection
        kwargs
            Additional keyword arguments passed directly to the
            `trino.dbapi.connect` API.

        Examples
        --------
        >>> catalog = "hive"
        >>> schema = "default"

        Connect using a URL, with the default user, password, host and port

        >>> con = ibis.connect(f"trino:///{catalog}/{schema}")

        Connect using a URL

        >>> con = ibis.connect(f"trino://user@localhost:8080/{catalog}/{schema}")

        Connect using keyword arguments

        >>> con = ibis.trino.connect(database=catalog, schema=schema)
        >>> con = ibis.trino.connect(database=catalog, schema=schema, source="my-app")
        """
        if (
            isinstance(auth, str)
            and (scheme := urlparse(host).scheme)
            and scheme != "http"
        ):
            auth = BasicAuthentication(user, auth)

        self.con = trino.dbapi.connect(
            user=user,
            host=host,
            port=port,
            catalog=database,
            schema=schema,
            source=source or "ibis",
            timezone=timezone,
            auth=auth,
            **kwargs,
        )

    @util.experimental
    @classmethod
    def from_connection(cls, con: trino.dbapi.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to a Trino database.

        Parameters
        ----------
        con
            An existing connection to a Trino database.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        return new_backend

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        name = util.gen_name(f"{self.name}_metadata")
        with self.begin() as cur:
            cur.execute(f"PREPARE {name} FROM {query}")
            try:
                cur.execute(f"DESCRIBE OUTPUT {name}")
                info = cur.fetchall()
            finally:
                cur.execute(f"DEALLOCATE PREPARE {name}")

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(trino_type).copy(
                    # trino types appear to be always nullable
                    nullable=True
                )
                for name, _, _, _, trino_type, *_ in info
            }
        )

    def create_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        with self._safe_raw_sql(
            sge.Create(
                this=sg.table(name, catalog=catalog, quoted=self.compiler.quoted),
                kind="SCHEMA",
                exists=force,
            )
        ):
            pass

    def drop_database(
        self, name: str, /, *, catalog: str | None = None, force: bool = False
    ) -> None:
        with self._safe_raw_sql(
            sge.Drop(
                this=sg.table(name, catalog=catalog, quoted=self.compiler.quoted),
                kind="SCHEMA",
                exists=force,
            )
        ):
            pass

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
        comment: str | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> ir.Table:
        """Create a table in Trino.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but one of `obj`
            or `schema` must be specified
        schema
            The schema of the table to create; optional, but one of `obj` or
            `schema` must be specified
        database
            The database to insert the table into.
            If not provided, the current database is used.
            You can provide a single database name, like `"mydb"`. For
            multi-level hierarchies, you can pass in a dotted string path like
            `"catalog.database"` or a tuple of strings like `("catalog",
            "database")`.
        temp
            This parameter is not yet supported in the Trino backend, because
            Trino doesn't implement temporary tables
        overwrite
            If `True`, replace the table if it already exists, otherwise fail if
            the table exists
        comment
            Add a comment to the table
        properties
            Table properties to set on creation
        """
        if obj is None and schema is None:
            raise com.IbisError("One of the `schema` or `obj` parameter is required")
        if schema is not None:
            schema = ibis.schema(schema)

        if temp:
            raise NotImplementedError(
                "Temporary tables are not supported in the Trino backend"
            )

        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        quoted = self.compiler.quoted
        orig_table_ref = sg.table(name, catalog=catalog, db=db, quoted=quoted)

        if overwrite:
            name = util.gen_name(f"{self.name}_overwrite")

        table_ref = sg.table(name, catalog=catalog, db=db, quoted=quoted)

        if schema is not None and obj is None:
            target = sge.Schema(
                this=table_ref,
                expressions=schema.to_sqlglot_column_defs(self.dialect),
            )
        else:
            target = table_ref

        property_list = []

        for k, v in (properties or {}).items():
            name = sg.to_identifier(k)
            expr = ibis.literal(v)
            value = self.compiler.visit_Literal(expr.op(), value=v, dtype=expr.type())
            property_list.append(sge.Property(this=name, value=value))

        if comment:
            property_list.append(sge.SchemaCommentProperty(this=sge.convert(comment)))

        if obj is not None:
            if isinstance(obj, ir.Table):
                table = obj
            else:
                table = ibis.memtable(obj, schema=schema)

            self._run_pre_execute_hooks(table)

            # cast here because trino doesn't allow specifying a schema in
            # CTAS, e.g., `CREATE TABLE (schema) AS SELECT`
            select = sg.select(
                *(
                    self.compiler.cast(sg.column(name, quoted=quoted), typ).as_(
                        name, quoted=quoted
                    )
                    for name, typ in (schema or table.schema()).items()
                )
            ).from_(self.compiler.to_sqlglot(table).subquery())
        else:
            select = None

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            expression=select,
            properties=(
                sge.Properties(expressions=property_list) if property_list else None
            ),
        )

        with self._safe_raw_sql(create_stmt) as cur:
            if overwrite:
                # drop the original table
                cur.execute(
                    sge.Drop(kind="TABLE", this=orig_table_ref, exists=True).sql(
                        self.name
                    )
                )

                # rename the new table to the original table name
                cur.execute(
                    AlterTable(
                        this=table_ref,
                        exists=True,
                        actions=[RenameTable(this=orig_table_ref, exists=True)],
                    ).sql(self.name)
                )

        return self.table(orig_table_ref.name, database=(catalog, db))

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.backends.trino.converter import TrinoPandasData

        try:
            df = pd.DataFrame.from_records(
                cursor.fetchall(), columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = TrinoPandasData.convert_table(df, schema)
        return df

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "Trino cannot yet reliably handle `null` typed columns; "
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
        ).sql(self.name)

        data = op.data.to_frame().itertuples(index=False)
        insert_stmt = self._build_insert_template(name, schema=schema)
        with self.begin() as cur:
            cur.execute(create_stmt)
            for row in data:
                cur.execute(insert_stmt, row)
