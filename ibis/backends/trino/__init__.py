"""Trino backend."""

from __future__ import annotations

import contextlib
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge
import trino

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanListDatabases, NoUrl
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.base.sqlglot.compiler import C
from ibis.backends.trino.compiler import TrinoCompiler

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pandas as pd
    import pyarrow as pa

    import ibis.expr.operations as ops


class Backend(SQLGlotBackend, CanListDatabases, NoUrl):
    name = "trino"
    compiler = TrinoCompiler()
    supports_create_or_replace = False
    supports_temporary_tables = False

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
        self, table_name: str, schema: str | None = None, database: str | None = None
    ) -> sch.Schema:
        """Compute the schema of a `table`.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        schema
            Schema name
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema

        """
        conditions = [sg.column("table_name").eq(sge.convert(table_name))]

        if schema is not None:
            conditions.append(sg.column("table_schema").eq(sge.convert(schema)))

        query = (
            sg.select(
                "column_name",
                "data_type",
                sg.column("is_nullable").eq(sge.convert("YES")).as_("nullable"),
            )
            .from_(sg.table("columns", db="information_schema", catalog=database))
            .where(sg.and_(*conditions))
            .order_by("ordinal_position")
        )

        with self._safe_raw_sql(query) as cur:
            meta = cur.fetchall()

        if not meta:
            fqn = sg.table(table_name, db=schema, catalog=database).sql(self.name)
            raise com.IbisError(f"Table not found: {fqn}")

        return sch.Schema(
            {
                name: self.compiler.type_mapper.from_string(typ, nullable=nullable)
                for name, typ, nullable in meta
            }
        )

    @cached_property
    def version(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.version())) as cur:
            [(version,)] = cur.fetchall()
        return version

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(C.current_catalog)) as cur:
            [(database,)] = cur.fetchall()
        return database

    @property
    def current_schema(self) -> str:
        with self._safe_raw_sql(sg.select(C.current_schema)) as cur:
            [(schema,)] = cur.fetchall()
        return schema

    def list_databases(self, like: str | None = None) -> list[str]:
        query = "SHOW CATALOGS"
        with self._safe_raw_sql(query) as cur:
            catalogs = cur.fetchall()
        return self._filter_with_like(list(map(itemgetter(0), catalogs)), like=like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if database is not None:
            database = sg.to_identifier(database, quoted=self.compiler.quoted).sql(
                self.name
            )
            query += f" IN {database}"

        with self._safe_raw_sql(query) as cur:
            schemata = cur.fetchall()
        return self._filter_with_like(list(map(itemgetter(0), schemata)), like)

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
        query = "SHOW TABLES"

        if database is not None and schema is None:
            raise com.IbisInputError(
                f"{self.name} cannot list tables only using `database` specifier. "
                "Include a `schema` argument."
            )
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sg.exp.Table).sql(dialect=self.name)
        else:
            database = sg.table(schema, db=database).sql(dialect=self.name) or None
        if database is not None:
            query += f" IN {database}"

        with self._safe_raw_sql(query) as cur:
            tables = cur.fetchall()

        return self._filter_with_like(list(map(itemgetter(0), tables)), like=like)

    def do_connect(
        self,
        user: str = "user",
        password: str | None = None,
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
        password
            Password to connect with
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

        >>> con = ibis.connect(f"trino://user:password@host:port/{catalog}/{schema}")

        Connect using keyword arguments

        >>> con = ibis.trino.connect(database=catalog, schema=schema)
        >>> con = ibis.trino.connect(database=catalog, schema=schema, source="my-app")

        """
        self.con = trino.dbapi.connect(
            user=user,
            auth=password,
            host=host,
            port=port,
            catalog=database,
            schema=schema,
            source=source or "ibis",
            timezone=timezone,
            **kwargs,
        )

    @contextlib.contextmanager
    def _prepare_metadata(self, query: str) -> Iterator[dict[str, str]]:
        name = util.gen_name(f"{self.name}_metadata")
        with self.begin() as cur:
            cur.execute(f"PREPARE {name} FROM {query}")
            try:
                cur.execute(f"DESCRIBE OUTPUT {name}")
                yield cur.fetchall()
            finally:
                cur.execute(f"DEALLOCATE PREPARE {name}")

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self._prepare_metadata(query) as info:
            yield from (
                # trino types appear to be always nullable
                (
                    name,
                    self.compiler.type_mapper.from_string(trino_type).copy(
                        nullable=True
                    ),
                )
                for name, _, _, _, trino_type, *_ in info
            )

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        with self._safe_raw_sql(
            sge.Create(
                this=sg.table(name, catalog=database, quoted=self.compiler.quoted),
                kind="SCHEMA",
                exists=force,
            )
        ):
            pass

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        with self._safe_raw_sql(
            sge.Drop(
                this=sg.table(name, catalog=database, quoted=self.compiler.quoted),
                kind="SCHEMA",
                exists=force,
            )
        ):
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
            Not yet implemented.
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

        if temp:
            raise NotImplementedError(
                "Temporary tables are not supported in the Trino backend"
            )

        quoted = self.compiler.quoted
        orig_table_ref = sg.to_identifier(name, quoted=quoted)

        if overwrite:
            name = util.gen_name(f"{self.name}_overwrite")

        table_ref = sg.table(name, catalog=database, quoted=quoted)

        if schema is not None and obj is None:
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(name, quoted=self.compiler.quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    # TODO(cpcloud): not null constraints are unreliable in
                    # trino, so we ignore them
                    # https://github.com/trinodb/trino/issues/2923
                    constraints=None,
                )
                for name, typ in schema.items()
            ]
            target = sge.Schema(this=table_ref, expressions=column_defs)
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
            import pandas as pd
            import pyarrow as pa
            import pyarrow_hotfix  # noqa: F401

            if isinstance(obj, (pd.DataFrame, pa.Table)):
                table = ibis.memtable(obj, schema=schema)
            else:
                table = obj

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
            ).from_(self._to_sqlglot(table).subquery())
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
                    sge.AlterTable(
                        this=table_ref,
                        exists=True,
                        actions=[sge.RenameTable(this=orig_table_ref, exists=True)],
                    ).sql(self.name)
                )

        return self.table(orig_table_ref.name)

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
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                "Trino cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                    # we don't support `NOT NULL` constraints in trino because
                    # because each trino connector differs in whether it
                    # supports nullability constraints, and whether the
                    # connector supports it isn't visible to ibis via a
                    # metadata query
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
            ).sql(self.name, pretty=True)

            data = op.data.to_frame().itertuples(index=False)
            specs = ", ".join("?" * len(schema))
            table = sg.table(name, quoted=quoted).sql(self.name)
            insert_stmt = f"INSERT INTO {table} VALUES ({specs})"
            with self.begin() as cur:
                cur.execute(create_stmt)
                for row in data:
                    cur.execute(insert_stmt, row)
