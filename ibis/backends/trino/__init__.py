"""Trino backend."""

from __future__ import annotations

import collections
import contextlib
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlalchemy as sa
import sqlglot as sg
import toolz
from trino.sqlalchemy.datatype import ROW as _ROW
from trino.sqlalchemy.dialect import TrinoDialect

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanListDatabases
from ibis.backends.base.sql.alchemy import (
    AlchemyCanCreateSchema,
    AlchemyCrossSchemaBackend,
)
from ibis.backends.base.sql.alchemy.datatypes import ArrayType
from ibis.backends.trino.compiler import TrinoSQLCompiler
from ibis.backends.trino.datatypes import INTERVAL, ROW, TrinoType

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pyarrow as pa

    import ibis.expr.schema as sch


class Backend(AlchemyCrossSchemaBackend, AlchemyCanCreateSchema, CanListDatabases):
    name = "trino"
    compiler = TrinoSQLCompiler
    supports_create_or_replace = False
    supports_temporary_tables = False

    @cached_property
    def version(self) -> str:
        return self._scalar_query(sa.select(sa.func.version()))

    @property
    def current_database(self) -> str:
        return self._scalar_query(sa.select(sa.literal_column("current_catalog")))

    def list_databases(self, like: str | None = None) -> list[str]:
        query = "SHOW CATALOGS"
        with self.begin() as con:
            catalogs = list(con.exec_driver_sql(query).scalars())
        return self._filter_with_like(catalogs, like=like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = "SHOW SCHEMAS"

        if database is not None:
            query += f" IN {self._quote(database)}"

        with self.begin() as con:
            schemata = list(con.exec_driver_sql(query).scalars())
        return self._filter_with_like(schemata, like)

    @property
    def current_schema(self) -> str:
        return self._scalar_query(sa.select(sa.literal_column("current_schema")))

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
            util.warn_deprecated(
                "database",
                instead=(
                    f"{self.name} cannot list tables only using `database` specifier. "
                    "Include a `schema` argument."
                ),
                as_of="7.1",
                removed_in="8.0",
            )
            database = sg.parse_one(database, into=sg.exp.Table).sql(dialect=self.name)
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sg.exp.Table).sql(dialect=self.name)
        else:
            database = sg.table(schema, db=database).sql(dialect=self.name) or None
        if database is not None:
            query += f" IN {database}"

        with self.begin() as con:
            tables = list(con.exec_driver_sql(query).scalars())

        return self._filter_with_like(tables, like=like)

    def do_connect(
        self,
        user: str = "user",
        password: str | None = None,
        host: str = "localhost",
        port: int = 8080,
        database: str | None = None,
        schema: str | None = None,
        source: str | None = None,
        **connect_args,
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
        connect_args
            Additional keyword arguments passed directly to SQLAlchemy's
            `create_engine`

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
        database = "/".join(filter(None, (database, schema)))
        url = sa.engine.URL.create(
            drivername="trino",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
            query=dict(source="ibis" if source is None else source),
        )
        connect_args.setdefault("timezone", "UTC")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The dbapi\(\) classmethod on dialect classes has been renamed",
                category=sa.exc.SADeprecationWarning,
            )
            super().do_connect(
                sa.create_engine(
                    url, connect_args=connect_args, poolclass=sa.pool.StaticPool
                )
            )

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            if isinstance(typ := column_info["type"], _ROW):
                column_info["type"] = ROW(typ.attr_types)
            elif isinstance(typ, sa.ARRAY):
                column_info["type"] = toolz.nth(
                    typ.dimensions or 1, toolz.iterate(ArrayType, typ.item_type)
                )
            elif isinstance(typ, sa.Interval):
                column_info["type"] = INTERVAL(
                    native=typ.native,
                    day_precision=typ.day_precision,
                    second_precision=typ.second_precision,
                )

        return meta

    @contextlib.contextmanager
    def _prepare_metadata(self, query: str) -> Iterator[dict[str, str]]:
        name = util.gen_name("trino_metadata")
        with self.begin() as con:
            con.exec_driver_sql(f"PREPARE {name} FROM {query}")
            try:
                yield con.exec_driver_sql(f"DESCRIBE OUTPUT {name}").mappings()
            finally:
                con.exec_driver_sql(f"DEALLOCATE PREPARE {name}")

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self._prepare_metadata(query) as mappings:
            yield from (
                # trino types appear to be always nullable
                (name, TrinoType.from_string(trino_type).copy(nullable=True))
                for name, trino_type in toolz.pluck(["Column Name", "Type"], mappings)
            )

    def _execute_view_creation(self, name, definition):
        from sqlalchemy_views import CreateView

        # NB: trino doesn't support temporary views so we use the less
        # desirable method of cleaning up when the Python process exits using
        # an atexit hook
        #
        # the method that defines the atexit hook is defined in the parent
        # class
        view = CreateView(sa.table(name), definition, or_replace=True)

        with self.begin() as con:
            con.execute(view)

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        name = ".".join(map(self._quote, filter(None, [database, name])))
        if_not_exists = "IF NOT EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA {if_not_exists}{name}")

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        name = ".".join(map(self._quote, filter(None, [database, name])))
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP SCHEMA {if_exists}{name}")

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
                "Temporary tables in the Trino backend are not yet supported"
            )

        orig_table_ref = name

        if overwrite:
            name = util.gen_name("trino_overwrite")

        create_stmt = "CREATE TABLE"

        table_ref = self._quote(name)

        create_stmt += f" {table_ref}"

        if schema is not None and obj is None:
            schema_str = ", ".join(
                (
                    f"{self._quote(name)} {TrinoType.to_string(typ)}"
                    + " NOT NULL" * (not typ.nullable)
                )
                for name, typ in schema.items()
            )
            create_stmt += f" ({schema_str})"

        if comment is not None:
            create_stmt += f" COMMENT {comment!r}"

        if properties:

            def literal_compile(v):
                if isinstance(v, collections.abc.Mapping):
                    return f"MAP(ARRAY{list(v.keys())!r}, ARRAY{list(v.values())!r})"
                elif util.is_iterable(v):
                    return f"ARRAY{list(v)!r}"
                else:
                    return repr(v)

            pairs = ", ".join(
                f"{k} = {literal_compile(v)}" for k, v in properties.items()
            )
            create_stmt += f" WITH ({pairs})"

        if obj is not None:
            import pyarrow as pa

            if isinstance(obj, (pd.DataFrame, pa.Table)):
                table = ibis.memtable(obj, schema=schema)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            compiled_table = self.compile(table)

            # cast here because trino doesn't allow specifying a schema in
            # CTAS, e.g., `CREATE TABLE (schema) AS SELECT`
            subquery = compiled_table.subquery()
            columns = subquery.columns
            select = sa.select(
                *(
                    sa.cast(columns[name], TrinoType.from_ibis(typ))
                    for name, typ in (schema or table.schema()).items()
                )
            )

            compiled = select.compile(
                dialect=TrinoDialect(), compile_kwargs=dict(literal_binds=True)
            )

            create_stmt += f" AS {compiled}"

        with self.begin() as con:
            con.exec_driver_sql(create_stmt)

            if overwrite:
                # drop the original table
                con.exec_driver_sql(
                    f"DROP TABLE IF EXISTS {self._quote(orig_table_ref)}"
                )

                # rename the new table to the original table name
                con.exec_driver_sql(
                    f"ALTER TABLE IF EXISTS {table_ref} RENAME TO {self._quote(orig_table_ref)}"
                )

        return self.table(orig_table_ref)

    def _table_from_schema(
        self,
        name: str,
        schema: sch.Schema,
        temp: bool = False,
        database: str | None = None,
        **kwargs: Any,
    ) -> sa.Table:
        return super()._table_from_schema(
            name,
            schema,
            temp=temp,
            trino_catalog=database or self.current_database,
            **kwargs,
        )
