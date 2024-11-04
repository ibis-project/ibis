"""Databricks backend."""

from __future__ import annotations

import contextlib
import functools
import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import databricks.sql
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, UrlFromPath
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR, AlterTable, C, RenameTable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import polars as pl

    from ibis.expr.schema import SchemaLike


class Backend(SQLBackend, CanCreateDatabase, UrlFromPath):
    name = "databricks"
    compiler = sc.databricks.compiler

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_catalog())) as cur:
            [(db,)] = cur.fetchall()
        return db

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_database())) as cur:
            [(db,)] = cur.fetchall()
        return db

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)
        cur = self.con.cursor()
        try:
            cur.execute(query, **kwargs)
        except Exception:
            cur.close()
            raise
        return cur

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
        schema: SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        using: str = "delta",
        location: str | None = None,
        tblproperties: Mapping[str, str] | None = None,
    ):
        """Create a table in Databricks.

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
            The name of the database in which to create the table; if not
            passed, the current database is used.

            For multi-level table hierarchies, you can pass in a dotted string
            path like `"catalog.database"` or a tuple of strings like
            `("catalog", "database")`.
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists
        using
            Data source format
        location
            Storage location for the table
        tblproperties
            Table properties
        """
        if temp:
            raise exc.UnsupportedOperationError("Temporary tables not yet supported")

        table_loc = self._to_sqlglot_table(database)

        catalog = table_loc.catalog or self.current_catalog
        database = table_loc.db or self.current_database

        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = [sge.FileFormatProperty(this=self.compiler.v[using.upper()])]

        if location is not None:
            properties.append(sge.LocationProperty(this=sge.convert(location)))

        for key, value in (tblproperties or {}).items():
            properties.append(
                sge.Property(this=sge.convert(str(key)), value=sge.convert(str(value)))
            )

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
            temp_name = util.gen_name("databricks_table")
        else:
            temp_name = name

        quoted = self.compiler.quoted
        dialect = self.dialect

        initial_table = sg.table(temp_name, catalog=catalog, db=database, quoted=quoted)
        target = sge.Schema(
            this=initial_table,
            expressions=(schema or table.schema()).to_sqlglot(dialect),
        )

        properties = sge.Properties(expressions=properties)
        create_stmt = sge.Create(kind="TABLE", this=target, properties=properties)

        # This is the same table as initial_table unless overwrite == True
        final_table = sg.table(name, catalog=catalog, db=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.insert(query, into=initial_table).sql(dialect)
                cur.execute(insert_stmt).fetchall()

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=final_table, exists=True).sql(dialect)
                )
                if temp:
                    cur.execute(
                        sge.Create(
                            kind="TABLE",
                            this=final_table,
                            expression=sg.select(STAR).from_(initial_table),
                            properties=properties,
                        ).sql(dialect)
                    )
                    cur.execute(
                        sge.Drop(kind="TABLE", this=initial_table, exists=True).sql(
                            dialect
                        )
                    )
                else:
                    cur.execute(
                        AlterTable(
                            this=initial_table, actions=[RenameTable(this=final_table)]
                        ).sql(dialect)
                    )

        return self.table(name, database=(catalog, database))

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        database
            Database name

        Returns
        -------
        Table
            Table expression

        """
        table_loc = self._to_sqlglot_table(database)

        # TODO: set these to better defaults
        catalog = table_loc.catalog or None
        database = table_loc.db or None

        table_schema = self.get_schema(name, catalog=catalog, database=database)
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=database),
        ).to_expr()

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
        table = sg.table(
            table_name, db=database, catalog=catalog, quoted=self.compiler.quoted
        )
        sql = sge.Describe(kind="TABLE", this=table).sql(self.dialect)
        try:
            with self.con.cursor() as cur:
                out = cur.execute(sql).fetchall_arrow()
        except databricks.sql.exc.ServerOperationError as e:
            raise exc.TableNotFound(
                f"Table {table_name!r} not found in "
                f"{catalog or self.current_catalog}.{database or self.current_database}"
            ) from e

        names = out["col_name"].to_pylist()
        types = out["data_type"].to_pylist()

        return sch.Schema(
            dict(zip(names, map(self.compiler.type_mapper.from_string, types)))
        )

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)
        with self.con.cursor() as cur:
            yield cur.execute(query, *args, **kwargs)

    def list_catalogs(self, like: str | None = None) -> list[str]:
        with self.con.cursor() as cur:
            out = cur.catalogs().fetchall_arrow()
        return self._filter_with_like(out["TABLE_CAT"].to_pylist(), like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        with self.con.cursor() as cur:
            out = cur.schemas(
                catalog_name=catalog or self.current_catalog
            ).fetchall_arrow()
        return self._filter_with_like(out["TABLE_SCHEM"].to_pylist(), like=like)

    @functools.cached_property
    def version(self) -> str:
        query = sg.select(self.compiler.f.current_version())
        with self._safe_raw_sql(query) as cur:
            [(version_info,)] = cur.fetchall()
        return version_info["dbsql_version"]

    def do_connect(
        self,
        *,
        server_hostname: str | None = None,
        http_path: str | None = None,
        access_token: str | None = None,
        auth_type: str | None = None,
        credentials_provider: str | None = None,
        password: str | None = None,
        username: str | None = None,
        session_configuration: Mapping[str, str] | None = None,
        http_headers: list[tuple[str, str]] | None = None,
        catalog: str | None = None,
        schema: str = "default",
        use_cloud_fetch: bool = False,
        memtable_volume: str | None = None,
        staging_allowed_local_path: str | None = None,
        **config: Any,
    ) -> None:
        """Create an Ibis client connected to a Databricks cloud instance."""
        if staging_allowed_local_path is None:
            staging_allowed_local_path = tempfile.gettempdir()
        self.con = databricks.sql.connect(
            server_hostname=(
                server_hostname or os.environ.get("DATABRICKS_SERVER_HOSTNAME")
            ),
            http_path=http_path or os.environ.get("DATABRICKS_HTTP_PATH"),
            access_token=access_token or os.environ.get("DATABRICKS_TOKEN"),
            auth_type=auth_type,
            credentials_provider=credentials_provider,
            password=password,
            username=username,
            session_configuration=session_configuration,
            http_headers=http_headers,
            catalog=catalog,
            schema=schema,
            use_cloud_fetch=use_cloud_fetch,
            staging_allowed_local_path=staging_allowed_local_path,
            **config,
        )
        if memtable_volume is None:
            short_version = "".join(map(str, sys.version_info[:3]))
            memtable_volume = (
                f"{getpass.getuser()}-py={short_version}-pid={os.getpid()}"
            )
        self._memtable_volume = memtable_volume
        self._memtable_catalog = self.current_catalog
        self._memtable_database = self.current_database
        self._post_connect(memtable_volume=memtable_volume)

    @contextlib.contextmanager
    def begin(self):
        with self.con.cursor() as cur:
            yield cur

    @util.experimental
    @classmethod
    def from_connection(
        cls,
        con,
        memtable_volume: str | None = None,
    ) -> Backend:
        """Create an Ibis client from an existing connection to a Databricks cloud instance.

        Parameters
        ----------
        con
            An existing connection to a Databricks database.
        memtable_volume
            The volume to use for Ibis memtables.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect(memtable_volume=memtable_volume)
        return new_backend

    def _post_connect(self, *, memtable_volume: str) -> None:
        sql = f"CREATE VOLUME IF NOT EXISTS `{memtable_volume}` COMMENT 'Ibis memtable storage volume'"
        with self.con.cursor() as cur:
            cur.execute(sql)

    @functools.cached_property
    def _memtable_volume_path(self) -> str:
        return f"/Volumes/{self._memtable_catalog}/{self._memtable_database}/{self._memtable_volume}"

    def _in_memory_table_exists(self, name: str) -> bool:
        sql = (
            sg.select(self.compiler.f.count(STAR))
            .from_(
                sg.table("views", db="information_schema", catalog=self.current_catalog)
            )
            .where(
                C.table_name.eq(sge.convert(name)),
                C.table_schema.eq(self.compiler.f.current_database()),
            )
        )
        with self._safe_raw_sql(sql) as cur:
            [(out,)] = cur.fetchall()

        assert 0 <= out <= 1, str(out)
        return out == 1

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        import pyarrow.parquet as pq

        quoted = self.compiler.quoted
        name = op.name
        stem = f"{name}.parquet"

        upstream_path = f"{self._memtable_volume_path}/{stem}"
        sql = sge.Create(
            kind="VIEW",
            this=sg.table(
                name,
                db=self.current_database,
                catalog=self.current_catalog,
                quoted=quoted,
            ),
            expression=sge.select(STAR).from_(
                sg.table(upstream_path, db="parquet", quoted=quoted)
            ),
        ).sql(self.dialect)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            data = op.data.to_pyarrow(schema=op.schema)
            path = Path(tmpdir, stem)
            put_into = f"PUT '{path}' INTO '{upstream_path}' OVERWRITE"
            # optimize for bandwidth so use zstd which typically compresses
            # better than the other options without much loss in speed
            pq.write_table(data, path, compression="zstd")
            with self.con.cursor() as cur:
                cur.execute(put_into)
                cur.execute(sql)

    def _finalize_memtable(self, name: str) -> None:
        path = f"{self._memtable_volume_path}/{name}.parquet"
        sql = sge.Drop(
            kind="VIEW",
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            exists=True,
        ).sql(self.dialect)
        with self.con.cursor() as cur:
            cur.execute(sql)
            cur.execute(f"REMOVE '{path}'")

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Create(this=name, kind="SCHEMA", replace=force)):
            pass

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Drop(this=name, kind="SCHEMA", replace=force)):
            pass

    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        """List tables and views.

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
            Regex to filter by table/view name.
        database
            Database location. If not passed, uses the current database.

            By default uses the current `database` (`self.current_database`) and
            `catalog` (`self.current_catalog`).

            To specify a table in a separate catalog, you can pass in the
            catalog and database as a string `"catalog.database"`, or as a tuple of
            strings `("catalog", "database")`.

        Returns
        -------
        list[str]
            List of table and view names.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.databricks.connect()
        >>> foo = con.create_table("foo", schema=ibis.schema(dict(a="int")))
        >>> con.list_tables()
        ['foo']
        >>> bar = con.create_view("bar", foo)
        >>> con.list_tables()
        ['bar', 'foo']
        >>> con.create_database("my_database")
        >>> con.list_tables(database="my_database")
        []
        >>> con.raw_sql("CREATE TABLE my_database.baz (a INTEGER)")  # doctest: +ELLIPSIS
        <... object at 0x...>
        >>> con.list_tables(database="my_database")
        ['baz']

        """
        table_loc = self._to_sqlglot_table(database)

        catalog = table_loc.catalog or self.current_catalog
        database = table_loc.db or self.current_database

        with self.con.cursor() as cur:
            cur.tables(catalog_name=catalog, schema_name=database)
            out = cur.fetchall_arrow()

        return self._filter_with_like(out["TABLE_NAME"].to_pylist(), like)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Return a stream of record batches.

        The returned `RecordBatchReader` contains a cursor with an unbounded lifetime.

        For analytics use cases this is usually nothing to fret about. In some cases you
        may need to explicit release the cursor.

        Parameters
        ----------
        expr
            Ibis expression
        params
            Bound parameters
        limit
            Limit the result to this number of rows
        chunk_size
            The number of rows to fetch per batch
        """
        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params)

        def batch_producer(con, sql):
            with con.cursor() as cur:
                batched_cur = cur.execute(sql)
                while batch := batched_cur.fetchmany_arrow(size=chunk_size):
                    yield from batch.to_batches()

        pyarrow_schema = expr.as_table().schema().to_pyarrow()
        producer = batch_producer(self.con, sql)
        return pa.ipc.RecordBatchReader.from_batches(pyarrow_schema, producer)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        self._run_pre_execute_hooks(expr)

        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        with self._safe_raw_sql(sql) as cur:
            res = cur.fetchall_arrow()

        target_schema = expr.as_table().schema().to_pyarrow()
        if res is None:
            res = target_schema.empty_table()

        return expr.__pyarrow_result__(res)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if (table := cursor.fetchall_arrow()) is None:
            table = schema.to_pyarrow().empty_table()
        df = table.to_pandas(timestamp_as_object=True)
        df.columns = list(schema.names)
        return df

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with self._safe_raw_sql(
            sge.Describe(this=sg.parse_one(query, read=self.dialect))
        ) as cur:
            rows = cur.fetchall_arrow()

        rows = rows.to_pydict()

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(typ, nullable=True)
                for name, typ in zip(rows["col_name"], rows["data_type"])
            }
        )

    def _get_temp_view_definition(self, name: str, definition: str) -> str:
        return sge.Create(
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            kind="VIEW",
            expression=definition,
            replace=True,
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
        )

    def _create_temp_view(self, table_name, source):
        with self._safe_raw_sql(self._get_temp_view_definition(table_name, source)):
            pass

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.

        """
        old = sg.table(old_name, quoted=True)
        new = sg.table(new_name, quoted=True)
        query = AlterTable(
            this=old, exists=False, actions=[RenameTable(this=new, exists=True)]
        )
        with self._safe_raw_sql(query):
            pass
