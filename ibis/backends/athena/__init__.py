"""Databricks backend."""

from __future__ import annotations

import contextlib
import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec
import pyarrow_hotfix  # noqa: F401
import pyathena
import sqlglot as sg
import sqlglot.expressions as sge
from pyathena.arrow.cursor import ArrowCursor

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, UrlFromPath
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import AlterTable, RenameTable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from pyathena.cursor import Cursor


class Backend(SQLBackend, CanCreateDatabase, UrlFromPath):
    name = "athena"
    compiler = sc.athena.compiler

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(
            sg.select(self.compiler.v.current_catalog), unload=False
        ) as cur:
            [(db,)] = cur.fetchall()
        return db

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(
            sg.select(self.compiler.v.current_schema), unload=False
        ) as cur:
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
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool | None = None,
        comment: str | None = None,
        properties: Mapping[str, Any] | None = None,
        location: str | None = None,
        stored_as: str = "PARQUET",
    ) -> ir.Table:
        """Create a table in Amazon Athena.

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
            This parameter is not yet supported in the Amazon Athena backend, because
            Amazon Athena doesn't implement temporary tables
        overwrite
            If `True`, replace the table if it already exists, otherwise fail if
            the table exists
        comment
            Add a comment to the table
        properties
            Table properties to set on creation
        location
            s3 location to store table data. Defaults to the `s3_staging_dir`
            bucket with the table name as the bucket key.
        stored_as
            The file format in which to store table data. Defaults to parquet.
        """
        if overwrite is not None:
            raise com.UnsupportedOperationError(
                "Amazon Athena does not support REPLACE syntax, nor does it "
                "support syntax for alternative implementations that would use ALTER TABLE RENAME TO"
            )
        if obj is None and schema is None:
            raise com.IbisError("One of the `schema` or `obj` parameter is required")
        if schema is not None:
            schema = ibis.schema(schema)

        if temp:
            raise NotImplementedError(
                "Temporary tables are not supported in the Amazon Athena backend"
            )

        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        compiler = self.compiler
        quoted = compiler.quoted
        orig_table_ref = sg.table(name, catalog=catalog, db=db, quoted=quoted)

        if overwrite:
            name = util.gen_name(f"{self.name}_overwrite")

        table_ref = sg.table(name, catalog=catalog, db=db, quoted=quoted)

        if schema is not None and obj is None:
            target = sge.Schema(
                this=table_ref, expressions=schema.to_sqlglot(self.dialect)
            )
        else:
            target = table_ref

        if location is None:
            location = f"{self._s3_staging_dir}/{name}"

        property_list = [
            sge.ExternalProperty(),
            sge.FileFormatProperty(this=compiler.v[stored_as]),
            sge.LocationProperty(this=sge.convert(location)),
        ]

        for k, v in (properties or {}).items():
            name = sg.to_identifier(k)
            expr = ibis.literal(v)
            value = compiler.visit_Literal(expr.op(), value=v, dtype=expr.type())
            property_list.append(sge.Property(this=name, value=value))

        if comment:
            property_list.append(sge.SchemaCommentProperty(this=sge.convert(comment)))

        if obj is not None:
            if isinstance(obj, ir.Table):
                table = obj
            else:
                table = ibis.memtable(obj, schema=schema)

            self._run_pre_execute_hooks(table)

            # cast here because trino (and therefore athena) doesn't allow
            # specifying a schema in CTAS
            #
            # e.g., `CREATE TABLE (schema) AS SELECT`
            select = sg.select(
                *(
                    compiler.cast(sg.column(name, quoted=quoted), typ).as_(
                        name, quoted=quoted
                    )
                    for name, typ in (schema or table.schema()).items()
                )
            ).from_(compiler.to_sqlglot(table).subquery())
        else:
            select = None

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            expression=select,
            properties=sge.Properties(expressions=property_list),
        )

        with self._safe_raw_sql(create_stmt, unload=False):
            pass

        return self.table(orig_table_ref.name, database=(catalog, db))

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
        with self.con.cursor() as cur:
            tables = cur.list_table_metadata(
                catalog_name=catalog, schema_name=database, expression=table_name
            )

        if not tables:
            raise com.TableNotFound(table.sql(self.dialect))

        (table,) = tables

        return sch.Schema(
            {
                metacol.name: self.compiler.type_mapper.from_string(metacol.type)
                for metacol in table.columns
            }
        )

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, *args, unload: bool = True, **kwargs):
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)
        with self.con.cursor(unload=unload) as cur:
            yield cur.execute(query, *args, **kwargs)

    def list_catalogs(self, like: str | None = None) -> list[str]:
        response = self.con.client.list_data_catalogs()
        catalogs = [
            element["CatalogName"] for element in response["DataCatalogsSummary"]
        ]
        return self._filter_with_like(catalogs, like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        """List databases.

        Parameters
        ----------
        like
            Regular expression to use to match database names.
        catalog
            Catalog in which to search for databases.

        Returns
        -------
        list[str]
            A list of databases in `catalog` matching the pattern `like`.
        """
        if catalog is None:
            catalog = self.current_catalog
        with self.con.cursor() as cur:
            dbs = [db.name for db in cur.list_databases(catalog_name=catalog)]
        return self._filter_with_like(dbs, like=like)

    @property
    def version(self) -> str:
        raise NotImplementedError(
            "athena does not provide a way to programmatically access its version"
        )

    def do_connect(
        self,
        *,
        s3_staging_dir: str,
        cursor_class: type[Cursor] = ArrowCursor,
        memtable_volume: str | None = None,
        **config: Any,
    ) -> None:
        """Create an Ibis client connected to an Amazon Athena instance."""
        self.con = pyathena.connect(
            s3_staging_dir=s3_staging_dir, cursor_class=cursor_class, **config
        )

        if memtable_volume is None:
            short_version = "".join(map(str, sys.version_info[:3]))
            memtable_volume = (
                f"{getpass.getuser()}-py={short_version}-pid={os.getpid()}"
            )
        self._s3_staging_dir = s3_staging_dir.removesuffix("/")
        self._memtable_volume = memtable_volume
        self._memtable_catalog = self.current_catalog
        self._memtable_database = self.current_database
        self._memtable_volume_path = "/".join(  # noqa: FLY002
            (
                self._s3_staging_dir,
                self._memtable_catalog,
                self._memtable_database,
                self._memtable_volume,
            )
        )
        self._fs = fsspec.filesystem("s3")
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
        """Create an Ibis client from an existing connection to an Amazon Athena instance.

        Parameters
        ----------
        con
            An existing connection to an Amazon Athena instance.
        memtable_volume
            The volume to use for Ibis memtables.
        """
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect(memtable_volume=memtable_volume)
        return new_backend

    def _post_connect(self, *, memtable_volume: str) -> None:
        pass

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        import pyarrow.parquet as pq

        compiler = self.compiler
        quoted = compiler.quoted
        name = op.name
        schema = op.schema

        upstream_path = f"{self._memtable_volume_path}/{name}"
        sql = sge.Create(
            kind="TABLE",
            this=sge.Schema(
                this=sg.table(
                    name,
                    db=self.current_database,
                    catalog=self.current_catalog,
                    quoted=quoted,
                ),
                expressions=schema.to_sqlglot(self.dialect),
            ),
            properties=sge.Properties(
                expressions=[
                    sge.ExternalProperty(),
                    sge.FileFormatProperty(this=compiler.v.PARQUET),
                    sge.LocationProperty(this=sge.convert(upstream_path)),
                ]
            ),
        )

        raw_upstream_dir = upstream_path.removeprefix("s3://")
        raw_upstream_path = f"{raw_upstream_dir}/data.parquet"

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            data = op.data.to_pyarrow(schema=schema)
            path = Path(tmpdir, name)
            # optimize for bandwidth so use zstd which typically compresses
            # better than the other options without much loss in speed
            pq.write_table(data, path, compression="zstd")
            self._fs.put(path, raw_upstream_path)

            with self._safe_raw_sql(sql, unload=False):
                pass

    def _finalize_memtable(self, name: str) -> None:
        path = f"{self._memtable_volume_path}/{name}"
        sql = sge.Drop(
            kind="TABLE",
            this=sg.to_identifier(name, quoted=self.compiler.quoted),
            exists=True,
        )

        with self._safe_raw_sql(sql, unload=False):
            pass

        self._fs.rm(path, recursive=True)

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        sql = sge.Create(this=name, kind="SCHEMA", replace=force)
        with self._safe_raw_sql(sql, unload=False):
            pass

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        sql = sge.Drop(this=name, kind="SCHEMA", replace=force)
        with self._safe_raw_sql(sql, unload=False):
            pass

    def list_tables(
        self, like: str | None = None, database: tuple[str, str] | str | None = None
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
        >>> con = ibis.athena.connect()
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
            tables = cur.list_table_metadata(
                catalog_name=catalog, schema_name=database, expression=like
            )

        return sorted(table.name for table in tables)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
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
        """
        return self.to_pyarrow(expr.as_table(), params=params, limit=limit).to_reader()

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
            res = cur.as_arrow()

        ibis_schema = expr.as_table().schema()
        target_schema = ibis_schema.to_pyarrow()

        if not res:
            res = target_schema.empty_table()
        else:
            res = res.rename_columns(list(ibis_schema.names))

        return expr.__pyarrow_result__(res)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        if not (table := cursor.as_arrow()):
            table = schema.to_pyarrow().empty_table()
        df = table.to_pandas(timestamp_as_object=True)
        df.columns = list(schema.names)
        return df

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        view_name = util.gen_name("temp_view_athena")

        compiler = self.compiler
        quoted = compiler.quoted
        dialect = self.dialect

        create_view = sge.Create(
            this=sg.to_identifier(view_name, quoted=quoted),
            kind="VIEW",
            expression=sg.parse_one(query, dialect=dialect),
        )
        drop_view = sge.Drop(
            kind="VIEW", this=sg.to_identifier(view_name, quoted=quoted)
        ).sql(dialect)

        catalog_name = self.current_catalog
        schema_name = self.current_database
        with self._safe_raw_sql(create_view, unload=False) as cur:
            (table_meta,) = cur.list_table_metadata(
                catalog_name=catalog_name,
                schema_name=schema_name,
                expression=view_name,
                max_results=1,
            )
            cur.execute(drop_view)

        type_mapper = compiler.type_mapper
        return sch.Schema(
            {
                metacol.name: type_mapper.from_string(metacol.type, nullable=True)
                for metacol in table_meta.columns
            }
        )

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
        with self._safe_raw_sql(query, unload=False):
            pass
