"""DuckDB backend."""

from __future__ import annotations

import ast
import contextlib
import urllib
import warnings
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge
from packaging.version import parse as vparse

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanCreateDatabase, UrlFromPath
from ibis.backends.duckdb.converter import DuckDBPandasData
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import STAR, AlterTable, C, RenameTable
from ibis.common.dispatch import lazy_singledispatch
from ibis.expr.operations.udf import InputType

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableMapping, Sequence

    import pandas as pd
    import polars as pl
    import torch
    from fsspec import AbstractFileSystem

    from ibis.expr.schema import SchemaLike


_UDF_INPUT_TYPE_MAPPING = {
    InputType.PYARROW: duckdb.functional.ARROW,
    InputType.PYTHON: duckdb.functional.NATIVE,
}


class _Settings:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con

    def __getitem__(self, key: str) -> Any:
        maybe_value = self.con.execute(
            "select value from duckdb_settings() where name = $1", [key]
        ).fetchone()
        if maybe_value is not None:
            return maybe_value[0]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.con.execute(f"SET {key} = {str(value)!r}")

    def __repr__(self):
        return repr(self.con.sql("from duckdb_settings()"))


class Backend(SQLBackend, CanCreateDatabase, UrlFromPath):
    name = "duckdb"
    compiler = sc.duckdb.compiler

    @property
    def settings(self) -> _Settings:
        return _Settings(self.con)

    @property
    def current_catalog(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_database())) as cur:
            [(db,)] = cur.fetchall()
        return db

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_schema())) as cur:
            [(db,)] = cur.fetchall()
        return db

    # TODO(kszucs): should be moved to the base SQLGLot backend
    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)
        return self.con.execute(query, **kwargs)

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
    ):
        """Create a table in DuckDB.

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

        """
        table_loc = self._to_sqlglot_table(database)

        if getattr(table_loc, "catalog", False) and temp:
            raise exc.UnsupportedArgumentError(
                "DuckDB can only create temporary tables in the `temp` catalog. "
                "Don't specify a catalog to enable temp table creation."
            )

        catalog = table_loc.catalog or self.current_catalog
        database = table_loc.db or self.current_database

        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        quoted = self.compiler.quoted
        dialect = self.dialect

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())
            catalog = "temp"

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
            temp_name = util.gen_name("duckdb_table")
        else:
            temp_name = name

        initial_table = sg.table(temp_name, catalog=catalog, db=database, quoted=quoted)
        target = sge.Schema(
            this=initial_table,
            expressions=(schema or table.schema()).to_sqlglot(dialect),
        )

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

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
                # TODO: This branching should be removed once DuckDB >=0.9.3 is
                # our lower bound (there's an upstream bug in 0.9.2 that
                # disallows renaming temp tables)
                # We should (pending that release) be able to remove the if temp
                # branch entirely.
                if temp:
                    cur.execute(
                        sge.Create(
                            kind="TABLE",
                            this=final_table,
                            expression=sg.select(STAR).from_(initial_table),
                            properties=sge.Properties(expressions=properties),
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
        # load geospatial only if geo columns
        if any(typ.is_geospatial() for typ in table_schema.types):
            self.load_extension("spatial")
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
        query = sge.Describe(
            this=sg.table(
                table_name, db=database, catalog=catalog, quoted=self.compiler.quoted
            )
        ).sql(self.dialect)

        try:
            result = self.con.sql(query)
        except duckdb.CatalogException:
            raise exc.TableNotFound(table_name)
        else:
            meta = result.fetch_arrow_table()

        names = meta["column_name"].to_pylist()
        types = meta["column_type"].to_pylist()
        nullables = meta["null"].to_pylist()

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(typ, nullable=null == "YES")
                for name, typ, null in zip(names, types, nullables)
            }
        )

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        yield self.raw_sql(*args, **kwargs)

    def list_catalogs(self, like: str | None = None) -> list[str]:
        col = "catalog_name"
        query = sg.select(sge.Distinct(expressions=[sg.column(col)])).from_(
            sg.table("schemata", db="information_schema")
        )
        with self._safe_raw_sql(query) as cur:
            result = cur.fetch_arrow_table()
        dbs = result[col]
        return self._filter_with_like(dbs.to_pylist(), like)

    def list_databases(
        self, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        col = "schema_name"
        query = sg.select(sge.Distinct(expressions=[sg.column(col)])).from_(
            sg.table("schemata", db="information_schema")
        )

        if catalog is not None:
            query = query.where(sg.column("catalog_name").eq(sge.convert(catalog)))

        with self._safe_raw_sql(query) as cur:
            out = cur.fetch_arrow_table()
        return self._filter_with_like(out[col].to_pylist(), like=like)

    @staticmethod
    def _convert_kwargs(kwargs: MutableMapping) -> None:
        read_only = str(kwargs.pop("read_only", "False")).capitalize()
        try:
            kwargs["read_only"] = ast.literal_eval(read_only)
        except ValueError as e:
            raise ValueError(
                f"invalid value passed to ast.literal_eval: {read_only!r}"
            ) from e

    @property
    def version(self) -> str:
        # TODO: there is a `PRAGMA version` we could use instead
        import importlib.metadata

        return importlib.metadata.version("duckdb")

    def do_connect(
        self,
        database: str | Path = ":memory:",
        read_only: bool = False,
        extensions: Sequence[str] | None = None,
        **config: Any,
    ) -> None:
        """Create an Ibis client connected to a DuckDB database.

        ::: {.callout-note title="Changed in version 10.0.0"}
        Before, we had special handling if the user passed the `temp_directory`
        parameter, setting a custom default, and creating intermediate
        directories if necessary. Now, we do nothing, and just pass the value
        directly to DuckDB. You may need to add
        `Path(your_temp_dir).mkdir(exists_ok=True, parents=True)`
        to your code to maintain the old behavior.
        :::

        Parameters
        ----------
        database
            Path to a duckdb database.
        read_only
            Whether the database is read-only.
        extensions
            A list of duckdb extensions to install/load upon connection.
        config
            DuckDB configuration parameters. See the [DuckDB configuration
            documentation](https://duckdb.org/docs/sql/configuration) for
            possible configuration values.

        Examples
        --------
        >>> import ibis
        >>> ibis.duckdb.connect(threads=4, memory_limit="1GB")  # doctest: +ELLIPSIS
        <ibis.backends.duckdb.Backend object at 0x...>
        """
        if not isinstance(database, Path) and not database.startswith(
            ("md:", "motherduck:", ":memory:")
        ):
            database = Path(database).absolute()
        self.con = duckdb.connect(str(database), config=config, read_only=read_only)
        self._post_connect(extensions)

    @util.experimental
    @classmethod
    def from_connection(
        cls,
        con: duckdb.DuckDBPyConnection,
        extensions: Sequence[str] | None = None,
    ) -> Backend:
        """Create an Ibis client from an existing connection to a DuckDB database.

        Parameters
        ----------
        con
            An existing connection to a DuckDB database.
        extensions
            A list of duckdb extensions to install/load upon connection.
        """
        new_backend = cls(extensions=extensions)
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect(extensions)
        return new_backend

    def _post_connect(self, extensions: Sequence[str] | None = None) -> None:
        # Load any pre-specified extensions
        if extensions is not None:
            self._load_extensions(extensions)

        # Default timezone, can't be set with `config`
        self.settings["timezone"] = "UTC"

        # setting this to false disables magic variables-as-tables discovery,
        # hopefully eliminating large classes of bugs
        if vparse(self.version) > vparse("1"):
            self.settings["python_enable_replacements"] = False

        self._record_batch_readers_consumed = {}

    def _load_extensions(
        self, extensions: list[str], force_install: bool = False
    ) -> None:
        f = self.compiler.f
        query = (
            sg.select(f.anon.unnest(f.list_append(C.aliases, C.extension_name)))
            .from_(f.duckdb_extensions())
            .where(C.installed, C.loaded)
        )
        with self._safe_raw_sql(query) as cur:
            installed = map(itemgetter(0), cur.fetchall())
            # Install and load all other extensions
            todo = frozenset(extensions).difference(installed)
            for extension in todo:
                cur.install_extension(extension, force_install=force_install)
                cur.load_extension(extension)

    def load_extension(self, extension: str, force_install: bool = False) -> None:
        """Install and load a duckdb extension by name or path.

        Parameters
        ----------
        extension
            The extension name or path.
        force_install
            Force reinstallation of the extension.

        """
        self._load_extensions([extension], force_install=force_install)

    def create_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot create a database in another catalog."
            )

        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Create(this=name, kind="SCHEMA", replace=force)):
            pass

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        if catalog is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot drop a database in another catalog."
            )

        name = sg.table(name, catalog=catalog, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Drop(this=name, kind="SCHEMA", replace=force)):
            pass

    @util.experimental
    def read_json(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
        columns: Mapping[str, str] | None = None,
        **kwargs,
    ) -> ir.Table:
        """Read newline-delimited JSON into an ibis table.

        ::: {.callout-note}
        ## This feature requires duckdb>=0.7.0
        :::

        Parameters
        ----------
        source_list
            File or list of files
        table_name
            Optional table name
        columns
            Optional mapping from string column name to duckdb type string.
        **kwargs
            Additional keyword arguments passed to DuckDB's `read_json_auto` function.

            See https://duckdb.org/docs/data/json/overview.html#json-loading
            for parameters and more information about reading JSON.

        Returns
        -------
        Table
            An ibis table expression

        """
        if not table_name:
            table_name = util.gen_name("read_json")

        options = [
            sg.to_identifier(key).eq(sge.convert(val)) for key, val in kwargs.items()
        ]

        if columns:
            options.append(
                sg.to_identifier("columns").eq(
                    sge.Struct.from_arg_list(
                        [
                            sge.PropertyEQ(
                                this=sg.to_identifier(key),
                                expression=sge.convert(value),
                            )
                            for key, value in columns.items()
                        ]
                    )
                )
            )

        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(
                self.compiler.f.read_json_auto(
                    util.normalize_filenames(source_list), *options
                )
            ),
        )

        return self.table(table_name)

    def read_csv(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
        columns: Mapping[str, str | dt.DataType] | None = None,
        types: Mapping[str, str | dt.DataType] | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        source_list
            The data source(s). May be a path to a file or directory of CSV
            files, or an iterable of CSV files.
        table_name
            An optional name to use for the created table. This defaults to a
            sequentially generated name.
        columns
            An optional mapping of **all** column names to their types.
        types
            An optional mapping of a **subset** of column names to their types.
        **kwargs
            Additional keyword arguments passed to DuckDB loading function. See
            https://duckdb.org/docs/data/csv for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        Examples
        --------
        Generate some data

        >>> import tempfile
        >>> data = b'''
        ... lat,lon,geom
        ... 1.0,2.0,POINT (1 2)
        ... 2.0,3.0,POINT (2 3)
        ... '''
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nbytes = f.write(data)

        Import Ibis

        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> con = ibis.duckdb.connect()

        Read the raw CSV file

        >>> t = con.read_csv(f.name)
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ lat     ┃ lon     ┃ geom        ┃
        ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ float64 │ float64 │ string      │
        ├─────────┼─────────┼─────────────┤
        │     1.0 │     2.0 │ POINT (1 2) │
        │     2.0 │     3.0 │ POINT (2 3) │
        └─────────┴─────────┴─────────────┘

        Load the `spatial` extension and read the CSV file again, using
        specific column types

        >>> con.load_extension("spatial")
        >>> t = con.read_csv(f.name, types={"geom": "geometry"})
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ lat     ┃ lon     ┃ geom                 ┃
        ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ float64 │ float64 │ geospatial:geometry  │
        ├─────────┼─────────┼──────────────────────┤
        │     1.0 │     2.0 │ <POINT (1 2)>        │
        │     2.0 │     3.0 │ <POINT (2 3)>        │
        └─────────┴─────────┴──────────────────────┘
        """
        source_list = util.normalize_filenames(source_list)

        if not table_name:
            table_name = util.gen_name("read_csv")

        # auto_detect and columns collide, so we set auto_detect=True
        # unless COLUMNS has been specified
        if any(
            source.startswith(("http://", "https://", "s3://"))
            for source in source_list
        ):
            self._load_extensions(["httpfs"])

        kwargs.setdefault("header", True)
        kwargs["auto_detect"] = kwargs.pop("auto_detect", columns is None)
        # TODO: clean this up
        # We want to _usually_ quote arguments but if we quote `columns` it messes
        # up DuckDB's struct parsing.
        options = [C[key].eq(sge.convert(val)) for key, val in kwargs.items()]

        def make_struct_argument(obj: Mapping[str, str | dt.DataType]) -> sge.Struct:
            expressions = []
            geospatial = False
            type_mapper = self.compiler.type_mapper

            for name, typ in obj.items():
                typ = dt.dtype(typ)
                geospatial |= typ.is_geospatial()
                sgtype = type_mapper.from_ibis(typ)
                prop = sge.PropertyEQ(
                    this=sge.to_identifier(name), expression=sge.convert(sgtype)
                )
                expressions.append(prop)

            if geospatial:
                self._load_extensions(["spatial"])
            return sge.Struct(expressions=expressions)

        if columns is not None:
            options.append(C.columns.eq(make_struct_argument(columns)))

        if types is not None:
            options.append(C.types.eq(make_struct_argument(types)))

        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(self.compiler.f.read_csv(source_list, *options)),
        )

        return self.table(table_name)

    def read_geo(
        self,
        source: str,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a GEO file as a table in the current database.

        Parameters
        ----------
        source
            The data source(s). Path to a file of geospatial files supported
            by duckdb.
            See https://duckdb.org/docs/extensions/spatial.html#st_read---read-spatial-data-from-files
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DuckDB loading function.
            See https://duckdb.org/docs/extensions/spatial.html#st_read---read-spatial-data-from-files
            for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """

        if not table_name:
            table_name = util.gen_name("read_geo")

        # load geospatial extension
        self.load_extension("spatial")

        source = util.normalize_filename(source)
        if source.startswith(("http://", "https://", "s3://")):
            self._load_extensions(["httpfs"])

        source_expr = sg.select(STAR).from_(
            self.compiler.f.st_read(
                source,
                *(sg.to_identifier(key).eq(val) for key, val in kwargs.items()),
            )
        )

        view = sge.Create(
            kind="VIEW",
            this=sg.table(table_name, quoted=self.compiler.quoted),
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
            expression=source_expr,
        )
        with self._safe_raw_sql(view):
            pass
        return self.table(table_name)

    def read_parquet(
        self,
        source_list: str | Iterable[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        source_list
            The data source(s). May be a path to a file, an iterable of files,
            or directory of parquet files.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DuckDB loading function.
            See https://duckdb.org/docs/data/parquet for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source_list = util.normalize_filenames(source_list)

        table_name = table_name or util.gen_name("read_parquet")

        # Default to using the native duckdb parquet reader
        # If that fails because of auth issues, fall back to ingesting via
        # pyarrow dataset
        try:
            self._read_parquet_duckdb_native(source_list, table_name, **kwargs)
        except duckdb.IOException:
            self._read_parquet_pyarrow_dataset(source_list, table_name, **kwargs)

        return self.table(table_name)

    def _read_parquet_duckdb_native(
        self, source_list: str | Iterable[str], table_name: str, **kwargs: Any
    ) -> None:
        if any(
            source.startswith(("http://", "https://", "s3://"))
            for source in source_list
        ):
            self._load_extensions(["httpfs"])

        options = [
            sg.to_identifier(key).eq(sge.convert(val)) for key, val in kwargs.items()
        ]
        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(self.compiler.f.read_parquet(source_list, *options)),
        )

    def _read_parquet_pyarrow_dataset(
        self, source_list: str | Iterable[str], table_name: str, **kwargs: Any
    ) -> None:
        import pyarrow.dataset as ds

        dataset = ds.dataset(list(map(ds.dataset, source_list)), **kwargs)
        self._load_extensions(["httpfs"])
        # We don't create a view since DuckDB special cases Arrow Datasets
        # so if we also create a view we end up with both a "lazy table"
        # and a view with the same name
        self.con.register(table_name, dataset)
        # DuckDB normally auto-detects Arrow Datasets that are defined
        # in local variables but the `dataset` variable won't be local
        # by the time we execute against this so we register it
        # explicitly.

    def read_delta(
        self,
        source_table: str,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a Delta Lake table as a table in the current database.

        Parameters
        ----------
        source_table
            The data source. Must be a directory
            containing a Delta Lake table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to deltalake.DeltaTable.

        Returns
        -------
        ir.Table
            The just-registered table.

        """
        source_table = util.normalize_filenames(source_table)[0]

        table_name = table_name or util.gen_name("read_delta")

        try:
            from deltalake import DeltaTable
        except ImportError:
            raise ImportError(
                "The deltalake extra is required to use the "
                "read_delta method. You can install it using pip:\n\n"
                "pip install 'ibis-framework[deltalake]'\n"
            )

        delta_table = DeltaTable(source_table, **kwargs)

        self.con.register(table_name, delta_table.to_pyarrow_dataset())
        return self.table(table_name)

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
        >>> con = ibis.duckdb.connect()
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
        <duckdb.duckdb.DuckDBPyConnection object at 0x...>
        >>> con.list_tables(database="my_database")
        ['baz']

        """
        table_loc = self._to_sqlglot_table(database)

        catalog = table_loc.catalog or self.current_catalog
        database = table_loc.db or self.current_database

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(
                C.table_catalog.isin(sge.convert(catalog), sge.convert("temp")),
                C.table_schema.eq(sge.convert(database)),
            )
            .sql(self.dialect)
        )
        out = self.con.execute(sql).fetch_arrow_table()

        return self._filter_with_like(out[col].to_pylist(), like)

    def read_postgres(
        self, uri: str, *, table_name: str | None = None, database: str = "public"
    ) -> ir.Table:
        """Register a table from a postgres instance into a DuckDB table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        uri
            A postgres URI of the form `postgres://user:password@host:port`
        table_name
            The table to read
        database
            PostgreSQL database (schema) where `table_name` resides

        Returns
        -------
        ir.Table
            The just-registered table.

        """
        if table_name is None:
            raise ValueError(
                "`table_name` is required when registering a postgres table"
            )
        self._load_extensions(["postgres_scanner"])

        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(
                self.compiler.f.postgres_scan_pushdown(uri, database, table_name)
            ),
        )

        return self.table(table_name)

    def read_mysql(
        self,
        uri: str,
        *,
        catalog: str,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a table from a MySQL instance into a DuckDB table.

        Parameters
        ----------
        uri
            A mysql URI of the form `mysql://user:password@host:port/database`
        catalog
            User-defined alias given to the MySQL database that is being attached
            to DuckDB
        table_name
            The table to read

        Returns
        -------
        ir.Table
            The just-registered table.
        """

        parsed = urllib.parse.urlparse(uri)

        if table_name is None:
            raise ValueError("`table_name` is required when registering a mysql table")

        self._load_extensions(["mysql"])

        database = parsed.path.strip("/")

        query_con = f"""ATTACH 'host={parsed.hostname} user={parsed.username} password={parsed.password} port={parsed.port} database={database}' AS {catalog} (TYPE mysql)"""

        with self._safe_raw_sql(query_con):
            pass

        return self.table(table_name, database=(catalog, database))

    def read_sqlite(
        self, path: str | Path, *, table_name: str | None = None
    ) -> ir.Table:
        """Register a table from a SQLite database into a DuckDB table.

        Parameters
        ----------
        path
            The path to the SQLite database
        table_name
            The table to read

        Returns
        -------
        ir.Table
            The just-registered table.

        Examples
        --------
        >>> import ibis
        >>> import sqlite3
        >>> ibis.options.interactive = True
        >>> con = sqlite3.connect("/tmp/sqlite.db")
        >>> with con:
        ...     con.execute("DROP TABLE IF EXISTS t")  # doctest: +ELLIPSIS
        ...     con.execute("CREATE TABLE t (a INT, b TEXT)")  # doctest: +ELLIPSIS
        ...     con.execute(
        ...         "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        ...     )  # doctest: +ELLIPSIS
        <...>
        >>> con.close()
        >>> con = ibis.connect("duckdb://")
        >>> t = con.read_sqlite(path="/tmp/sqlite.db", table_name="t")
        >>> t
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ a      │
        │     2 │ b      │
        │     3 │ c      │
        └───────┴────────┘

        """

        if table_name is None:
            raise ValueError("`table_name` is required when registering a sqlite table")
        self._load_extensions(["sqlite"])

        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(
                self.compiler.f.sqlite_scan(
                    sg.to_identifier(str(path), quoted=True), table_name
                )
            ),
        )

        return self.table(table_name)

    def attach(
        self, path: str | Path, name: str | None = None, read_only: bool = False
    ) -> None:
        """Attach another DuckDB database to the current DuckDB session.

        Parameters
        ----------
        path
            Path to the database to attach.
        name
            Name to attach the database as. Defaults to the basename of `path`.
        read_only
            Whether to attach the database as read-only.

        """
        code = f"ATTACH '{path}'"

        if name is not None:
            name = sg.to_identifier(name).sql(self.name)
            code += f" AS {name}"

        if read_only:
            code += " (READ_ONLY)"

        self.con.execute(code).fetchall()

    def detach(self, name: str) -> None:
        """Detach a database from the current DuckDB session.

        Parameters
        ----------
        name
            The name of the database to detach.

        """
        name = sg.to_identifier(name).sql(self.name)
        self.con.execute(f"DETACH {name}").fetchall()

    def attach_sqlite(
        self, path: str | Path, overwrite: bool = False, all_varchar: bool = False
    ) -> None:
        """Attach a SQLite database to the current DuckDB session.

        Parameters
        ----------
        path
            The path to the SQLite database.
        overwrite
            Allow overwriting any tables or views that already exist in your current
            session with the contents of the SQLite database.
        all_varchar
            Set all SQLite columns to type `VARCHAR` to avoid type errors on ingestion.

        Examples
        --------
        >>> import ibis
        >>> import sqlite3
        >>> con = sqlite3.connect("/tmp/attach_sqlite.db")
        >>> with con:
        ...     con.execute("DROP TABLE IF EXISTS t")  # doctest: +ELLIPSIS
        ...     con.execute("CREATE TABLE t (a INT, b TEXT)")  # doctest: +ELLIPSIS
        ...     con.execute(
        ...         "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        ...     )  # doctest: +ELLIPSIS
        <...>
        >>> con.close()
        >>> con = ibis.connect("duckdb://")
        >>> con.list_tables()
        []
        >>> con.attach_sqlite("/tmp/attach_sqlite.db")
        >>> con.list_tables()
        ['t']

        """
        self.load_extension("sqlite")
        with self._safe_raw_sql(f"SET GLOBAL sqlite_all_varchar={all_varchar}") as cur:
            cur.execute(
                f"CALL sqlite_attach('{path}', overwrite={overwrite})"
            ).fetchall()

    def register_filesystem(self, filesystem: AbstractFileSystem):
        """Register an `fsspec` filesystem object with DuckDB.

        This allow a user to read from any `fsspec` compatible filesystem using
        `read_csv`, `read_parquet`, `read_json`, etc.


        ::: {.callout-note}
        Creating an `fsspec` filesystem requires that the corresponding
        backend-specific `fsspec` helper library is installed.

        e.g. to connect to Google Cloud Storage, `gcsfs` must be installed.
        :::

        Parameters
        ----------
        filesystem
            The fsspec filesystem object to register with DuckDB.
            See https://duckdb.org/docs/guides/python/filesystems for details.

        Examples
        --------
        >>> import ibis
        >>> import fsspec
        >>> ibis.options.interactive = True
        >>> gcs = fsspec.filesystem("gcs")
        >>> con = ibis.duckdb.connect()
        >>> con.register_filesystem(gcs)
        >>> t = con.read_csv(
        ...     "gcs://ibis-examples/data/band_members.csv.gz",
        ...     table_name="band_members",
        ... )
        >>> t
        ┏━━━━━━━━┳━━━━━━━━━┓
        ┃ name   ┃ band    ┃
        ┡━━━━━━━━╇━━━━━━━━━┩
        │ string │ string  │
        ├────────┼─────────┤
        │ Mick   │ Stones  │
        │ John   │ Beatles │
        │ Paul   │ Beatles │
        └────────┴─────────┘
        """
        self.con.register_filesystem(filesystem)

    def _run_pre_execute_hooks(self, expr: ir.Expr) -> None:
        # Warn for any tables depending on RecordBatchReaders that have already
        # started being consumed.
        for t in expr.op().find(ops.PhysicalTable):
            started = self._record_batch_readers_consumed.get(t.name)
            if started is True:
                warnings.warn(
                    f"Table {t.name!r} is backed by a `pyarrow.RecordBatchReader` "
                    "that has already been partially consumed. This may lead to "
                    "unexpected results. Either recreate the table from a new "
                    "`pyarrow.RecordBatchReader`, or use `Table.cache()`/"
                    "`con.create_table()` to consume and store the results in "
                    "the backend to reuse later."
                )
            elif started is False:
                self._record_batch_readers_consumed[t.name] = True

        if expr.op().find((ops.GeoSpatialUnOp, ops.GeoSpatialBinOp)):
            self.load_extension("spatial")

        super()._run_pre_execute_hooks(expr)

    def _to_duckdb_relation(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
    ):
        """Preprocess the expr, and return a `duckdb.DuckDBPyRelation` object.

        When retrieving in-memory results, it's faster to use `duckdb_con.sql`
        than `duckdb_con.execute`, as the query planner can take advantage of
        knowing the output type. Since the relation objects aren't compatible
        with the dbapi, we choose to only use them in select internal methods
        where performance might matter, and use the standard
        `duckdb_con.execute` everywhere else.
        """
        self._run_pre_execute_hooks(expr)
        table_expr = expr.as_table()
        sql = self.compile(table_expr, limit=limit, params=params)
        if table_expr.schema().geospatial:
            self._load_extensions(["spatial"])
        return self.con.sql(sql)

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

        def batch_producer(cur):
            yield from cur.fetch_record_batch(rows_per_batch=chunk_size)

        result = self.raw_sql(sql)
        return pa.ipc.RecordBatchReader.from_batches(
            expr.as_table().schema().to_pyarrow(), batch_producer(result)
        )

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **_: Any,
    ) -> pa.Table:
        table = self._to_duckdb_relation(expr, params=params, limit=limit).arrow()
        return expr.__pyarrow_result__(table)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **_: Any,
    ) -> Any:
        """Execute an expression."""
        import pandas as pd
        import pyarrow.types as pat

        table = self._to_duckdb_relation(expr, params=params, limit=limit).arrow()

        df = pd.DataFrame(
            {
                name: (
                    col.to_pylist()
                    if (
                        pat.is_nested(col.type)
                        or
                        # pyarrow / duckdb type null literals columns as int32?
                        # but calling `to_pylist()` will render it as None
                        col.null_count
                    )
                    else col.to_pandas()
                )
                for name, col in zip(table.column_names, table.columns)
            }
        )
        df = DuckDBPandasData.convert_table(df, expr.as_table().schema())
        return expr.__pandas_result__(df)

    @util.experimental
    def to_torch(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Execute an expression and return results as a dictionary of torch tensors.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Parameters to substitute into the expression.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        kwargs
            Keyword arguments passed into the backend's `to_torch` implementation.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of torch tensors, keyed by column name.

        """
        return self._to_duckdb_relation(expr, params=params, limit=limit).torch()

    @util.experimental
    def to_parquet(
        self,
        expr: ir.Table,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to parquet.
        path
            The data source. A string or Path to the parquet file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            DuckDB Parquet writer arguments. See
            https://duckdb.org/docs/data/parquet/overview.html#writing-to-parquet-files
            for details.

        Examples
        --------
        Write out an expression to a single parquet file.

        >>> import ibis
        >>> penguins = ibis.examples.penguins.fetch()
        >>> con = ibis.get_backend(penguins)
        >>> con.to_parquet(penguins, "/tmp/penguins.parquet")

        Write out an expression to a hive-partitioned parquet file.

        >>> import tempfile
        >>> penguins = ibis.examples.penguins.fetch()
        >>> con = ibis.get_backend(penguins)

        Partition on a single column.

        >>> con.to_parquet(penguins, tempfile.mkdtemp(), partition_by="year")

        Partition on multiple columns.

        >>> con.to_parquet(penguins, tempfile.mkdtemp(), partition_by=("year", "island"))

        """
        self._run_pre_execute_hooks(expr)
        query = self.compile(expr, params=params)
        args = ["FORMAT 'parquet'", *(f"{k.upper()} {v!r}" for k, v in kwargs.items())]
        copy_cmd = f"COPY ({query}) TO {str(path)!r} ({', '.join(args)})"
        with self._safe_raw_sql(copy_cmd):
            pass

    @util.experimental
    def to_csv(
        self,
        expr: ir.Table,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        header: bool = True,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            The ibis expression to execute and persist to CSV.
        path
            The data source. A string or Path to the CSV file.
        params
            Mapping of scalar parameter expressions to value.
        header
            Whether to write the column names as the first line of the CSV file.
        **kwargs
            DuckDB CSV writer arguments. https://duckdb.org/docs/data/csv/overview.html#parameters

        """
        self._run_pre_execute_hooks(expr)
        query = self.compile(expr, params=params)
        args = [
            "FORMAT 'csv'",
            f"HEADER {int(header)}",
            *(f"{k.upper()} {v!r}" for k, v in kwargs.items()),
        ]
        copy_cmd = f"COPY ({query}) TO {str(path)!r} ({', '.join(args)})"
        with self._safe_raw_sql(copy_cmd):
            pass

    @util.experimental
    def to_geo(
        self,
        expr: ir.Table,
        path: str | Path,
        *,
        format: str,
        layer_creation_options: Mapping[str, Any] | None = None,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing `expr` to a geospatial output.

        Parameters
        ----------
        expr
            Ibis expression to execute and persist to geospatial output.
        path
            A string or Path to the desired output file location.
        format
            The format of the geospatial output. One of GDAL's supported vector formats.
            The list of vector formats is located here: https://gdal.org/en/latest/drivers/vector/index.html
        layer_creation_options
            A mapping of layer creation options.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        kwargs
            Additional keyword arguments passed to the DuckDB `COPY` command.

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> from ibis import _

        Load some geospatial data

        >>> con = ibis.duckdb.connect()
        >>> zones = ibis.examples.zones.fetch(backend=con)
        >>> zones[["zone", "geom"]].head()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ zone                                  ┃ geom                                 ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                                │ geospatial:geometry                  │
        ├───────────────────────────────────────┼──────────────────────────────────────┤
        │                                       │ <POLYGON ((933100.918 192536.086,    │
        │ Newark Airport                        │ 933091.011 192572.175, 933088.585    │
        │                                       │ 192604.9...>                         │
        │                                       │ <MULTIPOLYGON (((1033269.244         │
        │ Jamaica Bay                           │ 172126.008, 1033439.643 170883.946,  │
        │                                       │ 1033473.265...>                      │
        │                                       │ <POLYGON ((1026308.77 256767.698,    │
        │ Allerton/Pelham Gardens               │ 1026495.593 256638.616, 1026567.23   │
        │                                       │ 256589....>                          │
        │                                       │ <POLYGON ((992073.467 203714.076,    │
        │ Alphabet City                         │ 992068.667 203711.502, 992061.716    │
        │                                       │ 203711.7...>                         │
        │                                       │ <POLYGON ((935843.31 144283.336,     │
        │ Arden Heights                         │ 936046.565 144173.418, 936387.922    │
        │                                       │ 143967.75...>                        │
        └───────────────────────────────────────┴──────────────────────────────────────┘

        Write to a GeoJSON file

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     con.to_geo(
        ...         zones,
        ...         path=os.path.join(tmpdir, "zones.geojson"),
        ...         format="geojson",
        ...     )

        Write to a Shapefile

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     con.to_geo(
        ...         zones,
        ...         path=os.path.join(tmpdir, "zones.shp"),
        ...         format="ESRI Shapefile",
        ...     )
        """
        self._run_pre_execute_hooks(expr)
        query = self.compile(expr, params=params, limit=limit)

        args = ["FORMAT GDAL", f"DRIVER '{format}'"]

        if layer_creation_options := " ".join(
            f"{k.upper()}={v}" for k, v in (layer_creation_options or {}).items()
        ):
            args.append(f"LAYER_CREATION_OPTIONS '{layer_creation_options}'")

        args.extend(f"{k.upper()} {v!r}" for k, v in (kwargs or {}).items())

        copy_cmd = f"COPY ({query}) TO {str(path)!r} ({', '.join(args)})"

        with self._safe_raw_sql(copy_cmd):
            pass

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        with self._safe_raw_sql(f"DESCRIBE {query}") as cur:
            rows = cur.fetch_arrow_table()

        rows = rows.to_pydict()

        type_mapper = self.compiler.type_mapper
        return sch.Schema(
            {
                name: type_mapper.from_string(typ, nullable=null == "YES")
                for name, typ, null in zip(
                    rows["column_name"], rows["column_type"], rows["null"]
                )
            }
        )

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        data = op.data
        schema = op.schema

        try:
            obj = data.to_pyarrow_dataset(schema)
        except AttributeError:
            obj = data.to_pyarrow(schema)

        self.con.register(op.name, obj)

    def _finalize_memtable(self, name: str) -> None:
        # if we don't aggressively unregister tables duckdb will keep a
        # reference to every memtable ever registered, even if there's no
        # way for a user to access the operation anymore, resulting in a
        # memory leak
        #
        # we can't use drop_table, because self.con.register creates a view, so
        # use the corresponding unregister method
        self.con.unregister(name)

    def _register_udfs(self, expr: ir.Expr) -> None:
        con = self.con

        for udf_node in expr.op().find(ops.ScalarUDF):
            register_func = getattr(
                self, f"_register_{udf_node.__input_type__.name.lower()}_udf"
            )
            with contextlib.suppress(duckdb.InvalidInputException):
                con.remove_function(udf_node.__class__.__name__)

            registration_func = register_func(udf_node)
            if registration_func is not None:
                registration_func(con)

    def _register_udf(self, udf_node: ops.ScalarUDF):
        type_mapper = self.compiler.type_mapper
        input_types = [
            type_mapper.to_string(param.annotation.pattern.dtype)
            for param in udf_node.__signature__.parameters.values()
        ]

        def register_udf(con):
            return con.create_function(
                name=type(udf_node).__name__,
                function=udf_node.__func__,
                parameters=input_types,
                return_type=type_mapper.to_string(udf_node.dtype),
                type=_UDF_INPUT_TYPE_MAPPING[udf_node.__input_type__],
                **udf_node.__config__,
            )

        return register_udf

    _register_python_udf = _register_udf
    _register_pyarrow_udf = _register_udf

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


@lazy_singledispatch
def _read_in_memory(source: Any, table_name: str, _conn: Backend, **_: Any):
    raise NotImplementedError(
        f"The `{_conn.name}` backend currently does not support "
        f"reading data of {type(source)!r}"
    )


@_read_in_memory.register("polars.DataFrame")
@_read_in_memory.register("polars.LazyFrame")
@_read_in_memory.register("pyarrow.Table")
@_read_in_memory.register("pandas.DataFrame")
@_read_in_memory.register("pyarrow.dataset.Dataset")
def _default(source, table_name, _conn, **_: Any):
    _conn.con.register(table_name, source)


@_read_in_memory.register("pyarrow.RecordBatchReader")
def _pyarrow_rbr(source, table_name, _conn, **_: Any):
    _conn.con.register(table_name, source)
    # Ensure the reader isn't marked as started, in case the name is
    # being overwritten.
    _conn._record_batch_readers_consumed[table_name] = False
