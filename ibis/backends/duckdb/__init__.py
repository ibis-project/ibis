"""DuckDB backend."""

from __future__ import annotations

import ast
import contextlib
import os
import warnings
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateSchema, UrlFromPath
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.base.sqlglot.compiler import STAR, C, F
from ibis.backends.duckdb.compiler import DuckDBCompiler
from ibis.backends.duckdb.converter import DuckDBPandasData
from ibis.expr.operations.udf import InputType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence

    import pandas as pd
    import torch
    from fsspec import AbstractFileSystem


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


_UDF_INPUT_TYPE_MAPPING = {
    InputType.PYARROW: duckdb.functional.ARROW,
    InputType.PYTHON: duckdb.functional.NATIVE,
}


class _Settings:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con = con

    def __getitem__(self, key: str) -> Any:
        maybe_value = self.con.execute(
            f"select value from duckdb_settings() where name = '{key}'"
        ).fetchone()
        if maybe_value is not None:
            return maybe_value[0]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.con.execute(f"SET {key} = '{value}'")

    def __repr__(self):
        ((kv,),) = self.con.execute(
            "select map(array_agg(name), array_agg(value)) from duckdb_settings()"
        ).fetch()

        return repr(dict(zip(kv["key"], kv["value"])))


class Backend(SQLGlotBackend, CanCreateSchema, UrlFromPath):
    name = "duckdb"
    compiler = DuckDBCompiler()

    def _define_udf_translation_rules(self, expr):
        """No-op: UDF translation rules are defined in the compiler."""

    @property
    def settings(self) -> _Settings:
        return _Settings(self.con)

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_database())) as cur:
            [(db,)] = cur.fetchall()
        return db

    @property
    def current_schema(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.current_schema())) as cur:
            [(schema,)] = cur.fetchall()
        return schema

    # TODO(kszucs): should be moved to the base SQLGLot backend
    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)
        return self.con.execute(query, **kwargs)

    def _to_sqlglot(
        self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any
    ):
        sql = super()._to_sqlglot(expr, limit=limit, params=params)

        table_expr = expr.as_table()
        geocols = frozenset(
            name for name, typ in table_expr.schema().items() if typ.is_geospatial()
        )

        if not geocols:
            return sql

        return sg.select(
            *(
                self.compiler.f.st_aswkb(
                    sg.column(col, quoted=self.compiler.quoted)
                ).as_(col)
                if col in geocols
                else col
                for col in table_expr.columns
            )
        ).from_(sql.subquery())

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
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
        temp
            Create a temporary table
        overwrite
            If `True`, replace the table if it already exists, otherwise fail
            if the table exists

        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
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
            temp_name = util.gen_name("duckdb_table")
        else:
            temp_name = name

        initial_table = sg.table(
            temp_name, catalog=database, quoted=self.compiler.quoted
        )
        target = sge.Schema(this=initial_table, expressions=column_defs)

        create_stmt = sge.Create(
            kind="TABLE",
            this=target,
            properties=sge.Properties(expressions=properties),
        )

        # This is the same table as initial_table unless overwrite == True
        final_table = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                insert_stmt = sge.insert(query, into=initial_table).sql(self.name)
                cur.execute(insert_stmt).fetchall()

            if overwrite:
                cur.execute(
                    sge.Drop(kind="TABLE", this=final_table, exists=True).sql(self.name)
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
                        ).sql(self.name)
                    )
                    cur.execute(
                        sge.Drop(kind="TABLE", this=initial_table, exists=True).sql(
                            self.name
                        )
                    )
                else:
                    cur.execute(
                        sge.AlterTable(
                            this=initial_table,
                            actions=[sge.RenameTable(this=final_table)],
                        ).sql(self.name)
                    )

        return self.table(name, schema=database)

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(op.name)

    def table(
        self, name: str, schema: str | None = None, database: str | None = None
    ) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        schema
            Schema name
        database
            Database name

        Returns
        -------
        Table
            Table expression

        """
        table_schema = self.get_schema(name, schema=schema, database=database)
        # load geospatial only if geo columns
        if any(typ.is_geospatial() for typ in table_schema.types):
            self.load_extension("spatial")
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(database=database, schema=schema),
        ).to_expr()

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

        if database is not None:
            conditions.append(sg.column("table_catalog").eq(sge.convert(database)))

        if schema is not None:
            conditions.append(sg.column("table_schema").eq(sge.convert(schema)))

        query = (
            sg.select(
                "column_name",
                "data_type",
                sg.column("is_nullable").eq(sge.convert("YES")).as_("nullable"),
            )
            .from_(sg.table("columns", db="information_schema"))
            .where(sg.and_(*conditions))
            .order_by("ordinal_position")
        )

        with self._safe_raw_sql(query) as cur:
            meta = cur.fetch_arrow_table()

        if not meta:
            raise exc.IbisError(f"Table not found: {table_name!r}")

        names = meta["column_name"].to_pylist()
        types = meta["data_type"].to_pylist()
        nullables = meta["nullable"].to_pylist()

        return sch.Schema(
            {
                name: self.compiler.type_mapper.from_string(typ, nullable=nullable)
                for name, typ, nullable in zip(names, types, nullables)
            }
        )

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        yield self.raw_sql(*args, **kwargs)

    def list_databases(self, like: str | None = None) -> list[str]:
        col = "catalog_name"
        query = sg.select(sge.Distinct(expressions=[sg.column(col)])).from_(
            sg.table("schemata", db="information_schema")
        )
        with self._safe_raw_sql(query) as cur:
            result = cur.fetch_arrow_table()
        dbs = result[col]
        return self._filter_with_like(dbs.to_pylist(), like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        col = "schema_name"
        query = sg.select(sge.Distinct(expressions=[sg.column(col)])).from_(
            sg.table("schemata", db="information_schema")
        )

        if database is not None:
            query = query.where(sg.column("catalog_name").eq(sge.convert(database)))

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
        temp_directory: str | Path | None = None,
        extensions: Sequence[str] | None = None,
        **config: Any,
    ) -> None:
        """Create an Ibis client connected to a DuckDB database.

        Parameters
        ----------
        database
            Path to a duckdb database.
        read_only
            Whether the database is read-only.
        temp_directory
            Directory to use for spilling to disk. Only set by default for
            in-memory connections.
        extensions
            A list of duckdb extensions to install/load upon connection.
        config
            DuckDB configuration parameters. See the [DuckDB configuration
            documentation](https://duckdb.org/docs/sql/configuration) for
            possible configuration values.

        Examples
        --------
        >>> import ibis
        >>> ibis.duckdb.connect("database.ddb", threads=4, memory_limit="1GB")
        <ibis.backends.duckdb.Backend object at ...>

        """
        if (
            not isinstance(database, Path)
            and database != ":memory:"
            and not database.startswith(("md:", "motherduck:"))
        ):
            database = Path(database).absolute()

        if temp_directory is None:
            temp_directory = (
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "ibis-duckdb"
                / str(os.getpid())
            )
        else:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            config["temp_directory"] = str(temp_directory)

        self.con = duckdb.connect(str(database), config=config, read_only=read_only)

        # Load any pre-specified extensions
        if extensions is not None:
            self._load_extensions(extensions)

        # Default timezone
        with self._safe_raw_sql("SET TimeZone = 'UTC'"):
            pass

        self._record_batch_readers_consumed = {}

    def _load_extensions(
        self, extensions: list[str], force_install: bool = False
    ) -> None:
        f = self.compiler.f
        query = (
            sg.select(f.unnest(f.list_append(C.aliases, C.extension_name)))
            .from_(f.duckdb_extensions())
            .where(sg.and_(C.installed, C.loaded))
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

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot create a schema in another database."
            )

        name = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Create(this=name, kind="SCHEMA", replace=force)):
            pass

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot drop a schema in another database."
            )

        name = sg.table(name, catalog=database, quoted=self.compiler.quoted)
        with self._safe_raw_sql(sge.Drop(this=name, kind="SCHEMA", replace=force)):
            pass

    def register(
        self,
        source: str | Path | Any,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a data source as a table in the current database.

        Parameters
        ----------
        source
            The data source(s). May be a path to a file or directory of
            parquet/csv files, an iterable of parquet or CSV files, a pandas
            dataframe, a pyarrow table or dataset, or a postgres URI.
        table_name
            An optional name to use for the created table. This defaults to a
            sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DuckDB loading functions for
            CSV or parquet.  See https://duckdb.org/docs/data/csv and
            https://duckdb.org/docs/data/parquet for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """

        if isinstance(source, (str, Path)):
            first = str(source)
        elif isinstance(source, (list, tuple)):
            first = source[0]
        else:
            try:
                return self.read_in_memory(source, table_name=table_name, **kwargs)
            except (duckdb.InvalidInputException, NameError):
                self._register_failure()

        if first.startswith(("parquet://", "parq://")) or first.endswith(
            ("parq", "parquet")
        ):
            return self.read_parquet(source, table_name=table_name, **kwargs)
        elif first.startswith(
            ("csv://", "csv.gz://", "txt://", "txt.gz://")
        ) or first.endswith(("csv", "csv.gz", "tsv", "tsv.gz", "txt", "txt.gz")):
            return self.read_csv(source, table_name=table_name, **kwargs)
        elif first.startswith(("postgres://", "postgresql://")):
            return self.read_postgres(source, table_name=table_name, **kwargs)
        elif first.startswith("sqlite://"):
            return self.read_sqlite(
                first[len("sqlite://") :], table_name=table_name, **kwargs
            )
        else:
            self._register_failure()  # noqa: RET503

    def _register_failure(self):
        import inspect

        msg = ", ".join(
            name for name, _ in inspect.getmembers(self) if name.startswith("read_")
        )
        raise ValueError(
            f"Cannot infer appropriate read function for input, "
            f"please call one of {msg} directly"
        )

    @util.experimental
    def read_json(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
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
        **kwargs
            Additional keyword arguments passed to DuckDB's `read_json_auto` function

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

        self._create_temp_view(
            table_name,
            sg.select(STAR).from_(
                self.compiler.f.read_json_auto(
                    normalize_filenames(source_list), *options
                )
            ),
        )

        return self.table(table_name)

    def read_csv(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Register a CSV file as a table in the current database.

        Parameters
        ----------
        source_list
            The data source(s). May be a path to a file or directory of CSV files, or an
            iterable of CSV files.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.
        **kwargs
            Additional keyword arguments passed to DuckDB loading function.
            See https://duckdb.org/docs/data/csv for more information.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        source_list = normalize_filenames(source_list)

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
        kwargs["auto_detect"] = kwargs.pop("auto_detect", "columns" not in kwargs)
        # TODO: clean this up
        # We want to _usually_ quote arguments but if we quote `columns` it messes
        # up DuckDB's struct parsing.
        options = [
            sg.to_identifier(key).eq(sge.convert(val)) for key, val in kwargs.items()
        ]

        if (columns := kwargs.pop("columns", None)) is not None:
            options.append(
                sg.to_identifier("columns").eq(
                    sge.Struct(
                        expressions=[
                            sge.Slice(
                                this=sge.convert(key), expression=sge.convert(value)
                            )
                            for key, value in columns.items()
                        ]
                    )
                )
            )

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
        source_list = normalize_filenames(source_list)

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

    def read_in_memory(
        self,
        source: pd.DataFrame | pa.Table | pa.RecordBatchReader,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a Pandas DataFrame or pyarrow object as a table in the current database.

        Parameters
        ----------
        source
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        table_name = table_name or util.gen_name("read_in_memory")
        self.con.register(table_name, source)

        if isinstance(source, pa.RecordBatchReader):
            # Ensure the reader isn't marked as started, in case the name is
            # being overwritten.
            self._record_batch_readers_consumed[table_name] = False

        return self.table(table_name)

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
        source_table = normalize_filenames(source_table)[0]

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

        return self.read_in_memory(
            delta_table.to_pyarrow_dataset(), table_name=table_name
        )

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> list[str]:
        """List tables and views.

        Parameters
        ----------
        like
            Regex to filter by table/view name.
        database
            Database name. If not passed, uses the current database. Only
            supported with MotherDuck.
        schema
            Schema name. If not passed, uses the current schema.

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
        >>> con.create_schema("my_schema")
        >>> con.list_tables(schema="my_schema")
        []
        >>> with con.begin() as c:
        ...     c.exec_driver_sql("CREATE TABLE my_schema.baz (a INTEGER)")  # doctest: +ELLIPSIS
        <...>
        >>> con.list_tables(schema="my_schema")
        ['baz']

        """
        database = F.current_database() if database is None else sge.convert(database)
        schema = F.current_schema() if schema is None else sge.convert(schema)

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(
                C.table_catalog.eq(database).or_(
                    C.table_catalog.eq(sge.convert("temp"))
                ),
                C.table_schema.eq(schema),
            )
            .sql(self.name, pretty=True)
        )

        out = self.con.execute(sql).fetch_arrow_table()

        return self._filter_with_like(out[col].to_pylist(), like)

    def read_postgres(
        self, uri: str, table_name: str | None = None, schema: str = "public"
    ) -> ir.Table:
        """Register a table from a postgres instance into a DuckDB table.

        Parameters
        ----------
        uri
            A postgres URI of the form `postgres://user:password@host:port`
        table_name
            The table to read
        schema
            PostgreSQL schema where `table_name` resides

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
                self.compiler.f.postgres_scan_pushdown(uri, schema, table_name)
            ),
        )

        return self.table(table_name)

    def read_sqlite(self, path: str | Path, table_name: str | None = None) -> ir.Table:
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
        >>> with sqlite3.connect("/tmp/sqlite.db") as con:
        ...     con.execute("DROP TABLE IF EXISTS t")  # doctest: +ELLIPSIS
        ...     con.execute("CREATE TABLE t (a INT, b TEXT)")  # doctest: +ELLIPSIS
        ...     con.execute(
        ...         "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        ...     )  # doctest: +ELLIPSIS
        <...>
        >>> con = ibis.connect("duckdb://")
        >>> t = con.read_sqlite("/tmp/sqlite.db", table_name="t")
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
        >>> with sqlite3.connect("/tmp/attach_sqlite.db") as con:
        ...     con.execute("DROP TABLE IF EXISTS t")  # doctest: +ELLIPSIS
        ...     con.execute("CREATE TABLE t (a INT, b TEXT)")  # doctest: +ELLIPSIS
        ...     con.execute(
        ...         "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        ...     )  # doctest: +ELLIPSIS
        <...>
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
        >>> gcs = fsspec.filesystem("gcs")
        >>> con = ibis.duckdb.connect()
        >>> con.register_filesystem(gcs)
        >>> t = con.read_csv(
        ...     "gcs://ibis-examples/data/band_members.csv.gz",
        ...     table_name="band_members",
        ... )
        DatabaseTable: band_members
          name string
          band string

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
        super()._run_pre_execute_hooks(expr)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.RecordBatchReader:
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
            ::: {.callout-warning}
            ## DuckDB returns 1024 size batches regardless of what argument is passed.
            :::

        """
        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params)

        def batch_producer(cur):
            yield from cur.fetch_record_batch(rows_per_batch=chunk_size)

        # TODO: check that this is still handled correctly
        # batch_producer keeps the `self.con` member alive long enough to
        # exhaust the record batch reader, even if the backend or connection
        # have gone out of scope in the caller
        result = self.raw_sql(sql)

        return pa.RecordBatchReader.from_batches(
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
        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params)

        with self._safe_raw_sql(sql) as cur:
            table = cur.fetch_arrow_table()

        return expr.__pyarrow_result__(table)

    @util.experimental
    def to_torch(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
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
        compiled = self.compile(expr, limit=limit, params=params, **kwargs)
        with self._safe_raw_sql(compiled) as cur:
            return cur.torch()

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
            https://duckdb.org/docs/data/parquet#writing-to-parquet-files for
            details

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
        query = self._to_sql(expr, params=params)
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
            DuckDB CSV writer arguments. https://duckdb.org/docs/data/csv.html#parameters

        """
        self._run_pre_execute_hooks(expr)
        query = self._to_sql(expr, params=params)
        args = [
            "FORMAT 'csv'",
            f"HEADER {int(header)}",
            *(f"{k.upper()} {v!r}" for k, v in kwargs.items()),
        ]
        copy_cmd = f"COPY ({query}) TO {str(path)!r} ({', '.join(args)})"
        with self._safe_raw_sql(copy_cmd):
            pass

    def _fetch_from_cursor(
        self, cursor: duckdb.DuckDBPyConnection, schema: sch.Schema
    ) -> pd.DataFrame:
        import pandas as pd
        import pyarrow.types as pat

        table = cursor.fetch_arrow_table()

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
                    else col.to_pandas(timestamp_as_object=True)
                )
                for name, col in zip(table.column_names, table.columns)
            }
        )
        return DuckDBPandasData.convert_table(df, schema)

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self._safe_raw_sql(f"DESCRIBE {query}") as cur:
            rows = cur.fetch_arrow_table()

        rows = rows.to_pydict()

        for name, typ, null in zip(
            rows["column_name"], rows["column_type"], rows["null"]
        ):
            yield (
                name,
                self.compiler.type_mapper.from_string(typ, nullable=null == "YES"),
            )

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        for memtable in expr.op().find(ops.InMemoryTable):
            self._register_in_memory_table(memtable)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            self.con.register(name, op.data.to_pyarrow(op.schema))

    def _register_udfs(self, expr: ir.Expr) -> None:
        import ibis.expr.operations as ops

        con = self.con

        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            with contextlib.suppress(duckdb.InvalidInputException):
                con.remove_function(udf_node.__class__.__name__)

            registration_func = compile_func(udf_node)
            if registration_func is not None:
                registration_func(con)

    def _compile_udf(self, udf_node: ops.ScalarUDF) -> None:
        func = udf_node.__func__
        name = type(udf_node).__name__
        type_mapper = self.compiler.type_mapper
        input_types = [
            type_mapper.to_string(param.annotation.pattern.dtype)
            for param in udf_node.__signature__.parameters.values()
        ]
        output_type = type_mapper.to_string(udf_node.dtype)

        def register_udf(con):
            return con.create_function(
                name,
                func,
                input_types,
                output_type,
                type=_UDF_INPUT_TYPE_MAPPING[udf_node.__input_type__],
            )

        return register_udf

    _compile_python_udf = _compile_udf
    _compile_pyarrow_udf = _compile_udf

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:
        """No op."""

    def _compile_pandas_udf(self, _: ops.ScalarUDF) -> None:
        raise NotImplementedError("duckdb doesn't support pandas UDFs")

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
