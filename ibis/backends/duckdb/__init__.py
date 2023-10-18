"""DuckDB backend."""

from __future__ import annotations

import ast
import contextlib
import os
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

import duckdb
import pyarrow as pa
import sqlalchemy as sa
import sqlglot as sg
import toolz

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateSchema
from ibis.backends.base.sql.alchemy import AlchemyCrossSchemaBackend
from ibis.backends.base.sqlglot import C, F
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import DuckDBType
from ibis.expr.operations.relations import PandasDataFrameProxy
from ibis.expr.operations.udf import InputType
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence

    import pandas as pd
    import torch


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


def _format_kwargs(kwargs: Mapping[str, Any]):
    bindparams, pieces = [], []
    for name, value in kwargs.items():
        bindparam = sa.bindparam(name, value)
        if isinstance(paramtype := bindparam.type, sa.String):
            # special case strings to avoid double escaping backslashes
            pieces.append(f"{name} = '{value!s}'")
        elif not isinstance(paramtype, sa.types.NullType):
            bindparams.append(bindparam)
            pieces.append(f"{name} = :{name}")
        else:  # fallback to string strategy
            pieces.append(f"{name} = {value!r}")

    return sa.text(", ".join(pieces)).bindparams(*bindparams)


_UDF_INPUT_TYPE_MAPPING = {
    InputType.PYARROW: duckdb.functional.ARROW,
    InputType.PYTHON: duckdb.functional.NATIVE,
}


class Backend(AlchemyCrossSchemaBackend, CanCreateSchema):
    name = "duckdb"
    compiler = DuckDBSQLCompiler
    supports_create_or_replace = True

    @property
    def current_database(self) -> str:
        return self._scalar_query(sa.select(sa.func.current_database()))

    def list_databases(self, like: str | None = None) -> list[str]:
        s = sa.table(
            "schemata",
            sa.column("catalog_name", sa.TEXT()),
            schema="information_schema",
        )

        query = sa.select(sa.distinct(s.c.catalog_name))
        with self.begin() as con:
            results = list(con.execute(query).scalars())
        return self._filter_with_like(results, like=like)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        # override duckdb because all databases are always visible
        text = """\
SELECT schema_name
FROM information_schema.schemata
WHERE catalog_name = :database"""
        query = sa.text(text).bindparams(
            database=database if database is not None else self.current_database
        )

        with self.begin() as con:
            schemas = list(con.execute(query).scalars())
        return self._filter_with_like(schemas, like=like)

    @property
    def current_schema(self) -> str:
        return self._scalar_query(sa.select(sa.func.current_schema()))

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

    @staticmethod
    def _new_sa_metadata():
        meta = sa.MetaData()

        # _new_sa_metadata is invoked whenever `_get_sqla_table` is called, so
        # it's safe to store columns as keys, that is, columns from different
        # tables with the same name won't collide
        complex_type_info_cache = {}

        @sa.event.listens_for(meta, "column_reflect")
        def column_reflect(inspector, table, column_info):
            import duckdb_engine.datatypes as ddt

            # duckdb_engine as of 0.7.2 doesn't expose the inner types of any
            # complex types so we have to extract it from duckdb directly
            ddt_struct_type = getattr(ddt, "Struct", sa.types.NullType)
            ddt_map_type = getattr(ddt, "Map", sa.types.NullType)
            if isinstance(
                column_info["type"], (sa.ARRAY, ddt_struct_type, ddt_map_type)
            ):
                engine = inspector.engine
                colname = column_info["name"]
                if (coltype := complex_type_info_cache.get(colname)) is None:
                    quote = engine.dialect.identifier_preparer.quote
                    quoted_colname = quote(colname)
                    quoted_tablename = quote(table.name)
                    with engine.connect() as con:
                        # The .connection property is used to avoid creating a
                        # nested transaction
                        con.connection.execute(
                            f"DESCRIBE SELECT {quoted_colname} FROM {quoted_tablename}"
                        )
                        _, typ, *_ = con.connection.fetchone()
                    complex_type_info_cache[colname] = coltype = DuckDBType.from_string(
                        typ
                    )

                column_info["type"] = DuckDBType.from_ibis(coltype)

        return meta

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

        engine = sa.create_engine(
            f"duckdb:///{database}",
            connect_args=dict(read_only=read_only, config=config),
            poolclass=sa.pool.StaticPool,
        )

        @sa.event.listens_for(engine, "connect")
        def configure_connection(dbapi_connection, connection_record):
            if extensions is not None:
                self._sa_load_extensions(dbapi_connection, extensions)
            dbapi_connection.execute("SET TimeZone = 'UTC'")

        self._record_batch_readers_consumed = {}

        # TODO(cpcloud): remove this when duckdb is >0.8.1
        # this is here to workaround https://github.com/duckdb/duckdb/issues/8735
        with contextlib.suppress(duckdb.InvalidInputException):
            duckdb.execute("SELECT ?", (1,))

        engine.dialect._backslash_escapes = False
        super().do_connect(engine)

    @staticmethod
    def _sa_load_extensions(dbapi_con, extensions):
        query = """
        WITH exts AS (
          SELECT extension_name AS name, aliases FROM duckdb_extensions()
          WHERE installed AND loaded
        )
        SELECT name FROM exts
        UNION (SELECT UNNEST(aliases) AS name FROM exts)
        """
        installed = (name for (name,) in dbapi_con.sql(query).fetchall())
        # Install and load all other extensions
        todo = set(extensions).difference(installed)
        for extension in todo:
            dbapi_con.install_extension(extension)
            dbapi_con.load_extension(extension)

    def _load_extensions(self, extensions):
        with self.begin() as con:
            self._sa_load_extensions(con.connection, extensions)

    def load_extension(self, extension: str) -> None:
        """Install and load a duckdb extension by name or path.

        Parameters
        ----------
        extension
            The extension name or path.
        """
        self._load_extensions([extension])

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot create a schema in another database."
            )
        name = self._quote(name)
        if_not_exists = "IF NOT EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"CREATE SCHEMA {if_not_exists}{name}")

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot drop a schema in another database."
            )
        name = self._quote(name)
        if_exists = "IF EXISTS " * force
        with self.begin() as con:
            con.exec_driver_sql(f"DROP SCHEMA {if_exists}{name}")

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
            except sa.exc.ProgrammingError:
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

    def _compile_temp_view(self, table_name, source):
        raw_source = source.compile(
            dialect=self.con.dialect, compile_kwargs=dict(literal_binds=True)
        )
        return f'CREATE OR REPLACE TEMPORARY VIEW "{table_name}" AS {raw_source}'

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

        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.read_json_auto(
                sa.func.list_value(*normalize_filenames(source_list)),
                _format_kwargs(kwargs),
            )
        )
        view = self._compile_temp_view(table_name, source)
        with self.begin() as con:
            con.exec_driver_sql(view)

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
        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.read_csv(sa.func.list_value(*source_list), _format_kwargs(kwargs))
        )

        view = self._compile_temp_view(table_name, source)
        with self.begin() as con:
            con.exec_driver_sql(view)
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
        except sa.exc.OperationalError as e:
            if isinstance(e.orig, duckdb.IOException):
                self._read_parquet_pyarrow_dataset(source_list, table_name, **kwargs)
            else:
                raise e

        return self.table(table_name)

    def _read_parquet_duckdb_native(
        self, source_list: str | Iterable[str], table_name: str, **kwargs: Any
    ) -> None:
        if any(
            source.startswith(("http://", "https://", "s3://"))
            for source in source_list
        ):
            self._load_extensions(["httpfs"])

        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.read_parquet(
                sa.func.list_value(*source_list), _format_kwargs(kwargs)
            )
        )
        view = self._compile_temp_view(table_name, source)
        with self.begin() as con:
            con.exec_driver_sql(view)

    def _read_parquet_pyarrow_dataset(
        self, source_list: str | Iterable[str], table_name: str, **kwargs: Any
    ) -> None:
        import pyarrow.dataset as ds

        dataset = ds.dataset(list(map(ds.dataset, source_list)), **kwargs)
        self._load_extensions(["httpfs"])
        # We don't create a view since DuckDB special cases Arrow Datasets
        # so if we also create a view we end up with both a "lazy table"
        # and a view with the same name
        with self.begin() as con:
            # DuckDB normally auto-detects Arrow Datasets that are defined
            # in local variables but the `dataset` variable won't be local
            # by the time we execute against this so we register it
            # explicitly.
            con.connection.register(table_name, dataset)

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
        with self.begin() as con:
            con.connection.register(table_name, source)

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
        ...     c.exec_driver_sql(
        ...         "CREATE TABLE my_schema.baz (a INTEGER)"
        ...     )  # doctest: +ELLIPSIS
        ...
        <...>
        >>> con.list_tables(schema="my_schema")
        ['baz']
        """
        database = (
            F.current_database() if database is None else sg.exp.convert(database)
        )
        schema = F.current_schema() if schema is None else sg.exp.convert(schema)

        sql = (
            sg.select(C.table_name)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(
                C.table_catalog.eq(database).or_(
                    C.table_catalog.eq(sg.exp.convert("temp"))
                ),
                C.table_schema.eq(schema),
            )
            .sql(self.name, pretty=True)
        )

        with self.begin() as con:
            out = con.exec_driver_sql(sql).cursor.fetch_arrow_table()

        return self._filter_with_like(out["table_name"].to_pylist(), like)

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
        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.postgres_scan_pushdown(uri, schema, table_name)
        )
        view = self._compile_temp_view(table_name, source)
        with self.begin() as con:
            con.exec_driver_sql(view)

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
        ...
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

        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.sqlite_scan(str(path), table_name)
        )
        view = self._compile_temp_view(table_name, source)
        with self.begin() as con:
            con.exec_driver_sql(view)

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

        with self.begin() as con:
            con.exec_driver_sql(code)

    def detach(self, name: str) -> None:
        """Detach a database from the current DuckDB session.

        Parameters
        ----------
        name
            The name of the database to detach.
        """
        name = sg.to_identifier(name).sql(self.name)
        with self.begin() as con:
            con.exec_driver_sql(f"DETACH {name}")

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
        ...
        <...>
        >>> con = ibis.connect("duckdb://")
        >>> con.list_tables()
        []
        >>> con.attach_sqlite("/tmp/attach_sqlite.db")
        >>> con.list_tables()
        ['t']
        """
        self._load_extensions(["sqlite"])
        with self.begin() as con:
            con.execute(sa.text(f"SET GLOBAL sqlite_all_varchar={all_varchar}"))
            con.execute(sa.text(f"CALL sqlite_attach('{path}', overwrite={overwrite})"))

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
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        def batch_producer(con):
            with con.begin() as c, contextlib.closing(c.execute(sql)) as cur:
                yield from cur.cursor.fetch_record_batch(rows_per_batch=chunk_size)

        # batch_producer keeps the `self.con` member alive long enough to
        # exhaust the record batch reader, even if the backend or connection
        # have gone out of scope in the caller
        return pa.RecordBatchReader.from_batches(
            expr.as_table().schema().to_pyarrow(), batch_producer(self.con)
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
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)

        # We use `.sql` instead of `.execute` below for performance - in
        # certain cases duckdb query -> arrow table can be significantly faster
        # in this configuration. Currently `.sql` doesn't support parametrized
        # queries, so we need to compile with literal_binds for now.
        sql = str(
            query_ast.compile().compile(
                dialect=self.con.dialect, compile_kwargs={"literal_binds": True}
            )
        )

        with self.begin() as con:
            table = con.connection.sql(sql).to_arrow_table()

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
            return cur.connection.connection.torch()

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

        >>> con.to_parquet(
        ...     penguins, tempfile.mkdtemp(), partition_by=("year", "island")
        ... )
        """
        self._run_pre_execute_hooks(expr)
        query = self._to_sql(expr, params=params)
        args = ["FORMAT 'parquet'", *(f"{k.upper()} {v!r}" for k, v in kwargs.items())]
        copy_cmd = f"COPY ({query}) TO {str(path)!r} ({', '.join(args)})"
        with self.begin() as con:
            con.exec_driver_sql(copy_cmd)

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
        with self.begin() as con:
            con.exec_driver_sql(copy_cmd)

    def fetch_from_cursor(
        self, cursor: duckdb.DuckDBPyConnection, schema: sch.Schema
    ) -> pd.DataFrame:
        import pandas as pd
        import pyarrow.types as pat

        table = cursor.cursor.fetch_arrow_table()

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
        return PandasData.convert_table(df, schema)

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self.begin() as con:
            rows = con.exec_driver_sql(f"DESCRIBE {query}")

            for name, type, null in toolz.pluck(
                ["column_name", "column_type", "null"], rows.mappings()
            ):
                nullable = null.lower() == "yes"
                ibis_type = DuckDBType.from_string(type, nullable=nullable)
                yield name, ibis_type

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        # in theory we could use pandas dataframes, but when using dataframes
        # with pyarrow datatypes later reads of this data segfault
        import pandas as pd

        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise exc.IbisTypeError(
                "DuckDB cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            if isinstance(data := op.data, PandasDataFrameProxy):
                table = data.to_frame()

                # convert to object string dtypes because duckdb is either
                # 1. extremely slow to register DataFrames with not-pyarrow
                #    string dtypes
                # 2. broken for string[pyarrow] dtypes (segfault)
                if conversions := {
                    colname: "str"
                    for colname, col in table.items()
                    if isinstance(col.dtype, pd.StringDtype)
                }:
                    table = table.astype(conversions)
            else:
                table = data.to_pyarrow(schema)

            # register creates a transaction, and we can't nest transactions so
            # we create a function to encapsulate the whole shebang
            def _register(name, table):
                with self.begin() as con:
                    con.connection.register(name, table)

            try:
                _register(name, table)
            except duckdb.NotImplementedException:
                _register(name, data.to_pyarrow(schema))

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"

    def _register_udfs(self, expr: ir.Expr) -> None:
        import ibis.expr.operations as ops

        with self.con.connect() as con:
            for udf_node in expr.op().find(ops.ScalarUDF):
                compile_func = getattr(
                    self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
                )
                with contextlib.suppress(duckdb.InvalidInputException):
                    con.connection.remove_function(udf_node.__class__.__name__)

                registration_func = compile_func(udf_node)
                if registration_func is not None:
                    registration_func(con)

    def _compile_udf(self, udf_node: ops.ScalarUDF) -> None:
        func = udf_node.__func__
        name = func.__name__
        input_types = [DuckDBType.to_string(arg.dtype) for arg in udf_node.args]
        output_type = DuckDBType.to_string(udf_node.dtype)

        def register_udf(con):
            return con.connection.create_function(
                name,
                func,
                input_types,
                output_type,
                type=_UDF_INPUT_TYPE_MAPPING[udf_node.__input_type__],
            )

        return register_udf

    _compile_python_udf = _compile_udf
    _compile_pyarrow_udf = _compile_udf

    def _compile_pandas_udf(self, _: ops.ScalarUDF) -> None:
        raise NotImplementedError("duckdb doesn't support pandas UDFs")

    def _get_compiled_statement(self, view: sa.Table, definition: sa.sql.Selectable):
        # TODO: remove this once duckdb supports CTAS prepared statements
        return super()._get_compiled_statement(
            view, definition, compile_kwargs={"literal_binds": True}
        )

    def _insert_dataframe(
        self, table_name: str, df: pd.DataFrame, overwrite: bool
    ) -> None:
        columns = list(df.columns)
        t = sa.table(table_name, *map(sa.column, columns))

        table_name = self._quote(table_name)

        # the table name df here matters, and *must* match the input variable's
        # name because duckdb will look up this name in the outer scope of the
        # insert call and pull in that variable's data to scan
        source = sa.table("df", *map(sa.column, columns))

        with self.begin() as con:
            if overwrite:
                con.execute(t.delete())
            con.execute(t.insert().from_select(columns, sa.select(source)))
