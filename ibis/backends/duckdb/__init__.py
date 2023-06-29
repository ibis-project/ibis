"""DuckDB backend."""

from __future__ import annotations

import ast
import contextlib
import os
import warnings
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)

import duckdb
import pyarrow as pa
import sqlalchemy as sa
import toolz
from packaging.version import parse as vparse

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import DuckDBType, parse
from ibis.expr.operations.relations import PandasDataFrameProxy
from ibis.expr.operations.udf import InputType
from ibis.formats.pandas import PandasData

if TYPE_CHECKING:
    import pandas as pd
    import torch

    import ibis.expr.operations as ops


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


class Backend(BaseAlchemyBackend):
    name = "duckdb"
    compiler = DuckDBSQLCompiler
    supports_create_or_replace = True

    def current_database(self) -> str:
        return "main"

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
                    quote = engine.dialect.identifier_preparer.quote_identifier
                    quoted_colname = quote(colname)
                    quoted_tablename = quote(table.name)
                    with engine.connect() as con:
                        # The .connection property is used to avoid creating a
                        # nested transaction
                        con.connection.execute(
                            f"DESCRIBE SELECT {quoted_colname} FROM {quoted_tablename}"
                        )
                        _, typ, *_ = con.connection.fetchone()
                    complex_type_info_cache[colname] = coltype = parse(typ)

                column_info["type"] = DuckDBType.from_ibis(coltype)

        return meta

    def do_connect(
        self,
        database: str | Path = ":memory:",
        read_only: bool = False,
        temp_directory: str | Path | None = None,
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
            dbapi_connection.execute("SET TimeZone = 'UTC'")
            # the progress bar in duckdb <0.8.0 causes kernel crashes in
            # jupyterlab, fixed in https://github.com/duckdb/duckdb/pull/6831
            if vparse(duckdb.__version__) < vparse("0.8.0"):
                dbapi_connection.execute("SET enable_progress_bar = false")

        self._record_batch_readers_consumed = {}
        super().do_connect(engine)

    def _load_extensions(self, extensions):
        extension_name = sa.column("extension_name")
        loaded = sa.column("loaded")
        installed = sa.column("installed")
        aliases = sa.column("aliases")
        query = (
            sa.select(extension_name)
            .select_from(sa.func.duckdb_extensions())
            .where(
                sa.and_(
                    # extension isn't loaded or isn't installed
                    sa.not_(loaded & installed),
                    # extension is one that we're requesting, or an alias of it
                    sa.or_(
                        extension_name.in_(extensions),
                        *map(partial(sa.func.array_has, aliases), extensions),
                    ),
                )
            )
        )
        with self.begin() as con:
            c = con.connection
            for extension in con.execute(query).scalars():
                c.install_extension(extension)
                c.load_extension(extension)

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

        !!! note "This feature requires duckdb>=0.7.0"

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
        if (version := vparse(self.version)) < vparse("0.7.0"):
            raise exc.IbisError(
                f"`read_json` requires duckdb >= 0.7.0, duckdb {version} is installed"
            )
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
        if any(source.startswith(("http://", "https://")) for source in source_list):
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

    def list_tables(self, like=None, database=None):
        tables = self.inspector.get_table_names(schema=database)
        views = self.inspector.get_view_names(schema=database)
        # workaround for GH5503
        temp_views = self.inspector.get_view_names(
            schema="temp" if database is None else database
        )
        return self._filter_with_like(tables + views + temp_views, like)

    def read_postgres(self, uri, table_name: str | None = None, schema: str = "public"):
        """Register a table from a postgres instance into a DuckDB table.

        Parameters
        ----------
        uri
            The postgres URI in form 'postgres://user:password@host:port'
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
        >>> con = ibis.connect("duckdb://")
        >>> t = con.read_sqlite("ci/ibis-testing-data/ibis_testing.db", table_name="diamonds")
        >>> t.head().execute()
                carat      cut color clarity  depth  table  price     x     y     z
            0   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
            1   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
            2   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31
            3   0.29  Premium     I     VS2   62.4   58.0    334  4.20  4.23  2.63
            4   0.31     Good     J     SI2   63.3   58.0    335  4.34  4.35  2.75
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

        Returns
        -------
        None

        Examples
        --------
        >>> import ibis
        >>> con = ibis.connect("duckdb://")
        >>> con.attach_sqlite("ci/ibis-testing-data/ibis_testing.db")
        >>> con.list_tables()
        ['functional_alltypes', 'awards_players', 'batting', 'diamonds']
        """
        self._load_extensions(["sqlite"])
        with self.begin() as con:
            con.execute(sa.text(f"SET GLOBAL sqlite_all_varchar={all_varchar}"))
            con.execute(sa.text(f"CALL sqlite_attach('{path}', overwrite={overwrite})"))

    def _run_pre_execute_hooks(self, expr: ir.Expr) -> None:
        from ibis.expr.analysis import find_physical_tables

        # Warn for any tables depending on RecordBatchReaders that have already
        # started being consumed.
        for t in find_physical_tables(expr.op()):
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
            !!! warning "DuckDB returns 1024 size batches regardless of what argument is passed."
        """
        self._run_pre_execute_hooks(expr)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        # handle the argument name change in duckdb 0.8.0
        fetch_record_batch = (
            (lambda cur: cur.fetch_record_batch(rows_per_batch=chunk_size))
            if vparse(duckdb.__version__) >= vparse("0.8.0")
            else (lambda cur: cur.fetch_record_batch(chunk_size=chunk_size))
        )

        def batch_producer(con):
            with con.begin() as c, contextlib.closing(c.execute(sql)) as cur:
                yield from fetch_record_batch(cur.cursor)

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
        sql = query_ast.compile()

        with self.begin() as con:
            cursor = con.execute(sql)
            table = cursor.cursor.fetch_arrow_table()

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
        >>> con.to_parquet(penguins, "penguins.parquet")

        Write out an expression to a hive-partitioned parquet file.

        >>> import ibis
        >>> penguins = ibis.examples.penguins.fetch()
        >>> con = ibis.get_backend(penguins)
        >>> con.to_parquet(penguins, "penguins_hive_dir", partition_by="year")  # doctest: +SKIP
        >>> # partition on multiple columns
        >>> con.to_parquet(penguins, "penguins_hive_dir", partition_by=("year", "island"))  # doctest: +SKIP
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
                ibis_type = parse(type)
                yield name, ibis_type.copy(nullable=null.lower() == "yes")

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

    def _get_sqla_table(
        self, name: str, schema: str | None = None, **kwargs: Any
    ) -> sa.Table:
        with warnings.catch_warnings():
            # We don't rely on index reflection, ignore this warning
            warnings.filterwarnings(
                "ignore",
                message="duckdb-engine doesn't yet support reflection on indices",
            )
            return super()._get_sqla_table(name, schema, **kwargs)

    def _get_temp_view_definition(
        self, name: str, definition: sa.sql.compiler.Compiled
    ) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"

    def _register_udfs(self, expr: ir.Expr) -> None:
        import ibis.expr.operations as ops

        with self.begin() as con:
            for udf_node in expr.op().find(ops.ScalarUDF):
                compile_func = getattr(
                    self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
                )
                with contextlib.suppress(duckdb.InvalidInputException):
                    con.connection.remove_function(udf_node.__class__.__name__)

                registration_func = compile_func(udf_node)
                registration_func(con)

    def _compile_udf(self, udf_node: ops.ScalarUDF) -> None:
        func = udf_node.__func__
        name = func.__name__
        input_types = [DuckDBType.to_string(arg.output_dtype) for arg in udf_node.args]
        output_type = DuckDBType.to_string(udf_node.output_dtype)

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

        quote = self.con.dialect.identifier_preparer.quote
        table_name = quote(table_name)

        # the table name df here matters, and *must* match the input variable's
        # name because duckdb will look up this name in the outer scope of the
        # insert call and pull in that variable's data to scan
        source = sa.table("df", *map(sa.column, columns))

        with self.begin() as con:
            if overwrite:
                con.execute(t.delete())
            con.execute(t.insert().from_select(columns, sa.select(source)))
