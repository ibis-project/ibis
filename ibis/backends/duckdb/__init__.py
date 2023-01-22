"""DuckDB backend."""

from __future__ import annotations

import ast
import itertools
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, MutableMapping

import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
from ibis import util

if TYPE_CHECKING:
    import duckdb
    import pandas as pd
    import pyarrow as pa

import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import parse

# counters for in-memory, parquet, csv, and json reads
# used if no table name is specified
pd_n = itertools.count(0)
pa_n = itertools.count(0)
csv_n = itertools.count(0)
json_n = itertools.count(0)


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


def _create_view(*args, **kwargs):
    import sqlalchemy_views as sav

    return sav.CreateView(*args, **kwargs)


def _format_kwargs(kwargs: Mapping[str, Any]):
    bindparams, pieces = [], []
    for name, value in kwargs.items():
        bindparams.append(sa.bindparam(name, value))
        pieces.append(f"{name} = :{name}")
    return sa.text(", ".join(pieces)).bindparams(*bindparams)


class Backend(BaseAlchemyBackend):
    name = "duckdb"
    compiler = DuckDBSQLCompiler

    def current_database(self) -> str:
        return "main"

    @staticmethod
    def _convert_kwargs(kwargs: MutableMapping) -> None:
        read_only = kwargs.pop("read_only", "False").capitalize()
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
        path: str | Path = None,
        read_only: bool = False,
        temp_directory: Path | str | None = None,
        **config: Any,
    ) -> None:
        """Create an Ibis client connected to a DuckDB database.

        Parameters
        ----------
        database
            Path to a duckdb database.
        path
            Deprecated, use `database` instead.
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
        """
        if path is not None:
            warnings.warn(
                "The `path` argument is deprecated in 4.0. Use `database=...` "
                "instead."
            )
            database = path
        if not (in_memory := database == ":memory:"):
            database = Path(database).absolute()
        else:
            if temp_directory is None:
                temp_directory = (
                    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                    / "duckdb"
                )

        if temp_directory is not None:
            config["temp_directory"] = str(temp_directory)

        super().do_connect(
            sa.create_engine(
                f"duckdb:///{database}",
                connect_args=dict(read_only=read_only, config=config),
                poolclass=sa.pool.SingletonThreadPool if in_memory else None,
            )
        )
        self._meta = sa.MetaData()
        self._extensions = set()

    def _load_extensions(self, extensions):
        for extension in extensions:
            if extension not in self._extensions:
                with self.begin() as con:
                    c = con.connection
                    c.install_extension(extension)
                    c.load_extension(extension)
                self._extensions.add(extension)

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
            An optional name to use for the created table. This defaults to the
            filename if a path (with hyphens replaced with underscores), or
            sequentially generated name otherwise.
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

    def read_json(
        self,
        source_list: str | list[str] | tuple[str],
        table_name: str | None = None,
        **kwargs,
    ) -> ir.Table:
        """Read newline-delimited JSON into an ibis table.

        Parameters
        ----------
        source_list
            File or list of files
        table_name
            Optional table name
        kwargs
            Additional keyword arguments passed to DuckDB's `read_json_objects` function

        Returns
        -------
        Table
            An ibis table expression
        """
        if not table_name:
            table_name = f"ibis_read_json_{next(json_n)}"

        source_list = normalize_filenames(source_list)

        objects = (
            sa.func.read_json_objects(
                sa.func.list_value(*source_list), _format_kwargs(kwargs)
            )
            .table_valued("raw")
            .render_derived()
        )
        # read a single row out to get the schema, assumes the first row is representative
        json_structure_query = sa.select(sa.func.json_structure(objects.c.raw)).limit(1)

        with self.begin() as con:
            json_structure = con.execute(json_structure_query).scalar()
            data = sa.select(sa.literal_column("s.*")).select_from(
                sa.select(
                    sa.func.json_transform(objects.c.raw, json_structure).label("s")
                ).subquery()
            )
            view = _create_view(sa.table(table_name), data, or_replace=True)
            con.execute(view)

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
        kwargs
            Additional keyword arguments passed to DuckDB loading function.
            See https://duckdb.org/docs/data/csv for more information.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        source_list = normalize_filenames(source_list)

        if not table_name:
            table_name = f"ibis_read_csv_{next(csv_n)}"

        # auto_detect and columns collide, so we set auto_detect=True
        # unless COLUMNS has been specified
        if any(source.startswith(("http://", "https://")) for source in source_list):
            self._load_extensions(["httpfs"])

        kwargs["auto_detect"] = kwargs.pop("auto_detect", "columns" not in kwargs)
        source = sa.select(sa.literal_column("*")).select_from(
            sa.func.read_csv(sa.func.list_value(*source_list), _format_kwargs(kwargs))
        )
        view = _create_view(sa.table(table_name), source, or_replace=True)
        with self.begin() as con:
            con.execute(view)
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
        kwargs
            Additional keyword arguments passed to DuckDB loading function.
            See https://duckdb.org/docs/data/parquet for more information.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        source_list = normalize_filenames(source_list)

        if any(source.startswith("s3://") for source in source_list):
            if len(source_list) > 1:
                raise ValueError("only single s3 paths are supported")

            import pyarrow.dataset as ds

            dataset = ds.dataset(source_list[0])
            table_name = table_name or f"ibis_read_parquet_{next(pa_n)}"
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
        else:
            if any(
                source.startswith(("http://", "https://")) for source in source_list
            ):
                self._load_extensions(["httpfs"])

            if table_name is None:
                table_name = f"ibis_read_parquet_{next(pa_n)}"

            source = sa.select(sa.literal_column("*")).select_from(
                sa.func.read_parquet(
                    sa.func.list_value(*source_list), _format_kwargs(kwargs)
                )
            )
            view = _create_view(sa.table(table_name), source, or_replace=True)
            with self.begin() as con:
                con.execute(view)

        return self.table(table_name)

    def read_in_memory(
        self, dataframe: pd.DataFrame | pa.Table, table_name: str | None = None
    ) -> ir.Table:
        """Register a Pandas DataFrame or pyarrow Table as a table in the current database.

        Parameters
        ----------
        dataframe
            The data source.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table
        """
        table_name = table_name or f"ibis_read_in_memory_{next(pd_n)}"
        with self.begin() as con:
            con.connection.register(table_name, dataframe)

        return self.table(table_name)

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
        view = _create_view(sa.table(table_name), source, or_replace=True)
        with self.begin() as con:
            con.execute(view)

        return self.table(table_name)

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
            !!! warning "DuckDB returns 1024 size batches regardless of what argument is passed."
        """
        self._import_pyarrow()

        from ibis.backends.duckdb.pyarrow import IbisRecordBatchReader

        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        cursor = self.con.connect().execute(sql)

        reader = cursor.cursor.fetch_record_batch(chunk_size=chunk_size)
        return IbisRecordBatchReader(reader, cursor)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **_: Any,
    ) -> pa.Table:
        pa = self._import_pyarrow()
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        with self.begin() as con:
            cursor = con.execute(sql)
            table = cursor.cursor.fetch_arrow_table()

        if isinstance(expr, ir.Table):
            return table
        elif isinstance(expr, ir.Column):
            # Column will be a ChunkedArray, `combine_chunks` will
            # flatten it
            if len(table.columns[0]):
                return table.columns[0].combine_chunks()
            else:
                return pa.array(table.columns[0])
        elif isinstance(expr, ir.Scalar):
            return table.columns[0][0]
        else:
            raise ValueError

    def fetch_from_cursor(
        self,
        cursor: duckdb.DuckDBPyConnection,
        schema: sch.Schema,
    ):
        import pandas as pd
        import pyarrow.types as pat

        table = cursor.cursor.fetch_arrow_table()

        df = pd.DataFrame(
            {
                name: (
                    col.to_pylist()
                    if pat.is_nested(col.type)
                    else col.to_pandas(timestamp_as_object=True)
                )
                for name, col in zip(table.column_names, table.columns)
            }
        )
        return schema.apply_to(df)

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        with self.begin() as con:
            rows = con.exec_driver_sql(f"DESCRIBE {query}")

            for name, type, null in toolz.pluck(
                ["column_name", "column_type", "null"], rows.mappings()
            ):
                ibis_type = parse(type)
                yield name, ibis_type.copy(nullable=null.lower() == "yes")

    def _register_in_memory_table(self, table_op):
        df = table_op.data.to_frame()
        with self.begin() as con:
            con.connection.register(table_op.name, df)

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

    def _get_compiled_statement(self, view: sa.Table, definition: sa.sql.Selectable):
        # TODO: remove this once duckdb supports CTAS prepared statements
        return super()._get_compiled_statement(
            view, definition, compile_kwargs={"literal_binds": True}
        )
