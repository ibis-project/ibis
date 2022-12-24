"""DuckDB backend."""

from __future__ import annotations

import ast
import itertools
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, MutableMapping

import pyarrow as pa
import pyarrow.types as pat
import sqlalchemy as sa
import toolz

import ibis.expr.datatypes as dt
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type

if TYPE_CHECKING:
    import duckdb

import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy import BaseAlchemyBackend
from ibis.backends.duckdb.compiler import DuckDBSQLCompiler
from ibis.backends.duckdb.datatypes import parse
from ibis.common.dispatch import RegexDispatcher

_generate_view_code = RegexDispatcher("_register")
_dialect = sa.dialects.postgresql.dialect()

_gen_table_names = (f"registered_table{i:d}" for i in itertools.count())


def _name_from_path(path: str | Path) -> str:
    # https://github.com/duckdb/duckdb/issues/5203
    return str(path).replace(".", "_")


def _name_from_dataset(dataset: pa.dataset.FileSystemDataset) -> str:
    return _name_from_path(Path(os.path.commonprefix(dataset.files)))


def _quote(name: str):
    return _dialect.identifier_preparer.quote(name)


def _get_scheme(scheme):
    if scheme is None or scheme == "file://":
        return ""
    return scheme


def _format_kwargs(kwargs):
    return (
        f"{k}='{v}'" if isinstance(v, str) else f"{k}={v!r}" for k, v in kwargs.items()
    )


@_generate_view_code.register(r"parquet://(?P<path>.+)", priority=13)
def _parquet(_, path, table_name=None, scheme=None, **kwargs):
    scheme = _get_scheme(scheme)
    if not scheme:
        path = os.path.abspath(path)
    if not table_name:
        table_name = _name_from_path(path)
    quoted_table_name = _quote(table_name)
    args = [f"'{scheme}{path}'", *_format_kwargs(kwargs)]
    code = f"""\
CREATE OR REPLACE VIEW {quoted_table_name} AS
SELECT * FROM read_parquet({', '.join(args)})"""
    return code, table_name, ["parquet"] + ["httpfs"] if scheme else []


@_generate_view_code.register(r"(c|t)sv://(?P<path>.+)", priority=13)
def _csv(_, path, table_name=None, scheme=None, **kwargs):
    scheme = _get_scheme(scheme)
    if not scheme:
        path = os.path.abspath(path)
    if not table_name:
        table_name = _name_from_path(path)
    quoted_table_name = _quote(table_name)
    # auto_detect and columns collide, so we set auto_detect=True
    # unless COLUMNS has been specified
    args = [
        f"'{scheme}{path}'",
        f"auto_detect={kwargs.pop('auto_detect', 'columns' not in kwargs)}",
        *_format_kwargs(kwargs),
    ]
    code = f"""\
CREATE OR REPLACE VIEW {quoted_table_name} AS
SELECT * FROM read_csv({', '.join(args)})"""
    return code, table_name, ["httpfs"] if scheme else []


@_generate_view_code.register(
    r"(?P<scheme>(?:file|https?)://)?(?P<path>.+?\.((?:c|t)sv|txt)(?:\.gz)?)",
    priority=12,
)
def _csv_file_or_url(_, path, table_name=None, **kwargs):
    return _csv(f"csv://{path}", path=path, table_name=table_name, **kwargs)


@_generate_view_code.register(
    r"(?P<scheme>(?:file|https?)://)?(?P<path>.+?\.parquet)", priority=12
)
def _parquet_file_or_url(_, path, table_name=None, **kwargs):
    return _parquet(f"parquet://{path}", path=path, table_name=table_name, **kwargs)


@_generate_view_code.register(r"s3://.+", priority=13)
def _s3(full_path, table_name=None):
    # TODO: gate this import once the ResultHandler mixin is merged #4454
    import pyarrow.dataset as ds

    dataset = ds.dataset(full_path)
    table_name = table_name or _name_from_dataset(dataset)
    return table_name, dataset, []


@_generate_view_code.register(r"postgres(ql)?://.+", priority=10)
def _postgres(uri, table_name=None):
    if table_name is None:
        raise ValueError("`table_name` is required when registering a postgres table")
    quoted_table_name = _quote(table_name)
    sql = (
        f"CREATE OR REPLACE VIEW {quoted_table_name} AS "
        f"SELECT * FROM postgres_scan_pushdown('{uri}', 'public', '{table_name}')"
    )
    return sql, table_name, ["postgres_scanner"]


@_generate_view_code.register(r".+", priority=1)
def _default(path, **kwargs):
    raise ValueError(
        f"""Unrecognized file type or extension: {path}.

Valid prefixes are parquet://, csv://, tsv://, s3://, or file://
Supported file extensions are parquet, csv, tsv, txt, csv.gz, tsv.gz, and txt.gz
    """
    )


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
        self._meta = sa.MetaData(bind=self.con)
        self._extensions = set()

    def _load_extensions(self, extensions):
        for extension in extensions:
            if extension not in self._extensions:
                with self.con.connect() as con:
                    con.execute(f"INSTALL '{extension}'")
                    con.execute(f"LOAD '{extension}'")
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
            The data source. May be a path to a file or directory of
            parquet/csv files, a pandas dataframe, or a pyarrow table or
            dataset.
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
        if isinstance(source, str) and source.startswith("s3://"):
            table_name, dataset, extensions_required = _generate_view_code(
                source, table_name=table_name
            )
            self._load_extensions(extensions_required)
            # We don't create a view since DuckDB special cases Arrow Datasets
            # so if we also create a view we end up with both a "lazy table"
            # and a view with the same name
            with self._safe_raw_sql("SELECT 1") as cursor:
                # DuckDB normally auto-detects Arrow Datasets that are defined
                # in local variables but the `dataset` variable won't be local
                # by the time we execute against this so we register it
                # explicitly.
                cursor.cursor.c.register(table_name, dataset)
        elif isinstance(source, (str, Path)):
            sql, table_name, extensions_required = _generate_view_code(
                str(source), table_name=table_name, **kwargs
            )
            self._load_extensions(extensions_required)
            self.con.execute(sql)
        else:
            if table_name is None:
                table_name = next(_gen_table_names)
            self.con.execute("register", (table_name, source))

        _table = self.table(table_name)
        with warnings.catch_warnings():
            # don't fail or warn if duckdb-engine fails to discover types
            # mostly (tinyint)
            warnings.filterwarnings(
                "ignore",
                message="Did not recognize type",
                category=sa.exc.SAWarning,
            )
            # We don't rely on index reflection, ignore this warning
            warnings.filterwarnings(
                "ignore",
                message="duckdb-engine doesn't yet support reflection on indices",
            )
            self.inspector.reflect_table(_table.op().sqla_table, _table.columns)
        return self.table(table_name)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> IbisRecordBatchReader:
        _ = self._import_pyarrow()
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        cursor = self.raw_sql(sql)

        _reader = cursor.cursor.fetch_record_batch(chunk_size=chunk_size)
        # Horrible hack to make sure cursor isn't garbage collected
        # before batches are streamed out of the RecordBatchReader
        batches = IbisRecordBatchReader(_reader, cursor)
        return batches
        # TODO: duckdb seems to not care about the `chunk_size` argument
        # and returns batches in 1024 row chunks

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        _ = self._import_pyarrow()
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()

        cursor = self.raw_sql(sql)
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
        for name, type, null in toolz.pluck(
            ["column_name", "column_type", "null"],
            self.con.execute(f"DESCRIBE {query}"),
        ):
            ibis_type = parse(type)
            yield name, ibis_type(nullable=null.lower() == "yes")

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a DuckDB SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

    def _register_in_memory_table(self, table_op):
        df = table_op.data.to_frame()
        self.con.execute("register", (table_op.name, df))

    def _get_sqla_table(
        self,
        name: str,
        schema: str | None = None,
        **kwargs: Any,
    ) -> sa.Table:
        with warnings.catch_warnings():
            # don't fail or warn if duckdb-engine fails to discover types
            warnings.filterwarnings(
                "ignore",
                message="Did not recognize type",
                category=sa.exc.SAWarning,
            )
            # We don't rely on index reflection, ignore this warning
            warnings.filterwarnings(
                "ignore",
                message="duckdb-engine doesn't yet support reflection on indices",
            )

            table = super()._get_sqla_table(name, schema, **kwargs)

        nulltype_cols = frozenset(
            col.name for col in table.c if isinstance(col.type, sa.types.NullType)
        )

        if not nulltype_cols:
            return table

        quoted_name = self.con.dialect.identifier_preparer.quote(name)

        for colname, type in self._metadata(quoted_name):
            if colname in nulltype_cols:
                # replace null types discovered by sqlalchemy with non null
                # types
                table.append_column(
                    sa.Column(
                        colname,
                        to_sqla_type(type),
                        nullable=type.nullable,
                    ),
                    replace_existing=True,
                )
        return table

    def _get_temp_view_definition(
        self,
        name: str,
        definition: sa.sql.compiler.Compiled,
    ) -> str:
        return f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"


class IbisRecordBatchReader(pa.ipc.RecordBatchReader):
    def __init__(self, reader, cursor):
        self.reader = reader
        self.cursor = cursor

    def close(self):
        self.reader.close()
        del self.cursor

    def read_all(self):
        return self.reader.read_all()

    def read_next_batch(self):
        return self.reader.read_next_batch()

    def read_pandas(self):
        return self.reader.read_pandas()

    def schema(self):
        return self.reader.schema
