"""DuckDB backend."""

from __future__ import annotations

import ast
import contextlib
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd
import pyarrow as pa
import sqlglot as sg
import toolz
from packaging.version import parse as vparse

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateSchema
from ibis.backends.base.sql import BaseBackend
from ibis.backends.base.sqlglot.datatypes import DuckDBType
from ibis.backends.duckdb.compiler import translate
from ibis.backends.duckdb.datatypes import DuckDBPandasData
from ibis.expr.operations.relations import PandasDataFrameProxy
from ibis.expr.operations.udf import InputType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence

    import torch

    from ibis.common.typing import SupportsSchema


def normalize_filenames(source_list):
    # Promote to list
    source_list = util.promote_list(source_list)

    return list(map(util.normalize_filename, source_list))


_UDF_INPUT_TYPE_MAPPING = {
    InputType.PYARROW: duckdb.functional.ARROW,
    InputType.PYTHON: duckdb.functional.NATIVE,
}


class DuckDBTable(ir.Table):
    """References a physical table in DuckDB."""

    @property
    def _client(self):
        return self.op().source

    @property
    def name(self):
        return self.op().name


class Backend(BaseBackend, CanCreateSchema):
    name = "duckdb"
    supports_create_or_replace = True

    def _define_udf_translation_rules(self, expr):
        # TODO:
        ...

    @property
    def current_database(self) -> str:
        return self.raw_sql("SELECT CURRENT_DATABASE()").arrow()[0][0].as_py()

    @property
    def current_schema(self) -> str:
        return self.raw_sql("SELECT CURRENT_SCHEMA()").arrow()[0][0].as_py()

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect="duckdb")
        return self.con.execute(query, **kwargs)

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
        if temp and overwrite:
            raise exc.IbisInputError("Cannot specify both temp and overwrite")

        if obj is None and schema is None:
            raise exc.IbisError("The schema or obj parameter is required")

        table_identifier = sg.to_identifier(name, quoted=True)
        create_expr = sg.expressions.Create(
            kind="TABLE",  # TABLE
            replace=overwrite,  # OR REPLACE
        )

        if temp:
            create_expr.args["properties"] = sg.expressions.Properties(
                expressions=[sg.expressions.TemporaryProperty()]  # TEMPORARY
            )

        if obj is not None and not isinstance(obj, ir.Expr):
            # pd.DataFrame or pa.Table
            obj = ibis.memtable(obj, schema=schema)
            self._register_in_memory_table(obj.op())
            create_expr.args["expression"] = self.compile(obj)  # AS ...
            create_expr.args["this"] = table_identifier  # t0
        elif obj is not None:
            self._register_in_memory_tables(obj)
            # If both `obj` and `schema` are specified, `obj` overrides `schema`
            # DuckDB doesn't support `create table (schema) AS select * ...`
            create_expr.args["expression"] = self.compile(obj)  # AS ...
            create_expr.args["this"] = table_identifier  # t0
        else:
            # Schema -> Table -> [ColumnDefs]
            schema_expr = sg.expressions.Schema(
                this=sg.expressions.Table(this=table_identifier),
                expressions=[
                    sg.expressions.ColumnDef(
                        this=sg.to_identifier(key, quoted=False),
                        kind=DuckDBType.from_ibis(typ),
                    )
                    if typ.nullable
                    else sg.expressions.ColumnDef(
                        this=sg.to_identifier(key, quoted=False),
                        kind=DuckDBType.from_ibis(typ),
                        constraints=[
                            sg.expressions.ColumnConstraint(
                                kind=sg.expressions.NotNullColumnConstraint()
                            )
                        ],
                    )
                    for key, typ in schema.items()
                ],
            )
            create_expr.args["this"] = schema_expr

        # create the table
        self.raw_sql(create_expr)

        return self.table(name, database=database)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        qualname = self._fully_qualified_name(name, database)
        replace = "OR REPLACE " * overwrite
        query = self.compile(obj)
        code = f"CREATE {replace}VIEW {qualname} AS {query}"
        self.raw_sql(code)

        return self.table(name, database=database)

    def drop_table(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        ident = self._fully_qualified_name(name, database)
        self.raw_sql(f"DROP TABLE {'IF EXISTS ' * force}{ident}")

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        name = self._fully_qualified_name(name, database)
        if_exists = "IF EXISTS " * force
        self.raw_sql(f"DROP VIEW {if_exists}{name}")

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(op.name)

    def list_schemas(self):
        out = self.raw_sql("SELECT current_schemas(True) as schemas").arrow()
        return list(set(out["schemas"].to_pylist()[0]))

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
        schema = self.get_schema(name, database=database)
        qname = self._fully_qualified_name(name, database)
        return DuckDBTable(ops.DatabaseTable(qname, schema, self))

    def _fully_qualified_name(self, name: str, database: str | None) -> str:
        return name
        # TODO: make this less bad
        # calls to here from `drop_table` already have `main` prepended to the table name
        # so what's the more robust way to deduplicate that identifier?
        db = database or self.current_database
        if name.startswith(db):
            # This is a hack to get around nested quoting of table name
            # e.g. '"main._ibis_temp_table_2"'
            return name
        return sg.table(name, db=db)  # .sql(dialect="duckdb")

    def get_schema(self, table_name: str, database: str | None = None) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)
        if isinstance(qualified_name, str):
            qualified_name = sg.expressions.Identifier(this=qualified_name, quoted=True)
        query = sg.expressions.Describe(this=qualified_name)
        results = self.raw_sql(query)
        names, types, nulls, *_ = results.fetch_arrow_table()
        names = names.to_pylist()
        types = types.to_pylist()
        # DuckDB gives back "YES", "NO" for nullability
        # TODO: remove code crime
        # nulls = [bool(null[:-2]) for null in nulls.to_pylist()]
        nulls = [null == "YES" for null in nulls.to_pylist()]
        return sch.Schema(
            dict(
                zip(
                    names,
                    (
                        DuckDBType.from_string(typ, nullable=null)
                        for typ, null in zip(types, nulls)
                    ),
                )
            )
        )

    def list_databases(self, like: str | None = None) -> list[str]:
        result = self.raw_sql("PRAGMA database_list;")
        results = result.fetch_arrow_table()

        if results:
            _, databases, *_ = results
            databases = databases.to_pylist()
        else:
            databases = []
        return self._filter_with_like(databases, like)

    def list_tables(self, like: str | None = None) -> list[str]:
        result = self.raw_sql("PRAGMA show_tables;")
        results = result.fetch_arrow_table()

        if results:
            tables, *_ = results
            tables = tables.to_pylist()
        else:
            tables = []
        return self._filter_with_like(tables, like)

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        from ibis.backends.duckdb.compiler.values import translate_val

        return translate_val.dispatch(operation) is not translate_val.dispatch(object)

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

        import duckdb

        self.con = duckdb.connect(str(database), config=config)

        # Load any pre-specified extensions
        if extensions is not None:
            self._load_extensions(extensions)

        # Default timezone
        self.con.execute("SET TimeZone = 'UTC'")
        # the progress bar in duckdb <0.8.0 causes kernel crashes in
        # jupyterlab, fixed in https://github.com/duckdb/duckdb/pull/6831
        if vparse(duckdb.__version__) < vparse("0.8.0"):
            self.con.execute("SET enable_progress_bar = false")

        self._record_batch_readers_consumed = {}

    def _from_url(self, url: str, **kwargs) -> BaseBackend:
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments

        Returns
        -------
        BaseBackend
            A backend instance
        """
        import sqlalchemy as sa

        url = sa.engine.make_url(url)

        kwargs = toolz.merge(
            {
                name: value
                for name in ("database", "read_only", "temp_directory")
                if (value := getattr(url, name, None))
            },
            kwargs,
        )

        kwargs.update(url.query)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    def compile(self, expr: ir.Expr, limit: str | None = None, params=None, **_: Any):
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = translate(table_expr.op(), params=params)
        assert not isinstance(sql, sg.exp.Subquery)

        if isinstance(sql, sg.exp.Table):
            sql = sg.select("*").from_(sql)

        assert not isinstance(sql, sg.exp.Subquery)
        return sql.sql(dialect="duckdb", pretty=True)

    def _to_sql(self, expr: ir.Expr, **kwargs) -> str:
        return str(self.compile(expr, **kwargs))

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        util.log(sql)

    def execute(
        self,
        expr: ir.Expr,
        limit: str | None = "default",
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, **kwargs)

        schema = table.schema()
        self._log(sql)

        try:
            result = self.con.execute(sql)
        except duckdb.CatalogException as e:
            raise exc.IbisError(e)

        # TODO: should we do this in arrow?
        # also what is pandas doing with dates?
        # 🡅 is because of https://github.com/duckdb/duckdb/issues/8539

        pandas_df = result.fetch_df()
        result = DuckDBPandasData.convert_table(pandas_df, schema)
        return expr.__pandas_result__(result)

    def load_extension(self, extension: str) -> None:
        """Install and load a duckdb extension by name or path.

        Parameters
        ----------
        extension
            The extension name or path.
        """
        self._load_extensions([extension])

    def _load_extensions(self, extensions):
        query = """
        WITH exts AS (
          SELECT extension_name AS name, aliases FROM duckdb_extensions()
          WHERE installed AND loaded
        )
        SELECT name FROM exts
        UNION (SELECT UNNEST(aliases) AS name FROM exts)
        """
        installed = (name for (name,) in self.con.sql(query).fetchall())
        # Install and load all other extensions
        todo = set(extensions).difference(installed)
        for extension in todo:
            self.con.install_extension(extension)
            self.con.load_extension(extension)

    def create_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot create a schema in another database."
            )

        name = sg.to_identifier(database, quoted=True)
        return sg.expressions.Create(
            this=name,
            kind="SCHEMA",
            replace=force,
        )

    def drop_schema(
        self, name: str, database: str | None = None, force: bool = False
    ) -> None:
        if database is not None:
            raise exc.UnsupportedOperationError(
                "DuckDB cannot drop a schema in another database."
            )

        name = sg.to_identifier(database, quoted=True)
        return sg.expressions.Drop(
            this=name,
            kind="SCHEMA",
            replace=force,
        )

    def sql(
        self,
        query: str,
        schema: SupportsSchema | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        query = self._transpile_sql(query, dialect=dialect)
        if schema is None:
            schema = self._get_schema_using_query(query)
        return ops.SQLQueryResult(query, ibis.schema(schema), self).to_expr()

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string."""
        return sch.Schema.from_tuples(self._metadata(query))

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

    def _compile_temp_view(self, table_name, source):
        return sg.expressions.Create(
            this=sg.expressions.Identifier(
                this=table_name, quoted=True
            ),  # CREATE ... 'table_name'
            kind="VIEW",  # VIEW
            replace=True,  # OR REPLACE
            properties=sg.expressions.Properties(
                expressions=[sg.expressions.TemporaryProperty()]  # TEMPORARY
            ),
            expression=source,  # AS ...
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
        if (version := vparse(self.version)) < vparse("0.7.0"):
            raise exc.IbisError(
                f"`read_json` requires duckdb >= 0.7.0, duckdb {version} is installed"
            )
        if not table_name:
            table_name = util.gen_name("read_json")

        options = [f"{key}={val}" for key, val in kwargs.items()]

        sg_view_expr = self._compile_temp_view(
            table_name,
            sg.select("*").from_(
                sg.func("read_json_auto", normalize_filenames(source_list), *options)
            ),
        )

        self.raw_sql(sg_view_expr)
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
            f'{key}="{val}"' if key != "columns" else f"{key}={val}"
            for key, val in kwargs.items()
        ]

        sg_view_expr = self._compile_temp_view(
            table_name,
            sg.select("*").from_(sg.func("read_csv", source_list, *options)),
        )

        self.raw_sql(sg_view_expr)
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

        if kwargs:
            options = [f"{key}={val}" for key, val in kwargs.items()]
            pq_func = sg.func("read_parquet", source_list, *options)
        else:
            pq_func = sg.func("read_parquet", source_list)

        sg_view_expr = self._compile_temp_view(
            table_name,
            sg.select("*").from_(pq_func),
        )

        self.raw_sql(sg_view_expr)

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

        sg_view_expr = self._compile_temp_view(
            table_name,
            sg.select("*").from_(
                sg.func("postgres_scan_pushdown", uri, schema, table_name)
            ),
        )
        self.raw_sql(sg_view_expr)

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
        ...     _ = con.execute("DROP TABLE IF EXISTS t")
        ...     _ = con.execute("CREATE TABLE t (a INT, b TEXT)")
        ...     _ = con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        ...
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

        sg_view_expr = self._compile_temp_view(
            table_name,
            sg.select("*").from_(
                sg.func(
                    "sqlite_scan", sg.to_identifier(str(path), quoted=True), table_name
                )
            ),
        )
        self.raw_sql(sg_view_expr)

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

        Examples
        --------
        >>> import ibis
        >>> import sqlite3
        >>> with sqlite3.connect("/tmp/attach_sqlite.db") as con:
        ...     _ = con.execute("DROP TABLE IF EXISTS t")
        ...     _ = con.execute("CREATE TABLE t (a INT, b TEXT)")
        ...     _ = con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        ...
        >>> con = ibis.connect("duckdb://")
        >>> con.list_tables()
        []
        >>> con.attach_sqlite("/tmp/attach_sqlite.db")
        >>> con.list_tables()
        ['t']
        """
        self.load_extension("sqlite")
        self.raw_sql(f"SET GLOBAL sqlite_all_varchar={all_varchar}")
        self.raw_sql(f"CALL sqlite_attach('{path}', overwrite={overwrite})")

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

        # handle the argument name change in duckdb 0.8.0
        fetch_record_batch = (
            (lambda cur: cur.fetch_record_batch(rows_per_batch=chunk_size))
            if vparse(duckdb.__version__) >= vparse("0.8.0")
            else (lambda cur: cur.fetch_record_batch(chunk_size=chunk_size))
        )

        def batch_producer(table):
            yield from fetch_record_batch(table)

        # TODO: check that this is still handled correctly
        # batch_producer keeps the `self.con` member alive long enough to
        # exhaust the record batch reader, even if the backend or connection
        # have gone out of scope in the caller
        table = self.raw_sql(sql)

        return pa.RecordBatchReader.from_batches(
            expr.as_table().schema().to_pyarrow(), batch_producer(table)
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

        table = self.raw_sql(sql).fetch_arrow_table()

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
        return self.raw_sql(compiled).torch()

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
        self.raw_sql(copy_cmd)

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
        self.raw_sql(copy_cmd)

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
        return DuckDBPandasData.convert_table(df, schema)

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        rows = self.raw_sql(f"DESCRIBE {query}").fetch_arrow_table()

        as_py = lambda val: val.as_py()
        for name, type, null in zip(
            map(as_py, rows["column_name"]),
            map(as_py, rows["column_type"]),
            map(as_py, rows["null"]),
        ):
            ibis_type = DuckDBType.from_string(type, nullable=null.lower() == "yes")
            yield name, ibis_type.copy(nullable=null.lower() == "yes")

    def _register_in_memory_tables(self, expr: ir.Expr) -> None:
        for memtable in expr.op().find(ops.InMemoryTable):
            self._register_in_memory_table(memtable)

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
                self.con.register(name, table)

            try:
                _register(name, table)
            except duckdb.NotImplementedException:
                _register(name, data.to_pyarrow(schema))

    def _get_temp_view_definition(self, name: str, definition) -> str:
        yield f"CREATE OR REPLACE TEMPORARY VIEW {name} AS {definition}"

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
            registration_func(con)

    def _compile_udf(self, udf_node: ops.ScalarUDF) -> None:
        func = udf_node.__func__
        name = func.__name__
        input_types = [DuckDBType.to_string(arg.dtype) for arg in udf_node.args]
        output_type = DuckDBType.to_string(udf_node.dtype)

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

    def _compile_pandas_udf(self, _: ops.ScalarUDF) -> None:
        raise NotImplementedError("duckdb doesn't support pandas UDFs")

    def _get_compiled_statement(self, view, definition):
        # TODO: remove this once duckdb supports CTAS prepared statements
        return super()._get_compiled_statement(
            view, definition, compile_kwargs={"literal_binds": True}
        )

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.
        overwrite
            If `True` then replace existing contents of table

        Raises
        ------
        NotImplementedError
            If inserting data from a different database
        ValueError
            If the type of `obj` isn't supported
        """
        con = self.con

        table = sg.table(table_name, db=database)

        if overwrite:
            con.execute(f"TRUNCATE TABLE {table.sql('duckdb')}")

        if isinstance(obj, ir.Table):
            query = sg.exp.insert(
                expression=self.compile(obj), into=table, dialect="duckdb"
            )
            con.execute(query.sql("duckdb"))
        elif isinstance(obj, pd.DataFrame):
            con.append(table_name, obj)
        else:
            con.append(table_name, pd.DataFrame(obj))
