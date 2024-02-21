from __future__ import annotations

import ast
import contextlib
import glob
from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlparse

import clickhouse_connect as cc
import pyarrow as pa
import pyarrow_hotfix  # noqa: F401
import sqlglot as sg
import sqlglot.expressions as sge
import toolz
from clickhouse_connect.driver.external import ExternalData

import ibis
import ibis.common.exceptions as com
import ibis.config
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import BaseBackend, CanCreateDatabase
from ibis.backends.clickhouse.compiler import ClickHouseCompiler
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compiler import C

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

    import pandas as pd


def _to_memtable(v):
    return ibis.memtable(v).op() if not isinstance(v, ops.InMemoryTable) else v


class Backend(SQLBackend, CanCreateDatabase):
    name = "clickhouse"
    compiler = ClickHouseCompiler()

    # ClickHouse itself does, but the client driver does not
    supports_temporary_tables = False

    class Options(ibis.config.Config):
        """Clickhouse options.

        Attributes
        ----------
        bool_type : str
            Type to use for boolean columns

        """

        bool_type: Literal["Bool", "UInt8", "Int8"] = "Bool"

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
        url = urlparse(url)
        database = url.path[1:]
        query_params = parse_qs(url.query)

        connect_args = {
            "user": url.username,
            "password": url.password or "",
            "host": url.hostname,
            "database": database or "",
        }

        for name, value in query_params.items():
            if len(value) > 1:
                connect_args[name] = value
            elif len(value) == 1:
                connect_args[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        kwargs.update(connect_args)
        self._convert_kwargs(kwargs)

        return self.connect(**kwargs)

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["secure"] = bool(ast.literal_eval(kwargs["secure"]))

    def do_connect(
        self,
        host: str = "localhost",
        port: int | None = None,
        database: str = "default",
        user: str = "default",
        password: str = "",
        client_name: str = "ibis",
        secure: bool | None = None,
        compression: str | bool = True,
        settings: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Create a ClickHouse client for use with Ibis.

        Parameters
        ----------
        host
            Host name of the clickhouse server
        port
            ClickHouse HTTP server's port. If not passed, the value depends on
            whether `secure` is `True` or `False`.
        database
            Default database when executing queries
        user
            User to authenticate with
        password
            Password to authenticate with
        client_name
            Name of client that will appear in clickhouse server logs
        secure
            Whether or not to use an authenticated endpoint
        compression
            The kind of compression to use for requests. See
            https://clickhouse.com/docs/en/integrations/python#compression for
            more information.
        settings
            ClickHouse session settings
        kwargs
            Client specific keyword arguments

        Examples
        --------
        >>> import ibis
        >>> client = ibis.clickhouse.connect()
        >>> client
        <ibis.clickhouse.client.ClickhouseClient object at 0x...>

        """
        if settings is None:
            settings = {}
        settings.setdefault("session_timezone", "UTC")

        self.con = cc.get_client(
            host=host,
            # 8123 is the default http port 443 is https
            port=port if port is not None else 443 if secure else 8123,
            database=database,
            user=user,
            password=password,
            client_name=client_name,
            query_limit=0,
            compress=compression,
            **kwargs,
        )

    @property
    def version(self) -> str:
        return self.con.server_version

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as result:
            yield result

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql(sg.select(self.compiler.f.currentDatabase())) as result:
            [(db,)] = result.result_rows
        return db

    def list_databases(self, like: str | None = None) -> list[str]:
        with self._safe_raw_sql(
            sg.select(C.name).from_(sg.table("databases", db="system"))
        ) as result:
            results = result.result_columns

        if results:
            (databases,) = results
        else:
            databases = []
        return self._filter_with_like(databases, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        query = sg.select(C.name).from_(sg.table("tables", db="system"))

        if database is None:
            database = self.compiler.f.currentDatabase()
        else:
            database = sge.convert(database)

        query = query.where(C.database.eq(database).or_(C.is_temporary))

        with self._safe_raw_sql(query) as result:
            results = result.result_columns

        if results:
            (tables,) = results
        else:
            tables = []
        return self._filter_with_like(tables, like)

    def _normalize_external_tables(self, external_tables=None) -> ExternalData | None:
        """Merge registered external tables with any new external tables."""
        external_data = ExternalData()
        n = 0
        type_mapper = self.compiler.type_mapper
        for name, obj in (external_tables or {}).items():
            n += 1
            if not (schema := obj.schema):
                raise TypeError(f"Schema is empty for external table {name}")

            structure = [
                f"{name} {type_mapper.to_string(typ.copy(nullable=not typ.is_nested()))}"
                for name, typ in schema.items()
            ]
            external_data.add_file(
                file_name=name,
                data=obj.data.to_pyarrow_bytes(schema=schema),
                structure=structure,
                fmt="Arrow",
            )
        if not n:
            return None
        return external_data

    def _collect_in_memory_tables(
        self, expr: ir.Table | None, external_tables: Mapping | None = None
    ):
        memtables = {op.name: op for op in expr.op().find(ops.InMemoryTable)}
        externals = toolz.valmap(_to_memtable, external_tables or {})
        return toolz.merge(memtables, externals)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        external_tables: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        # we convert to batches first to avoid a bunch of rigmarole resulting
        # from the following rough edges
        #
        # 1. clickhouse's awkward
        #    client-settings-are-permissioned-on-the-server "feature"
        # 2. the bizarre conversion of `DateTime64` without scale to arrow
        #    uint32 inside of clickhouse
        # 3. the fact that uint32 cannot be cast to pa.timestamp without first
        #    casting it to int64
        #
        # the extra code to make this dance work without first converting to
        # record batches isn't worth it without some benchmarking
        with self.to_pyarrow_batches(
            expr=expr,
            params=params,
            limit=limit,
            external_tables=external_tables,
            **kwargs,
        ) as reader:
            table = reader.read_all()

        return expr.__pyarrow_result__(table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Scalar, Any] | None = None,
        external_tables: Mapping[str, Any] | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return an iterator of pyarrow record batches.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        external_tables
            External data
        chunk_size
            Maximum number of row to return in a single chunk

        Returns
        -------
        results
            RecordBatchReader

        Notes
        -----
        There are a variety of ways to implement clickhouse -> record batches.

        1. FORMAT ArrowStream -> record batches via raw_query
           This has the same type conversion problem(s) as `to_pyarrow`.
           It's harder to address due to lack of `cast` on `RecordBatch`.
           However, this is a ClickHouse problem: we should be able to get
           string data out without a bunch of settings/permissions rigmarole.
        2. Native -> Python objects -> pyarrow batches
           This is what is implemented, using `query_column_block_stream`.
        3. Native -> Python objects -> DataFrame chunks -> pyarrow batches
           This is not implemented because it adds an unnecessary pandas step in
           between Python object -> arrow. We can go directly to record batches
           without pandas in the middle.

        """
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params)

        external_tables = self._collect_in_memory_tables(expr, external_tables)
        external_data = self._normalize_external_tables(external_tables)

        def batcher(sql: str, *, schema: pa.Schema) -> Iterator[pa.RecordBatch]:
            settings = {}

            # readonly != 1 means that the server setting is writable
            if self.con.server_settings["max_block_size"].readonly != 1:
                settings["max_block_size"] = chunk_size
            with self.con.query_column_block_stream(
                sql, external_data=external_data, settings=settings
            ) as blocks:
                yield from map(
                    partial(pa.RecordBatch.from_arrays, schema=schema), blocks
                )

        self._log(sql)
        schema = table.schema().to_pyarrow()
        return pa.RecordBatchReader.from_batches(schema, batcher(sql, schema=schema))

    def execute(
        self,
        expr: ir.Expr,
        limit: str | None = "default",
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""
        import pandas as pd

        table = expr.as_table()
        sql = self.compile(table, limit=limit, **kwargs)

        schema = table.schema()
        self._log(sql)

        external_tables = self._collect_in_memory_tables(expr, external_tables)
        external_data = self._normalize_external_tables(external_tables)
        df = self.con.query_df(
            sql, external_data=external_data, use_na_values=False, use_none=True
        )

        if df.empty:
            df = pd.DataFrame(columns=schema.names)
        else:
            df.columns = list(schema.names)

        # TODO: remove the extra conversion
        #
        # the extra __pandas_result__ call is to work around slight differences
        # in single column conversion and whole table conversion
        return expr.__pandas_result__(table.__pandas_result__(df))

    def insert(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table,
        settings: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        import pandas as pd
        import pyarrow as pa

        if overwrite:
            self.truncate_table(name)

        if isinstance(obj, pa.Table):
            return self.con.insert_arrow(name, obj, settings=settings, **kwargs)
        elif isinstance(obj, pd.DataFrame):
            return self.con.insert_arrow(
                name, pa.Table.from_pandas(obj), settings=settings, **kwargs
            )
        elif not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        query = sge.insert(self.compile(obj), into=name, dialect=self.name)

        external_tables = self._collect_in_memory_tables(obj, {})
        external_data = self._normalize_external_tables(external_tables)
        return self.con.command(query.sql(self.name), external_data=external_data)

    def raw_sql(
        self,
        query: str | sge.Expression,
        external_tables: Mapping[str, pd.DataFrame] | None = None,
        **kwargs,
    ) -> Any:
        """Execute a SQL string `query` against the database.

        Parameters
        ----------
        query
            Raw SQL string
        external_tables
            Mapping of table name to pandas DataFrames providing
            external datasources for the query
        kwargs
            Backend specific query arguments

        Returns
        -------
        Cursor
            Clickhouse cursor

        """
        external_tables = toolz.valmap(_to_memtable, external_tables or {})
        external_data = self._normalize_external_tables(external_tables)
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name, pretty=True)
        self._log(query)
        return self.con.query(query, external_data=external_data, **kwargs)

    def disconnect(self) -> None:
        """Close ClickHouse connection."""
        self.con.close()

    def get_schema(
        self, table_name: str, database: str | None = None, schema: str | None = None
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            May **not** be fully qualified. Use `database` if you want to
            qualify the identifier.
        database
            Database name
        schema
            Schema name, not supported by ClickHouse

        Returns
        -------
        sch.Schema
            Ibis schema

        """
        if schema is not None:
            raise com.UnsupportedBackendFeatureError(
                "`schema` namespaces are not supported by clickhouse"
            )
        query = sge.Describe(this=sg.table(table_name, db=database))
        with self._safe_raw_sql(query) as results:
            names, types, *_ = results.result_columns
        return sch.Schema(
            dict(zip(names, map(self.compiler.type_mapper.from_string, types)))
        )

    def _metadata(self, query: str) -> sch.Schema:
        name = util.gen_name("clickhouse_metadata")
        with closing(self.raw_sql(f"CREATE VIEW {name} AS {query}")):
            pass
        try:
            with closing(self.raw_sql(f"DESCRIBE {name}")) as results:
                names, types, *_ = results.result_columns
        finally:
            with closing(self.raw_sql(f"DROP VIEW {name}")):
                pass
        return zip(names, map(self.compiler.type_mapper.from_string, types))

    def create_database(
        self, name: str, *, force: bool = False, engine: str = "Atomic"
    ) -> None:
        src = sge.Create(
            this=sg.to_identifier(name),
            kind="DATABASE",
            exists=force,
            properties=sge.Properties(
                expressions=[sge.EngineProperty(this=sg.to_identifier(engine))]
            ),
        )
        with self._safe_raw_sql(src):
            pass

    def drop_database(self, name: str, *, force: bool = False) -> None:
        src = sge.Drop(this=sg.to_identifier(name), kind="DATABASE", exists=force)
        with self._safe_raw_sql(src):
            pass

    def truncate_table(self, name: str, database: str | None = None) -> None:
        ident = sg.table(name, db=database).sql(self.name)
        with self._safe_raw_sql(f"TRUNCATE TABLE {ident}"):
            pass

    def read_parquet(
        self,
        path: str | Path,
        table_name: str | None = None,
        engine: str = "MergeTree",
        **kwargs: Any,
    ) -> ir.Table:
        import pyarrow.dataset as ds
        from clickhouse_connect.driver.tools import insert_file

        from ibis.formats.pyarrow import PyArrowSchema

        paths = list(glob.glob(str(path)))
        schema = PyArrowSchema.to_ibis(ds.dataset(paths, format="parquet").schema)

        name = table_name or util.gen_name("read_parquet")
        table = self.create_table(name, engine=engine, schema=schema, temp=True)

        for file_path in paths:
            insert_file(
                client=self.con,
                table=name,
                file_path=file_path,
                fmt="Parquet",
                **kwargs,
            )
        return table

    def read_csv(
        self,
        path: str | Path,
        table_name: str | None = None,
        engine: str = "MergeTree",
        **kwargs: Any,
    ) -> ir.Table:
        import pyarrow.dataset as ds
        from clickhouse_connect.driver.tools import insert_file

        from ibis.formats.pyarrow import PyArrowSchema

        paths = list(glob.glob(str(path)))
        schema = PyArrowSchema.to_ibis(ds.dataset(paths, format="csv").schema)

        name = table_name or util.gen_name("read_csv")
        table = self.create_table(name, engine=engine, schema=schema, temp=True)

        for file_path in paths:
            insert_file(client=self.con, table=name, file_path=file_path, **kwargs)
        return table

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        # backend specific arguments
        engine: str = "MergeTree",
        order_by: Iterable[str] | None = None,
        partition_by: Iterable[str] | None = None,
        sample_by: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> ir.Table:
        """Create a table in a ClickHouse database.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            Optional data to create the table with
        schema
            Optional names and types of the table
        database
            Database to create the table in
        temp
            Create a temporary table. This is not yet supported, and exists for
            API compatibility.
        overwrite
            Whether to overwrite the table
        engine
            The table engine to use. See [ClickHouse's `CREATE TABLE` documentation](https://clickhouse.com/docs/en/sql-reference/statements/create/table)
            for specifics. Defaults to [`MergeTree`](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree)
            with `ORDER BY tuple()` because `MergeTree` is the most
            feature-complete engine.
        order_by
            String column names to order by. Required for some table engines like `MergeTree`.
        partition_by
            String column names to partition by
        sample_by
            String column names to sample by
        settings
            Key-value pairs of settings for table creation

        Returns
        -------
        Table
            The new table

        """
        if temp and overwrite:
            raise com.IbisInputError(
                "Cannot specify both `temp=True` and `overwrite=True` for ClickHouse"
            )

        if obj is None and schema is None:
            raise com.IbisError("The `schema` or `obj` parameter is required")

        if obj is not None and not isinstance(obj, ir.Expr):
            obj = ibis.memtable(obj, schema=schema)

        if schema is None:
            schema = obj.schema()

        this = sge.Schema(
            this=sg.table(name, db=database),
            expressions=[
                sge.ColumnDef(
                    this=sg.to_identifier(name),
                    kind=self.compiler.type_mapper.from_ibis(typ),
                )
                for name, typ in schema.items()
            ],
        )
        properties = [
            # the engine cannot be quoted, since clickhouse won't allow e.g.,
            # "File(Native)"
            sge.EngineProperty(this=sg.to_identifier(engine, quoted=False))
        ]

        if temp:
            properties.append(sge.TemporaryProperty())

        if order_by is not None or engine == "MergeTree":
            # engine == "MergeTree" requires an order by clause, which is the
            # empty tuple if order_by is False-y
            properties.append(
                sge.Order(
                    expressions=[
                        sge.Ordered(
                            this=sge.Tuple(
                                expressions=list(map(sg.column, order_by or ()))
                            )
                        )
                    ]
                )
            )

        if partition_by is not None:
            properties.append(
                sge.PartitionedByProperty(
                    this=sge.Schema(
                        expressions=list(map(sg.to_identifier, partition_by))
                    )
                )
            )

        if sample_by is not None:
            properties.append(
                sge.SampleProperty(
                    this=sge.Tuple(expressions=list(map(sg.column, sample_by)))
                )
            )

        if settings:
            properties.append(
                sge.SettingsProperty(
                    expressions=[
                        sge.SetItem(
                            this=sge.EQ(
                                this=sg.to_identifier(name),
                                expression=sge.convert(value),
                            )
                        )
                        for name, value in settings.items()
                    ]
                )
            )

        external_tables = {}
        expression = None

        if obj is not None:
            expression = self._to_sqlglot(obj)
            external_tables.update(self._collect_in_memory_tables(obj))

        code = sge.Create(
            this=this,
            kind="TABLE",
            replace=overwrite,
            expression=expression,
            properties=sge.Properties(expressions=properties),
        )

        external_data = self._normalize_external_tables(external_tables)

        # create the table
        sql = code.sql(self.name, pretty=True)
        self.con.raw_query(sql, external_data=external_data)

        return self.table(name, database=database)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        expression = self._to_sqlglot(obj)
        src = sge.Create(
            this=sg.table(name, db=database),
            kind="VIEW",
            replace=overwrite,
            expression=expression,
        )
        external_tables = self._collect_in_memory_tables(obj)
        with self._safe_raw_sql(src, external_tables=external_tables):
            pass
        return self.table(name, database=database)
